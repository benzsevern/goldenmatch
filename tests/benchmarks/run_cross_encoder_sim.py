"""Simulate Level 3 cross-encoder boost on Leipzig benchmarks.

Uses ground truth (with noise) to simulate LLM labels,
then trains cross-encoder with Ditto-style augmentation.
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import polars as pl

from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_fuzzy_matches
from goldenmatch.core.boost import _sample_initial_pairs
from goldenmatch.core.cross_encoder import (
    serialize_record, train_cross_encoder, score_pairs,
    augment_training_data, merge_scores,
)
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField,
    BlockingConfig, BlockingKeyConfig,
)

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_ground_truth(mapping_path, id_col_a, id_col_b):
    df = pl.read_csv(mapping_path)
    return {(str(row[id_col_a]).strip(), str(row[id_col_b]).strip()) for row in df.to_dicts()}


def run_pipeline_get_pairs(df_a, df_b, matchkeys, blocking, standardization=None):
    df_a = df_a.cast({col: pl.Utf8 for col in df_a.columns})
    df_b = df_b.cast({col: pl.Utf8 for col in df_b.columns})
    df_a = df_a.with_columns(pl.lit("source_a").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("source_b").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    combined, _ = auto_fix_dataframe(combined)
    if standardization:
        lf = combined.lazy()
        lf = apply_standardization(lf, standardization)
        combined = lf.collect()
    lf = combined.lazy()
    lf = compute_matchkeys(lf, matchkeys)
    combined = lf.collect()

    all_pairs = []
    for mk in matchkeys:
        if mk.type == "weighted" and blocking:
            blocks = build_blocks(combined.lazy(), blocking)
            for block in blocks:
                bdf = block.df.collect()
                p = find_fuzzy_matches(bdf, mk, pre_scored_pairs=block.pre_scored_pairs)
                all_pairs.extend(p)
    return all_pairs, combined


def simulate_labels(pairs, combined_df, gt_pairs, n_labels, noise_rate=0.05):
    row_to_source = {}
    row_to_id = {}
    for row in combined_df.select("__row_id__", "__source__", "id").to_dicts():
        row_to_source[row["__row_id__"]] = row["__source__"]
        row_to_id[row["__row_id__"]] = str(row["id"]).strip()

    indices = _sample_initial_pairs(pairs, n=min(n_labels, len(pairs)))
    if len(indices) < n_labels:
        rng = np.random.default_rng(42)
        remaining = [i for i in range(len(pairs)) if i not in set(indices)]
        extra = rng.choice(remaining, min(n_labels - len(indices), len(remaining)), replace=False).tolist()
        indices = indices + extra
    indices = indices[:n_labels]

    labels = []
    for idx in indices:
        a, b, _ = pairs[idx]
        src_a, src_b = row_to_source.get(a), row_to_source.get(b)
        id_a, id_b = row_to_id.get(a), row_to_id.get(b)
        if src_a == src_b:
            is_match = False
        elif src_a == "source_a":
            is_match = (id_a, id_b) in gt_pairs
        else:
            is_match = (id_b, id_a) in gt_pairs
        rng = np.random.default_rng(hash((a, b, 42)) % 2**32)
        if rng.random() < noise_rate:
            is_match = not is_match
        labels.append(is_match)
    return indices, labels


def evaluate(found_pairs, gt_pairs):
    tp = found_pairs & gt_pairs
    p = len(tp) / len(found_pairs) if found_pairs else 0.0
    r = len(tp) / len(gt_pairs) if gt_pairs else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def pairs_to_id_set(pairs, combined_df, threshold):
    row_to_source = {}
    row_to_id = {}
    for row in combined_df.select("__row_id__", "__source__", "id").to_dicts():
        row_to_source[row["__row_id__"]] = row["__source__"]
        row_to_id[row["__row_id__"]] = str(row["id"]).strip()
    found = set()
    for a, b, score in pairs:
        if score < threshold:
            continue
        src_a, src_b = row_to_source.get(a), row_to_source.get(b)
        if src_a != src_b:
            id_a, id_b = row_to_id.get(a), row_to_id.get(b)
            if src_a == "source_a":
                found.add((id_a, id_b))
            else:
                found.add((id_b, id_a))
    return found


def run_cross_encoder_sim(name, df_a, df_b, gt, text_col, extra_cols, standardization):
    print(f"\n{'='*70}")
    print(f"{name} — Level 3 Cross-Encoder Simulation")
    print(f"{'='*70}")

    all_cols = [text_col] + extra_cols
    fields = [MatchkeyField(column=text_col, transforms=["lowercase", "strip"], scorer="token_sort", weight=1.0)]

    pairs, combined = run_pipeline_get_pairs(df_a, df_b,
        matchkeys=[MatchkeyConfig(
            name="fuzzy", comparison="weighted", threshold=0.50,
            fields=fields,
        )],
        blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "substring:0:3"])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "substring:0:5"]),
                BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "soundex"]),
                BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "token_sort", "substring:0:8"]),
            ],
            max_block_size=500,
        ),
        standardization=standardization,
    )

    # Check blocking recall
    row_to_source = {}
    row_to_id = {}
    for row in combined.select("__row_id__", "__source__", "id").to_dicts():
        row_to_source[row["__row_id__"]] = row["__source__"]
        row_to_id[row["__row_id__"]] = str(row["id"]).strip()

    pair_id_set = set()
    for a, b, _ in pairs:
        src_a, src_b = row_to_source.get(a), row_to_source.get(b)
        if src_a != src_b:
            id_a, id_b = row_to_id.get(a), row_to_id.get(b)
            if src_a == "source_a":
                pair_id_set.add((id_a, id_b))
            else:
                pair_id_set.add((id_b, id_a))

    blocking_recall = len(pair_id_set & gt) / len(gt) if gt else 0
    print(f"\n  Candidate pairs: {len(pairs):,}")
    print(f"  Blocking recall: {blocking_recall:.1%} ({len(pair_id_set & gt)}/{len(gt)})")

    # Baseline
    for thresh in [0.50, 0.60, 0.70, 0.80]:
        found = pairs_to_id_set(pairs, combined, thresh)
        p, r, f1 = evaluate(found, gt)
        if f1 > 0.05:
            print(f"  Baseline (thresh={thresh}): P={p:.1%} R={r:.1%} F1={f1:.1%}")

    rows = combined.to_dicts()
    id_to_idx = {row["__row_id__"]: i for i, row in enumerate(rows)}
    matchable = [c for c in combined.columns if not c.startswith("__")]

    # Cross-encoder at different label counts
    print(f"\n  --- Cross-Encoder (Ditto-style) ---")
    print(f"  {'Labels':<8} {'Noise':<8} {'Aug':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Thresh':<8} {'Time'}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for n_labels in [300, 500, 800]:
        if n_labels > len(pairs):
            continue
        for noise in [0.0, 0.05]:
            indices, labels = simulate_labels(pairs, combined, gt, n_labels, noise)

            # Build serialized training pairs
            train_pairs = []
            for idx, label in zip(indices, labels):
                id_a, id_b, _ = pairs[idx]
                idx_a = id_to_idx.get(id_a)
                idx_b = id_to_idx.get(id_b)
                if idx_a is not None and idx_b is not None:
                    text_a = serialize_record(rows[idx_a], matchable)
                    text_b = serialize_record(rows[idx_b], matchable)
                    train_pairs.append((text_a, text_b, label))

            # Augment
            augmented = augment_training_data(train_pairs, n_augments=3)

            try:
                t0 = time.perf_counter()
                with tempfile.TemporaryDirectory() as tmpdir:
                    model = train_cross_encoder(
                        augmented,
                        epochs=4,  # reduced from 10
                        save_dir=Path(tmpdir) / "model",
                    )

                    # Score only uncertain pairs (0.3-0.9 from baseline scorer)
                    # Plus sample of high/low for calibration
                    all_text_pairs = []
                    pair_indices = []
                    for i, (a, b, score) in enumerate(pairs):
                        if 0.2 <= score <= 0.9:  # uncertain zone
                            idx_a = id_to_idx.get(a)
                            idx_b = id_to_idx.get(b)
                            if idx_a is not None and idx_b is not None:
                                text_a = serialize_record(rows[idx_a], matchable)
                                text_b = serialize_record(rows[idx_b], matchable)
                                all_text_pairs.append((text_a, text_b))
                                pair_indices.append(i)

                    cross_scores_subset = score_pairs(model, all_text_pairs)

                    # Build full score list: use cross-encoder for uncertain, baseline for rest
                    cross_scores = []
                    subset_idx = 0
                    for i, (a, b, score) in enumerate(pairs):
                        if subset_idx < len(pair_indices) and pair_indices[subset_idx] == i:
                            cross_scores.append(cross_scores_subset[subset_idx])
                            subset_idx += 1
                        else:
                            cross_scores.append(score)

                elapsed = time.perf_counter() - t0

                rescored = [(a, b, s) for (a, b, _), s in zip(pairs, cross_scores)]

                best_f1, best_thresh, best_p, best_r = 0, 0.5, 0, 0
                for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                    found = pairs_to_id_set(rescored, combined, thresh)
                    p, r, f1 = evaluate(found, gt)
                    if f1 > best_f1:
                        best_f1, best_thresh, best_p, best_r = f1, thresh, p, r

                noise_label = f"{noise:.0%}" if noise > 0 else "0%"
                print(f"  {n_labels:<8} {noise_label:<8} {len(augmented):<8} {best_p:<7.1%} {best_r:<7.1%} {best_f1:<7.1%} {best_thresh:<8} {elapsed:.0f}s")
            except Exception as e:
                print(f"  {n_labels:<8} — Failed: {e}")


def main():
    print("=" * 70)
    print("LEVEL 3 CROSS-ENCODER SIMULATION")
    print("Ditto-style: cross-attention + data augmentation")
    print("=" * 70)

    # Abt-Buy
    df_a = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Buy.csv", encoding="utf8-lossy")
    gt = load_ground_truth(DATASETS_DIR / "Abt-Buy" / "abt_buy_perfectMapping.csv", "idAbt", "idBuy")
    run_cross_encoder_sim("Abt-Buy", df_a, df_b, gt, "name", [], {"name": ["strip", "trim_whitespace"]})

    # Amazon-Google
    df_a = pl.read_csv(DATASETS_DIR / "Amazon-GoogleProducts" / "Amazon.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b = pl.read_csv(DATASETS_DIR / "Amazon-GoogleProducts" / "GoogleProducts.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b = df_b.rename({"name": "title"})
    gt = load_ground_truth(DATASETS_DIR / "Amazon-GoogleProducts" / "Amzon_GoogleProducts_perfectMapping.csv", "idAmazon", "idGoogleBase")
    run_cross_encoder_sim("Amazon-Google", df_a, df_b, gt, "title", ["manufacturer"], {"title": ["strip", "trim_whitespace"], "manufacturer": ["strip", "trim_whitespace"]})


if __name__ == "__main__":
    main()
