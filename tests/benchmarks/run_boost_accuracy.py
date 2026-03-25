"""Benchmark: LLM boost Level 2 (fine-tuning) + Level 3 (cross-encoder) on Abt-Buy.

Simulates the full boost pipeline using ground truth labels instead of LLM calls.
Tests the actual accuracy improvement from fine-tuning and cross-encoder reranking.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import polars as pl
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.boost import extract_feature_matrix, finetune_and_rescore
from goldenmatch.core.cross_encoder import (
    serialize_record, train_cross_encoder, score_pairs as ce_score_pairs,
    augment_training_data, merge_scores,
)
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_fuzzy_matches
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
)

DATASETS = Path(__file__).parent / "datasets"


def evaluate(found, gt, label=""):
    tp = found & gt
    prec = len(tp) / len(found) if found else 0
    rec = len(tp) / len(gt) if gt else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    if label:
        print(f"    {label}: P={prec:.1%} R={rec:.1%} F1={f1:.1%} ({len(found)} pairs)")
    return f1


def run_abt_buy():
    print("=" * 70)
    print("ABT-BUY: Level 2 (fine-tuning) + Level 3 (cross-encoder)")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS / "Abt-Buy/Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS / "Abt-Buy/Buy.csv", encoding="utf8-lossy")
    gt = set(
        (str(r["idAbt"]).strip(), str(r["idBuy"]).strip())
        for r in pl.read_csv(DATASETS / "Abt-Buy/abt_buy_perfectMapping.csv").to_dicts()
    )

    # Combine
    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns}).with_columns(pl.lit("a").alias("__source__"))
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns}).with_columns(pl.lit("b").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    combined, _ = auto_fix_dataframe(combined)

    ids = combined["id"].to_list()
    srcs = combined["__source__"].to_list()
    rows = combined.to_dicts()
    row_ids = combined["__row_id__"].to_list()
    id_to_idx = {rid: i for i, rid in enumerate(row_ids)}
    matchable = [c for c in combined.columns if not c.startswith("__")]

    # Generate candidate pairs with multi-pass blocking + token_sort
    print("\n  Step 1: Generating candidates (multi_pass + token_sort t=0.50)...")
    mk = MatchkeyConfig(name="ts", comparison="weighted", threshold=0.50, fields=[
        MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="token_sort", weight=1.0),
    ])
    blk = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "token_sort", "substring:0:5"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "first_token"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "last_token"]),
        ],
        max_block_size=500,
    )
    lf = compute_matchkeys(combined.lazy(), [mk])
    cdf = lf.collect()
    blocks = build_blocks(cdf.lazy(), blk)
    candidate_pairs = []
    for b in blocks:
        bdf = b.df.collect()
        candidate_pairs.extend(find_fuzzy_matches(bdf, mk, pre_scored_pairs=b.pre_scored_pairs))

    # Filter to cross-source only
    row_to_src = {r["__row_id__"]: r["__source__"] for r in cdf.select("__row_id__", "__source__").to_dicts()}
    row_to_id = {r["__row_id__"]: str(r["id"]).strip() for r in cdf.select("__row_id__", "id").to_dicts()}
    cross_pairs = [(a, b, s) for a, b, s in candidate_pairs if row_to_src.get(a) != row_to_src.get(b)]
    print(f"    {len(cross_pairs)} cross-source candidate pairs")

    # Baseline
    print("\n  Step 2: Baseline (token_sort scores)...")
    for t in [0.50, 0.60, 0.65, 0.70]:
        found = set()
        for a, b, s in cross_pairs:
            if s >= t:
                if row_to_src[a] == "a":
                    found.add((row_to_id[a], row_to_id[b]))
                else:
                    found.add((row_to_id[b], row_to_id[a]))
        evaluate(found, gt, f"token_sort t={t}")

    # Label pairs using ground truth (simulates LLM)
    print("\n  Step 3: Labeling 300 pairs using ground truth...")
    scores_arr = np.array([s for _, _, s in cross_pairs])

    # Stratified sampling
    buckets = [(0.90, 1.01, 40), (0.80, 0.90, 40), (0.70, 0.80, 50),
               (0.60, 0.70, 60), (0.50, 0.60, 60), (0.00, 0.50, 50)]
    labeled_indices = []
    for lo, hi, count in buckets:
        bidx = [i for i, s in enumerate(scores_arr) if lo <= s < hi]
        if bidx:
            np.random.seed(42 + len(labeled_indices))
            chosen = list(np.random.choice(bidx, size=min(count, len(bidx)), replace=False))
            labeled_indices.extend(chosen)
    labeled_indices = labeled_indices[:300]

    labels = []
    n_pos = 0
    for idx in labeled_indices:
        a, b, s = cross_pairs[idx]
        id_a = row_to_id[a]
        id_b = row_to_id[b]
        is_match = (id_a, id_b) in gt or (id_b, id_a) in gt
        labels.append(is_match)
        if is_match:
            n_pos += 1
    print(f"    {len(labels)} labeled: {n_pos} matches, {len(labels)-n_pos} non-matches")

    # Level 2: Fine-tune MiniLM bi-encoder
    print("\n  Step 4: Level 2 — Fine-tuning bi-encoder on 300 labels...")
    t0 = time.perf_counter()
    try:
        level2_pairs = finetune_and_rescore(
            cross_pairs, combined, matchable,
            labeled_indices, labels,
            base_model="all-MiniLM-L6-v2",
            epochs=3,
        )
        t_ft = time.perf_counter() - t0
        print(f"    Fine-tuning took {t_ft:.1f}s")

        print("\n    Level 2 results:")
        for t in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
            found = set()
            for a, b, s in level2_pairs:
                if s >= t:
                    if row_to_src[a] == "a":
                        found.add((row_to_id[a], row_to_id[b]))
                    else:
                        found.add((row_to_id[b], row_to_id[a]))
            evaluate(found, gt, f"Level 2 t={t}")
    except Exception as e:
        print(f"    Level 2 failed: {e}")
        level2_pairs = cross_pairs

    # Level 3: Cross-encoder on uncertain pairs
    print("\n  Step 5: Level 3 — Cross-encoder training + reranking...")
    t0 = time.perf_counter()
    try:
        # Build training data
        train_pairs = []
        for idx, label in zip(labeled_indices, labels):
            a, b, _ = cross_pairs[idx]
            idx_a = id_to_idx.get(a)
            idx_b = id_to_idx.get(b)
            if idx_a is not None and idx_b is not None:
                text_a = serialize_record(rows[idx_a], matchable)
                text_b = serialize_record(rows[idx_b], matchable)
                train_pairs.append((text_a, text_b, label))

        augmented = augment_training_data(train_pairs, n_augments=3)
        print(f"    Training cross-encoder on {len(augmented)} examples...")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cross_model = train_cross_encoder(augmented, epochs=10, save_dir=Path(tmpdir) / "model")

        # Score uncertain pairs (0.3-0.8 range in Level 2 scores)
        uncertain_texts = []
        uncertain_keys = []
        for a, b, s in level2_pairs:
            if 0.3 <= s <= 0.8:
                idx_a = id_to_idx.get(a)
                idx_b = id_to_idx.get(b)
                if idx_a is not None and idx_b is not None:
                    uncertain_texts.append((
                        serialize_record(rows[idx_a], matchable),
                        serialize_record(rows[idx_b], matchable),
                    ))
                    uncertain_keys.append((min(a, b), max(a, b)))

        print(f"    Scoring {len(uncertain_texts)} uncertain pairs with cross-encoder...")
        ce_scores = ce_score_pairs(cross_model, uncertain_texts)
        ce_map = {k: s for k, s in zip(uncertain_keys, ce_scores)}

        level3_pairs = merge_scores(level2_pairs, ce_map)
        t_ce = time.perf_counter() - t0
        print(f"    Cross-encoder took {t_ce:.1f}s")

        print("\n    Level 3 results:")
        for t in [0.40, 0.50, 0.60, 0.70, 0.80]:
            found = set()
            for a, b, s in level3_pairs:
                if s >= t:
                    if row_to_src[a] == "a":
                        found.add((row_to_id[a], row_to_id[b]))
                    else:
                        found.add((row_to_id[b], row_to_id[a]))
            evaluate(found, gt, f"Level 3 t={t}")
    except Exception as e:
        print(f"    Level 3 failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_abt_buy()
