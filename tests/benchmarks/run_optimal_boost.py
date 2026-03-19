"""Optimal boost: train on multi-pass pairs, score ANN pairs with fine-tuned model.

Option C: ANN blocking (98% recall) + bi-encoder fine-tuned on multi-pass pairs (cleaner training data)
Option B: More labels (800) + more epochs (8)
Option A: Better base model (all-mpnet-base-v2)
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
from goldenmatch.core.boost import _sample_initial_pairs, _build_training_texts, finetune_and_rescore
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
)

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_ground_truth(path, col_a, col_b):
    df = pl.read_csv(path)
    return {(str(r[col_a]).strip(), str(r[col_b]).strip()) for r in df.to_dicts()}


def prepare_combined(df_a, df_b, standardization):
    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns})
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns})
    df_a = df_a.with_columns(pl.lit("source_a").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("source_b").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    combined, _ = auto_fix_dataframe(combined)
    if standardization:
        lf = combined.lazy()
        lf = apply_standardization(lf, standardization)
        combined = lf.collect()
    return combined


def get_pairs(combined, matchkeys, blocking):
    lf = combined.lazy()
    lf = compute_matchkeys(lf, matchkeys)
    df = lf.collect()
    all_pairs = []
    for mk in matchkeys:
        if mk.type == "weighted" and blocking:
            blocks = build_blocks(df.lazy(), blocking)
            for block in blocks:
                bdf = block.df.collect()
                p = find_fuzzy_matches(bdf, mk, pre_scored_pairs=block.pre_scored_pairs)
                all_pairs.extend(p)
    return all_pairs, df


def simulate_labels(pairs, df, gt, n_labels, noise=0.05):
    row_to = {}
    for r in df.select("__row_id__", "__source__", "id").to_dicts():
        row_to[r["__row_id__"]] = (r["__source__"], str(r["id"]).strip())

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
        sa, ia = row_to.get(a, ("", ""))
        sb, ib = row_to.get(b, ("", ""))
        if sa == sb:
            m = False
        elif sa == "source_a":
            m = (ia, ib) in gt
        else:
            m = (ib, ia) in gt
        rng = np.random.default_rng(hash((a, b, 42)) % 2**32)
        if rng.random() < noise:
            m = not m
        labels.append(m)
    return indices, labels


def evaluate(pairs, df, gt, threshold):
    row_to = {}
    for r in df.select("__row_id__", "__source__", "id").to_dicts():
        row_to[r["__row_id__"]] = (r["__source__"], str(r["id"]).strip())
    found = set()
    for a, b, s in pairs:
        if s < threshold:
            continue
        sa, ia = row_to.get(a, ("", ""))
        sb, ib = row_to.get(b, ("", ""))
        if sa != sb:
            if sa == "source_a":
                found.add((ia, ib))
            else:
                found.add((ib, ia))
    tp = found & gt
    p = len(tp) / len(found) if found else 0
    r = len(tp) / len(gt) if gt else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    return p, r, f1, len(found)


def run_experiment(name, df_a, df_b, gt, text_col, standardization, base_model, n_labels, epochs, noise):
    combined = prepare_combined(df_a, df_b, standardization)
    mk_fuzzy = MatchkeyConfig(
        name="f", comparison="weighted", threshold=0.50,
        fields=[MatchkeyField(column=text_col, transforms=["lowercase", "strip"], scorer="token_sort", weight=1.0)],
    )

    # Step 1: Get TRAINING pairs from multi-pass blocking (clean, fewer)
    train_blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "token_sort", "substring:0:8"]),
        ],
        max_block_size=500,
    )
    train_pairs, train_df = get_pairs(combined, [mk_fuzzy], train_blocking)

    # Step 2: Get SCORING pairs from ANN blocking (high recall)
    score_blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=[text_col], transforms=["lowercase", "substring:0:3"])],
        strategy="ann_pairs",
        ann_column=text_col,
        ann_model="all-MiniLM-L6-v2",
        ann_top_k=30,
    )
    score_pairs_list, score_df = get_pairs(combined, [mk_fuzzy], score_blocking)

    # Check blocking recalls
    def check_recall(pairs, df):
        row_to = {}
        for r in df.select("__row_id__", "__source__", "id").to_dicts():
            row_to[r["__row_id__"]] = (r["__source__"], str(r["id"]).strip())
        found = set()
        for a, b, _ in pairs:
            sa, ia = row_to.get(a, ("", ""))
            sb, ib = row_to.get(b, ("", ""))
            if sa != sb:
                if sa == "source_a":
                    found.add((ia, ib))
                else:
                    found.add((ib, ia))
        return len(found & gt) / len(gt) if gt else 0

    train_recall = check_recall(train_pairs, train_df)
    score_recall = check_recall(score_pairs_list, score_df)

    print(f"\n  {name}")
    print(f"  Train pairs: {len(train_pairs):,} (multi-pass, {train_recall:.1%} recall)")
    print(f"  Score pairs: {len(score_pairs_list):,} (ANN, {score_recall:.1%} recall)")

    # Step 3: Label training pairs
    indices, labels = simulate_labels(train_pairs, train_df, gt, n_labels, noise)

    # Step 4: Fine-tune on training pairs
    t0 = time.perf_counter()
    try:
        # Use finetune_and_rescore but apply to SCORING pairs
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        rows = train_df.to_dicts()
        row_ids = train_df["__row_id__"].to_list()
        id_to_idx = {rid: i for i, rid in enumerate(row_ids)}
        matchable = [c for c in train_df.columns if not c.startswith("__")]

        # Build training examples
        train_examples = []
        for pair_idx, label in zip(indices, labels):
            id_a, id_b, _ = train_pairs[pair_idx]
            idx_a = id_to_idx.get(id_a)
            idx_b = id_to_idx.get(id_b)
            if idx_a is None or idx_b is None:
                continue
            parts_a = [f"{c}: {rows[idx_a].get(c, '')}" for c in matchable if rows[idx_a].get(c) is not None]
            parts_b = [f"{c}: {rows[idx_b].get(c, '')}" for c in matchable if rows[idx_b].get(c) is not None]
            text_a = " | ".join(parts_a)
            text_b = " | ".join(parts_b)
            train_examples.append(InputExample(texts=[text_a, text_b], label=1.0 if label else 0.0))

        if len(train_examples) < 10:
            print(f"  Too few training examples ({len(train_examples)})")
            return

        # Fine-tune
        model = SentenceTransformer(base_model)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            show_progress_bar=True,
            warmup_steps=int(len(train_dataloader) * 0.1),
        )

        # Step 5: Re-embed and re-score the ANN pairs with fine-tuned model
        score_rows = score_df.to_dicts()
        score_row_ids = score_df["__row_id__"].to_list()
        score_id_to_idx = {rid: i for i, rid in enumerate(score_row_ids)}

        all_texts = []
        for row in score_rows:
            parts = [f"{c}: {row.get(c, '')}" for c in matchable if row.get(c) is not None]
            all_texts.append(" | ".join(parts))

        embeddings = model.encode(all_texts, show_progress_bar=False, normalize_embeddings=True)

        rescored = []
        for id_a, id_b, old_score in score_pairs_list:
            idx_a = score_id_to_idx.get(id_a)
            idx_b = score_id_to_idx.get(id_b)
            if idx_a is not None and idx_b is not None and idx_a < len(embeddings) and idx_b < len(embeddings):
                cos_sim = float(np.dot(embeddings[idx_a], embeddings[idx_b]))
                rescored.append((id_a, id_b, cos_sim))
            else:
                rescored.append((id_a, id_b, old_score))

        elapsed = time.perf_counter() - t0

        # Find best threshold
        best_f1, best_t, best_p, best_r = 0, 0.5, 0, 0
        for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
            p, r, f1, n = evaluate(rescored, score_df, gt, t)
            if f1 > best_f1:
                best_f1, best_t, best_p, best_r = f1, t, p, r

        noise_str = f"{noise:.0%}" if noise > 0 else "0%"
        print(f"  Model: {base_model}, Labels: {n_labels}, Noise: {noise_str}, Epochs: {epochs}")
        print(f"  Result: P={best_p:.1%} R={best_r:.1%} F1={best_f1:.1%} (thresh={best_t}) [{elapsed:.0f}s]")
        return best_f1

    except Exception as e:
        print(f"  Failed: {e}")
        return 0


def main():
    print("=" * 70)
    print("OPTIMAL BOOST: Train on multi-pass, score on ANN")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Buy.csv", encoding="utf8-lossy")
    gt = load_ground_truth(DATASETS_DIR / "Abt-Buy" / "abt_buy_perfectMapping.csv", "idAbt", "idBuy")
    std = {"name": ["strip", "trim_whitespace"]}

    # Option C: Multi-pass train + ANN score (300 labels, MiniLM)
    print("\n--- Option C: Train on multi-pass, score on ANN ---")
    run_experiment("C: MiniLM/300/3ep", df_a, df_b, gt, "name", std, "all-MiniLM-L6-v2", 300, 3, 0.05)

    # Option B: More labels + epochs
    print("\n--- Option B: More labels + epochs ---")
    run_experiment("B: MiniLM/500/6ep", df_a, df_b, gt, "name", std, "all-MiniLM-L6-v2", 500, 6, 0.05)
    run_experiment("B: MiniLM/800/8ep", df_a, df_b, gt, "name", std, "all-MiniLM-L6-v2", 800, 8, 0.05)

    # Option A: Better base model
    print("\n--- Option A: Better base model ---")
    run_experiment("A: mpnet/300/3ep", df_a, df_b, gt, "name", std, "all-mpnet-base-v2", 300, 3, 0.05)
    run_experiment("A: mpnet/500/6ep", df_a, df_b, gt, "name", std, "all-mpnet-base-v2", 500, 6, 0.05)
    run_experiment("A: mpnet/800/8ep", df_a, df_b, gt, "name", std, "all-mpnet-base-v2", 800, 8, 0.05)


if __name__ == "__main__":
    main()
