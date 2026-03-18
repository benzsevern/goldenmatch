"""Simulate LLM boost at different label budgets using ground truth.

Simulates what an LLM would label by using ground truth with configurable
noise levels (LLMs aren't perfect). Measures F1 improvement at different
label counts: 50, 100, 200, 300, 500.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from goldenmatch.core.ingest import load_file
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.core.boost import extract_feature_matrix, _sample_initial_pairs, finetune_and_rescore
from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    BlockingConfig, BlockingKeyConfig, OutputConfig,
    GoldenRulesConfig,
)

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_ground_truth(mapping_path, id_col_a, id_col_b):
    df = pl.read_csv(mapping_path)
    pairs = set()
    for row in df.to_dicts():
        a = str(row[id_col_a]).strip()
        b = str(row[id_col_b]).strip()
        pairs.add((a, b))
    return pairs


def run_pipeline_get_pairs(df_a, df_b, matchkeys, blocking, standardization=None):
    """Run pipeline through scoring, return candidate pairs + combined df."""
    df_a = df_a.cast({col: pl.Utf8 for col in df_a.columns})
    df_b = df_b.cast({col: pl.Utf8 for col in df_b.columns})
    df_a = df_a.with_columns(pl.lit("source_a").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("source_b").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(
        pl.col("__row_id__").cast(pl.Int64)
    )
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
        if mk.type == "exact":
            pairs = find_exact_matches(combined.lazy(), mk)
            all_pairs.extend(pairs)
        elif mk.type == "weighted" and blocking:
            blocks = build_blocks(combined.lazy(), blocking)
            for block in blocks:
                bdf = block.df.collect()
                pairs = find_fuzzy_matches(bdf, mk, pre_scored_pairs=block.pre_scored_pairs)
                all_pairs.extend(pairs)

    return all_pairs, combined


def simulate_llm_labels(
    pairs, combined_df, gt_pairs, n_labels, noise_rate=0.05,
):
    """Simulate LLM labeling using ground truth with noise.

    noise_rate: probability of flipping the true label (simulates LLM errors).
    """
    row_to_source = {}
    row_to_id = {}
    for row in combined_df.select("__row_id__", "__source__", "id").to_dicts():
        row_to_source[row["__row_id__"]] = row["__source__"]
        row_to_id[row["__row_id__"]] = str(row["id"]).strip()

    # Sample pairs using the same strategy as boost.py
    indices = _sample_initial_pairs(pairs, n=min(n_labels, len(pairs)))
    if len(indices) < n_labels:
        # Add random indices
        rng = np.random.default_rng(42)
        remaining = [i for i in range(len(pairs)) if i not in set(indices)]
        extra = rng.choice(remaining, min(n_labels - len(indices), len(remaining)), replace=False).tolist()
        indices = indices + extra

    indices = indices[:n_labels]

    labels = []
    for idx in indices:
        a, b, _ = pairs[idx]
        src_a = row_to_source.get(a)
        src_b = row_to_source.get(b)
        id_a = row_to_id.get(a)
        id_b = row_to_id.get(b)

        if src_a == src_b:
            is_match = False
        else:
            if src_a == "source_a":
                is_match = (id_a, id_b) in gt_pairs
            else:
                is_match = (id_b, id_a) in gt_pairs

        # Add noise
        rng = np.random.default_rng(hash((a, b, 42)) % 2**32)
        if rng.random() < noise_rate:
            is_match = not is_match

        labels.append(is_match)

    return indices, labels


def train_and_rescore(
    pairs, combined_df, columns, indices, labels, include_embeddings=False,
):
    """Train classifier on labeled subset, re-score all pairs."""
    matchable = [c for c in columns if not c.startswith("__")]
    all_features = extract_feature_matrix(pairs, combined_df, matchable, include_embeddings=include_embeddings)

    X_train = np.array([all_features[i] for i in indices])
    y_train = np.array(labels, dtype=int)

    if len(set(y_train)) < 2:
        return pairs  # can't train

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)

    # Cross-val F1
    cv_folds = min(5, min(sum(y_train), sum(1 - y_train)))
    if cv_folds >= 2:
        cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv_folds, scoring="f1").mean()
    else:
        cv_f1 = 0.0

    probs = clf.predict_proba(all_features)[:, 1]
    rescored = [(a, b, float(prob)) for (a, b, _), prob in zip(pairs, probs)]

    return rescored, cv_f1


def evaluate(found_pairs, gt_pairs):
    tp = found_pairs & gt_pairs
    fp = found_pairs - gt_pairs
    fn = gt_pairs - found_pairs
    precision = len(tp) / len(found_pairs) if found_pairs else 0.0
    recall = len(tp) / len(gt_pairs) if gt_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def pairs_to_id_set(pairs, combined_df, threshold):
    """Convert scored pairs to (id_a, id_b) set for evaluation."""
    row_to_source = {}
    row_to_id = {}
    for row in combined_df.select("__row_id__", "__source__", "id").to_dicts():
        row_to_source[row["__row_id__"]] = row["__source__"]
        row_to_id[row["__row_id__"]] = str(row["id"]).strip()

    found = set()
    for a, b, score in pairs:
        if score < threshold:
            continue
        src_a = row_to_source.get(a)
        src_b = row_to_source.get(b)
        if src_a != src_b:
            id_a = row_to_id.get(a)
            id_b = row_to_id.get(b)
            if src_a == "source_a":
                found.add((id_a, id_b))
            else:
                found.add((id_b, id_a))
    return found


def run_boost_sim(dataset_name, df_a, df_b, gt, matchkeys, blocking, standardization, columns, include_embeddings=False, try_finetune=False):
    """Run boost simulation at different label budgets."""
    print(f"\n{'='*70}")
    print(f"{dataset_name} — LLM Boost Simulation")
    print(f"{'='*70}")

    # Get candidate pairs
    t0 = time.perf_counter()
    pairs, combined = run_pipeline_get_pairs(df_a, df_b, matchkeys, blocking, standardization)
    pipeline_time = time.perf_counter() - t0

    # Baseline (no boost)
    for thresh in [0.50, 0.60, 0.70, 0.80]:
        found = pairs_to_id_set(pairs, combined, thresh)
        p, r, f1 = evaluate(found, gt)
        if f1 > 0:
            print(f"\n  Baseline (threshold={thresh}): P={p:.1%} R={r:.1%} F1={f1:.1%} ({len(found)} pairs)")

    # Boost at different label counts
    label_counts = [50, 100, 200, 300, 500]
    noise_rates = [0.0, 0.05, 0.10]  # perfect LLM, good LLM, noisy LLM

    print(f"\n  {'Labels':<8} {'Noise':<8} {'CV F1':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Best Thresh'}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*11}")

    for n_labels in label_counts:
        if n_labels > len(pairs):
            continue

        for noise in noise_rates:
            indices, labels = simulate_llm_labels(pairs, combined, gt, n_labels, noise)
            rescored, cv_f1 = train_and_rescore(pairs, combined, combined.columns, indices, labels, include_embeddings=include_embeddings)

            # Find best threshold
            best_f1 = 0
            best_thresh = 0.5
            best_p = 0
            best_r = 0
            for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
                found = pairs_to_id_set(rescored, combined, thresh)
                p, r, f1 = evaluate(found, gt)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
                    best_p = p
                    best_r = r

            noise_label = f"{noise:.0%}" if noise > 0 else "0%"
            print(f"  {n_labels:<8} {noise_label:<8} {cv_f1:<7.1%} {best_p:<7.1%} {best_r:<7.1%} {best_f1:<7.1%} {best_thresh}")

    # Fine-tuning simulation
    if try_finetune:
        print(f"\n  --- Fine-Tuning Results ---")
        print(f"  {'Labels':<8} {'Noise':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Best Thresh'}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*11}")

        for n_labels in [100, 200, 300, 500]:
            if n_labels > len(pairs):
                continue
            for noise in [0.0, 0.05]:
                indices, labels = simulate_llm_labels(pairs, combined, gt, n_labels, noise)

                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        rescored = finetune_and_rescore(
                            pairs, combined, combined.columns,
                            indices, labels,
                            base_model="all-MiniLM-L6-v2",
                            epochs=3,
                            save_dir=Path(tmpdir) / "model",
                        )

                    best_f1 = 0
                    best_thresh = 0.5
                    best_p = 0
                    best_r = 0
                    for thresh in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
                        found = pairs_to_id_set(rescored, combined, thresh)
                        p, r, f1 = evaluate(found, gt)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thresh = thresh
                            best_p = p
                            best_r = r

                    noise_label = f"{noise:.0%}" if noise > 0 else "0%"
                    print(f"  {n_labels:<8} {noise_label:<8} {best_p:<7.1%} {best_r:<7.1%} {best_f1:<7.1%} {best_thresh}")
                except Exception as e:
                    print(f"  {n_labels:<8} — Fine-tuning failed: {e}")


def main():
    print("=" * 70)
    print("GOLDENMATCH — LLM Boost Simulation")
    print("Uses ground truth to simulate LLM labels at different budgets/noise")
    print("=" * 70)

    # DBLP-ACM
    df_a = pl.read_csv(DATASETS_DIR / "DBLP-ACM" / "DBLP2.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS_DIR / "DBLP-ACM" / "ACM.csv", encoding="utf8-lossy")
    gt = load_ground_truth(DATASETS_DIR / "DBLP-ACM" / "DBLP-ACM_perfectMapping.csv", "idDBLP", "idACM")

    run_boost_sim("DBLP-ACM", df_a, df_b, gt,
        matchkeys=[MatchkeyConfig(
            name="fuzzy", comparison="weighted", threshold=0.70,
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"], scorer="jaro_winkler", weight=0.5),
                MatchkeyField(column="authors", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.3),
                MatchkeyField(column="year", transforms=["strip"], scorer="exact", weight=0.2),
            ],
        )],
        blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"]),
                BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
                BlockingKeyConfig(fields=["authors"], transforms=["lowercase", "first_token"]),
            ],
            max_block_size=500,
        ),
        standardization={"title": ["strip", "trim_whitespace"], "authors": ["strip", "trim_whitespace"]},
        columns=["title", "authors", "year"],
        include_embeddings=False,
    )

    # Abt-Buy
    df_a = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Buy.csv", encoding="utf8-lossy")
    gt = load_ground_truth(DATASETS_DIR / "Abt-Buy" / "abt_buy_perfectMapping.csv", "idAbt", "idBuy")

    run_boost_sim("Abt-Buy", df_a, df_b, gt,
        matchkeys=[MatchkeyConfig(
            name="fuzzy", comparison="weighted", threshold=0.50,
            fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="token_sort", weight=1.0),
            ],
        )],
        blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"]),
                BlockingKeyConfig(fields=["name"], transforms=["lowercase", "soundex"]),
                BlockingKeyConfig(fields=["name"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            ],
            max_block_size=500,
        ),
        standardization={"name": ["strip", "trim_whitespace"]},
        columns=["name"],
        include_embeddings=True,
        try_finetune=True,
    )

    # Amazon-Google
    df_a = pl.read_csv(DATASETS_DIR / "Amazon-GoogleProducts" / "Amazon.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b = pl.read_csv(DATASETS_DIR / "Amazon-GoogleProducts" / "GoogleProducts.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b = df_b.rename({"name": "title"})
    gt = load_ground_truth(DATASETS_DIR / "Amazon-GoogleProducts" / "Amzon_GoogleProducts_perfectMapping.csv", "idAmazon", "idGoogleBase")

    run_boost_sim("Amazon-Google", df_a, df_b, gt,
        matchkeys=[MatchkeyConfig(
            name="fuzzy", comparison="weighted", threshold=0.50,
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.7),
                MatchkeyField(column="manufacturer", transforms=["lowercase", "strip"], scorer="jaro_winkler", weight=0.3),
            ],
        )],
        blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"]),
                BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
                BlockingKeyConfig(fields=["manufacturer"], transforms=["lowercase", "substring:0:3"]),
            ],
            max_block_size=500,
        ),
        standardization={"title": ["strip", "trim_whitespace"], "manufacturer": ["strip", "trim_whitespace"]},
        columns=["title", "manufacturer"],
        include_embeddings=True,
        try_finetune=True,
    )


if __name__ == "__main__":
    main()
