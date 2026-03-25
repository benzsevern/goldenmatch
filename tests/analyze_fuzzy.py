"""Analyze fuzzy matching accuracy on 1M synthetic data.

Uses last_name (jaro_winkler) + zip (exact) with blocking on last_name[:3].
This is the hard case — typos in names, different zips, threshold tuning.
"""

import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    BlockingConfig, BlockingKeyConfig, OutputConfig,
    GoldenRulesConfig, GoldenFieldRule, StandardizationConfig,
    ValidationConfig,
)
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.core.cluster import build_clusters


def run_fuzzy_analysis(threshold: float, sample_size: int | None = None):
    data_path = Path("D:/show_case/goldenmatch/tests/fixtures/synthetic_1m.csv")

    # Config: fuzzy name + zip matching with blocking
    cfg = GoldenMatchConfig(
        matchkeys=[
            # Keep exact email as baseline
            MatchkeyConfig(
                name="email_exact",
                fields=[MatchkeyField(column="email", transforms=["lowercase", "strip"])],
                comparison="exact",
            ),
            # Fuzzy: last_name (jaro_winkler) + first_name (jaro_winkler) + zip (exact)
            MatchkeyConfig(
                name="fuzzy_name_zip",
                fields=[
                    MatchkeyField(column="last_name", transforms=["lowercase", "strip"],
                                  scorer="jaro_winkler", weight=0.4),
                    MatchkeyField(column="first_name", transforms=["lowercase", "strip"],
                                  scorer="jaro_winkler", weight=0.3),
                    MatchkeyField(column="zip", transforms=["strip"],
                                  scorer="exact", weight=0.3),
                ],
                comparison="weighted",
                threshold=threshold,
            ),
        ],
        blocking=BlockingConfig(
            keys=[
                BlockingKeyConfig(fields=["last_name"], transforms=["lowercase", "substring:0:3"]),
            ],
            max_block_size=5000,
            skip_oversized=True,
        ),
        standardization=StandardizationConfig(rules={
            "email": ["email"],
            "phone": ["phone"],
            "first_name": ["strip", "name_proper"],
            "last_name": ["strip", "name_proper"],
            "zip": ["zip5"],
        }),
        validation=ValidationConfig(auto_fix=True),
        output=OutputConfig(format="csv", directory=".", run_name="fuzzy_test"),
        golden_rules=GoldenRulesConfig(default=GoldenFieldRule(strategy="most_complete")),
    )

    matchkeys = cfg.get_matchkeys()

    # Load
    print(f"\n{'='*70}")
    print(f"FUZZY MATCHING ANALYSIS — threshold={threshold}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    df = pl.read_csv(data_path, infer_schema_length=10000, ignore_errors=True)
    df = df.with_columns(pl.lit("bench").alias("__source__"))
    df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))

    if sample_size and sample_size < df.height:
        df = df.head(sample_size)
        print(f"Using sample of {sample_size:,} records")

    # Auto-fix
    df, _ = auto_fix_dataframe(df)

    # Standardize
    lf = df.lazy()
    lf = apply_standardization(lf, cfg.standardization.rules)

    # Matchkeys
    lf = compute_matchkeys(lf, matchkeys)
    df = lf.collect()

    # Exact matching
    print("\nRunning exact email matching...")
    t1 = time.perf_counter()
    exact_pairs = []
    for mk in matchkeys:
        if mk.type == "exact":
            pairs = find_exact_matches(df.lazy(), mk)
            exact_pairs.extend(pairs)
    t2 = time.perf_counter()
    print(f"  Exact pairs: {len(exact_pairs):,} ({t2-t1:.2f}s)")

    # Fuzzy matching
    print("Running fuzzy name+zip matching...")
    fuzzy_pairs = []
    fuzzy_scores = []
    for mk in matchkeys:
        if mk.type == "weighted":
            blocks = build_blocks(df.lazy(), cfg.blocking)
            print(f"  Blocks created: {len(blocks):,}")
            block_sizes = [b.df.collect().height for b in blocks[:100]]
            if block_sizes:
                print(f"  Avg block size (sample): {sum(block_sizes)/len(block_sizes):.1f}")

            for i, block in enumerate(blocks):
                block_df = block.df.collect()
                pairs = find_fuzzy_matches(block_df, mk)
                for a, b, s in pairs:
                    fuzzy_pairs.append((a, b, s))
                    fuzzy_scores.append(s)
                if (i + 1) % 5000 == 0:
                    print(f"    Processed {i+1:,}/{len(blocks):,} blocks, {len(fuzzy_pairs):,} pairs so far...")

    t3 = time.perf_counter()
    print(f"  Fuzzy pairs: {len(fuzzy_pairs):,} ({t3-t2:.2f}s)")

    # Combine all pairs
    all_pairs = exact_pairs + fuzzy_pairs
    # Deduplicate pairs (same pair may match on both exact and fuzzy)
    pair_set = {}
    for a, b, s in all_pairs:
        key = (min(a, b), max(a, b))
        if key not in pair_set or s > pair_set[key]:
            pair_set[key] = s
    deduped_pairs = [(a, b, s) for (a, b), s in pair_set.items()]

    print(f"\n  Combined unique pairs: {len(deduped_pairs):,}")
    print(f"    From exact only: {len(exact_pairs):,}")
    print(f"    From fuzzy only: {len(fuzzy_pairs):,}")
    overlap = len(exact_pairs) + len(fuzzy_pairs) - len(deduped_pairs)
    print(f"    Overlap (found by both): {overlap:,}")

    # Cluster
    print("\nClustering...")
    all_ids = df["__row_id__"].to_list()
    clusters = build_clusters(deduped_pairs, all_ids, max_cluster_size=100)
    multi = {k: v for k, v in clusters.items() if v["size"] > 1}
    t4 = time.perf_counter()
    print(f"  Clusters: {len(multi):,} ({t4-t3:.2f}s)")

    # Cluster size distribution
    sizes = [v["size"] for v in multi.values()]
    size_dist = Counter(sizes)
    print(f"\n  Cluster size distribution:")
    for size in sorted(size_dist.keys()):
        print(f"    Size {size}: {size_dist[size]:,}")

    # Score distribution for fuzzy pairs
    if fuzzy_scores:
        print(f"\n  Fuzzy score distribution:")
        buckets = [0] * 10
        for s in fuzzy_scores:
            idx = min(int((s - threshold) / ((1.0 - threshold) / 10)), 9)
            idx = max(0, idx)
            buckets[idx] += 1
        step = (1.0 - threshold) / 10
        for i, count in enumerate(buckets):
            lo = threshold + i * step
            hi = lo + step
            bar = "#" * min(int(count / max(max(buckets), 1) * 40), 40)
            print(f"    {lo:.2f}-{hi:.2f}: {count:>6,} {bar}")

    # ── Accuracy: Check using email as ground truth ──
    print(f"\n{'='*70}")
    print("ACCURACY (using email as ground truth)")
    print(f"{'='*70}")

    # True duplicates: records sharing the same standardized email
    mk_email_col = "__mk_email_exact__"
    if mk_email_col not in df.columns:
        print("  Cannot compute accuracy — no email matchkey column")
        return

    # Build ground truth: which row_ids should be clustered together?
    email_groups = {}
    for row in df.select("__row_id__", mk_email_col).to_dicts():
        email = row[mk_email_col]
        rid = row["__row_id__"]
        if email is not None:
            email_groups.setdefault(email, []).append(rid)

    true_clusters = {email: ids for email, ids in email_groups.items() if len(ids) > 1}
    true_pairs = set()
    for ids in true_clusters.values():
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                true_pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))

    found_pairs = set()
    for a, b, s in deduped_pairs:
        found_pairs.add((min(a, b), max(a, b)))

    true_positives = found_pairs & true_pairs
    false_positives = found_pairs - true_pairs
    false_negatives = true_pairs - found_pairs

    precision = len(true_positives) / len(found_pairs) if found_pairs else 0
    recall = len(true_positives) / len(true_pairs) if true_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  True duplicate pairs (ground truth): {len(true_pairs):,}")
    print(f"  Found pairs:                         {len(found_pairs):,}")
    print(f"  True positives:                      {len(true_positives):,}")
    print(f"  False positives:                     {len(false_positives):,}")
    print(f"  False negatives (missed):            {len(false_negatives):,}")
    print(f"\n  Precision: {precision:.4%}")
    print(f"  Recall:    {recall:.4%}")
    print(f"  F1 Score:  {f1:.4%}")

    # Sample false positives
    if false_positives:
        print(f"\n  Sample FALSE POSITIVES (first 10):")
        for a, b in list(false_positives)[:10]:
            row_a = df.filter(pl.col("__row_id__") == a).select(
                ["id", "first_name", "last_name", "email", "zip"]
            ).to_dicts()[0]
            row_b = df.filter(pl.col("__row_id__") == b).select(
                ["id", "first_name", "last_name", "email", "zip"]
            ).to_dicts()[0]
            print(f"    Pair ({a}, {b}):")
            print(f"      A: {row_a}")
            print(f"      B: {row_b}")

    # Sample false negatives
    if false_negatives:
        print(f"\n  Sample FALSE NEGATIVES (first 10):")
        for a, b in list(false_negatives)[:10]:
            row_a = df.filter(pl.col("__row_id__") == a).select(
                ["id", "first_name", "last_name", "email", "zip"]
            ).to_dicts()[0]
            row_b = df.filter(pl.col("__row_id__") == b).select(
                ["id", "first_name", "last_name", "email", "zip"]
            ).to_dicts()[0]
            print(f"    Pair ({a}, {b}):")
            print(f"      A: {row_a}")
            print(f"      B: {row_b}")

    total_time = time.perf_counter() - t0
    print(f"\n  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    # Run on 100K sample first (fuzzy is much slower than exact)
    print("=" * 70)
    print("PHASE 1: 100K sample at threshold=0.85")
    print("=" * 70)
    run_fuzzy_analysis(threshold=0.85, sample_size=100_000)

    print("\n\n")
    print("=" * 70)
    print("PHASE 2: 100K sample at threshold=0.75 (looser)")
    print("=" * 70)
    run_fuzzy_analysis(threshold=0.75, sample_size=100_000)

    print("\n\n")
    print("=" * 70)
    print("PHASE 3: 100K sample at threshold=0.95 (stricter)")
    print("=" * 70)
    run_fuzzy_analysis(threshold=0.95, sample_size=100_000)
