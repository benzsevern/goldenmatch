"""Analyze 1M benchmark results for matching accuracy."""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from goldenmatch.config.loader import load_config
from goldenmatch.core.ingest import load_file
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.scorer import find_exact_matches
from goldenmatch.core.cluster import build_clusters


def main():
    data_path = Path("D:/show_case/goldenmatch/tests/fixtures/synthetic_1m.csv")
    config_path = Path("D:/show_case/goldenmatch/tests/fixtures/bench_config.yaml")

    cfg = load_config(config_path)
    matchkeys = cfg.get_matchkeys()

    # Load and process
    print("Loading and processing...")
    df = pl.read_csv(data_path, infer_schema_length=10000, ignore_errors=True)
    df = df.with_columns(pl.lit("bench").alias("__source__"))
    df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))

    df, _ = auto_fix_dataframe(df)
    lf = df.lazy()
    lf = apply_standardization(lf, cfg.standardization.rules)
    lf = compute_matchkeys(lf, matchkeys)
    df = lf.collect()

    # Match
    print("Matching...")
    all_pairs = []
    for mk in matchkeys:
        if mk.type == "exact":
            pairs = find_exact_matches(df.lazy(), mk)
            all_pairs.extend(pairs)

    all_ids = df["__row_id__"].to_list()
    clusters = build_clusters(all_pairs, all_ids, max_cluster_size=100)
    multi = {k: v for k, v in clusters.items() if v["size"] > 1}

    print(f"\n{'='*60}")
    print(f"ACCURACY ANALYSIS")
    print(f"{'='*60}")
    print(f"\nTotal records: {df.height:,}")
    print(f"Total pairs found: {len(all_pairs):,}")
    print(f"Multi-member clusters: {len(multi):,}")
    print(f"Singletons: {len(clusters) - len(multi):,}")

    # ── Cluster size distribution ──
    sizes = [v["size"] for v in multi.values()]
    size_dist = Counter(sizes)
    print(f"\nCluster size distribution:")
    for size in sorted(size_dist.keys()):
        print(f"  Size {size}: {size_dist[size]:,} clusters")

    # ── Sample clusters: inspect actual data ──
    print(f"\n{'='*60}")
    print("SAMPLE CLUSTERS (first 10)")
    print(f"{'='*60}")

    sample_clusters = list(multi.items())[:10]
    for cid, cinfo in sample_clusters:
        members = cinfo["members"]
        cluster_df = df.filter(pl.col("__row_id__").is_in(members))

        print(f"\n--- Cluster {cid} (size {cinfo['size']}) ---")
        display_cols = ["id", "first_name", "last_name", "email", "phone", "zip"]
        for row in cluster_df.select([c for c in display_cols if c in cluster_df.columns]).to_dicts():
            print(f"  {row}")

    # ── Accuracy metrics ──
    # The synthetic data has a known structure:
    # - IDs 1 through 850,000 are unique base records
    # - IDs 850,001+ are duplicates of base records
    # - Duplicates share the same base email (before messiness)
    # We can check: are clustered records actually duplicates?

    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")

    # Since we matched on email after standardization, check if emails within
    # clusters are actually the same after lowercasing/stripping
    mk_col = "__mk_email_exact__"

    correct_clusters = 0
    incorrect_clusters = 0
    total_checked = 0

    for cid, cinfo in multi.items():
        members = cinfo["members"]
        cluster_df = df.filter(pl.col("__row_id__").is_in(members))

        if mk_col in cluster_df.columns:
            keys = cluster_df[mk_col].drop_nulls().unique().to_list()
            if len(keys) == 1:
                correct_clusters += 1
            else:
                incorrect_clusters += 1
                if incorrect_clusters <= 5:
                    print(f"\n  MISMATCH cluster {cid}: {len(keys)} distinct keys")
                    for row in cluster_df.select(["id", "email", mk_col]).to_dicts():
                        print(f"    {row}")
        total_checked += 1

    print(f"\nClusters where all members share the same matchkey: {correct_clusters:,} / {total_checked:,}")
    print(f"Clusters with mismatched matchkeys: {incorrect_clusters:,}")
    if total_checked > 0:
        print(f"Precision: {correct_clusters / total_checked:.4%}")

    # ── Check for missed duplicates (recall) ──
    # Group original data by standardized email to find "true" duplicates
    print(f"\n{'='*60}")
    print("RECALL ANALYSIS")
    print(f"{'='*60}")

    if mk_col in df.columns:
        # Count how many distinct standardized emails appear more than once
        email_counts = (
            df.filter(pl.col(mk_col).is_not_null())
            .group_by(mk_col)
            .agg(pl.len().alias("count"))
        )
        true_dupe_emails = email_counts.filter(pl.col("count") > 1)
        true_dupe_count = true_dupe_emails.height
        true_dupe_records = true_dupe_emails["count"].sum()

        # How many of those are in our clusters?
        clustered_emails = set()
        for cinfo in multi.values():
            members = cinfo["members"]
            cluster_df = df.filter(pl.col("__row_id__").is_in(members))
            if mk_col in cluster_df.columns:
                emails = cluster_df[mk_col].drop_nulls().unique().to_list()
                clustered_emails.update(emails)

        true_dupe_email_list = true_dupe_emails[mk_col].to_list()
        found = sum(1 for e in true_dupe_email_list if e in clustered_emails)
        missed = true_dupe_count - found

        print(f"Distinct emails appearing 2+ times (true duplicates): {true_dupe_count:,}")
        print(f"Total records in true duplicate groups: {true_dupe_records:,}")
        print(f"True duplicate emails found by matching: {found:,}")
        print(f"True duplicate emails missed: {missed:,}")
        if true_dupe_count > 0:
            print(f"Recall: {found / true_dupe_count:.4%}")

    # ── Check for false positives ──
    print(f"\n{'='*60}")
    print("FALSE POSITIVE CHECK")
    print(f"{'='*60}")

    # Sample 20 random clusters and check if they look correct
    import random
    random.seed(42)
    sample_ids = random.sample(list(multi.keys()), min(20, len(multi)))

    fp_count = 0
    for cid in sample_ids:
        cinfo = multi[cid]
        members = cinfo["members"]
        cluster_df = df.filter(pl.col("__row_id__").is_in(members))

        # Check: do all members have the same last_name (after standardization)?
        last_names = cluster_df["last_name"].drop_nulls().unique().to_list()
        emails_raw = cluster_df["email"].drop_nulls().to_list()

        # For exact email matching, all standardized emails should be the same
        if mk_col in cluster_df.columns:
            mk_vals = cluster_df[mk_col].drop_nulls().unique().to_list()
            if len(mk_vals) > 1:
                fp_count += 1
                print(f"  FALSE POSITIVE? Cluster {cid}:")
                for row in cluster_df.select(["id", "first_name", "last_name", "email"]).to_dicts():
                    print(f"    {row}")

    print(f"\nRandom sample of 20 clusters: {fp_count} potential false positives")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records:      {df.height:,}")
    print(f"  Clusters found:     {len(multi):,}")
    print(f"  Pairs found:        {len(all_pairs):,}")
    if total_checked > 0:
        print(f"  Precision:          {correct_clusters / total_checked:.2%}")
    if true_dupe_count > 0:
        print(f"  Recall:             {found / true_dupe_count:.2%}")
    print(f"  False positives:    {fp_count}/20 sampled")


if __name__ == "__main__":
    main()
