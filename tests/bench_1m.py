"""Benchmark: 1M record dedupe pipeline with timing per stage."""

import time
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from goldenmatch.config.loader import load_config
from goldenmatch.core.ingest import load_file
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.core.cluster import build_clusters
from goldenmatch.core.golden import build_golden_record


def timed(label):
    """Context manager for timing."""
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            print(f"  {label}: {elapsed:.2f}s")
            self.elapsed = elapsed
    return Timer()


def main():
    data_path = Path("D:/show_case/goldenmatch/tests/fixtures/synthetic_1m.csv")
    config_path = Path("D:/show_case/goldenmatch/tests/fixtures/bench_config.yaml")
    output_dir = Path("D:/show_case/goldenmatch/tests/fixtures/bench_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    matchkeys = cfg.get_matchkeys()

    print(f"=== GoldenMatch 1M Benchmark ===\n")

    # Stage 1: Ingest
    with timed("INGEST (load + row IDs)") as t_ingest:
        df = pl.read_csv(data_path, infer_schema_length=10000, ignore_errors=True)
        df = df.with_columns(pl.lit("bench").alias("__source__"))
        df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    print(f"    Rows: {df.height:,}, Cols: {len(df.columns)}")
    print(f"    Memory: {df.estimated_size('mb'):.1f} MB")

    # Stage 2: Auto-fix
    with timed("AUTO-FIX") as t_fix:
        df, fixes = auto_fix_dataframe(df)
    print(f"    Fixes applied: {len(fixes)}")
    for fix in fixes:
        print(f"      - {fix['fix']}: {fix['rows_affected']} rows")

    # Stage 3: Standardize
    with timed("STANDARDIZE") as t_std:
        lf = df.lazy()
        lf = apply_standardization(lf, cfg.standardization.rules)
        df = lf.collect()

    # Stage 4: Matchkeys
    with timed("MATCHKEYS (compute)") as t_mk:
        lf = df.lazy()
        lf = compute_matchkeys(lf, matchkeys)
        df = lf.collect()

    # Stage 5: Exact matching
    with timed("EXACT MATCHING") as t_match:
        all_pairs = []
        for mk in matchkeys:
            if mk.type == "exact":
                pairs = find_exact_matches(df.lazy(), mk)
                all_pairs.extend(pairs)
    print(f"    Pairs found: {len(all_pairs):,}")

    # Stage 6: Clustering
    with timed("CLUSTERING") as t_cluster:
        all_ids = df["__row_id__"].to_list()
        max_cs = cfg.golden_rules.max_cluster_size if cfg.golden_rules else 100
        clusters = build_clusters(all_pairs, all_ids, max_cluster_size=max_cs)
    multi = {k: v for k, v in clusters.items() if v["size"] > 1}
    print(f"    Total clusters: {len(multi):,}")
    print(f"    Singletons: {len(clusters) - len(multi):,}")
    if multi:
        sizes = [v["size"] for v in multi.values()]
        print(f"    Avg cluster size: {sum(sizes)/len(sizes):.1f}")
        print(f"    Max cluster size: {max(sizes)}")

    # Stage 7: Golden records (sample — do first 1000 clusters)
    sample_clusters = dict(list(multi.items())[:1000])
    with timed(f"GOLDEN RECORDS ({len(sample_clusters)} clusters)") as t_golden:
        golden_count = 0
        for cid, cinfo in sample_clusters.items():
            if not cinfo["oversized"]:
                cluster_df = df.filter(pl.col("__row_id__").is_in(cinfo["members"]))
                golden = build_golden_record(cluster_df, cfg.golden_rules)
                golden_count += 1
    print(f"    Golden records built: {golden_count}")

    # Summary
    total = t_ingest.elapsed + t_fix.elapsed + t_std.elapsed + t_mk.elapsed + t_match.elapsed + t_cluster.elapsed + t_golden.elapsed
    print(f"\n=== TOTAL: {total:.2f}s ===")
    print(f"\nBreakdown:")
    print(f"  Ingest:       {t_ingest.elapsed:.2f}s ({t_ingest.elapsed/total*100:.0f}%)")
    print(f"  Auto-fix:     {t_fix.elapsed:.2f}s ({t_fix.elapsed/total*100:.0f}%)")
    print(f"  Standardize:  {t_std.elapsed:.2f}s ({t_std.elapsed/total*100:.0f}%)")
    print(f"  Matchkeys:    {t_mk.elapsed:.2f}s ({t_mk.elapsed/total*100:.0f}%)")
    print(f"  Matching:     {t_match.elapsed:.2f}s ({t_match.elapsed/total*100:.0f}%)")
    print(f"  Clustering:   {t_cluster.elapsed:.2f}s ({t_cluster.elapsed/total*100:.0f}%)")
    print(f"  Golden:       {t_golden.elapsed:.2f}s ({t_golden.elapsed/total*100:.0f}%)")


if __name__ == "__main__":
    main()
