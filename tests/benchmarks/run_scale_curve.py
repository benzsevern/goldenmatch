"""Scale curve benchmark — throughput from 1K to max memory allows.

Measures records/second, pipeline stage breakdown, and memory usage
at each scale point. Projects performance for larger scales.
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import polars as pl


def generate_synthetic(n: int, dupe_rate: float = 0.15) -> pl.DataFrame:
    """Generate synthetic records with controlled duplicate rate."""
    rng = np.random.default_rng(42)

    first_names = ["John", "Jane", "Bob", "Alice", "Mike", "Sarah", "Chris", "Lisa",
                   "Tom", "Emma", "David", "Susan", "James", "Karen", "Robert", "Mary",
                   "Richard", "Emily", "William", "Laura", "Daniel", "Anna", "Paul", "Helen"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                  "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore",
                  "Jackson", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "test.com", "health.net", "clinic.com"]
    specialties = ["Cardiology", "Radiology", "Neurology", "Pediatrics", "Oncology",
                   "Dermatology", "Surgery", "Psychiatry", "Orthopedics", "Urology"]

    n_unique = int(n * (1 - dupe_rate))
    n_dupes = n - n_unique

    records = []
    for i in range(n_unique):
        fn = first_names[i % len(first_names)]
        ln = last_names[i % len(last_names)]
        records.append({
            "id": i,
            "first_name": fn,
            "last_name": ln,
            "email": f"{fn.lower()}.{ln.lower()}{i}@{domains[i % len(domains)]}",
            "phone": f"555-{rng.integers(1000, 9999):04d}",
            "zip": f"{rng.integers(10000, 99999)}",
            "specialty": specialties[i % len(specialties)],
        })

    # Generate dupes with variations
    for i in range(n_dupes):
        src = records[i % n_unique].copy()
        src["id"] = n_unique + i

        # Apply random variation
        variation = rng.integers(0, 4)
        if variation == 0:
            src["first_name"] = src["first_name"][:3]  # truncate
        elif variation == 1:
            src["email"] = src["email"].upper()
        elif variation == 2:
            src["phone"] = src["phone"][:7] + str(rng.integers(0, 9))
        # else: exact dupe

        records.append(src)

    rng.shuffle(records)
    return pl.DataFrame(records)


def run_benchmark(n: int) -> dict:
    """Run full pipeline benchmark at scale n."""
    from goldenmatch.core.autofix import auto_fix_dataframe
    from goldenmatch.core.standardize import apply_standardization
    from goldenmatch.core.matchkey import compute_matchkeys
    from goldenmatch.core.blocker import build_blocks
    from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
    from goldenmatch.core.cluster import build_clusters
    from goldenmatch.core.golden import build_golden_record
    from goldenmatch.config.schemas import (
        MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
        GoldenRulesConfig,
    )

    import psutil
    mem_before = psutil.Process().memory_info().rss / 1e6

    timings = {}

    # Generate data
    t0 = time.perf_counter()
    df = generate_synthetic(n)
    df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    df = df.with_columns(pl.lit("source").alias("__source__"))
    timings["generate"] = time.perf_counter() - t0

    # Auto-fix
    t0 = time.perf_counter()
    df, _ = auto_fix_dataframe(df)
    timings["autofix"] = time.perf_counter() - t0

    # Standardize
    t0 = time.perf_counter()
    std = {"first_name": ["strip", "trim_whitespace"], "last_name": ["strip", "trim_whitespace"]}
    lf = df.lazy()
    lf = apply_standardization(lf, std)
    df = lf.collect()
    timings["standardize"] = time.perf_counter() - t0

    # Matchkeys
    mk_exact = MatchkeyConfig(
        name="phone_exact",
        type="exact",
        fields=[MatchkeyField(field="phone", transforms=["strip"])],
    )
    mk_fuzzy = MatchkeyConfig(
        name="name_zip",
        type="weighted",
        threshold=0.85,
        fields=[
            MatchkeyField(field="first_name", scorer="jaro_winkler", weight=0.3, transforms=["lowercase"]),
            MatchkeyField(field="last_name", scorer="jaro_winkler", weight=0.3, transforms=["lowercase"]),
            MatchkeyField(field="zip", scorer="exact", weight=0.2),
            MatchkeyField(field="specialty", scorer="exact", weight=0.2),
        ],
    )

    t0 = time.perf_counter()
    lf = df.lazy()
    lf = compute_matchkeys(lf, [mk_exact, mk_fuzzy])
    df = lf.collect()
    timings["matchkeys"] = time.perf_counter() - t0

    # Exact matching
    t0 = time.perf_counter()
    exact_pairs = find_exact_matches(df.lazy(), mk_exact)
    timings["exact_match"] = time.perf_counter() - t0

    # Blocking + fuzzy matching
    blocking = BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])])
    t0 = time.perf_counter()
    blocks = build_blocks(df.lazy(), blocking)
    timings["blocking"] = time.perf_counter() - t0

    matched_pairs = {(min(a, b), max(a, b)) for a, b, _ in exact_pairs}
    all_pairs = list(exact_pairs)

    t0 = time.perf_counter()
    for block in blocks:
        bdf = block.df.collect()
        pairs = find_fuzzy_matches(bdf, mk_fuzzy, exclude_pairs=matched_pairs,
                                    pre_scored_pairs=block.pre_scored_pairs)
        all_pairs.extend(pairs)
        for a, b, _ in pairs:
            matched_pairs.add((min(a, b), max(a, b)))
    timings["fuzzy_match"] = time.perf_counter() - t0

    # Clustering
    t0 = time.perf_counter()
    all_ids = df["__row_id__"].to_list()
    clusters = build_clusters(all_pairs, all_ids, max_cluster_size=100)
    timings["cluster"] = time.perf_counter() - t0

    # Golden records
    t0 = time.perf_counter()
    golden_rules = GoldenRulesConfig(default_strategy="most_complete")
    golden_count = 0
    multi_clusters = {k: v for k, v in clusters.items() if v["size"] > 1}
    for cid, info in multi_clusters.items():
        if not info.get("oversized"):
            cluster_df = df.filter(pl.col("__row_id__").is_in(info["members"]))
            g = build_golden_record(cluster_df, golden_rules)
            if g:
                golden_count += 1
    timings["golden"] = time.perf_counter() - t0

    total = sum(timings.values()) - timings["generate"]
    mem_after = psutil.Process().memory_info().rss / 1e6

    return {
        "n": n,
        "total_seconds": round(total, 2),
        "records_per_second": round(n / total) if total > 0 else 0,
        "pairs_found": len(all_pairs),
        "clusters": len(multi_clusters),
        "golden_records": golden_count,
        "memory_mb": round(mem_after - mem_before),
        "peak_memory_mb": round(mem_after),
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }


def main():
    print("=" * 70)
    print("GoldenMatch Scale Curve Benchmark")
    print("=" * 70)
    print()

    scales = [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
    results = []

    for n in scales:
        print(f"  Scale: {n:>12,} records ... ", end="", flush=True)
        gc.collect()

        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / 1e9
            est_needed_gb = n * 150 / 1e9  # ~150 bytes/row estimate
            if est_needed_gb > avail_gb * 0.8:
                print(f"SKIP (need ~{est_needed_gb:.1f}GB, have {avail_gb:.1f}GB)")
                continue

            result = run_benchmark(n)
            results.append(result)

            print(f"{result['total_seconds']:>8.2f}s  "
                  f"{result['records_per_second']:>10,} rec/s  "
                  f"{result['pairs_found']:>8,} pairs  "
                  f"{result['peak_memory_mb']:>6,}MB")

        except MemoryError:
            print("OUT OF MEMORY")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            break

    # Summary table
    print()
    print("=" * 70)
    print(f"{'Scale':>12} {'Time':>8} {'Rec/sec':>12} {'Pairs':>10} {'Clusters':>10} {'Memory':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['n']:>12,} {r['total_seconds']:>7.2f}s {r['records_per_second']:>11,} "
              f"{r['pairs_found']:>10,} {r['clusters']:>10,} {r['peak_memory_mb']:>7,}MB")

    # Stage breakdown for largest successful run
    if results:
        last = results[-1]
        print()
        print(f"Pipeline breakdown at {last['n']:,} records:")
        total = last['total_seconds']
        for stage, t in last['timings'].items():
            if stage == "generate":
                continue
            pct = t / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
            print(f"  {stage:<15} {t:>7.3f}s  {bar} {pct:>5.1f}%")

    # Projection to 500M
    if len(results) >= 2:
        print()
        print("Projected performance (extrapolated):")
        # Use last two points to estimate scaling factor
        r1, r2 = results[-2], results[-1]
        scale_factor = r2['total_seconds'] / r1['total_seconds']
        size_factor = r2['n'] / r1['n']
        # Estimate O(n log n) or O(n) scaling
        for target in [50_000_000, 100_000_000, 500_000_000]:
            ratio = target / r2['n']
            # Assume near-linear scaling (Polars + blocking keeps it sub-quadratic)
            est_time = r2['total_seconds'] * ratio * 1.1  # 10% overhead factor
            est_rps = target / est_time if est_time > 0 else 0
            est_mem = r2['peak_memory_mb'] * ratio
            print(f"  {target:>12,} records: ~{est_time:>8.0f}s ({est_time/60:>5.1f} min)  "
                  f"~{est_rps:>10,.0f} rec/s  ~{est_mem/1024:>5.1f}GB RAM")


if __name__ == "__main__":
    main()
