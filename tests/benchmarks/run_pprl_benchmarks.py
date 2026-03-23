#!/usr/bin/env python
"""PPRL benchmarks on standard person-data datasets.

FEBRL4: 5K vs 5K synthetic person records with controlled corruption.
NCVR: North Carolina Voter Registration (if downloaded).

These are the industry-standard benchmarks for privacy-preserving record linkage.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import polars as pl

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_febrl4():
    """Load FEBRL4 link dataset: 5K vs 5K person records."""
    try:
        from recordlinkage.datasets import load_febrl4 as _load
    except ImportError:
        print("  [SKIP] pip install recordlinkage for FEBRL datasets")
        return None, None, None

    df_a_pd, df_b_pd = _load()

    # Build ground truth from ID scheme: rec-NNN-org in A matches rec-NNN-dup-0 in B
    gt = set()
    a_ids = {idx: i for i, idx in enumerate(df_a_pd.index)}
    b_ids = {idx: i for i, idx in enumerate(df_b_pd.index)}
    for b_idx in df_b_pd.index:
        # rec-561-dup-0 -> rec-561
        base = b_idx.replace("-dup-0", "").replace("-dup-1", "").replace("-dup-2", "")
        org_key = base + "-org"
        if org_key in a_ids:
            gt.add((a_ids[org_key], b_ids[b_idx]))

    # Convert to Polars
    df_a_pd = df_a_pd.reset_index()
    df_b_pd = df_b_pd.reset_index()
    df_a = pl.from_pandas(df_a_pd)
    df_b = pl.from_pandas(df_b_pd)

    return df_a, df_b, gt


def load_ncvr():
    """Load NCVR sample if downloaded."""
    ncvr_dir = DATASETS_DIR / "NCVR"
    if not ncvr_dir.exists():
        print("  [SKIP] NCVR not found. Download from NC State Board of Elections")
        print("         and place files in tests/benchmarks/datasets/NCVR/")
        return None, None, None

    # Look for common file patterns
    files = list(ncvr_dir.glob("*.csv")) + list(ncvr_dir.glob("*.txt"))
    if not files:
        print("  [SKIP] No CSV/TXT files found in NCVR directory")
        return None, None, None

    print(f"  Found {len(files)} NCVR files")
    # NCVR doesn't have pre-built ground truth -- would need to create
    # synthetic duplicates from the real data for benchmarking
    return None, None, None


def evaluate_pprl(result, gt):
    """Evaluate PPRL result against ground truth (row-index based)."""
    tp = fp = 0
    for cid, members in result.clusters.items():
        a_ids = [rid for pid, rid in members if pid == "party_a"]
        b_ids = [rid for pid, rid in members if pid == "party_b"]
        for aid in a_ids:
            for bid in b_ids:
                if (aid, bid) in gt:
                    tp += 1
                else:
                    fp += 1
    fn = len(gt) - tp
    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    return pr, rc, f1


def run_normal_benchmark(df_a, df_b, gt, fields, threshold=0.85):
    """Run normal (non-PPRL) fuzzy matching for comparison."""
    from goldenmatch.core.autofix import auto_fix_dataframe
    from goldenmatch.core.standardize import apply_standardization
    from goldenmatch.core.matchkey import compute_matchkeys
    from goldenmatch.core.blocker import build_blocks
    from goldenmatch.core.scorer import score_blocks_parallel
    from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig

    df_a2 = df_a.with_columns(pl.lit("a").alias("__source__"))
    df_b2 = df_b.with_columns(pl.lit("b").alias("__source__"))
    df = pl.concat([df_a2, df_b2], how="diagonal")
    df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    df, _ = auto_fix_dataframe(df)

    mk_fields = []
    for f in fields:
        mk_fields.append(MatchkeyField(
            field=f, scorer="jaro_winkler", weight=1.0 / len(fields),
            transforms=["lowercase", "strip"],
        ))

    mk = MatchkeyConfig(name="fuzzy", type="weighted", threshold=threshold, fields=mk_fields)
    blocking = BlockingConfig(keys=[BlockingKeyConfig(fields=[fields[0]], transforms=["lowercase", "substring:0:3"])])

    lf = df.lazy()
    lf = compute_matchkeys(lf, [mk])
    collected = lf.collect()

    t0 = time.perf_counter()
    blocks = build_blocks(collected.lazy(), blocking)
    all_pairs = score_blocks_parallel(blocks, mk, set())
    elapsed = time.perf_counter() - t0

    # Cross-source filter
    src = {r["__row_id__"]: r["__source__"] for r in collected.select("__row_id__", "__source__").to_dicts()}
    all_pairs = [(a, b, s) for a, b, s in all_pairs if src.get(a) != src.get(b)]

    # Map row_ids back to original indices for gt comparison
    row_to_orig_idx = {}
    a_count = df_a.height
    for r in collected.to_dicts():
        rid = r["__row_id__"]
        if r["__source__"] == "a":
            # Find original index by position
            row_to_orig_idx[rid] = ("a", rid)
        else:
            row_to_orig_idx[rid] = ("b", rid - a_count)

    tp = fp = 0
    for a, b, s in all_pairs:
        info_a = row_to_orig_idx.get(a)
        info_b = row_to_orig_idx.get(b)
        if not info_a or not info_b:
            continue
        src_a, idx_a = info_a
        src_b, idx_b = info_b
        if src_a == src_b:
            continue
        if src_a == "a":
            pair = (idx_a, idx_b)
        else:
            pair = (idx_b, idx_a)
        if pair in gt:
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    return pr, rc, f1, len(all_pairs), elapsed


def run_pprl_benchmark(df_a, df_b, gt, fields, threshold, security_level):
    """Run PPRL benchmark."""
    from goldenmatch.pprl.protocol import PPRLConfig, run_pprl

    config = PPRLConfig(
        fields=fields,
        threshold=threshold,
        security_level=security_level,
        protocol="trusted_third_party",
    )
    levels = {"standard": (2, 20, 512), "high": (2, 30, 1024), "paranoid": (3, 40, 2048)}
    config.ngram_size, config.hash_functions, config.bloom_filter_size = levels[security_level]

    t0 = time.perf_counter()
    result = run_pprl(df_a, df_b, config)
    elapsed = time.perf_counter() - t0

    pr, rc, f1 = evaluate_pprl(result, gt)
    return pr, rc, f1, result.match_count, elapsed


# ── Main ──

print("=" * 80)
print("GoldenMatch PPRL Benchmarks: Industry-Standard Person Data")
print("=" * 80)

# ── FEBRL4 ──
print("\n" + "=" * 80)
print("Dataset: FEBRL4 (5K vs 5K synthetic person records)")
print("=" * 80)

df_a, df_b, gt = load_febrl4()
if df_a is not None:
    print(f"Loaded: {df_a.height} vs {df_b.height} records, {len(gt)} ground truth pairs")
    print(f"Columns: {[c for c in df_a.columns if c != 'rec_id']}")

    fields = ["given_name", "surname", "postcode", "state"]

    print(f"\n{'Strategy':<40s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>7s}  {'Time':>8s}")
    print(f"{'-'*40}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}")

    # Normal fuzzy baseline
    pr, rc, f1, pairs, t = run_normal_benchmark(df_a, df_b, gt, fields, threshold=0.80)
    print(f"{'Normal (fuzzy, t=0.80)':<40s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {pairs:>7d}  {t:>7.2f}s")

    # PPRL at each security level with tuned thresholds
    for level, thresh in [("standard", 0.80), ("high", 0.80), ("paranoid", 0.80)]:
        pr, rc, f1, pairs, t = run_pprl_benchmark(df_a, df_b, gt, fields, thresh, level)
        print(f"{'PPRL (' + level + ', t=' + str(thresh) + ')':<40s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {pairs:>7d}  {t:>7.2f}s")

    # Higher threshold sweep for precision-focused use
    print(f"\n  Threshold sweep (high security):")
    for thresh in [0.75, 0.80, 0.85, 0.90, 0.95]:
        pr, rc, f1, pairs, t = run_pprl_benchmark(df_a, df_b, gt, fields, thresh, "high")
        print(f"    t={thresh:.2f}:  P={pr:5.1%}  R={rc:5.1%}  F1={f1:5.1%}  pairs={pairs}")

# ── NCVR ──
print("\n" + "=" * 80)
print("Dataset: NCVR (North Carolina Voter Registration)")
print("=" * 80)
load_ncvr()

print(f"\n{'=' * 80}")
print("Benchmark complete.")
