#!/usr/bin/env python
"""Final PPRL benchmarks: FEBRL4 + NCVR optimized + auto-config validation.

Runs three benchmark suites:
1. FEBRL4 with vectorized code (verify numbers unchanged)
2. NCVR with optimized config (best fields + threshold from sweep)
3. Auto-config validation: does auto_configure_pprl pick good params?
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import polars as pl
from goldenmatch.pprl.protocol import PPRLConfig, run_pprl
from goldenmatch.pprl.autoconfig import auto_configure_pprl


def evaluate(result, gt):
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


def run_bench(label, df_a, df_b, gt, config):
    t0 = time.perf_counter()
    result = run_pprl(df_a, df_b, config)
    elapsed = time.perf_counter() - t0
    pr, rc, f1 = evaluate(result, gt)
    print(f"{label:<50s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {result.match_count:>7d}  {elapsed:>6.1f}s")
    return pr, rc, f1


# ===========================================================================
print("=" * 90)
print("GoldenMatch PPRL Final Benchmarks (vectorized, v0.6.0)")
print("=" * 90)

# -- 1. FEBRL4 -------------------------------------------------------------
print("\n" + "-" * 90)
print("1. FEBRL4 (5K vs 5K synthetic person records)")
print("-" * 90)

try:
    from recordlinkage.datasets import load_febrl4 as _load_febrl4
    df_a_pd, df_b_pd = _load_febrl4()

    gt_febrl = set()
    a_ids = {idx: i for i, idx in enumerate(df_a_pd.index)}
    b_ids = {idx: i for i, idx in enumerate(df_b_pd.index)}
    for b_idx in df_b_pd.index:
        base = b_idx.replace("-dup-0", "").replace("-dup-1", "").replace("-dup-2", "")
        org_key = base + "-org"
        if org_key in a_ids:
            gt_febrl.add((a_ids[org_key], b_ids[b_idx]))

    df_a_f = pl.from_pandas(df_a_pd.reset_index())
    df_b_f = pl.from_pandas(df_b_pd.reset_index())

    print(f"Loaded: {df_a_f.height} vs {df_b_f.height}, {len(gt_febrl)} ground truth pairs\n")

    print(f"{'Strategy':<50s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>7s}  {'Time':>6s}")
    print(f"{'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")

    fields_febrl = ["given_name", "surname", "postcode", "state"]

    for level in ["standard", "high", "paranoid"]:
        levels = {"standard": (2, 20, 512), "high": (2, 30, 1024), "paranoid": (3, 40, 2048)}
        ng, hf, bs = levels[level]
        config = PPRLConfig(fields=fields_febrl, threshold=0.80, security_level=level,
                            ngram_size=ng, hash_functions=hf, bloom_filter_size=bs)
        run_bench(f"PPRL {level} (t=0.80)", df_a_f, df_b_f, gt_febrl, config)

except ImportError:
    print("  [SKIP] pip install recordlinkage for FEBRL4")
    df_a_f = df_b_f = gt_febrl = None


# -- 2. NCVR Optimized ----------------------------------------------------
print("\n" + "-" * 90)
print("2. NCVR (5K real voter records + 2.5K corrupted duplicates)")
print("-" * 90)

ncvr_file = Path(__file__).parent / "datasets" / "NCVR" / "ncvoter_sample_10k.txt"
if ncvr_file.exists():
    df_ncvr = pl.read_csv(ncvr_file, separator="\t", ignore_errors=True, encoding="utf8-lossy")
    df_ncvr = df_ncvr.filter(
        (pl.col("last_name").str.len_chars() > 1) & (pl.col("first_name").str.len_chars() > 1)
    ).sample(n=5000, seed=42)

    from tests.benchmarks.run_ncvr_pprl import create_corrupted_duplicates
    df_a_n = df_ncvr.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    df_b_n, gt_ncvr = create_corrupted_duplicates(df_ncvr, n_dupes=2500, corruption_rate=0.3)
    df_b_n = df_b_n.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))

    print(f"Loaded: {df_a_n.height} vs {df_b_n.height}, {len(gt_ncvr)} ground truth pairs\n")

    print(f"{'Strategy':<50s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>7s}  {'Time':>6s}")
    print(f"{'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")

    # Original config (from first benchmark)
    config_orig = PPRLConfig(
        fields=["first_name", "last_name", "zip_code", "birth_year"],
        threshold=0.90, security_level="high",
        ngram_size=2, hash_functions=30, bloom_filter_size=1024,
    )
    run_bench("Original (4 fields, high, t=0.90)", df_a_n, df_b_n, gt_ncvr, config_orig)

    # Optimized config (from sweep)
    config_opt = PPRLConfig(
        fields=["first_name", "last_name", "zip_code", "birth_year", "gender_code"],
        threshold=0.88, security_level="standard",
        ngram_size=2, hash_functions=20, bloom_filter_size=512,
    )
    run_bench("Optimized (5 fields, 512b, t=0.88)", df_a_n, df_b_n, gt_ncvr, config_opt)

    # Best precision config
    config_prec = PPRLConfig(
        fields=["first_name", "last_name", "zip_code", "birth_year", "gender_code"],
        threshold=0.92, security_level="high",
        ngram_size=2, hash_functions=30, bloom_filter_size=1024,
    )
    run_bench("High precision (5 fields, high, t=0.92)", df_a_n, df_b_n, gt_ncvr, config_prec)

else:
    print("  [SKIP] NCVR sample not found")
    df_a_n = df_b_n = gt_ncvr = None


# -- 3. Auto-Config Validation ---------------------------------------------
print("\n" + "-" * 90)
print("3. Auto-Config Validation (does auto_configure_pprl pick good params?)")
print("-" * 90)

# FEBRL4 auto-config
if df_a_f is not None:
    print("\n  FEBRL4:")
    auto_f = auto_configure_pprl(df_a_f)
    print(f"    Auto-selected fields: {auto_f.recommended_fields}")
    print(f"    Threshold: {auto_f.recommended_config.threshold}")
    print(f"    BF: {auto_f.recommended_config.ngram_size}-gram, {auto_f.recommended_config.hash_functions}h, {auto_f.recommended_config.bloom_filter_size}b")

    print(f"\n    {'Strategy':<50s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>7s}  {'Time':>6s}")
    print(f"    {'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")

    # Manual best
    config_manual = PPRLConfig(fields=["given_name", "surname", "postcode", "state"],
                               threshold=0.80, security_level="high",
                               ngram_size=2, hash_functions=30, bloom_filter_size=1024)
    print("    ", end="")
    run_bench("Manual best (known optimal)", df_a_f, df_b_f, gt_febrl, config_manual)

    # Auto-config
    print("    ", end="")
    run_bench("Auto-configured", df_a_f, df_b_f, gt_febrl, auto_f.recommended_config)

# NCVR auto-config
if df_a_n is not None:
    print("\n  NCVR:")
    auto_n = auto_configure_pprl(df_a_n)
    print(f"    Auto-selected fields: {auto_n.recommended_fields}")
    print(f"    Threshold: {auto_n.recommended_config.threshold}")
    print(f"    BF: {auto_n.recommended_config.ngram_size}-gram, {auto_n.recommended_config.hash_functions}h, {auto_n.recommended_config.bloom_filter_size}b")

    print(f"\n    {'Strategy':<50s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>7s}  {'Time':>6s}")
    print(f"    {'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")

    # Manual best
    print("    ", end="")
    run_bench("Manual best (sweep-optimized)", df_a_n, df_b_n, gt_ncvr, config_opt)

    # Auto-config
    print("    ", end="")
    run_bench("Auto-configured", df_a_n, df_b_n, gt_ncvr, auto_n.recommended_config)


print(f"\n{'=' * 90}")
print("Final benchmarks complete.")
