#!/usr/bin/env python
"""NCVR PPRL benchmark: real voter data with synthetic corruption.

Takes a sample of NC voter registration records, creates corrupted duplicates
(typos, abbreviations, missing fields), then benchmarks PPRL linkage accuracy.
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import polars as pl

NCVR_DIR = Path(__file__).parent / "datasets" / "NCVR"


def load_ncvr_sample(n=5000, seed=42):
    """Load and sample NCVR records."""
    sample_file = NCVR_DIR / "ncvoter_sample_10k.txt"
    if not sample_file.exists():
        print(f"  [SKIP] {sample_file} not found. Run the extract step first.")
        return None

    df = pl.read_csv(sample_file, separator="\t", ignore_errors=True, encoding="utf8-lossy")
    # Filter to active voters with non-empty names
    df = df.filter(
        (pl.col("last_name").str.len_chars() > 1) &
        (pl.col("first_name").str.len_chars() > 1)
    )
    df = df.sample(n=min(n, df.height), seed=seed)
    return df


def corrupt_value(val: str, rng: random.Random) -> str:
    """Apply a random corruption to a string value."""
    if not val or len(val) < 2:
        return val

    corruption = rng.choice(["typo", "swap", "drop", "abbreviate", "case"])

    if corruption == "typo":
        pos = rng.randint(0, len(val) - 1)
        replacement = rng.choice("abcdefghijklmnopqrstuvwxyz")
        return val[:pos] + replacement + val[pos + 1:]
    elif corruption == "swap" and len(val) >= 3:
        pos = rng.randint(0, len(val) - 2)
        return val[:pos] + val[pos + 1] + val[pos] + val[pos + 2:]
    elif corruption == "drop" and len(val) >= 3:
        pos = rng.randint(0, len(val) - 1)
        return val[:pos] + val[pos + 1:]
    elif corruption == "abbreviate" and len(val) >= 3:
        return val[0] + "."
    elif corruption == "case":
        return val.lower() if rng.random() < 0.5 else val.upper()

    return val


def create_corrupted_duplicates(df, n_dupes=2500, corruption_rate=0.3, seed=42):
    """Create a second dataset with corrupted duplicates of selected records.

    Returns (df_b, ground_truth_pairs) where ground_truth_pairs maps
    original row index -> corrupted row index.
    """
    rng = random.Random(seed)
    rows = df.to_dicts()

    # Select records to duplicate
    dup_indices = rng.sample(range(len(rows)), min(n_dupes, len(rows)))

    corrupted_rows = []
    gt_pairs = set()  # (original_idx, corrupted_idx)

    corrupt_fields = ["first_name", "last_name", "middle_name", "res_street_address", "zip_code"]

    for orig_idx in dup_indices:
        row = dict(rows[orig_idx])
        # Corrupt some fields
        for field in corrupt_fields:
            if field in row and row[field] and rng.random() < corruption_rate:
                row[field] = corrupt_value(str(row[field]), rng)

        corrupted_idx = len(corrupted_rows)
        gt_pairs.add((orig_idx, corrupted_idx))
        corrupted_rows.append(row)

    # Add some non-matching records (noise)
    noise_indices = [i for i in range(len(rows)) if i not in set(dup_indices)]
    noise_sample = rng.sample(noise_indices, min(len(noise_indices), n_dupes // 2))
    for idx in noise_sample:
        corrupted_rows.append(dict(rows[idx]))

    rng.shuffle(corrupted_rows)

    # Rebuild ground truth after shuffle
    # Need to track where each corrupted record ended up
    df_b = pl.DataFrame(corrupted_rows)
    # Remap gt_pairs based on shuffle -- for simplicity, rebuild by matching ncid
    orig_ncids = {i: rows[i].get("ncid") for i in dup_indices}
    b_ncid_to_idx = {}
    for i, row in enumerate(corrupted_rows):
        ncid = row.get("ncid")
        if ncid and ncid not in b_ncid_to_idx:
            b_ncid_to_idx[ncid] = i

    gt_pairs_final = set()
    for orig_idx in dup_indices:
        ncid = orig_ncids[orig_idx]
        if ncid in b_ncid_to_idx:
            gt_pairs_final.add((orig_idx, b_ncid_to_idx[ncid]))

    return df_b, gt_pairs_final


def evaluate_pprl(result, gt):
    """Evaluate PPRL result against ground truth."""
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


# ── Main ──

print("=" * 80)
print("NCVR PPRL Benchmark: Real Voter Data with Synthetic Corruption")
print("=" * 80)

df = load_ncvr_sample(n=5000)
if df is None:
    sys.exit(1)

print(f"Loaded: {df.height} NCVR records")

# Create corrupted duplicates
df_a = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
df_b, gt = create_corrupted_duplicates(df, n_dupes=2500, corruption_rate=0.3)
df_b = df_b.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))

print(f"Party A: {df_a.height} records (original)")
print(f"Party B: {df_b.height} records ({len(gt)} corrupted duplicates + noise)")
print(f"Ground truth: {len(gt)} true match pairs")

from goldenmatch.pprl.protocol import PPRLConfig, run_pprl

fields = ["first_name", "last_name", "zip_code", "birth_year"]

print(f"\nLinkage fields: {fields}")
print(f"\n{'Strategy':<40s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>7s}  {'Time':>8s}")
print(f"{'-'*40}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}")

for level, thresh in [("standard", 0.80), ("high", 0.80), ("paranoid", 0.80)]:
    config = PPRLConfig(
        fields=fields, threshold=thresh, security_level=level,
        protocol="trusted_third_party",
    )
    levels = {"standard": (2, 20, 512), "high": (2, 30, 1024), "paranoid": (3, 40, 2048)}
    config.ngram_size, config.hash_functions, config.bloom_filter_size = levels[level]

    t0 = time.perf_counter()
    result = run_pprl(df_a, df_b, config)
    elapsed = time.perf_counter() - t0

    pr, rc, f1 = evaluate_pprl(result, gt)
    label = f"PPRL ({level}, t={thresh})"
    print(f"{label:<40s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {result.match_count:>7d}  {elapsed:>7.1f}s")

# Threshold sweep at high
print(f"\n  Threshold sweep (high security, NCVR):")
for thresh in [0.75, 0.80, 0.85, 0.90, 0.95]:
    config = PPRLConfig(
        fields=fields, threshold=thresh, security_level="high",
        ngram_size=2, hash_functions=30, bloom_filter_size=1024,
    )
    t0 = time.perf_counter()
    result = run_pprl(df_a, df_b, config)
    elapsed = time.perf_counter() - t0
    pr, rc, f1 = evaluate_pprl(result, gt)
    print(f"    t={thresh:.2f}:  P={pr:5.1%}  R={rc:5.1%}  F1={f1:5.1%}  pairs={result.match_count}  ({elapsed:.1f}s)")

print(f"\n{'=' * 80}")
print("Benchmark complete.")
