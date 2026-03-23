#!/usr/bin/env python
"""v0.6.0 benchmarks: normal pipeline vs PPRL (anonymized) on Leipzig datasets.

Compares matching accuracy with raw data vs privacy-preserving bloom filter linkage
at three security levels (standard, high, paranoid).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

import polars as pl
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches, score_blocks_parallel
from goldenmatch.core.cluster import build_clusters
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
)
from goldenmatch.pprl.protocol import PPRLConfig, run_pprl

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_dblp_acm():
    d = DATASETS_DIR / "DBLP-ACM"
    df_a = pl.read_csv(d / "DBLP2.csv", encoding="utf8-lossy", ignore_errors=True)
    df_b = pl.read_csv(d / "ACM.csv", encoding="utf8-lossy", ignore_errors=True)
    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns})
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns})
    df_a = df_a.with_columns(pl.lit("dblp").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("acm").alias("__source__"))
    df = pl.concat([df_a, df_b], how="diagonal")
    df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    df, _ = auto_fix_dataframe(df)

    gt_df = pl.read_csv(d / "DBLP-ACM_perfectMapping.csv")
    gt = set()
    for r in gt_df.to_dicts():
        gt.add((str(r["idDBLP"]).strip(), str(r["idACM"]).strip()))
    return df, df_a, df_b, gt


def load_abt_buy():
    d = DATASETS_DIR / "Abt-Buy"
    df_a = pl.read_csv(d / "Abt.csv", encoding="utf8-lossy", ignore_errors=True)
    df_b = pl.read_csv(d / "Buy.csv", encoding="utf8-lossy", ignore_errors=True)
    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns})
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns})
    df_a = df_a.with_columns(pl.lit("abt").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("buy").alias("__source__"))
    df = pl.concat([df_a, df_b], how="diagonal")
    df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    df, _ = auto_fix_dataframe(df)

    gt_df = pl.read_csv(d / "abt_buy_perfectMapping.csv")
    gt = set()
    for r in gt_df.to_dicts():
        gt.add((str(r["idAbt"]).strip(), str(r["idBuy"]).strip()))
    return df, df_a, df_b, gt


def evaluate_pairs(pairs, df, gt):
    """Evaluate predicted pairs against ground truth."""
    rows_d = df.to_dicts()
    idx = {r["__row_id__"]: i for i, r in enumerate(rows_d)}
    seen = set()
    tp = fp = 0
    for a, b, s in pairs:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        ra, rb = rows_d[ia], rows_d[ib]
        if ra.get("__source__") == rb.get("__source__"):
            continue
        id_a = str(ra.get("id", "")).strip()
        id_b = str(rb.get("id", "")).strip()
        canon = tuple(sorted([id_a, id_b]))
        if canon in seen:
            continue
        seen.add(canon)
        if (id_a, id_b) in gt or (id_b, id_a) in gt:
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    return pr, rc, f1


def evaluate_pprl_result(result, df_a, df_b, gt, source_a_name, source_b_name):
    """Evaluate PPRL linkage result against ground truth."""
    tp = fp = 0
    rows_a = df_a.to_dicts()
    rows_b = df_b.to_dicts()

    for cid, members in result.clusters.items():
        a_ids = [rid for pid, rid in members if pid == "party_a"]
        b_ids = [rid for pid, rid in members if pid == "party_b"]
        for aid in a_ids:
            for bid in b_ids:
                if aid < len(rows_a) and bid < len(rows_b):
                    id_a = str(rows_a[aid].get("id", "")).strip()
                    id_b = str(rows_b[bid].get("id", "")).strip()
                    if (id_a, id_b) in gt or (id_b, id_a) in gt:
                        tp += 1
                    else:
                        fp += 1
    fn = len(gt) - tp
    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    return pr, rc, f1


def run_normal_benchmark(name, df, gt, matchkeys, blocking):
    """Run normal (non-PPRL) pipeline benchmark."""
    lf = df.lazy()
    # Standardize all string columns
    std_rules = {}
    for col in df.columns:
        if not col.startswith("__") and df[col].dtype == pl.Utf8:
            std_rules[col] = ["strip", "trim_whitespace"]
    if std_rules:
        lf = apply_standardization(lf, std_rules)
    for mk in matchkeys:
        lf = compute_matchkeys(lf, [mk])
    collected = lf.collect()

    t0 = time.perf_counter()
    all_pairs = []

    # Exact matchkeys
    for mk in matchkeys:
        if mk.type == "exact":
            mk_col = f"__mk_{mk.name}__"
            if mk_col in collected.columns:
                pairs = find_exact_matches(collected.lazy(), mk)
                all_pairs.extend(pairs)

    # Fuzzy matchkeys
    matched = set()
    for mk in matchkeys:
        if mk.type == "weighted":
            blocks = build_blocks(collected.lazy(), blocking)
            pairs = score_blocks_parallel(blocks, mk, matched)
            all_pairs.extend(pairs)
            for a, b, s in pairs:
                matched.add((min(a, b), max(a, b)))

    # Cross-source filter
    src = {r["__row_id__"]: r["__source__"] for r in collected.select("__row_id__", "__source__").to_dicts()}
    all_pairs = [(a, b, s) for a, b, s in all_pairs if src.get(a) != src.get(b)]

    elapsed = time.perf_counter() - t0
    pr, rc, f1 = evaluate_pairs(all_pairs, collected, gt)
    return pr, rc, f1, len(all_pairs), elapsed


def run_pprl_benchmark(name, df_a, df_b, gt, fields, threshold, security_level):
    """Run PPRL benchmark at a given security level."""
    config = PPRLConfig(
        fields=fields,
        threshold=threshold,
        security_level=security_level,
        protocol="trusted_third_party",
    )
    # Set params from security level
    levels = {"standard": (2, 20, 512), "high": (2, 30, 1024), "paranoid": (3, 40, 2048)}
    config.ngram_size, config.hash_functions, config.bloom_filter_size = levels[security_level]

    t0 = time.perf_counter()
    result = run_pprl(df_a, df_b, config)
    elapsed = time.perf_counter() - t0

    pr, rc, f1 = evaluate_pprl_result(result, df_a, df_b, gt, "party_a", "party_b")
    return pr, rc, f1, result.match_count, elapsed


# ── Main ──

print("=" * 80)
print("GoldenMatch v0.6.0 Benchmarks: Normal vs PPRL (Anonymized)")
print("=" * 80)

# ── DBLP-ACM ──
print("\n" + "=" * 80)
print("Dataset: DBLP-ACM (bibliographic, 2.6K vs 2.3K records)")
print("=" * 80)

df, df_a, df_b, gt = load_dblp_acm()
print(f"Loaded: {df.height} records, {len(gt)} ground truth pairs")

mk_exact = MatchkeyConfig(
    name="exact_title", type="exact",
    fields=[MatchkeyField(field="title", transforms=["lowercase", "strip"])],
)
mk_fuzzy = MatchkeyConfig(
    name="fuzzy_title_authors", type="weighted", threshold=0.85,
    fields=[
        MatchkeyField(field="title", scorer="jaro_winkler", weight=0.5, transforms=["lowercase", "strip"]),
        MatchkeyField(field="authors", scorer="token_sort", weight=0.3, transforms=["lowercase", "strip"]),
        MatchkeyField(field="year", scorer="exact", weight=0.2),
    ],
)
blocking = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["year"])],
)

print(f"\n{'Strategy':<35s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>6s}  {'Time':>8s}")
print(f"{'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

pr, rc, f1, pairs, t = run_normal_benchmark("DBLP-ACM", df, gt, [mk_exact, mk_fuzzy], blocking)
print(f"{'Normal (fuzzy+exact)':<35s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {pairs:>6d}  {t:>7.2f}s")

for level, thresh in [("standard", 0.98), ("high", 0.95), ("paranoid", 0.90)]:
    pr, rc, f1, pairs, t = run_pprl_benchmark(
        "DBLP-ACM", df_a, df_b, gt,
        fields=["title", "authors", "year"],
        threshold=thresh,
        security_level=level,
    )
    print(f"{'PPRL (' + level + ', t=' + str(thresh) + ')':<35s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {pairs:>6d}  {t:>7.2f}s")

# ── Abt-Buy ──
print("\n" + "=" * 80)
print("Dataset: Abt-Buy (product, 1K vs 1K records)")
print("=" * 80)

df, df_a, df_b, gt = load_abt_buy()
print(f"Loaded: {df.height} records, {len(gt)} ground truth pairs")

mk_fuzzy_prod = MatchkeyConfig(
    name="fuzzy_name", type="weighted", threshold=0.80,
    fields=[
        MatchkeyField(field="name", scorer="jaro_winkler", weight=0.6, transforms=["lowercase", "strip"]),
        MatchkeyField(field="description", scorer="token_sort", weight=0.4, transforms=["lowercase", "strip"]),
    ],
)
blocking_prod = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
)

print(f"\n{'Strategy':<35s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>6s}  {'Time':>8s}")
print(f"{'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

pr, rc, f1, pairs, t = run_normal_benchmark("Abt-Buy", df, gt, [mk_fuzzy_prod], blocking_prod)
print(f"{'Normal (fuzzy)':<35s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {pairs:>6d}  {t:>7.2f}s")

for level, thresh in [("standard", 0.92), ("high", 0.90), ("paranoid", 0.85)]:
    pr, rc, f1, pairs, t = run_pprl_benchmark(
        "Abt-Buy", df_a, df_b, gt,
        fields=["name"],
        threshold=thresh,
        security_level=level,
    )
    print(f"{'PPRL (' + level + ', t=' + str(thresh) + ')':<35s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {pairs:>6d}  {t:>7.2f}s")

print(f"\n{'=' * 80}")
print("Benchmark complete.")
