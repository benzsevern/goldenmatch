#!/usr/bin/env python
"""Quick v0.3.0 benchmarks — F-S probabilistic vs weighted, learned vs static blocking.

Uses static blocking (not multi_pass) for speed, and smaller block sizes.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import polars as pl

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_dblp_acm():
    ds_dir = DATASETS_DIR / "DBLP-ACM"
    df_a = pl.read_csv(ds_dir / "DBLP2.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b = pl.read_csv(ds_dir / "ACM.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    # Cast all columns to String for safe concat
    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns})
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns})
    df_a = df_a.with_columns(pl.lit("dblp").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("acm").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__")

    gt_df = pl.read_csv(ds_dir / "DBLP-ACM_perfectMapping.csv")
    gt = set()
    for row in gt_df.to_dicts():
        gt.add((str(row["idDBLP"]).strip(), str(row["idACM"]).strip()))
    return combined, gt


def evaluate_pairs(pairs, df, gt):
    rows = df.to_dicts()
    id_to_idx = {r["__row_id__"]: i for i, r in enumerate(rows)}
    tp = fp = 0
    for a, b, s in pairs:
        idx_a, idx_b = id_to_idx.get(a), id_to_idx.get(b)
        if idx_a is None or idx_b is None:
            continue
        ra, rb = rows[idx_a], rows[idx_b]
        if ra.get("__source__") == rb.get("__source__"):
            continue
        if ra.get("__source__") == "dblp":
            pair = (str(ra.get("id", "")).strip(), str(rb.get("id", "")).strip())
        else:
            pair = (str(rb.get("id", "")).strip(), str(ra.get("id", "")).strip())
        if pair in gt:
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return tp, fp, fn, prec, rec, f1


print("=" * 70)
print("GoldenMatch v0.3.0 Quick Feature Benchmarks")
print("=" * 70)

df, gt = load_dblp_acm()
print(f"Loaded: {df.height} records, {len(gt)} ground truth pairs\n")

from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import score_blocks_parallel
from goldenmatch.core.probabilistic import train_em, score_probabilistic

# ── Benchmark 1: Weighted baseline ──
print("=" * 70)
print("Benchmark 1: Weighted vs Fellegi-Sunter (DBLP-ACM)")
print("=" * 70)

mk_w = MatchkeyConfig(
    name="fuzzy", type="weighted", threshold=0.85,
    fields=[
        MatchkeyField(field="title", scorer="token_sort", weight=1.0, transforms=["lowercase"]),
        MatchkeyField(field="authors", scorer="token_sort", weight=0.5, transforms=["lowercase"]),
        MatchkeyField(field="year", scorer="exact", weight=0.3),
    ],
)
blocking = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["year"])],
    max_block_size=1000,
)

print("  Running weighted baseline...", flush=True)
t0 = time.perf_counter()
lf = compute_matchkeys(df.lazy(), [mk_w])
collected = lf.collect()
blocks = build_blocks(collected.lazy(), blocking)
print(f"    {len(blocks)} blocks built", flush=True)
matched = set()
pairs_w = score_blocks_parallel(blocks, mk_w, matched)
t_w = time.perf_counter() - t0
tp, fp, fn, prec, rec, f1 = evaluate_pairs(pairs_w, df, gt)
print(f"  WEIGHTED:        P={prec:5.1%}  R={rec:5.1%}  F1={f1:5.1%}  {t_w:5.1f}s  ({len(pairs_w)} pairs)")

# ── Benchmark 1b: Fellegi-Sunter ──
print("  Running Fellegi-Sunter...", flush=True)
mk_fs = MatchkeyConfig(
    name="fs", type="probabilistic",
    fields=[
        MatchkeyField(field="title", scorer="token_sort", levels=3, partial_threshold=0.8, transforms=["lowercase"]),
        MatchkeyField(field="authors", scorer="token_sort", levels=3, partial_threshold=0.7, transforms=["lowercase"]),
        MatchkeyField(field="year", scorer="exact", levels=2),
    ],
)

t0 = time.perf_counter()
print("    Building blocks for EM training...", flush=True)
blocks_fs = build_blocks(df.lazy(), blocking)
print(f"    Training EM on within-block pairs from {len(blocks_fs)} blocks...", flush=True)
em = train_em(df, mk_fs, n_sample_pairs=15000, max_iterations=25, blocks=blocks_fs, blocking_fields=["year"])
print(f"    EM: converged={em.converged}, iters={em.iterations}, match_rate={em.proportion_matched:.4f}", flush=True)

print(f"    Scoring {len(blocks_fs)} blocks...", flush=True)
pairs_fs = []
for i, block in enumerate(blocks_fs):
    block_df = block.df.collect() if hasattr(block.df, 'collect') else block.df
    if block_df.height > 500:
        # Skip oversized blocks for speed
        continue
    p = score_probabilistic(block_df, mk_fs, em)
    pairs_fs.extend(p)
    if (i + 1) % 20 == 0:
        print(f"    Block {i+1}/{len(blocks_fs)}, {len(pairs_fs)} pairs so far...", flush=True)

t_fs = time.perf_counter() - t0
tp, fp, fn, prec, rec, f1_fs = evaluate_pairs(pairs_fs, df, gt)
print(f"  FELLEGI-SUNTER:  P={prec:5.1%}  R={rec:5.1%}  F1={f1_fs:5.1%}  {t_fs:5.1f}s  ({len(pairs_fs)} pairs)")

delta = f1_fs - f1
print(f"\n  Delta: {'+' if delta >= 0 else ''}{delta:.1%} F1")

# ── Benchmark 2: Static vs Learned Blocking ──
print(f"\n{'=' * 70}")
print("Benchmark 2: Static vs Learned Blocking (DBLP-ACM)")
print("=" * 70)

print("  Running static blocking...", flush=True)
t0 = time.perf_counter()
static_blocks = build_blocks(collected.lazy(), blocking)
n_static = len(static_blocks)
total_static = sum(b.df.collect().height for b in static_blocks)
matched2 = set()
pairs_static = score_blocks_parallel(static_blocks, mk_w, matched2)
t_static = time.perf_counter() - t0
tp, fp, fn, prec, rec, f1_s = evaluate_pairs(pairs_static, df, gt)
print(f"  STATIC:   P={prec:5.1%}  R={rec:5.1%}  F1={f1_s:5.1%}  {t_static:5.1f}s  blocks={n_static}, records_in_blocks={total_static}")

print("  Running learned blocking...", flush=True)
from goldenmatch.core.learned_blocking import learn_blocking_rules, apply_learned_blocks

t0 = time.perf_counter()
# Use a small sample for rule learning
sample = df.sample(min(500, df.height), seed=42)
# Score sample pairs quickly with static blocking
sample_lf = compute_matchkeys(sample.lazy(), [mk_w])
sample_collected = sample_lf.collect()
sample_blocks = build_blocks(sample_collected.lazy(), blocking)
sample_matched = set()
sample_pairs = score_blocks_parallel(sample_blocks, mk_w, sample_matched)
print(f"    Sample: {len(sample_pairs)} pairs from {sample.height} records", flush=True)

# Learn rules from just title and year
rules = learn_blocking_rules(
    sample_collected, sample_pairs,
    columns=["title", "year"],
    min_recall=0.50, min_reduction=0.20,
    predicate_depth=1,
)
print(f"    Learned {len(rules)} rules", flush=True)
for r in rules[:3]:
    print(f"      {r.key()}: recall={r.recall:.2f}, reduction={r.reduction_ratio:.2f}", flush=True)

learned_blocks = apply_learned_blocks(collected.lazy(), rules, max_block_size=1000)
n_learned = len(learned_blocks)
total_learned = sum(b.df.collect().height for b in learned_blocks)
matched3 = set()
pairs_learned_raw = score_blocks_parallel(learned_blocks, mk_w, matched3)
# Deduplicate pairs (same pair can appear in multiple learned blocks)
seen_pairs = {}
for a, b, s in pairs_learned_raw:
    key = (min(a, b), max(a, b))
    if key not in seen_pairs or s > seen_pairs[key]:
        seen_pairs[key] = s
pairs_learned = [(a, b, s) for (a, b), s in seen_pairs.items()]
t_learned = time.perf_counter() - t0
tp, fp, fn, prec, rec, f1_l = evaluate_pairs(pairs_learned, df, gt)
print(f"  LEARNED:  P={prec:5.1%}  R={rec:5.1%}  F1={f1_l:5.1%}  {t_learned:5.1f}s  blocks={n_learned}, records_in_blocks={total_learned}")

delta_b = f1_l - f1_s
print(f"\n  Delta: {'+' if delta_b >= 0 else ''}{delta_b:.1%} F1, blocks: {n_static} -> {n_learned}")

# ── Benchmark 3: LLM Budget Simulation ──
print(f"\n{'=' * 70}")
print("Benchmark 3: LLM Budget Simulation (Abt-Buy)")
print("=" * 70)

ds_dir = DATASETS_DIR / "Abt-Buy"
df_a = pl.read_csv(ds_dir / "Abt.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
df_b = pl.read_csv(ds_dir / "Buy.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
df_a = df_a.with_columns(pl.lit("abt").alias("__source__"))
df_b = df_b.with_columns(pl.lit("buy").alias("__source__"))
df_abt = pl.concat([df_a, df_b], how="diagonal").with_row_index("__row_id__")

gt_df = pl.read_csv(ds_dir / "abt_buy_perfectMapping.csv")
gt_abt = set()
for row in gt_df.to_dicts():
    gt_abt.add((str(row["idAbt"]).strip(), str(row["idBuy"]).strip()))

print(f"  Loaded: {df_abt.height} records, {len(gt_abt)} ground truth pairs")

mk_abt = MatchkeyConfig(
    name="fuzzy", type="weighted", threshold=0.50,
    fields=[MatchkeyField(field="name", scorer="token_sort", weight=1.0, transforms=["lowercase"])],
)
blocking_abt = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"])],
    max_block_size=200,
)

print("  Scoring pairs...", flush=True)
lf_abt = compute_matchkeys(df_abt.lazy(), [mk_abt])
coll_abt = lf_abt.collect()
blocks_abt = build_blocks(coll_abt.lazy(), blocking_abt)
m_abt = set()
pairs_abt = score_blocks_parallel(blocks_abt, mk_abt, m_abt)
print(f"  {len(pairs_abt)} total pairs found")

# Classify by tier
auto_accept = [(a,b,s) for a,b,s in pairs_abt if s >= 0.95]
candidates = [(a,b,s) for a,b,s in pairs_abt if 0.75 <= s < 0.95]
below = [(a,b,s) for a,b,s in pairs_abt if s < 0.75]
print(f"  Three-tier: {len(auto_accept)} auto-accept, {len(candidates)} candidates, {len(below)} below")

def eval_abt(pairs_subset, df, gt):
    rows = df.to_dicts()
    id_to_idx = {r["__row_id__"]: i for i, r in enumerate(rows)}
    tp = fp = 0
    for a, b, s in pairs_subset:
        ia, ib = id_to_idx.get(a), id_to_idx.get(b)
        if ia is None or ib is None:
            continue
        ra, rb = rows[ia], rows[ib]
        if ra.get("__source__") == rb.get("__source__"):
            continue
        if ra.get("__source__") == "abt":
            pair = (str(ra.get("id","")).strip(), str(rb.get("id","")).strip())
        else:
            pair = (str(rb.get("id","")).strip(), str(ra.get("id","")).strip())
        if pair in gt:
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

# ── Real LLM scoring with budget ──
import os
import subprocess

# Source the .testing/.env file (bash format with export statements)
env_file = Path(__file__).parent.parent.parent / ".testing" / ".env"
if env_file.exists():
    # Parse export KEY=VALUE lines (including commented-out ones we uncomment)
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("# export "):
            line = line[2:]  # uncomment
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("  WARNING: No OPENAI_API_KEY found. Skipping real LLM benchmark.")
else:
    from goldenmatch.core.llm_scorer import llm_score_pairs
    from goldenmatch.config.schemas import LLMScorerConfig, BudgetConfig

    print(f"\n  {'Budget':<12s}  {'LLM Calls':>10s}  {'Cost':>10s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*6}")

    # Baseline: no LLM (fuzzy only, filter >0.5)
    fuzzy_only = [(a,b,s) for a,b,s in pairs_abt if s > 0.5]
    prec, rec, f1 = eval_abt(fuzzy_only, df_abt, gt_abt)
    print(f"  {'no LLM':<12s}  {'0':>10s}  ${'0.0000':>8s}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}")

    for budget_usd in [0.10, 0.50, 999.0]:
        cfg = LLMScorerConfig(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            auto_threshold=0.95,
            candidate_lo=0.75,
            candidate_hi=0.95,
            budget=BudgetConfig(max_cost_usd=budget_usd),
        )
        print(f"  Running LLM scorer (budget=${budget_usd:.2f})...", flush=True)
        scored, budget_summary = llm_score_pairs(
            pairs_abt, df_abt,
            config=cfg, api_key=api_key, return_budget=True,
        )
        # Filter to matches
        llm_matches = [(a,b,s) for a,b,s in scored if s > 0.5]
        prec, rec, f1 = eval_abt(llm_matches, df_abt, gt_abt)

        cost = budget_summary["total_cost_usd"] if budget_summary else 0
        calls = budget_summary["total_calls"] if budget_summary else 0
        label = f"${budget_usd:.2f}" if budget_usd < 100 else "unlimited"
        print(f"  {label:<12s}  {calls:>10d}  ${cost:>8.4f}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}")

# ── Summary ──
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print("""
  Feature                          Finding
  -------------------------------- ------------------------------------------------
  Fellegi-Sunter vs Weighted       See F1 comparison above
  Learned vs Static Blocking       See block count and F1 comparison above
  LLM Budget                       See cost/accuracy table above
""")
