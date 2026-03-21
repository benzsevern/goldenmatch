#!/usr/bin/env python
"""LLM Scorer + Budget benchmark on Abt-Buy with embedding+ANN blocking.

Tests the LLM scorer on top of the best fuzzy baseline (rec_emb+ann_pairs)
with budget controls. Requires OPENAI_API_KEY in .testing/.env.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

# Load API key from .testing/.env
env_file = Path(__file__).parent.parent.parent / ".testing" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")

import polars as pl
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_fuzzy_matches
from goldenmatch.core.llm_scorer import llm_score_pairs
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField,
    BlockingConfig, BlockingKeyConfig,
    LLMScorerConfig, BudgetConfig,
)

DATASETS_DIR = Path(__file__).parent / "datasets"


def load_abt_buy():
    ds_dir = DATASETS_DIR / "Abt-Buy"
    df_a = pl.read_csv(ds_dir / "Abt.csv", encoding="utf8-lossy", ignore_errors=True)
    df_b = pl.read_csv(ds_dir / "Buy.csv", encoding="utf8-lossy", ignore_errors=True)
    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns})
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns})
    df_a = df_a.with_columns(pl.lit("abt").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("buy").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    combined, _ = auto_fix_dataframe(combined)

    gt_df = pl.read_csv(ds_dir / "abt_buy_perfectMapping.csv")
    gt = set()
    for row in gt_df.to_dicts():
        gt.add((str(row["idAbt"]).strip(), str(row["idBuy"]).strip()))
    return combined, gt


def evaluate(pairs, df, gt):
    rows = df.to_dicts()
    id_to_idx = {r["__row_id__"]: i for i, r in enumerate(rows)}
    tp = fp = 0
    for a, b, s in pairs:
        ia, ib = id_to_idx.get(a), id_to_idx.get(b)
        if ia is None or ib is None:
            continue
        ra, rb = rows[ia], rows[ib]
        if ra.get("__source__") == rb.get("__source__"):
            continue
        if ra.get("__source__") == "abt":
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
print("LLM Scorer Benchmark: Abt-Buy with Embedding+ANN Blocking")
print("=" * 70)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: No OPENAI_API_KEY found.")
    sys.exit(1)
print(f"API key loaded: {api_key[:10]}...{api_key[-4:]}")

df, gt = load_abt_buy()
print(f"Loaded: {df.height} records, {len(gt)} ground truth pairs\n")

# ── Embedding + ANN blocking ──
print("Building embedding+ANN baseline...", flush=True)

mk = MatchkeyConfig(
    name="rec_emb",
    fields=[
        MatchkeyField(
            scorer="record_embedding",
            columns=["name"],
            weight=1.0,
            model="all-MiniLM-L6-v2",
        ),
    ],
    comparison="weighted",
    threshold=0.80,
)

blocking = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
    strategy="ann_pairs",
    ann_column="name",
    ann_model="all-MiniLM-L6-v2",
    ann_top_k=20,
)

lf = df.lazy()
lf = apply_standardization(lf, {"name": ["strip", "trim_whitespace"]})
lf = compute_matchkeys(lf, [mk])
collected = lf.collect()

t0 = time.perf_counter()
blocks = build_blocks(collected.lazy(), blocking)
all_pairs = []
for block in blocks:
    bdf = block.df.collect()
    pairs = find_fuzzy_matches(bdf, mk, pre_scored_pairs=block.pre_scored_pairs)
    all_pairs.extend(pairs)

# Cross-source filter
row_src = {r["__row_id__"]: r["__source__"] for r in collected.select("__row_id__", "__source__").to_dicts()}
all_pairs = [(a, b, s) for a, b, s in all_pairs if row_src.get(a) != row_src.get(b)]
t_base = time.perf_counter() - t0

tp, fp, fn, prec, rec, f1_base = evaluate(all_pairs, df, gt)

auto_accept = [(a,b,s) for a,b,s in all_pairs if s >= 0.95]
candidates = [(a,b,s) for a,b,s in all_pairs if 0.75 <= s < 0.95]
below = [(a,b,s) for a,b,s in all_pairs if s < 0.75]

print(f"  Baseline: {len(all_pairs)} pairs ({len(auto_accept)} auto-accept, {len(candidates)} candidates, {len(below)} below)")
print(f"  P={prec:5.1%}  R={rec:5.1%}  F1={f1_base:5.1%}  Time={t_base:.1f}s\n")

# ── LLM scoring ──
print(f"{'Strategy':<35s}  {'Calls':>5s}  {'Cost':>8s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
print(f"{'-'*35}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
print(f"{'emb+ANN baseline (no LLM)':<35s}  {'0':>5s}  {'$0.00':>8s}  {prec:5.1%}  {rec:5.1%}  {f1_base:5.1%}")

for budget_usd, label in [(0.05, "$0.05"), (0.25, "$0.25"), (1.00, "$1.00"), (999.0, "unlimited")]:
    cfg = LLMScorerConfig(
        enabled=True,
        provider="openai",
        model="gpt-4o-mini",
        auto_threshold=0.95,
        candidate_lo=0.75,
        candidate_hi=0.95,
        budget=BudgetConfig(max_cost_usd=budget_usd),
    )

    print(f"  Running LLM (budget={label})...", flush=True, end="")
    scored, budget_summary = llm_score_pairs(
        all_pairs, collected,
        config=cfg, api_key=api_key, return_budget=True,
    )

    llm_matches = [(a, b, s) for a, b, s in scored if s > 0.5]
    tp, fp, fn, prec, rec, f1 = evaluate(llm_matches, df, gt)

    cost = budget_summary["total_cost_usd"] if budget_summary else 0
    calls = budget_summary["total_calls"] if budget_summary else 0
    display = f"emb+ANN+LLM ({label})"
    delta = f1 - f1_base
    print(f"\r{display:<35s}  {calls:>5d}  ${cost:>6.4f}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}  ({'+' if delta>=0 else ''}{delta:.1%})")

print(f"\n{'=' * 70}")
