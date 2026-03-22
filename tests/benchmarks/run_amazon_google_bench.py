#!/usr/bin/env python
"""Amazon-GoogleProducts benchmark: embedding+ANN + LLM scorer.

Clean pipeline: embedding blocking -> fuzzy scoring -> LLM precision filter.
No domain extraction (software titles lack precise identifiers).
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

env_file = Path(__file__).parent.parent.parent / ".testing" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

import polars as pl
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_fuzzy_matches
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    LLMScorerConfig, BudgetConfig,
)

DATASETS_DIR = Path(__file__).parent / "datasets" / "Amazon-GoogleProducts"

# Load data
df_a = pl.read_csv(DATASETS_DIR / "Amazon.csv", encoding="utf8-lossy", ignore_errors=True)
df_b = pl.read_csv(DATASETS_DIR / "GoogleProducts.csv", encoding="utf8-lossy", ignore_errors=True)
df_b = df_b.rename({"name": "title"})

df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns})
df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns})
df_a = df_a.with_columns(pl.lit("amazon").alias("__source__"))
df_b = df_b.with_columns(pl.lit("google").alias("__source__"))
df = pl.concat([df_a, df_b], how="diagonal")
df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
df, _ = auto_fix_dataframe(df)

gt_df = pl.read_csv(DATASETS_DIR / "Amzon_GoogleProducts_perfectMapping.csv")
gt = set((str(r["idAmazon"]).strip(), str(r["idGoogleBase"]).strip()) for r in gt_df.to_dicts())

row_src = {r["__row_id__"]: r["__source__"] for r in df.select("__row_id__", "__source__").to_dicts()}


def cross_filter(pairs):
    return [(a, b, s) for a, b, s in pairs if row_src.get(a) != row_src.get(b)]


def ev(pairs):
    rows_d = df.to_dicts()
    idx = {r["__row_id__"]: i for i, r in enumerate(rows_d)}
    tp = fp = 0
    for a, b, s in pairs:
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None:
            continue
        ra, rb = rows_d[ia], rows_d[ib]
        if ra.get("__source__") == rb.get("__source__"):
            continue
        if ra.get("__source__") == "amazon":
            p = (str(ra.get("id", "")).strip(), str(rb.get("id", "")).strip())
        else:
            p = (str(rb.get("id", "")).strip(), str(ra.get("id", "")).strip())
        if p in gt:
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    pr = tp / (tp + fp) if (tp + fp) else 0
    rc = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
    return pr, rc, f1


print("=" * 70)
print("Amazon-GoogleProducts Benchmark: emb+ANN + LLM")
print("=" * 70)
print(f"Loaded: {df.height} records, {len(gt)} ground truth pairs\n")

# Embedding + ANN blocking
mk_emb = MatchkeyConfig(
    name="rec_emb", comparison="weighted", threshold=0.80,
    fields=[MatchkeyField(scorer="record_embedding", columns=["title"], weight=1.0, model="all-MiniLM-L6-v2")],
)
blocking = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
    strategy="ann_pairs", ann_column="title", ann_model="all-MiniLM-L6-v2", ann_top_k=20,
)

lf = df.lazy()
lf = apply_standardization(lf, {"title": ["strip", "trim_whitespace"]})
lf = compute_matchkeys(lf, [mk_emb])
collected = lf.collect()

print("Building emb+ANN pairs...", flush=True)
t0 = time.perf_counter()
blocks = build_blocks(collected.lazy(), blocking)
all_pairs = []
for block in blocks:
    bdf = block.df.collect()
    all_pairs.extend(find_fuzzy_matches(bdf, mk_emb, pre_scored_pairs=block.pre_scored_pairs))
all_pairs = cross_filter(all_pairs)
t_base = time.perf_counter() - t0

pr, rc, f1 = ev(all_pairs)
auto_accept = sum(1 for _, _, s in all_pairs if s >= 0.95)
candidates = sum(1 for _, _, s in all_pairs if 0.75 <= s < 0.95)
below = sum(1 for _, _, s in all_pairs if s < 0.75)
print(f"  {len(all_pairs)} pairs ({auto_accept} auto-accept, {candidates} candidates, {below} below)")

print(f"\n{'Strategy':<35s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>6s}  {'Cost':>8s}")
print(f"{'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")
print(f"{'emb+ANN baseline':<35s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(all_pairs):>6d}  {'$0.00':>8s}")

# LLM scoring
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    from goldenmatch.core.llm_scorer import llm_score_pairs

    cfg = LLMScorerConfig(
        enabled=True, provider="openai", model="gpt-4o-mini",
        auto_threshold=0.95, candidate_lo=0.75, candidate_hi=0.95,
        budget=BudgetConfig(max_cost_usd=2.00),
    )
    print("Running LLM scorer...", flush=True, end="")
    scored, budget = llm_score_pairs(
        all_pairs, collected, config=cfg, api_key=api_key, return_budget=True,
    )
    llm_matches = [(a, b, s) for a, b, s in scored if s > 0.5]
    pr, rc, f1 = ev(llm_matches)
    cost = budget["total_cost_usd"] if budget else 0
    calls = budget["total_calls"] if budget else 0
    print(f"\r{'emb+ANN + LLM':<35s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(llm_matches):>6d}  ${cost:>7.4f}")
else:
    print("  [SKIPPED LLM -- no OPENAI_API_KEY]")

print(f"\n{'=' * 70}")
