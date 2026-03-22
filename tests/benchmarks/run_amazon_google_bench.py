#!/usr/bin/env python
"""Domain extraction benchmark on Amazon-GoogleProducts."""

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
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches, score_blocks_parallel
from goldenmatch.core.domain import detect_domain, extract_features, normalize_model
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    LLMScorerConfig, BudgetConfig,
)

DS = Path(__file__).parent / "datasets" / "Amazon-GoogleProducts"

# Load data
df_a = pl.read_csv(DS / "Amazon.csv", encoding="utf8-lossy", ignore_errors=True)
df_b = pl.read_csv(DS / "GoogleProducts.csv", encoding="utf8-lossy", ignore_errors=True)
df_b = df_b.rename({"name": "title"})

df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns})
df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns})
df_a = df_a.with_columns(pl.lit("amazon").alias("__source__"))
df_b = df_b.with_columns(pl.lit("google").alias("__source__"))
df = pl.concat([df_a, df_b], how="diagonal")
df = df.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
df, _ = auto_fix_dataframe(df)

gt_df = pl.read_csv(DS / "Amzon_GoogleProducts_perfectMapping.csv")
gt = set((str(r["idAmazon"]).strip(), str(r["idGoogleBase"]).strip()) for r in gt_df.to_dicts())

print("=" * 70)
print("Domain Extraction Benchmark: Amazon-GoogleProducts")
print("=" * 70)
print(f"Loaded: {df.height} records, {len(gt)} ground truth pairs")

# Domain extraction
domain = detect_domain([c for c in df.columns if not c.startswith("__")])
enhanced, low_conf = extract_features(df, domain, confidence_threshold=0.3)

if "__sw_name__" in enhanced.columns:
    sw_names = enhanced["__sw_name__"].drop_nulls()
    sw_versions = enhanced["__sw_version__"].drop_nulls()
    print(f"Domain: {domain.name} (subdomain: {domain.subdomain})")
    print(f"  SW names: {len(sw_names)}/{df.height}, versions: {len(sw_versions)}/{df.height}")
elif "__model_norm__" in enhanced.columns:
    models = enhanced["__model_norm__"].drop_nulls()
    brands = enhanced["__brand__"].drop_nulls()
    print(f"Domain: {domain.name} (subdomain: {domain.subdomain})")
    print(f"  Models: {len(models)}/{df.height}, brands: {len(brands)}/{df.height}")
else:
    print(f"Domain: {domain.name} -- no extraction columns")

row_src = {r["__row_id__"]: r["__source__"] for r in enhanced.select("__row_id__", "__source__").to_dicts()}


def cross_filter(pairs):
    return [(a, b, s) for a, b, s in pairs if row_src.get(a) != row_src.get(b)]


def dedupe(pairs):
    best = {}
    for a, b, s in pairs:
        key = (min(a, b), max(a, b))
        if key not in best or s > best[key]:
            best[key] = s
    return [(a, b, s) for (a, b), s in best.items()]


def ev(pairs):
    rows_d = enhanced.to_dicts()
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


print(f"\n{'Strategy':<45s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>6s}")
print(f"{'-' * 45}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}")

# A: Embedding+ANN baseline
print("  Building emb+ANN...", flush=True, end="")
mk_emb = MatchkeyConfig(
    name="rec_emb", comparison="weighted", threshold=0.80,
    fields=[MatchkeyField(scorer="record_embedding", columns=["title"], weight=1.0, model="all-MiniLM-L6-v2")],
)
blocking_ann = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
    strategy="ann_pairs", ann_column="title", ann_model="all-MiniLM-L6-v2", ann_top_k=20,
)
lf = enhanced.lazy()
lf = apply_standardization(lf, {"title": ["strip", "trim_whitespace"]})
lf = compute_matchkeys(lf, [mk_emb])
collected = lf.collect()

blocks = build_blocks(collected.lazy(), blocking_ann)
emb_pairs = []
for block in blocks:
    bdf = block.df.collect()
    emb_pairs.extend(find_fuzzy_matches(bdf, mk_emb, pre_scored_pairs=block.pre_scored_pairs))
emb_pairs = cross_filter(emb_pairs)
pr, rc, f1 = ev(emb_pairs)
print(f"\r{'A: emb+ANN baseline':<45s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(emb_pairs):>6d}")

# B: Software name exact (if software subdomain detected)
sw_name_pairs = []
if "__sw_name__" in collected.columns:
    mk_sw = MatchkeyConfig(name="sw_name_exact", comparison="exact", fields=[MatchkeyField(field="__sw_name__")])
    lf2 = compute_matchkeys(collected.lazy(), [mk_sw])
    collected2 = lf2.collect()
    sw_name_pairs = find_exact_matches(collected2.lazy(), mk_sw)
    sw_name_pairs = cross_filter(sw_name_pairs)
    pr, rc, f1 = ev(sw_name_pairs)
    print(f"{'B: sw_name_norm exact':<45s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(sw_name_pairs):>6d}")

# B2: Model norm exact (if electronics extraction ran)
model_pairs = []
if "__model_norm__" in collected.columns:
    mk_model = MatchkeyConfig(name="model_exact", comparison="exact", fields=[MatchkeyField(field="__model_norm__")])
    lf3 = compute_matchkeys(collected.lazy(), [mk_model])
    collected3 = lf3.collect()
    model_pairs = find_exact_matches(collected3.lazy(), mk_model)
    model_pairs = cross_filter(model_pairs)
    if model_pairs:
        pr, rc, f1 = ev(model_pairs)
        print(f"{'B2: model_norm exact':<45s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(model_pairs):>6d}")

# B3: sw_name blocking + title fuzzy (software-specific strategy)
sw_fuzzy_pairs = []
if "__sw_name__" in collected.columns:
    mk_sw_fuzzy = MatchkeyConfig(
        name="sw_title_fuzzy", comparison="weighted", threshold=0.65,
        fields=[MatchkeyField(field="title", scorer="token_sort", weight=1.0, transforms=["lowercase"])],
    )
    sw_blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["__sw_name__"])],
        max_block_size=200,
    )
    sw_lf = compute_matchkeys(collected.lazy(), [mk_sw_fuzzy])
    sw_collected = sw_lf.collect()
    sw_blocks = build_blocks(sw_collected.lazy(), sw_blocking)
    sw_matched = set()
    sw_fuzzy_pairs = score_blocks_parallel(sw_blocks, mk_sw_fuzzy, sw_matched)
    sw_fuzzy_pairs = cross_filter(sw_fuzzy_pairs)
    pr, rc, f1 = ev(sw_fuzzy_pairs)
    print(f"{'B3: sw_name_block + title_fuzzy(0.65)':<45s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(sw_fuzzy_pairs):>6d}")

# C: All extraction + emb combined
combined = dedupe(sw_name_pairs + sw_fuzzy_pairs + model_pairs + emb_pairs)
pr, rc, f1 = ev(combined)
print(f"{'C: extraction + emb+ANN':<45s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(combined):>6d}")

# D: Manufacturer blocking + title fuzzy
mfr_pairs = []
if "manufacturer" in collected.columns:
    mk_fuzzy = MatchkeyConfig(
        name="fuzzy_title", comparison="weighted", threshold=0.70,
        fields=[MatchkeyField(field="title", scorer="token_sort", weight=1.0, transforms=["lowercase"])],
    )
    mfr_blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["manufacturer"], transforms=["lowercase", "strip"])],
        max_block_size=500,
    )
    mfr_lf = compute_matchkeys(collected.lazy(), [mk_fuzzy])
    mfr_collected = mfr_lf.collect()
    mfr_blocks = build_blocks(mfr_collected.lazy(), mfr_blocking)
    matched_m = set()
    mfr_pairs = score_blocks_parallel(mfr_blocks, mk_fuzzy, matched_m)
    mfr_pairs = cross_filter(mfr_pairs)
    pr, rc, f1 = ev(mfr_pairs)
    print(f"{'D: mfr_block + title_fuzzy(0.70)':<45s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(mfr_pairs):>6d}")

# E: Everything combined
all_combined = dedupe(sw_name_pairs + sw_fuzzy_pairs + model_pairs + emb_pairs + mfr_pairs)
pr_e, rc_e, f1_e = ev(all_combined)
print(f"{'E: extraction + emb + mfr combined':<45s}  {pr_e:5.1%}  {rc_e:5.1%}  {f1_e:5.1%}  {len(all_combined):>6d}")

# F: + LLM
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    from goldenmatch.core.llm_scorer import llm_score_pairs

    cfg = LLMScorerConfig(
        enabled=True, provider="openai", model="gpt-4o-mini",
        auto_threshold=0.95, candidate_lo=0.75, candidate_hi=0.95,
        budget=BudgetConfig(max_cost_usd=2.00),
    )
    print("  Running LLM scorer...", flush=True, end="")
    scored, budget = llm_score_pairs(
        all_combined, collected, config=cfg, api_key=api_key, return_budget=True,
    )
    llm_matches = [(a, b, s) for a, b, s in scored if s > 0.5]
    pr, rc, f1 = ev(llm_matches)
    cost = budget["total_cost_usd"] if budget else 0
    calls = budget["total_calls"] if budget else 0
    print(f"\r{'F: model + emb + mfr + LLM':<45s}  {pr:5.1%}  {rc:5.1%}  {f1:5.1%}  {len(llm_matches):>6d}  ${cost:.4f} ({calls} calls)")
else:
    print("  [SKIPPED LLM -- no OPENAI_API_KEY]")

print(f"\n{'=' * 70}")
