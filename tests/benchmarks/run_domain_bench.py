#!/usr/bin/env python
"""Domain extraction + normalization benchmark on Abt-Buy.

Tests the impact of domain-aware feature extraction on product matching:
1. Baseline: embedding+ANN only
2. + Domain extraction (model exact match as additional matchkey)
3. + Model normalization
4. + Model containment
5. + LLM scorer on remaining pairs
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.stdout.reconfigure(line_buffering=True)

# Load API key
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
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.core.domain import (
    detect_domain, extract_features, normalize_model, model_contains,
)
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
print("Domain Extraction + Normalization Benchmark (Abt-Buy)")
print("=" * 70)

df, gt = load_abt_buy()
print(f"Loaded: {df.height} records, {len(gt)} ground truth pairs\n")

# ── Step 1: Domain extraction ──
print("Step 1: Domain-aware feature extraction...", flush=True)
domain = detect_domain([c for c in df.columns if not c.startswith("__")])
enhanced, low_conf = extract_features(df, domain, confidence_threshold=0.3)
print(f"  Domain: {domain.name}, {len(low_conf)} low-confidence records")

models = enhanced["__model_norm__"].drop_nulls()
brands = enhanced["__brand__"].drop_nulls()
print(f"  Models: {len(models)}/{df.height} extracted ({models.n_unique()} unique)")
print(f"  Brands: {len(brands)}/{df.height} extracted ({brands.n_unique()} unique)")

# ── Step 2: Build matching strategies ──

# Source lookup for cross-source filtering
row_src = {r["__row_id__"]: r["__source__"] for r in enhanced.select("__row_id__", "__source__").to_dicts()}

def cross_source_filter(pairs):
    return [(a, b, s) for a, b, s in pairs if row_src.get(a) != row_src.get(b)]

def dedupe_pairs(pairs):
    best = {}
    for a, b, s in pairs:
        key = (min(a, b), max(a, b))
        if key not in best or s > best[key]:
            best[key] = s
    return [(a, b, s) for (a, b), s in best.items()]

print(f"\n{'Strategy':<45s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Pairs':>6s}  {'Time':>6s}")
print(f"{'-'*45}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

# ── Strategy A: Embedding+ANN baseline (no domain extraction) ──
print("  Building embedding+ANN baseline...", flush=True, end="")
mk_emb = MatchkeyConfig(
    name="rec_emb", comparison="weighted", threshold=0.80,
    fields=[MatchkeyField(scorer="record_embedding", columns=["name"], weight=1.0, model="all-MiniLM-L6-v2")],
)
blocking_ann = BlockingConfig(
    keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
    strategy="ann_pairs", ann_column="name", ann_model="all-MiniLM-L6-v2", ann_top_k=20,
)
lf = enhanced.lazy()
lf = apply_standardization(lf, {"name": ["strip", "trim_whitespace"]})
lf = compute_matchkeys(lf, [mk_emb])
collected = lf.collect()

t0 = time.perf_counter()
blocks = build_blocks(collected.lazy(), blocking_ann)
emb_pairs = []
for block in blocks:
    bdf = block.df.collect()
    emb_pairs.extend(find_fuzzy_matches(bdf, mk_emb, pre_scored_pairs=block.pre_scored_pairs))
emb_pairs = cross_source_filter(emb_pairs)
t_emb = time.perf_counter() - t0
tp, fp, fn, prec, rec, f1 = evaluate(emb_pairs, enhanced, gt)
print(f"\r{'A: emb+ANN baseline':<45s}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}  {len(emb_pairs):>6d}  {t_emb:5.1f}s")

# ── Strategy B: Model normalized exact match only ──
print("  Building model norm exact...", flush=True, end="")
mk_model = MatchkeyConfig(
    name="model_exact", comparison="exact",
    fields=[MatchkeyField(field="__model_norm__")],
)
lf2 = compute_matchkeys(collected.lazy(), [mk_model])
collected2 = lf2.collect()

t0 = time.perf_counter()
model_pairs = find_exact_matches(collected2.lazy(), mk_model)
model_pairs = cross_source_filter(model_pairs)
t_model = time.perf_counter() - t0
tp, fp, fn, prec, rec, f1 = evaluate(model_pairs, enhanced, gt)
print(f"\r{'B: model_norm exact only':<45s}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}  {len(model_pairs):>6d}  {t_model:5.1f}s")

# ── Strategy C: Model containment (scored as pairs) ──
print("  Building model containment...", flush=True, end="")
t0 = time.perf_counter()
rows = enhanced.to_dicts()
# Build model index for containment matching
model_to_rows = {}
for r in rows:
    m = r.get("__model__")
    if m:
        mn = normalize_model(m)
        if mn:
            model_to_rows.setdefault(mn, []).append(r["__row_id__"])

# Find containment pairs
contain_pairs = []
seen = set()
mn_keys = list(model_to_rows.keys())
for i in range(len(mn_keys)):
    for j in range(i + 1, len(mn_keys)):
        ki, kj = mn_keys[i], mn_keys[j]
        if ki in kj or kj in ki:
            for ri in model_to_rows[ki]:
                for rj in model_to_rows[kj]:
                    pair = (min(ri, rj), max(ri, rj))
                    if pair not in seen:
                        seen.add(pair)
                        contain_pairs.append((pair[0], pair[1], 0.90))

contain_pairs = cross_source_filter(contain_pairs)
t_contain = time.perf_counter() - t0
tp, fp, fn, prec, rec, f1 = evaluate(contain_pairs, enhanced, gt)
print(f"\r{'C: model containment only':<45s}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}  {len(contain_pairs):>6d}  {t_contain:5.1f}s")

# ── Strategy D: Model exact + emb+ANN combined ──
print("  Combining model + embedding...", flush=True, end="")
combined_pairs = dedupe_pairs(model_pairs + emb_pairs)
tp, fp, fn, prec, rec, f1 = evaluate(combined_pairs, enhanced, gt)
print(f"\r{'D: model_norm exact + emb+ANN':<45s}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}  {len(combined_pairs):>6d}  {t_model + t_emb:5.1f}s")

# ── Strategy E: Model containment + emb+ANN combined ──
combined_contain = dedupe_pairs(contain_pairs + emb_pairs)
tp, fp, fn, prec, rec, f1 = evaluate(combined_contain, enhanced, gt)
print(f"{'E: model containment + emb+ANN':<45s}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}  {len(combined_contain):>6d}  {t_contain + t_emb:5.1f}s")

# ── Strategy F: Everything + LLM ──
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    from goldenmatch.core.llm_scorer import llm_score_pairs

    print("  Running LLM scorer on combined pairs...", flush=True, end="")
    cfg = LLMScorerConfig(
        enabled=True, provider="openai", model="gpt-4o-mini",
        auto_threshold=0.95, candidate_lo=0.75, candidate_hi=0.95,
        budget=BudgetConfig(max_cost_usd=1.00),
    )
    t0 = time.perf_counter()
    scored, budget = llm_score_pairs(
        combined_contain, collected,
        config=cfg, api_key=api_key, return_budget=True,
    )
    llm_matches = [(a, b, s) for a, b, s in scored if s > 0.5]
    t_llm = time.perf_counter() - t0
    tp, fp, fn, prec, rec, f1 = evaluate(llm_matches, enhanced, gt)
    cost = budget["total_cost_usd"] if budget else 0
    calls = budget["total_calls"] if budget else 0
    print(f"\r{'F: containment + emb+ANN + LLM':<45s}  {prec:5.1%}  {rec:5.1%}  {f1:5.1%}  {len(llm_matches):>6d}  {t_llm:5.1f}s  ${cost:.4f} ({calls} calls)")
else:
    print("  [SKIPPED LLM -- no OPENAI_API_KEY]")

print(f"\n{'=' * 70}")
