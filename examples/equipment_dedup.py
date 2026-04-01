"""Equipment auction deduplication with GoldenMatch.

Demonstrates the optimal configuration for large equipment datasets:
- Multi-pass string blocking + ANN fallback for oversized blocks
- Weighted fuzzy scoring (ensemble, jaro_winkler, token_sort, exact)
- Iterative LLM calibration (learns threshold from ~200 pairs, ~$0.01)
- Vertex AI embeddings for ANN sub-blocking (optional)

Required environment variables:
  OPENAI_API_KEY          For LLM calibration (required)
  DATA_PATH               Path to input CSV (default: equipment.csv)
  OUTPUT_DIR              Directory for output CSVs (default: current directory)

Optional environment variables (for ANN embeddings via Vertex AI):
  GOOGLE_CLOUD_PROJECT    GCP project ID
  GOLDENMATCH_GPU_MODE    Set to "vertex" to enable Vertex AI embeddings

Dataset: Kaggle Bulldozer Blue Book (401K rows)
  https://www.kaggle.com/datasets/farhanreynaldo/bulldozer-blue-book

Tip: For datasets >500K rows, use the DuckDB backend or chunked processing
to avoid OOM. See the GoldenMatch docs for --backend duckdb usage.

Usage:
    pip install goldenmatch
    export OPENAI_API_KEY=sk-...
    export DATA_PATH=equipment.csv
    python examples/equipment_dedup.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import Counter

import polars as pl

import goldenmatch
from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    BudgetConfig,
    GoldenMatchConfig,
    LLMScorerConfig,
    MatchkeyConfig,
    MatchkeyField,
    StandardizationConfig,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# Suppress noisy internal checker logs; keep goldenmatch INFO visible
logging.getLogger("goldencheck").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_PATH = os.environ.get("DATA_PATH", "equipment.csv")
print(f"Loading data from {DATA_PATH}...", flush=True)

# utf8-lossy handles encoding issues common in scraped auction exports;
# ignore_errors drops rows with structural CSV problems (extra commas, etc.)
df = pl.read_csv(DATA_PATH, encoding="utf8-lossy", ignore_errors=True)

# Cast everything to string so downstream transforms never see nulls-as-ints.
# Equipment CSVs often mix numeric and text in the same column (e.g. YearMade).
df = df.cast({col: pl.Utf8 for col in df.columns})

# ---------------------------------------------------------------------------
# Build a concatenated text column for ANN embedding fallback.
#
# When a blocking key produces a block larger than max_block_size (1 000 rows),
# the blocker falls back to ANN sub-blocking: it embeds ann_column, builds a
# FAISS index, and retrieves the top-k nearest neighbours per record instead of
# exhaustively comparing every pair. This keeps large "misc" blocks tractable.
# ---------------------------------------------------------------------------

df = df.with_columns(
    (
        pl.col("fiModelDesc").fill_null("") + " | " +
        pl.col("fiBaseModel").fill_null("") + " | " +
        pl.col("fiProductClassDesc").fill_null("") + " | " +
        pl.col("ProductGroup").fill_null("") + " | " +
        pl.col("ProductSize").fill_null("")
    ).alias("__equipment_text__")
)

print(f"Loaded {len(df):,} rows, {len(df.columns)} columns", flush=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

config = GoldenMatchConfig(

    # ------------------------------------------------------------------
    # Standardization
    # Normalise key fields before blocking and scoring so that
    # "CASE 580N" and "case 580n" land in the same block.
    # ------------------------------------------------------------------
    standardization=StandardizationConfig(rules={
        "fiModelDesc":        ["name_lower", "trim_whitespace"],
        "fiBaseModel":        ["name_lower", "strip"],
        "state":              ["strip"],
        "ProductGroup":       ["name_lower", "strip"],
    }),

    # ------------------------------------------------------------------
    # Multi-pass blocking with ANN fallback for oversized blocks.
    #
    # Pass 1 — model + state:    catches same-model machines in same region.
    # Pass 2 — model + category: catches cross-state same-equipment dupes.
    # Pass 3 — base model soundex: phonetic pass for model abbreviations
    #           (e.g. "D6T" vs "D-6T", "580N" vs "580-N").
    #
    # Any block that exceeds max_block_size uses ANN sub-blocking instead
    # of exhaustive comparison. ann_top_k controls how many neighbours to
    # retrieve per record inside those oversized blocks.
    # ------------------------------------------------------------------
    blocking=BlockingConfig(
        strategy="multi_pass",
        # `keys` is required even with multi_pass; use the primary pass here.
        keys=[
            BlockingKeyConfig(
                fields=["fiModelDesc", "state"],
                transforms=["lowercase", "strip"],
            ),
        ],
        passes=[
            BlockingKeyConfig(
                fields=["fiModelDesc", "state"],
                transforms=["lowercase", "strip"],
            ),
            BlockingKeyConfig(
                fields=["fiModelDesc", "ProductGroup"],
                transforms=["lowercase", "strip"],
            ),
            BlockingKeyConfig(
                fields=["fiBaseModel"],
                transforms=["lowercase", "soundex"],
            ),
        ],
        max_block_size=1000,
        skip_oversized=True,         # oversized blocks fall back to ANN
        ann_column="__equipment_text__",
        ann_top_k=20,
    ),

    # ------------------------------------------------------------------
    # Weighted fuzzy matchkey.
    #
    # Field weights reflect how discriminating each attribute is for
    # equipment identity:
    #   fiModelDesc   (2.0) — the full model string; most specific signal
    #   fiBaseModel   (1.5) — base chassis; strong but shared across variants
    #   fiProductClassDesc (1.0) — equipment category (wide); useful tie-breaker
    #   YearMade      (0.8) — year of manufacture; same model can span years
    #   ProductGroup  (0.5) — coarse grouping (e.g. "WL"); low discrimination
    #   state         (0.3) — geography; coincidental, lowest weight
    #
    # Scorer choices:
    #   ensemble      — combines jaro_winkler + levenshtein + token_sort + dice;
    #                   best for model strings that mix chars and numbers
    #   jaro_winkler  — best for short strings where prefix matters (base models)
    #   token_sort    — handles word-order differences in category descriptions
    #   exact         — hard match for structured fields (year, group, state)
    #
    # The overall threshold of 0.90 is intentionally high: equipment records
    # are detailed enough that false positives below 0.90 are common.
    # LLM calibration (below) will refine this automatically.
    # ------------------------------------------------------------------
    matchkeys=[
        MatchkeyConfig(
            name="equipment_match",
            type="weighted",
            threshold=0.90,
            fields=[
                MatchkeyField(
                    field="fiModelDesc",
                    scorer="ensemble",
                    weight=2.0,
                    transforms=["lowercase", "strip"],
                ),
                MatchkeyField(
                    field="fiBaseModel",
                    scorer="jaro_winkler",
                    weight=1.5,
                    transforms=["lowercase", "strip"],
                ),
                MatchkeyField(
                    field="fiProductClassDesc",
                    scorer="token_sort",
                    weight=1.0,
                    transforms=["lowercase", "strip"],
                ),
                MatchkeyField(
                    field="YearMade",
                    scorer="exact",
                    weight=0.8,
                ),
                MatchkeyField(
                    field="state",
                    scorer="exact",
                    weight=0.3,
                ),
                MatchkeyField(
                    field="ProductGroup",
                    scorer="exact",
                    weight=0.5,
                ),
            ],
        ),
    ],

    # ------------------------------------------------------------------
    # LLM calibration for borderline pairs.
    #
    # Pairs scoring between candidate_lo (0.85) and candidate_hi (0.95) are
    # considered borderline. The LLM samples ~200 of these, labels them, and
    # uses the labels to learn the optimal threshold (iterative calibration).
    # Cost for the calibration phase on a 400K-row dataset is roughly $0.01.
    #
    # After calibration, pairs above auto_threshold (0.95) are auto-accepted
    # without LLM calls; only genuine borderline pairs are reviewed.
    #
    # Budget cap: $1.00 maximum spend, with a warning at 80% of budget.
    # ------------------------------------------------------------------
    llm_scorer=LLMScorerConfig(
        enabled=True,
        provider="openai",          # reads OPENAI_API_KEY from environment
        auto_threshold=0.95,
        candidate_lo=0.85,
        candidate_hi=0.95,
        batch_size=75,
        max_workers=3,
        calibration_sample_size=100,
        calibration_max_rounds=5,
        calibration_convergence_delta=0.01,
        budget=BudgetConfig(
            max_cost_usd=1.00,
            max_calls=500,
            warn_at_pct=80,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Run deduplication
# ---------------------------------------------------------------------------

print(f"\nRunning deduplication on {len(df):,} records...", flush=True)
start = time.time()
result = goldenmatch.dedupe_df(df, config=config)
elapsed = time.time() - start

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

sizes = Counter(v["size"] for v in result.clusters.values())
multi = {k: v for k, v in result.clusters.items() if v["size"] >= 2}

print(f"\nResults ({elapsed:.1f}s):")
print(f"  Total clusters:       {result.total_clusters:,}")
print(f"  Multi-member clusters:{len(multi):,}")
print(f"  Records in clusters:  {sum(v['size'] for v in multi.values()):,}")
print(f"  Singletons:           {sizes.get(1, 0):,}")

# ---------------------------------------------------------------------------
# Write CSV outputs
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")

if result.golden is not None:
    path = f"{OUTPUT_DIR}/equipment_golden.csv"
    result.golden.write_csv(path)
    print(f"\nWrote {path} ({len(result.golden):,} golden records)")

if result.dupes is not None:
    path = f"{OUTPUT_DIR}/equipment_dupes.csv"
    result.dupes.write_csv(path)
    print(f"Wrote {path} ({len(result.dupes):,} duplicate records)")

if result.unique is not None:
    path = f"{OUTPUT_DIR}/equipment_unique.csv"
    result.unique.write_csv(path)
    print(f"Wrote {path} ({len(result.unique):,} unique records)")
