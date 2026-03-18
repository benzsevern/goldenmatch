# LLM Boost — Design Specification

## Overview

Add an optional LLM-powered accuracy boost to GoldenMatch. When a user provides an API key (Anthropic or OpenAI), GoldenMatch uses the LLM to label 100-500 candidate pairs, trains a logistic regression classifier on those labels, and re-scores all pairs with the trained model. The classifier learns the optimal way to combine GoldenMatch's existing scorer outputs for the specific dataset, achieving supervised-quality results without manual labeling. Total cost: $0.20-$0.50 per dataset. The trained model is saved locally and reused on subsequent runs — the LLM is only needed once.

## Problem Statement

GoldenMatch achieves 40-45% F1 on hard e-commerce datasets (Abt-Buy, Amazon-Google) using zero-shot methods. SOTA systems using supervised training achieve 70-90% F1. The gap exists because zero-shot scoring doesn't know what matters for a specific dataset — is price important? Should manufacturer mismatches be dealbreakers? A supervised model learns these weights from labeled examples.

Active learning tools like dedupe require a human to label 200+ pairs interactively. Most users won't do this. An LLM can label pairs instantly with near-human accuracy, eliminating the adoption barrier.

## Goals

- Optional `--llm-boost` flag that improves accuracy on hard datasets
- Support Anthropic (Claude) and OpenAI (GPT-4) APIs
- Total LLM cost under $1 for typical datasets
- Trained classifier persisted locally for instant reuse on subsequent runs
- No impact on the default zero-config experience for users without API keys
- Target: 60%+ F1 on Abt-Buy, 55%+ F1 on Amazon-Google (up from 44.5% and 40.5%)

## Non-Goals

- Fine-tuning embedding models (overkill for this approach)
- Supporting local LLMs in V1 (defer OpenAI-compatible endpoints to V2)
- Using the LLM for direct pair scoring at runtime (too slow/expensive)
- Replacing existing scorers — boost augments, not replaces

---

## Feature 1: LLM Labeler

### Design

New module `goldenmatch/core/llm_labeler.py` — handles API calls to label record pairs.

### API

```python
def label_pair(
    record_a: dict, record_b: dict, columns: list[str],
    context: str, provider: str, api_key: str, model: str,
) -> bool:
    """Ask the LLM if two records are the same entity. Returns True/False."""

def label_pairs(
    pairs: list[tuple[dict, dict]], columns: list[str],
    context: str, provider: str, api_key: str, model: str,
) -> list[bool]:
    """Label multiple pairs with progress bar and rate limiting."""
```

### Provider Detection

Check which API key is available (in order):
1. Environment variable: `ANTHROPIC_API_KEY` → use Claude
2. Environment variable: `OPENAI_API_KEY` → use GPT-4
3. Settings file: `~/.goldenmatch/settings.yaml` → `llm.anthropic_api_key` or `llm.openai_api_key`
4. Both available → prefer Claude (cheaper)

### Prompt Template

```
You are deduplicating a {context}. Determine if these two records
refer to the same real-world entity.

Record A:
  {col}: {val}
  ...

Record B:
  {col}: {val}
  ...

Same entity? Reply with only: yes or no
```

### Context Auto-Detection

Reuse column type profiles from `autoconfig.py`:
- name + email + phone → "contact list"
- title + manufacturer + price → "product catalog"
- title + authors + venue → "publication database"
- address + city + state → "address database"
- fallback → "dataset"

### Default Models

- Anthropic: `claude-haiku-4-5-20251001` (~$0.001/pair)
- OpenAI: `gpt-4o-mini` (~$0.001/pair)
- User can override via `llm.model` in settings

### Cost Estimation

Before labeling, display estimated cost and prompt for confirmation:
```
Will label ~100 pairs using Claude Haiku (~$0.10). Continue? [Y/n]
```

In `--no-tui` mode with `--yes` flag, skip confirmation.

### Dependencies

New optional dependency group in `pyproject.toml`:
```toml
[project.optional-dependencies]
llm = [
    "anthropic>=0.40",
    "openai>=1.0",
    "scikit-learn>=1.3",
]
```

Install via: `pip install goldenmatch[llm]`

Missing dependency triggers clear error:
```
LLM boost requires the anthropic or openai package.
Install with: pip install goldenmatch[llm]
```

### Response Parsing

Normalize LLM responses: strip whitespace, lowercase, check if starts with "y" or "n". If neither, retry once with a stricter prompt. If still ambiguous, skip that pair and log a warning.

### Error Handling

Exponential backoff on API errors (429, 500, 503): retry 3 times with 1s/2s/4s delays. Save intermediate labels to disk (`.goldenmatch_labels_partial.json`) so a crash mid-labeling can resume. On completion, delete the partial file.

---

## Feature 2: Feature Extraction + Classifier Training

### Design

New module `goldenmatch/core/boost.py` — feature extraction, classifier training, adaptive labeling loop.

### Feature Extraction

Batch feature extraction using vectorized operations — no per-pair Python loops:

```python
def extract_feature_matrix(
    pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Build (n_pairs, n_features) matrix using vectorized cdist."""
```

For each column, compute scorer matrices using `rapidfuzz.process.cdist` on the unique record values involved in pairs, then index into the matrices to extract per-pair features. This reuses the same vectorized approach as `_fuzzy_score_matrix` in `scorer.py`.

Features per column (5 per column, `5*N` total for N columns):
- Jaro-Winkler similarity (via `cdist`)
- Token sort ratio (via `cdist`)
- Levenshtein similarity (via `cdist`)
- Exact match (0/1)
- Length ratio (min_len / max_len)

If embeddings are available (model already loaded from pipeline), add cosine similarity as a 6th feature.

Feature extraction for 10K pairs takes <1 second using vectorized operations.

### Classifier

`sklearn.linear_model.LogisticRegression` with `class_weight='balanced'`.

- Trains in milliseconds on 500 samples
- Inference: ~100K pairs/second
- `predict_proba()[:, 1]` gives match probability as the new score

### Adaptive Training Loop

```python
def boost_accuracy(
    candidate_pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    columns: list[str],
    context: str,
    provider: str,
    api_key: str,
    model: str,
    max_labels: int = 500,
) -> list[tuple[int, int, float]]:
```

1. Extract features for all candidate pairs (vectorized)
2. **Initial batch:** Sample 100 pairs — 30 highest-scoring, 30 lowest-scoring, 40 from the uncertain middle (scores 0.4-0.7). This gives the model boundary examples from the start.
3. Send to LLM for yes/no labels
4. Train logistic regression, compute 5-fold cross-validation F1 score (`scoring='f1'`)
5. **Refinement loop** (if cross-val F1 < 0.75 and total labels < max_labels):
   - Find pairs where `abs(predicted_prob - 0.5)` is smallest (most uncertain)
   - Sample 100 of those, send to LLM, retrain
   - Repeat (max 3 refinement rounds)
6. Re-score all candidate pairs with `classifier.predict_proba()[:, 1]`
7. Return re-scored pairs with new probabilities

### Model Persistence

Save trained classifier as JSON to avoid deserialization security risks. Since LogisticRegression is fully defined by its coefficients, we store only those:

```json
{
  "coef": [[0.3, -0.1, ...]],
  "intercept": [0.5],
  "classes": [false, true],
  "columns": ["name", "email", "zip"],
  "column_hash": "a1b2c3..."
}
```

File: `.goldenmatch_model.json` in the project directory. On load, reconstruct `LogisticRegression` by setting `coef_`, `intercept_`, and `classes_` directly.

**Staleness detection:** Save a hash of the column names alongside the model. On load, compare against the current dataset's columns. If they differ, log a warning and force retraining.

On subsequent runs with `--llm-boost`:
- If `.goldenmatch_model.json` exists and column hash matches, load it and skip LLM calls
- Re-extract features for current candidate pairs, score with saved model
- `--llm-retrain` flag forces re-labeling and retraining

---

## Feature 3: Pipeline Integration

### Pipeline Position

Boost runs after scoring and before clustering:

```
ingest → block → score → [BOOST] → cluster → golden → output
```

It re-scores existing candidate pairs — it does not change blocking or pair generation.

### CLI Integration

In `goldenmatch/cli/dedupe.py`, add flags:

```python
llm_boost: bool = typer.Option(False, "--llm-boost", help="Boost accuracy with LLM-labeled training data")
llm_retrain: bool = typer.Option(False, "--llm-retrain", help="Force re-labeling (ignore saved model)")
llm_provider: Optional[str] = typer.Option(None, "--llm-provider", help="LLM provider: anthropic or openai")
```

### Settings Persistence

In `~/.goldenmatch/settings.yaml`:

```yaml
llm:
  anthropic_api_key: sk-ant-...   # or use ANTHROPIC_API_KEY env var
  openai_api_key: sk-...          # or use OPENAI_API_KEY env var
  provider: auto                  # auto | anthropic | openai
  model: auto                     # auto | specific model name
  max_labels: 500                 # budget cap
```

API key precedence: env vars > settings file.

### Config Option

In YAML config or `.goldenmatch.yaml`:

```yaml
llm_boost: true
```

### TUI Integration

After initial results display, show button: `[Boost with AI (~$0.30)]`. Only appears if an API key is detected. Clicking it:
1. Shows progress: "Labeling pairs... (42/100)"
2. Shows training: "Training classifier... 94% cross-val accuracy"
3. Shows re-scoring: "Re-scoring 5,432 pairs..."
4. Refreshes results view with improved matches

### Pipeline Code

In `pipeline.py` `run_dedupe()`, after the scoring loop and before `build_clusters()`:

```python
if config.llm_boost or llm_boost_flag:
    from goldenmatch.core.boost import boost_accuracy
    all_pairs = boost_accuracy(
        all_pairs, combined_df, matchable_columns,
        context, provider, api_key, model, max_labels,
    )
```

Same integration point in `engine.py` for TUI and in `run_match()` in `pipeline.py` (between pair generation and normalization).

### Schema Changes

Add `llm_boost: bool = False` to `GoldenMatchConfig` in `schemas.py` so it can be set in YAML config files.

### CLI Flags

Also add `--llm-max-labels` to override the budget per run, and accept `auto` for `--llm-provider`.

---

## Rollout Plan

1. **Phase 1: LLM Labeler**
   - `llm_labeler.py` with Anthropic + OpenAI support
   - Prompt template with context detection
   - Cost estimation
   - Tests with mocked API responses

2. **Phase 2: Boost Engine**
   - `boost.py` with feature extraction, classifier training, adaptive loop
   - Model persistence with joblib
   - Tests with synthetic labeled data

3. **Phase 3: Integration**
   - CLI flags (`--llm-boost`, `--llm-retrain`, `--llm-provider`)
   - Settings persistence for API keys
   - Pipeline + TUI wiring
   - Config option (`llm_boost: true`)

4. **Phase 4: Benchmark**
   - Run Abt-Buy and Amazon-Google with LLM boost
   - Target: 60%+ and 55%+ F1 respectively
   - Measure cost and latency
   - Update README

## Testing Strategy

### Unit Tests

- `test_llm_labeler.py` — prompt formatting, context detection, provider selection, cost estimation, mocked API responses for both providers
- `test_boost.py` — feature extraction correctness, classifier training on synthetic data, adaptive loop stops when accurate enough, model save/load round-trip, re-scoring produces valid probabilities
- `test_llm_integration.py` — end-to-end with mocked LLM: candidate pairs → boost → re-scored pairs

### Integration Tests

- Full pipeline with `llm_boost=True` and mocked LLM responses
- Model persistence: run once (labels), run again (loads saved model, skips LLM)
- Settings precedence: env var > settings file > default

### Manual Validation

- Run on Abt-Buy with real Claude API key
- Verify cost is under $1
- Verify F1 improvement

## Dependencies

New optional dependency group:
```toml
[project.optional-dependencies]
llm = ["anthropic>=0.40", "openai>=1.0", "scikit-learn>=1.3"]
```

Existing dependencies used:
- `numpy` for feature vectors
- `rapidfuzz` for vectorized feature extraction

## Cost Analysis

| Pairs Labeled | Claude Haiku | GPT-4o-mini | Quality |
|---|---|---|---|
| 100 (initial) | $0.10 | $0.10 | Usually sufficient |
| 200 (1 refinement) | $0.20 | $0.20 | Good |
| 300 (2 refinements) | $0.30 | $0.30 | Very good |
| 500 (max) | $0.50 | $0.50 | Diminishing returns |

Subsequent runs with saved model: $0.00.
