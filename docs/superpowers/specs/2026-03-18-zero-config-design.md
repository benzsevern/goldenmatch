# Zero-Config Mode — Design Specification

## Overview

Add a zero-config mode to GoldenMatch so users can run `goldenmatch dedupe file.csv` with no YAML config file and get good results automatically. The tool analyzes input columns, picks scorers, chooses blocking strategies, selects embedding models, and launches the TUI for review — all without user configuration. Includes accuracy improvements (ensemble scoring, TF-IDF pre-filter, adaptive thresholds) that don't sacrifice speed.

## Problem Statement

GoldenMatch currently requires users to write a YAML config file specifying matchkeys, scorers, weights, blocking keys, and thresholds. This is a significant adoption barrier — users need to understand entity resolution concepts before they can run their first dedupe. Competing tools like dedupe and Splink also require setup. A tool that "just works" on a CSV would be a differentiator.

Additionally, accuracy on hard datasets (Abt-Buy 44.5%, Amazon-Google 40.5%) lags behind SOTA. Algorithmic improvements that don't add latency can close part of this gap.

## Goals

- `goldenmatch dedupe file.csv` works with no config file
- Auto-detect column types and assign appropriate scorers
- Auto-select embedding model based on dataset size
- Launch TUI with pre-loaded config for user adjustment
- Persist user preferences across sessions (global + project-level)
- Improve accuracy on fuzzy/semantic matching without speed regression
- Maintain full backward compatibility with existing `--config` workflow

## Non-Goals

- LLM-based scoring in the default pipeline (too slow/expensive)
- Fine-tuning embedding models (deferred to V2)
- Auto-config for match mode (focus on dedupe first)
- Schema discovery for databases or APIs (CSV/Excel/Parquet only)

---

## Feature 1: Auto-Config Engine

### Design

New module `goldenmatch/core/autoconfig.py` with a single entry point:

```python
def auto_configure(files: list[Path]) -> GoldenMatchConfig:
```

### Column Classification

Two-phase detection:

**Phase 1 — Name heuristics:** Pattern match column names against known patterns:

| Pattern | Type |
|---------|------|
| `email`, `e_mail`, `email_address` | email |
| `name`, `full_name`, `first_name`, `last_name` | name |
| `phone`, `tel`, `mobile` | phone |
| `zip`, `postal`, `postcode` | zip |
| `address`, `street`, `addr` | address |
| `city`, `state`, `country` | geo |
| `id`, `key`, `code`, `sku` | identifier |

**Phase 2 — Data profiling:** Sample 1,000 rows and validate/augment:
- Contains `@` + `.` → email
- Mostly 5-digit or 5+4 digit → zip
- Mostly digits, 7-15 chars → phone
- Mixed alpha with common name patterns → name
- Long freetext (avg > 50 chars) → description
- If Phase 1 identified a type, Phase 2 validates it. If Phase 2 contradicts (e.g., column named "email" but no `@` signs), Phase 2 wins.

Columns that can't be classified are treated as generic string fields with `token_sort` scoring.

### Scorer Assignment

| Column Type | Scorer | Weight | Transforms |
|-------------|--------|--------|------------|
| email | exact | 1.0 | lowercase, strip |
| phone | exact | 0.8 | digits_only |
| zip | exact | 0.5 | strip |
| name | jaro_winkler | 1.0 | lowercase, strip |
| address | token_sort | 0.8 | lowercase, strip |
| identifier | exact | 1.0 | strip |
| description | record_embedding | 1.0 | — |
| geo | exact | 0.3 | lowercase, strip |
| (unclassified string) | token_sort | 0.5 | lowercase, strip |

### Blocking Key Selection

1. Pick the highest-cardinality exact-type column for primary blocking (email, phone, zip)
2. If no exact-type columns, use the best name column with `soundex` transform
3. Measure duplicate ratio in sample — if > 20% of blocking keys are shared by 3+ records, switch to `multi_pass` with `[substring:0:5, soundex, first_token]` passes
4. Fall back to `multi_pass` with substring + soundex + first_token for all-fuzzy configs
5. Last resort (no blockable columns at all, e.g., single freetext column): use `canopy` strategy on the best text column, leveraging the existing TF-IDF canopy blocker

### Model Selection

Only triggered when at least one column is classified as `description` (i.e., `record_embedding` scorer will be used). If no embedding columns, skip model selection entirely.

- Dataset < 50K rows → `gte-base-en-v1.5` (130MB, better accuracy)
- Dataset >= 50K rows → `all-MiniLM-L6-v2` (80MB, faster)
- User can override via settings or TUI
- `auto_model_threshold` configurable in settings (default: 50000)

### Implementation

New file: `goldenmatch/core/autoconfig.py`

Reuses the existing `profiler.py` `_guess_type` function for Phase 2 data profiling where applicable, extending it with the Phase 1 name heuristics layer.

```python
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    col_type: str  # email, name, phone, zip, address, geo, identifier, description, string
    confidence: float  # 0.0 to 1.0
    sample_values: list[str]

def profile_columns(df: pl.DataFrame, sample_size: int = 1000) -> list[ColumnProfile]:
    """Classify columns by type using name heuristics + data profiling.
    Samples randomly (not head) to avoid bias from header-adjacent rows."""

def build_matchkeys(profiles: list[ColumnProfile]) -> list[MatchkeyConfig]:
    """Generate matchkeys from column profiles.
    Applies adaptive threshold per-matchkey based on that matchkey's field types."""

def build_blocking(profiles: list[ColumnProfile], df: pl.DataFrame) -> BlockingConfig:
    """Generate blocking config from column profiles.
    Falls back to canopy strategy when no blockable columns exist."""

def select_model(row_count: int, has_embedding_columns: bool, threshold: int = 50000) -> str | None:
    """Select embedding model. Returns None if no embedding columns needed."""

def auto_configure(files: list[tuple[str, str]]) -> GoldenMatchConfig:
    """Main entry point: load files, profile, generate config.
    Accepts (path, source_name) tuples to match CLI file spec format.
    Always sets golden_rules with default_strategy='most_complete'."""
```

---

## Feature 2: User Preferences & Settings Persistence

### Design

Two-tier config system:

- `~/.goldenmatch/settings.yaml` — global user preferences
- `.goldenmatch.yaml` — project-level (current directory), overrides global

### Precedence

CLI flags > project `.goldenmatch.yaml` > global `~/.goldenmatch/settings.yaml` > auto-config defaults.

### Global Settings Schema

```yaml
# ~/.goldenmatch/settings.yaml
defaults:
  output_mode: tui          # tui | files | console
  output_dir: ./goldenmatch_output
  output_format: csv        # csv | parquet
  embedding_model: auto     # auto | all-MiniLM-L6-v2 | gte-base-en-v1.5 | all-mpnet-base-v2
  auto_model_threshold: 50000
```

### Project Settings Schema

`.goldenmatch.yaml` serializes the full `GoldenMatchConfig` in standard YAML format, reusing the existing `loader.py` parsing. This avoids introducing a second config format. Additional metadata (column type annotations, selected model) is stored under a `_autoconfig` key that the loader ignores but `autoconfig.py` reads.

```yaml
# .goldenmatch.yaml
matchkeys:
  - name: name_email
    type: weighted
    threshold: 0.82
    fields:
      - field: name
        scorer: ensemble
        weight: 1.0
        transforms: [lowercase, strip]
      - field: email
        scorer: exact
        weight: 1.0
        transforms: [lowercase, strip]
blocking:
  keys:
    - fields: [email]
      transforms: [lowercase, strip]
  strategy: static
golden_rules:
  default_strategy: most_complete

# Auto-config metadata (ignored by loader, used by autoconfig)
_autoconfig:
  column_types:
    name: name
    email: email
    zip: zip
  embedding_model: gte-base-en-v1.5
```

When `.goldenmatch.yaml` is present, it is loaded via the standard `load_config()` path. If the file references columns that no longer exist in the input data, auto-config logs a warning and falls back to fresh auto-detection.

### Implementation

New file: `goldenmatch/config/settings.py`

```python
@dataclass
class UserSettings:
    output_mode: str = "tui"
    output_dir: str = "./goldenmatch_output"
    output_format: str = "csv"
    embedding_model: str = "auto"
    auto_model_threshold: int = 50000

def load_settings() -> UserSettings:
    """Load from ~/.goldenmatch/settings.yaml, overlay .goldenmatch.yaml."""

def save_project_settings(config: GoldenMatchConfig, path: Path) -> None:
    """Save tuned config to .goldenmatch.yaml in current directory."""
```

### TUI "Save Settings" Action

After the user tunes column types, thresholds, or blocking in the TUI, they can save to `.goldenmatch.yaml`. Next run in the same directory picks up the saved config and skips auto-detection.

---

## Feature 3: Zero-Config CLI Flow

### Full Flow

1. **Load settings** — read global `~/.goldenmatch/settings.yaml`, overlay project `.goldenmatch.yaml` if present
2. **Check for saved config** — if `.goldenmatch.yaml` has column mappings for this file, skip auto-config and use them
3. **Auto-config** — if no saved config, run `auto_configure()`:
   - Load file, sample 1,000 rows
   - Classify columns (name heuristics + data profiling)
   - Assign scorers, weights, blocking keys
   - Select embedding model based on row count
   - Generate `GoldenMatchConfig`
4. **Launch TUI** — show auto-detected config with summary:
   ```
   Auto-detected columns:
     name      → name (jaro_winkler, weight: 1.0)
     email     → email (exact, weight: 1.0)
     zip       → zip (exact, weight: 0.5)
     field_3   → description (record_embedding)

   Blocking: multi_pass (soundex + substring:0:5)
   Model: gte-base-en-v1.5 (auto, <50K rows)

   [Run] [Edit Config] [Save Settings]
   ```
5. **User adjusts or runs** — modify column types, scorers, thresholds, model in TUI
6. **Run pipeline** — standard ingest → block → score → cluster → golden
7. **Show results** — clusters in TUI with option to export
8. **Offer to save** — prompt to save tuned settings to `.goldenmatch.yaml`

### CLI Flag Overrides

- `--no-tui` — skip TUI, run with auto-config directly, output files
- `--config config.yaml` — existing behavior, bypass auto-config entirely
- `--model <name>` — override model selection

### Modifications

- `goldenmatch/cli/dedupe.py` — make `--config` optional, add `--no-tui` and `--model` flags
- When `--config` is omitted, call `auto_configure()` instead of `load_config()`

---

## Feature 4: Accuracy Improvements (Speed-Neutral)

### Ensemble Scoring for Name Columns

Instead of picking one scorer for name-type columns, combine multiple scorers and take the weighted max:

```python
score = max(
    jaro_winkler(a, b),
    token_sort_ratio(a, b),
    0.8 * soundex_match(a, b),
)
```

This catches cases where one scorer fails — "John Smith" vs "Smith, John" scores poorly on jaro_winkler but perfectly on token_sort.

Implementation: Add `"ensemble"` to `VALID_SCORERS`. Add a new branch in `_fuzzy_score_matrix` in `scorer.py` that computes `max(jaro_winkler_matrix, token_sort_matrix, 0.8 * soundex_matrix)` element-wise and returns the result. This keeps the change contained to the scorer dispatcher — no schema changes beyond the new string in `VALID_SCORERS`. Cost: 3 RapidFuzz C calls instead of 1 — negligible overhead.

Auto-config assigns `ensemble` to name-type columns by default.

### TF-IDF Pre-Filter for Description Columns

Before running the embedding model on description/freetext columns, run a fast TF-IDF cosine similarity pass using scikit-learn (already installed). This serves as cheap blocking for text-heavy data where traditional blocking keys fail.

Implementation: New function in `blocker.py` or `autoconfig.py`:

```python
def tfidf_candidate_pairs(values: list[str], top_k: int = 20) -> list[tuple[int, int, float]]:
    """Fast TF-IDF blocking using character n-grams."""
```

Uses `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))` + sparse matrix top-K. Cost: ~2 seconds for 10K records. Can replace embedding-based ANN blocking for smaller datasets, saving model download time.

### Adaptive Threshold from Column Types

Instead of a fixed default threshold, set per-matchkey based on that matchkey's own field types:

| Field Types in Matchkey | Default Threshold |
|-------------------------|-------------------|
| All exact fields (email, phone) | 0.95 |
| Mix of exact + fuzzy | 0.80 |
| All fuzzy/embedding | 0.70 |
| Single field match | 0.85 |

Applied per-matchkey during `build_matchkeys()`, not globally. User can override per-matchkey in TUI or config.

---

## Rollout Plan

1. **Phase 1: Auto-Config Engine**
   - `autoconfig.py` with column profiling, scorer assignment, blocking selection, model selection
   - Tests with various CSV shapes (clean, messy, unknown columns)

2. **Phase 2: Settings Persistence**
   - `settings.py` with load/save for global + project settings
   - Tests for precedence rules

3. **Phase 3: CLI Integration**
   - Make `--config` optional in `dedupe.py`
   - Wire auto-config → TUI flow
   - Add `--no-tui` and `--model` flags

4. **Phase 4: Accuracy Improvements**
   - Ensemble scorer type
   - TF-IDF pre-filter
   - Adaptive threshold logic

5. **Phase 5: TUI Enhancements**
   - Auto-config summary screen
   - Edit Config panel
   - Save Settings action
   - Model cycling

6. **Phase 6: Benchmark & Polish**
   - Re-run Leipzig benchmarks with auto-config
   - Test pip install → first run experience
   - Update README with zero-config examples

## Testing Strategy

### Unit Tests

- `test_autoconfig.py` — column classification (email, name, phone, zip, address, description), scorer assignment, blocking key selection, model selection, edge cases (empty file, all numeric columns, single column)
- `test_settings.py` — load/save global settings, load/save project settings, precedence rules, missing files, invalid YAML
- `test_ensemble_scorer.py` — ensemble scoring vs individual scorers, null handling, NxN matrix shape

### Integration Tests

- End-to-end: `goldenmatch dedupe test.csv --no-tui` with no config → verify output files created
- Auto-config → pipeline → results for known test fixtures
- Settings save → re-run → verify saved config used

### Benchmark Validation

- Run Leipzig datasets through auto-config (no manual tuning)
- Target: within 5pts F1 of manually tuned configs
- Ensemble scoring should improve name-matching datasets (DBLP-ACM, DBLP-Scholar)

## Dependencies

- `scikit-learn` — add to core `dependencies` in `pyproject.toml` (currently only used by canopy via runtime import guard; zero-config makes TF-IDF a default-path feature)
- `sentence-transformers` (optional, existing `[embeddings]` extra) for embedding scoring
- `pyyaml` (already a direct dependency in `pyproject.toml`) for settings files
