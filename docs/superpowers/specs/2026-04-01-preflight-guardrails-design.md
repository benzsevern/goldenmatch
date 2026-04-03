# Preflight & Guardrails: Domain-Aware Auto-Config + Runtime Safety

**Date:** 2026-04-01
**Status:** Approved
**Scope:** GoldenMatch core pipeline

## Problem

Two issues discovered during a 401K-row equipment dedup run:

1. **Auto-config fails on complex scenarios.** `auto_configure_df()` builds configs from generic heuristics (column name regex + data profiling). It has no concept of data domain, so it picks wrong blocking strategies, scorer weights, and thresholds for domain-specific data.
2. **Misconfiguration costs are painful.** No pre-run validation or runtime monitoring means bad configs cause OOM crashes, runaway LLM API costs, and Polars lock contention with no warning.

## Approach: Layered Defense

Three layers, each catching what the previous missed:

1. **Domain-aware auto-config** — detect domain, load tuned preset, refine with LLM
2. **Sample preflight** — run 5K-row sample, measure resources, extrapolate, auto-downgrade
3. **Runtime circuit breakers** — monitor memory/cost/comparisons during full run, degrade gracefully

## Layer 1: Domain-Aware Auto-Config

### Domain Detection

After `profile_columns()`, a new `detect_domain()` step examines column profiles + sample values to identify the data domain. Reuses the existing domain registry (`core/domain_registry.py` + 7 built-in YAML packs in `goldenmatch/domains/`).

Detection logic:
- Score each registered domain against column profiles (field name overlap, sample value pattern matching)
- Return best-matching domain with confidence score
- Fall back to `"generic"` if no domain scores above threshold

### Domain Presets

Each domain YAML pack gains an optional `autoconfig_preset` section with tuned defaults:

```yaml
# domains/electronics.yaml (existing file, new section)
autoconfig_preset:
  blocking:
    strategy: ann
    ann_top_k: 20
    max_block_size: 1000
    skip_oversized: true
  scorer_overrides:
    description: {scorer: ensemble, weight: 2.0}
    brand: {scorer: jaro_winkler, weight: 1.5}
    category: {scorer: token_sort, weight: 1.0}
  threshold: 0.85
  recommended_standardization:
    description: [lowercase, strip, normalize_whitespace]
    brand: [lowercase, strip]
```

**Implementation detail:** The `DomainRulebook` dataclass in `core/domain_registry.py` gains a new field `autoconfig_preset: dict | None = None`. The `load_rulebook()` function is updated to read this key from YAML and pass it through. The preset is a plain dict matching the YAML structure above — no separate Pydantic model needed since it's only consumed internally by `auto_configure_df()` which maps it onto existing `BlockingConfig`/`MatchkeyField`/`StandardizationConfig` models.

### LLM Refinement

After applying the domain preset, if an LLM provider is available, send the preset + column samples to the LLM for refinement:
- Adjust weights for columns that are unusually noisy or clean
- Suggest additional blocking passes based on cardinality stats
- Recommend tightening/loosening thresholds based on data quality

Extends the existing `_llm_classify_columns()` pattern (same providers, same budget tracking).

### Flow Change in `auto_configure_df()`

```
profile_columns() -> detect_domain() -> load_preset() -> LLM_refine() -> build_matchkeys() -> build_blocking()
```

`build_matchkeys()` and `build_blocking()` receive the preset as a baseline to build on, rather than starting from scratch.

## Layer 2: Preflight System

### `RunPlan` Dataclass

```python
@dataclass
class RunPlan:
    original_config: GoldenMatchConfig
    adjusted_config: GoldenMatchConfig
    projections: ResourceProjection
    downgrades: list[Downgrade]
    sample_stats: SampleStats
    domain: str | None
    safety: str

@dataclass
class ResourceProjection:
    total_comparisons: int
    estimated_memory_mb: float
    estimated_llm_calls: int
    estimated_llm_cost_usd: float
    estimated_wall_time_seconds: float
    risk_level: str                       # "safe", "caution", "danger"

@dataclass
class Downgrade:
    component: str          # "blocking", "backend", "llm_scorer", "threshold", etc.
    original_value: Any
    new_value: Any
    reason: str             # human-readable: "projected 27M comparisons exceeds 5M limit"

@dataclass
class SampleStats:
    sample_size: int                      # rows in sample
    sample_comparisons: int               # pairs generated during sample run
    sample_peak_memory_mb: float          # peak RSS delta during sample (psutil)
    sample_llm_calls: int                 # LLM API calls during sample
    sample_llm_cost_usd: float            # LLM cost during sample
    sample_wall_time_seconds: float       # wall clock for sample pipeline
    blocking_stats: dict[str, int]        # {block_key: block_size} from full dataset group_by
```

### Sample Run Mechanics

`preflight()` takes a stratified sample (~5K rows, or 1% of dataset, whichever is larger up to 10K) and runs the full pipeline on it. Measurements:
- Peak process memory delta via `psutil.Process().memory_info().rss` (captures native allocations from Polars/Rust, rapidfuzz/C++, FAISS/C++, numpy — `tracemalloc` only tracks Python allocations and would significantly underreport)
- `tracemalloc` as a secondary signal for Python-side memory (e.g., intermediate dicts, pair lists)
- Total comparisons generated by blocking
- LLM calls + cost (from `BudgetTracker`)
- Wall clock time

**Memory measurement note:** `psutil` is used for all safety-critical memory checks (preflight projections, circuit breaker thresholds) because the heaviest memory consumers in the pipeline are native code. `tracemalloc` is logged as supplementary info but never used as the sole basis for a downgrade or stop decision. `psutil` is already an indirect dependency via Polars.

Extrapolation uses blocking statistics rather than naive linear scaling:
- Query actual blocking config against full dataset for real block counts/sizes (cheap `group_by().len()`)
- Total comparisons: `sum(n*(n-1)/2)` across all blocks
- Memory: `base_df_overhead + (comparisons * bytes_per_comparison)` where `base_df_overhead` accounts for DataFrame/block construction memory (measured from sample as `rss_after_blocking - rss_before_blocking`) and `bytes_per_comparison` is measured from the scoring phase. This two-term model avoids underestimating memory for large DataFrames with few comparisons.
- LLM cost: `(candidate_pairs / sample_candidate_pairs) * sample_llm_cost`

### `SafetyPolicy` and Auto-Downgrade

```python
@dataclass
class SafetyPolicy:
    max_comparisons: int = 10_000_000        # 10M pairs
    max_memory_mb: float = 4096              # 4GB
    max_llm_cost_usd: float = 5.00
    max_wall_time_seconds: float = 3600      # 1 hour
    mode: Literal["conservative", "aggressive", "none"] = "conservative"
```

**Valid modes:**
- `"conservative"` (default): only downgrades blocking strategy and backend. Match quality preserved.
- `"aggressive"`: also tightens thresholds, reduces LLM bands, drops low-utility fields. May reduce match quality.
- `"none"`: skip preflight entirely. Equivalent to today's behavior. Use when you've validated the config externally.

**Conservative downgrades** (default), applied in order until projections are safe:
1. Set `skip_oversized=True`
2. Halve `max_block_size`
3. Switch blocking strategy to `ann`
4. Switch backend to `duckdb` (skipped if `duckdb` optional dependency is not installed)
5. If still unsafe: raise `PreflightError` with projections + recommendations

**Aggressive downgrades** add:
6. Tighten `candidate_lo` on LLM scorer
7. Drop lowest-utility fuzzy fields
8. Raise match threshold

### Integration into `dedupe_df()`

By default, `dedupe_df()` calls `preflight()` internally when dataset exceeds 10K rows. Small datasets skip it.

```python
def dedupe_df(df, *, run_preflight=True, safety="conservative", plan=None, ...):
    if plan is not None:
        config = plan.adjusted_config
    elif run_preflight and df.height > 10_000:
        plan = preflight(df, config=config, safety=safety)
        config = plan.adjusted_config
    ...
```

**Note:** The parameter is named `run_preflight` (not `preflight`) to avoid shadowing the module-level `preflight()` function.

### Standalone `preflight()`

Exposed in public API:

```python
plan = gm.preflight(df, safety="conservative")
print(plan.projections)
print(plan.downgrades)
result = gm.dedupe_df(df, plan=plan)
```

## Layer 3: Runtime Circuit Breakers

### `CircuitBreaker` Class

Lightweight monitor called at natural checkpoints in the pipeline. Not a background thread.

```python
@dataclass
class CircuitBreaker:
    policy: SafetyPolicy
    budget_tracker: BudgetTracker | None = None

    def check(self, stage: str) -> CircuitAction:
        """Called at pipeline checkpoints. Returns CONTINUE, DOWNGRADE, or STOP."""
        ...

@dataclass
class CircuitAction:
    action: str              # "continue", "downgrade", "stop"
    reason: str | None
    downgrade: Downgrade | None
```

### Checkpoints

Inserted at three natural points (no new threads, no polling):

1. **Between blocks in `score_blocks_parallel()`** — after each block scored, check memory + comparison count
2. **Between LLM batches in `_batch_score()`** — after each batch, check cost against budget
3. **Between pipeline stages in `_run_dedupe_pipeline()`** — after blocking, scoring, clustering

### Actions on Breach

- **Memory:** Check via `psutil.Process().memory_info().rss` (same measurement as preflight). Force `gc.collect()`. If still over, drop to sequential processing. If critically over (>90% system RAM), raise `CircuitBreakerError` with partial results.
- **LLM cost:** Warning at 80%. Stop LLM scoring at 100%, fuzzy scores stand for remaining pairs. Graceful degradation, no crash.
- **Comparison count:** Skip remaining blocks if total exceeds `max_comparisons`. Log skipped blocks.

### Polars Lock Contention

Not addressed by circuit breaker (OS-level issue on Windows). Handled by preflight's backend downgrade (switch to DuckDB for large datasets).

### Integration

Circuit breaker created by `preflight()` and passed through pipeline as optional parameter. Functions gain `circuit_breaker: CircuitBreaker | None = None`. When `None`, zero overhead.

## Public API Surface

### New Functions

```python
gm.preflight(df, *, config=None, safety="conservative") -> RunPlan
```

### Extended `dedupe_df()` Signature

```python
def dedupe_df(
    df, *,
    config=None, exact=None, fuzzy=None, blocking=None, threshold=None,
    llm_scorer=False, backend=None, source_name="dataframe",
    run_preflight: bool = True,
    safety: Literal["conservative", "aggressive", "none"] = "conservative",
    plan: RunPlan | None = None,
) -> DedupeResult
```

### `DedupeResult` Addition

```python
plan: RunPlan | None = None  # populated when preflight ran
```

### YAML Config

```yaml
safety:
  mode: conservative
  max_comparisons: 10000000
  max_memory_mb: 4096
  max_llm_cost_usd: 5.00
```

### Backwards Compatibility

All new parameters have defaults preserving current behavior:
- `run_preflight=True` only activates for >10K rows
- `safety="conservative"` only downgrades when clearly unsafe
- `DedupeResult.plan=None` for runs without preflight

No existing tests break. No existing API signatures change.

### Logging

All preflight and circuit breaker activity through standard `logging`:
- `INFO`: projections, domain detected, sample stats
- `WARNING`: downgrades applied, 80% threshold warnings
- `ERROR`: `PreflightError` or `CircuitBreakerError`

No stdout/stderr. No interactive prompts.

## File Layout

### New Files

| File | Purpose |
|---|---|
| `goldenmatch/core/preflight.py` | `preflight()`, `RunPlan`, `ResourceProjection`, `SampleStats`, `Downgrade`, sample runner, extrapolation, auto-downgrade |
| `goldenmatch/core/circuit_breaker.py` | `CircuitBreaker`, `CircuitAction`, memory/cost/comparison checks (imports `SafetyPolicy` from `config/schemas.py`) |
| `goldenmatch/core/domain_detector.py` | `detect_domain()` — scores column profiles against registered domain packs |
| `tests/test_preflight.py` | Preflight unit tests |
| `tests/test_circuit_breaker.py` | Circuit breaker unit tests |
| `tests/test_domain_detector.py` | Domain detection unit tests |

### Modified Files

| File | Change |
|---|---|
| `goldenmatch/core/autoconfig.py` | `auto_configure_df()` gains domain detection + preset + LLM refine flow |
| `goldenmatch/domains/*.yaml` | Add `autoconfig_preset` section to 7 built-in packs |
| `goldenmatch/core/domain_registry.py` | Parse `autoconfig_preset` from YAML packs |
| `goldenmatch/_api.py` | New `preflight()`, extended `dedupe_df()`, `RunPlan` on `DedupeResult` |
| `goldenmatch/__init__.py` | Re-export `preflight`, `RunPlan`, `SafetyPolicy` |
| `goldenmatch/config/schemas.py` | `SafetyPolicy` Pydantic model (canonical location — imported by `circuit_breaker.py` and `preflight.py`), optional `safety` field on `GoldenMatchConfig` |
| `goldenmatch/core/pipeline.py` | Pass `CircuitBreaker` through pipeline stages |
| `goldenmatch/core/scorer.py` | `score_blocks_parallel()` checks circuit breaker between blocks |
| `goldenmatch/core/llm_scorer.py` | `_batch_score()` checks circuit breaker between batches |

### Not Modified

TUI, CLI, MCP, A2A, REST API, connectors, backends, PPRL.

## Testing Strategy

### Preflight Tests (`test_preflight.py`)
- Sample selection: verify stratified sampling (5K default, 1% for large, cap 10K)
- Extrapolation accuracy: 50K dataset with predictable blocking, verify projections within 2x
- Auto-downgrade cascade: config projecting >10M comparisons, verify conservative downgrades in order
- Aggressive vs conservative: same scenario, verify aggressive adds threshold/field changes
- `safety="none"` bypass: verify preflight skipped
- `RunPlan` passthrough: verify `dedupe_df(df, plan=plan)` uses adjusted config
- Small dataset skip: verify <10K rows skip preflight

### Circuit Breaker Tests (`test_circuit_breaker.py`)
- Memory: mock `psutil.Process().memory_info()`, verify STOP at threshold
- Cost: BudgetTracker at 79% -> CONTINUE, 81% -> DOWNGRADE warning, 100% -> STOP
- Comparison count: exceed `max_comparisons`, verify blocks skipped
- No-op: verify pipeline unchanged when `circuit_breaker=None`

### Domain Detection Tests (`test_domain_detector.py`)
- Electronics: brand/model/sku profiles -> `electronics`
- Person: first_name/last_name/email -> `people`
- Generic fallback: ambiguous profiles -> `generic`
- Preset loading: domain -> correct `autoconfig_preset` loaded
- LLM override: mock LLM adjusts weights, verify preset modified

### Integration Test
End-to-end with `generate_synthetic.py` (20K rows, people domain), `dedupe_df(run_preflight=True)`, verify `result.plan` populated with sensible projections and `result.plan.domain == "people"`.

### CI Safety
No real LLM calls. Mock `_call_llm_for_blocking()` (existing pattern).
