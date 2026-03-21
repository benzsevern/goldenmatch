# GoldenMatch Master Roadmap — v0.3 through v0.5+

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution model:** This is a **phased master plan**. Each phase has its own detailed implementation plan written and executed before the next phase begins. The master plan defines scope, dependencies, and contracts between phases.

**Goal:** Transform GoldenMatch from a strong entity resolution library into the definitive open-source ER tool — beating Splink on theory, Dedupe.io on ease-of-use, and Zingg on intelligence, while staying pip-installable and cheap to run.

**Architecture:** Nine independent subsystems delivered in dependency order across 4 phases. Each phase produces working, testable, releasable software. Later phases build on contracts established by earlier phases but each phase is independently valuable.

**Tech Stack:** Python 3.12, Polars, Pydantic, rapidfuzz, FAISS, scikit-learn, DuckDB (optional), OpenAI/Anthropic APIs, connector-specific SDKs as optional deps.

---

## Phase Overview

| Phase | Features | Release Target | Key Deliverable |
|-------|----------|---------------|-----------------|
| **Phase 1** | LLM Budget Controller, Fellegi-Sunter Probabilistic Model | v0.3.0 | Smart scoring with cost controls |
| **Phase 2** | Plugin Architecture, Learned Blocking, Auto-Config | v0.3.5 | Self-configuring, extensible framework |
| **Phase 3** | Connectors, Explainability, DuckDB Backend | v0.4.0 | Enterprise-ready data access + audit trail |
| **Phase 4** | Streaming/CDC, Multi-Table Graph ER | v0.5.0 | Real-time + relational entity resolution |

---

## Phase 1: Smart Scoring with Cost Controls

### 1A. LLM Budget Controller

**Why first:** Every subsequent feature that touches LLM scoring (auto-config, streaming, explainability) needs budget awareness. This is foundational infrastructure.

**Scope:**
- New `BudgetConfig` Pydantic model nested under `LLMScorerConfig`
- Real-time token/cost tracking during `llm_score_pairs`
- Graceful degradation: when budget exhausted, remaining pairs keep fuzzy-only scores
- Model tiering: cheap model default, expensive model for hardest pairs (configurable band)
- Cost report in `EngineStats` and lineage output
- Token estimation before sending batches

**Config contract:**
```yaml
llm_scorer:
  enabled: true
  model: gpt-4o-mini
  budget:
    max_cost_usd: 5.00          # hard cap
    max_calls: 500              # alternative cap
    escalation_model: gpt-4o   # expensive model for hardest pairs
    escalation_band: [0.80, 0.90]  # score range that triggers escalation
    escalation_budget_pct: 20  # max % of budget on expensive model
    warn_at_pct: 80            # log warning at this % consumed
```

**Signature evolution:** The current `llm_score_pairs` takes flat kwargs (`auto_threshold`, `candidate_lo`, etc.). Rather than adding `budget_tracker` as yet another kwarg, refactor to accept `LLMScorerConfig` directly: `llm_score_pairs(pairs, df, config: LLMScorerConfig)`. The pipeline call site already has `config.llm_scorer` available. This is a breaking internal change but no public API is affected. The `BudgetTracker` is instantiated from `config.llm_scorer.budget` and passed internally.

**New files:**
- `goldenmatch/core/llm_budget.py` — `BudgetTracker` class, token estimation, cost accounting
- Modify: `goldenmatch/core/llm_scorer.py` — refactor to accept `LLMScorerConfig`, integrate budget tracker
- Modify: `goldenmatch/config/schemas.py` — add `BudgetConfig` nested in `LLMScorerConfig`
- Modify: `goldenmatch/core/pipeline.py` — update `llm_score_pairs` call to pass config object
- Modify: `goldenmatch/tui/engine.py` — surface cost in `EngineStats`
- Tests: `tests/test_llm_budget.py`

**Interface contract (consumed by later phases):**
```python
class BudgetTracker:
    def __init__(self, config: BudgetConfig): ...
    def can_send(self, estimated_tokens: int) -> bool: ...
    def record_usage(self, input_tokens: int, output_tokens: int, model: str): ...
    def select_model(self, pair_score: float) -> str: ...  # tiering logic
    @property
    def total_cost_usd(self) -> float: ...
    @property
    def total_calls(self) -> int: ...
    @property
    def budget_exhausted(self) -> bool: ...
    @property
    def budget_remaining_pct(self) -> float: ...
    def summary(self) -> dict: ...  # for EngineStats / lineage
```

**Testing strategy:**
- Unit: BudgetTracker accounting math, tier selection, exhaustion behavior
- Integration: mock HTTP in `llm_score_pairs` with budget, verify degradation
- Edge: budget=0 (no LLM calls at all), budget=infinity (no change in behavior)

---

### 1B. Fellegi-Sunter Probabilistic Model

**Why:** Theoretical legitimacy. Automatic threshold/weight tuning via EM. Replaces hand-tuned thresholds with statistically optimal ones.

**Scope:**
- New matchkey type: `probabilistic` (alongside `exact` and `weighted`)
- EM algorithm to estimate m-probabilities (P(agree|match)) and u-probabilities (P(agree|random)) from unlabeled data
- Comparison vector generation: per-field agree/partial-agree/disagree
- Log-likelihood match weights per pair
- Fellegi-Sunter upper/lower thresholds (link / possible-link / non-link)
- Falls back gracefully: if EM doesn't converge, use weighted scoring with warnings

**Config contract:**
```yaml
matchkeys:
  - name: fs_names
    type: probabilistic
    fields:
      - field: first_name
        scorer: jaro_winkler
        levels: 3              # agree / partial / disagree
        partial_threshold: 0.8 # score >= this = partial agree
      - field: last_name
        scorer: jaro_winkler
        levels: 3
        partial_threshold: 0.8
      - field: zip
        scorer: exact
        levels: 2              # agree / disagree only
    em_iterations: 20          # max EM iterations
    convergence_threshold: 0.001
    # thresholds auto-computed; user can override:
    # link_threshold: 10.0     # log-likelihood above this = match
    # review_threshold: 2.0    # between review and link = possible match
```

**Schema changes required:**
- `MatchkeyConfig.type` Literal: extend from `"exact" | "weighted"` to `"exact" | "weighted" | "probabilistic"`
- `MatchkeyField`: add optional fields `levels: int | None = None`, `partial_threshold: float | None = None`
- `MatchkeyConfig`: add optional fields `em_iterations: int = 20`, `convergence_threshold: float = 0.001`, `link_threshold: float | None = None`, `review_threshold: float | None = None`
- `_validate_weighted` validator: add `elif self.type == "probabilistic"` branch — no threshold required upfront (EM computes it), fields need scorer but not weight

**Pipeline integration (CRITICAL — not a new stage, a third branch):**
- The existing pipeline loops over matchkeys with branches for `mk.type == "exact"` and `mk.type == "weighted"`. Probabilistic adds a **third branch in the same loop**, not a separate pipeline stage.
- Within the probabilistic branch: (1) build blocks, (2) train EM on a sample from blocks, (3) score all block pairs using EM weights. This is a two-pass within the branch: sample → train → score.
- The EM training step draws comparison vectors from a random sample of block pairs (configurable `em_sample_size`, default 10000 pairs). This avoids scoring all pairs twice.

**New files:**
- `goldenmatch/core/probabilistic.py` — EM training, comparison vectors, match-weight computation, threshold estimation
- Modify: `goldenmatch/config/schemas.py` — extend type Literal, add fields to `MatchkeyField` and `MatchkeyConfig`, update validator
- Modify: `goldenmatch/core/pipeline.py` — add third branch in matchkey loop for `type: probabilistic`
- Modify: `goldenmatch/core/scorer.py` — comparison vector generation reuses existing scorers
- Tests: `tests/test_probabilistic.py`

**Interface contract (consumed by later phases):**
```python
@dataclass
class EMResult:
    m_probs: dict[str, list[float]]     # field -> [P(level_i | match)]
    u_probs: dict[str, list[float]]     # field -> [P(level_i | non-match)]
    match_weights: dict[str, list[float]]  # field -> log2(m/u) per level
    converged: bool
    iterations: int
    proportion_matched: float           # estimated match rate

def train_em(
    df: pl.DataFrame,
    mk: MatchkeyConfig,
    blocks: list[BlockResult] | None = None,  # if None, sample random pairs
    max_iterations: int = 20,
    convergence: float = 0.001,
) -> EMResult: ...

def score_probabilistic(
    block_df: pl.DataFrame,
    mk: MatchkeyConfig,
    em_result: EMResult,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float]]: ...
# Returns pairs with match_weight normalized to 0-1 scale

def compute_thresholds(em_result: EMResult) -> tuple[float, float]:
    """Returns (link_threshold, review_threshold) on log-likelihood scale."""
```

**Testing strategy:**
- Unit: EM convergence on synthetic data with known match rate
- Unit: comparison vector generation for each scorer type
- Unit: threshold computation produces sensible link/review boundaries
- Integration: full pipeline with `type: probabilistic`, verify clusters match expected
- Benchmark: run on DBLP-ACM, compare F1 vs. weighted scoring

---

## Phase 2: Self-Configuring, Extensible Framework

**Depends on Phase 1:** Auto-config benefits from F-S for automatic thresholds. Plugin architecture defines extension points that Phase 1 features already use.

### 2A. Plugin Architecture

**Why before connectors:** Connectors become the first plugins. Learned blocking and auto-config can also be plugin-discoverable.

**Scope:**
- `goldenmatch.plugins` entry-point group for pip-installable extensions
- Plugin types: `scorer`, `transform`, `blocker`, `connector`, `golden_strategy`
- Registry pattern: plugins register via entry points, discovered at import time
- **Discovery-before-validation ordering:** `PluginRegistry.discover()` must be called before Pydantic config parsing. The registry populates sets (e.g., `PLUGIN_SCORERS`) that schema validators check as a fallback after `VALID_SCORERS`. This means the application entry point (CLI/TUI/API) calls `PluginRegistry.discover()` at startup, before loading any YAML config.
- Built-in features remain built-in (no regression); plugins extend the surface
- CLI: `goldenmatch plugins list` shows installed plugins

**Config contract:**
```yaml
# Plugins auto-discovered via entry points. User just references by name:
sources:
  - connector: snowflake          # resolved from goldenmatch.plugins.connector
    query: "SELECT * FROM customers"
    credentials_env: SNOWFLAKE_CONN

matchkeys:
  - name: custom
    type: weighted
    fields:
      - field: name
        scorer: my_custom_scorer  # resolved from goldenmatch.plugins.scorer
```

**New files:**
- `goldenmatch/plugins/__init__.py` — `PluginRegistry` singleton, discovery logic
- `goldenmatch/plugins/registry.py` — entry-point scanning, type validation, registration API
- `goldenmatch/plugins/base.py` — Protocol classes: `ScorerPlugin`, `TransformPlugin`, `BlockerPlugin`, `ConnectorPlugin`, `GoldenStrategyPlugin`
- Modify: `goldenmatch/config/schemas.py` — validators check plugin registry for unknown names
- Modify: `goldenmatch/core/scorer.py` — plugin dispatch fallback in `score_field` and `_fuzzy_score_matrix`
- Modify: `goldenmatch/core/blocker.py` — plugin dispatch fallback in `build_blocks`
- Modify: `goldenmatch/core/ingest.py` — connector dispatch for `connector:` sources
- Modify: `goldenmatch/cli/main.py` — add `plugins` command
- Tests: `tests/test_plugins.py`

**Interface contract (consumed by all later phases):**
```python
# Base protocols in goldenmatch/plugins/base.py

class ScorerPlugin(Protocol):
    name: str
    def score_pair(self, val_a: str | None, val_b: str | None) -> float | None: ...
    def score_matrix(self, values_a: list[str], values_b: list[str]) -> np.ndarray: ...
    # score_matrix optional — falls back to pairwise if not implemented

class TransformPlugin(Protocol):
    name: str
    def transform(self, value: str) -> str: ...
    def transform_series(self, series: pl.Series) -> pl.Series: ...
    # transform_series optional — falls back to map_elements if not implemented

class BlockerPlugin(Protocol):
    name: str
    def build_blocks(self, lf: pl.LazyFrame, config: BlockingConfig, **kwargs) -> list[BlockResult]: ...
    # kwargs allows learned blocker to receive scored_pairs, matchkeys, etc.
    # Standard blockers ignore kwargs; learned blocker uses them for training

class ConnectorPlugin(Protocol):
    name: str
    def read(self, config: dict) -> pl.LazyFrame: ...
    def write(self, df: pl.DataFrame, config: dict) -> None: ...

class GoldenStrategyPlugin(Protocol):
    name: str
    def merge(self, values: list, sources: list[str] | None = None) -> tuple[Any, float]: ...
    # Returns (merged_value, confidence)

class PluginRegistry:
    def discover() -> None: ...                     # scan entry points
    def get_scorer(name: str) -> ScorerPlugin | None: ...
    def get_blocker(name: str) -> BlockerPlugin | None: ...
    def get_connector(name: str) -> ConnectorPlugin | None: ...
    def get_transform(name: str) -> TransformPlugin | None: ...
    def get_golden_strategy(name: str) -> GoldenStrategyPlugin | None: ...
    def list_plugins() -> dict[str, list[str]]: ... # type -> [names]
```

**Entry point spec (for plugin authors):**
```toml
# In plugin's pyproject.toml:
[project.entry-points."goldenmatch.plugins.scorer"]
my_scorer = "my_package.scorers:MyScorer"

[project.entry-points."goldenmatch.plugins.connector"]
snowflake = "goldenmatch_snowflake:SnowflakeConnector"
```

**Testing strategy:**
- Unit: registry discovers mock entry points, returns correct plugin
- Unit: unknown scorer/blocker/connector falls through to plugin registry before raising ValueError
- Integration: create a test plugin package, install in editable mode, verify discovery
- Negative: verify built-in names still work, verify helpful error for truly unknown names

---

### 2B. Learned Blocking

**Why:** Blocking key selection dominates fuzzy performance. Learned blocking replaces human guesswork with data-driven predicate selection.

**Scope:**
- New blocking strategy: `learned`
- Training data from a small sample run (positive pairs from high-score matches, negatives from random)
- Predicate library: first-N-chars, soundex, exact, token-1, zip-prefix, etc.
- Evaluate candidate predicate sets by recall vs. reduction ratio
- Select the set with best recall at user-specified minimum reduction ratio
- Cache learned predicates for reuse across runs

**Config contract:**
```yaml
blocking:
  strategy: learned
  learned:
    sample_size: 5000           # records to sample for training
    min_recall: 0.95            # must capture 95% of true matches
    min_reduction: 0.90         # must eliminate 90% of comparisons
    predicate_depth: 2          # max predicates combined per blocking rule
    cache_path: .goldenmatch/learned_blocking.json  # persist for reuse
```

**Bootstrap problem resolution:** Learned blocking needs scored pairs for training, but blocking happens before scoring. Solution: **automatic two-pass approach**. Pass 1: run a fast sample (default 5000 records) with conservative static blocking (highest-cardinality column) to generate training pairs. Pass 2: train predicates on those pairs, then apply learned blocks to the full dataset. If a `cache_path` exists from a previous run, skip Pass 1 and reuse cached predicates. The sample run uses the same matchkeys as the main run but on a subset. This is transparent to the user — `strategy: learned` handles both passes internally.

**New files:**
- `goldenmatch/core/learned_blocking.py` — predicate library, training, evaluation, selection, two-pass orchestration
- Modify: `goldenmatch/config/schemas.py` — add `learned` to strategy Literal, add `LearnedBlockingConfig`
- Modify: `goldenmatch/core/blocker.py` — add `learned` dispatch
- Tests: `tests/test_learned_blocking.py`

**Interface contract:**
```python
@dataclass
class BlockingPredicate:
    field: str
    transform: str          # "exact", "first_3", "soundex", "first_token", "zip3", etc.

@dataclass
class BlockingRule:
    predicates: list[BlockingPredicate]   # conjunction (AND)
    recall: float
    reduction_ratio: float
    pair_completeness: float

def generate_predicates(columns: list[str]) -> list[BlockingPredicate]: ...

def evaluate_predicate_set(
    df: pl.DataFrame,
    predicates: list[BlockingPredicate],
    true_pairs: set[tuple[int, int]],
) -> tuple[float, float]: ...  # (recall, reduction_ratio)

def learn_blocking_rules(
    df: pl.DataFrame,
    scored_pairs: list[tuple[int, int, float]],
    config: LearnedBlockingConfig,
) -> list[BlockingRule]: ...

def apply_learned_blocks(
    lf: pl.LazyFrame,
    rules: list[BlockingRule],
) -> list[BlockResult]: ...
```

**Testing strategy:**
- Unit: predicate generation covers expected transforms per column type
- Unit: evaluation correctly computes recall and reduction ratio on known data
- Unit: rule selection respects min_recall and min_reduction constraints
- Integration: learned strategy produces blocks, pipeline completes end-to-end
- Benchmark: compare block counts and F1 vs. static blocking on DBLP-ACM

---

### 2C. Auto-Config / Zero-Config Mode

**Why:** Biggest adoption blocker. "Just throw data at it" is the Dedupe.io pitch we need to match.

**Scope:**
- `goldenmatch dedupe data.csv` with NO config file — auto-generates everything
- Column type detection: name, email, phone, address, zip, ID, date, free-text
- Auto-selects: matchkeys (exact for IDs/emails, weighted for names+addresses), blocking keys, thresholds
- If F-S is available (Phase 1B), uses EM for threshold tuning; otherwise conservative defaults
- Outputs generated YAML config alongside results for user to inspect/edit
- Extends existing `autoconfig.py` and `schema_match.py`

**Config contract (none needed — that's the point):**
```bash
# Zero-config:
goldenmatch dedupe customers.csv

# Auto with overrides:
goldenmatch dedupe customers.csv --threshold 0.85 --blocking-key zip
```

**Files to modify:**
- `goldenmatch/core/autoconfig.py` — expand column-type detection, matchkey generation, blocking selection
- Modify: `goldenmatch/cli/dedupe.py` — make config optional, call autoconfig if missing
- Modify: `goldenmatch/cli/match.py` — same
- New: `goldenmatch/core/column_classifier.py` — heuristic + regex column type classifier
- Tests: `tests/test_autoconfig.py`, `tests/test_column_classifier.py`

**Interface contract:**
```python
@dataclass
class ColumnProfile:
    name: str
    inferred_type: str          # "name", "email", "phone", "address", "zip", "id", "date", "text", "numeric", "unknown"
    cardinality: int
    null_rate: float
    avg_length: float
    sample_values: list[str]

def classify_columns(df: pl.DataFrame, sample_size: int = 1000) -> list[ColumnProfile]: ...

def generate_config(
    profiles: list[ColumnProfile],
    mode: str = "dedupe",       # "dedupe" or "match"
    use_probabilistic: bool = False,  # use F-S if available
) -> GoldenMatchConfig: ...

def save_generated_config(config: GoldenMatchConfig, path: Path) -> None: ...
```

**Testing strategy:**
- Unit: column classifier correctly identifies name/email/phone/zip/address on synthetic data
- Unit: generated config includes appropriate matchkeys for detected types
- Integration: zero-config dedupe on `tests/fixtures/` data produces reasonable clusters
- Edge: single-column file, all-null columns, mixed encodings

---

## Phase 3: Enterprise Data Access + Audit Trail

**Depends on Phase 2:** Connectors use plugin architecture. Explainability uses lineage contracts.

### 3A. Connectors (Snowflake, Databricks, BigQuery, HubSpot, Salesforce)

**Scope:**
- Each connector is a `ConnectorPlugin` (Phase 2A contract)
- Optional dependencies: `pip install goldenmatch[snowflake]`, `pip install goldenmatch[hubspot]`, etc.
- Read and write support (write-back is connector-specific)
- Credentials via environment variables (no secrets in config)
- Built-in connectors ship in `goldenmatch/connectors/` but register via the plugin system

**Schema integration with existing `InputConfig`:** The existing `InputConfig.files` uses `InputFileConfig` with `path` as the primary field. Connectors extend `InputFileConfig` with an optional `connector` field. When `connector` is set, `path` is ignored and data is read from the connector. This means `input.files` supports both file-based and connector-based sources in the same list — no new top-level `sources` key needed. The YAML example below uses `input.files` for consistency:

**Config contract:**
```yaml
input:
  files:
    - connector: snowflake
      source_name: sf_customers
    credentials_env: SNOWFLAKE_ACCOUNT  # reads SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
    query: "SELECT id, name, email, phone FROM customers WHERE active = true"

  - connector: hubspot
    source_name: hs_contacts
    credentials_env: HUBSPOT_API_KEY
    object_type: contacts
    properties: [firstname, lastname, email, phone, city, zip]
    filters:
      - property: createdate
        operator: GTE
        value: "2024-01-01"

output:
  write_back:
    connector: snowflake
    credentials_env: SNOWFLAKE_ACCOUNT
    table: customer_golden
    mode: upsert                # upsert | append | replace
```

**New files:**
- `goldenmatch/connectors/__init__.py`
- `goldenmatch/connectors/snowflake.py`
- `goldenmatch/connectors/databricks.py`
- `goldenmatch/connectors/bigquery.py`
- `goldenmatch/connectors/hubspot.py`
- `goldenmatch/connectors/salesforce.py`
- Modify: `goldenmatch/core/ingest.py` — detect `connector:` in source spec, dispatch to plugin
- Modify: `goldenmatch/output/writer.py` — support `write_back` config
- Modify: `pyproject.toml` — optional dependency groups
- Tests: `tests/test_connectors/` — one test file per connector (mocked)

**Connector-specific notes:**
- **Snowflake**: `snowflake-connector-python`, `write_pandas` for write-back
- **Databricks**: `databricks-sql-connector`, DBFS for large results
- **BigQuery**: `google-cloud-bigquery`, already have GCP credentials
- **HubSpot**: REST API v3, pagination via `after` cursor, rate limit 100 req/10s
- **Salesforce**: `simple-salesforce`, SOQL queries, bulk API for large reads

---

### 3B. Explainability-as-a-Feature

**Scope:**
- Upgrade lineage from a sidecar JSON to a first-class pipeline output
- Per-pair natural language explanation (template-based, not LLM — cheap)
- LLM-generated cluster narrative for borderline clusters (optional, budget-controlled)
- Remove 10,000 pair cap — **this is a partial rewrite of `lineage.py`**, not just a parameter change. Current `build_lineage` calls `df.to_dicts()` and iterates in memory. Replace with streaming JSON writer that appends pairs incrementally to disk, never holding all lineage in memory.
- Exportable audit trail format for compliance
- TUI: cluster detail view shows per-pair explanations

**Config contract:**
```yaml
explainability:
  enabled: true
  format: json                # json | html | csv
  natural_language: true      # template-based NL explanations
  llm_narratives: false       # LLM-generated cluster summaries (uses budget)
  max_pairs: null             # null = no cap (stream to disk)
  include_non_matches: false  # include rejected pairs in audit trail
```

**New files:**
- `goldenmatch/core/explain.py` — NL template engine, cluster narrative generator
- Modify: `goldenmatch/core/lineage.py` — streaming output, remove cap, add NL field
- Modify: `goldenmatch/config/schemas.py` — add `ExplainabilityConfig`
- Modify: `goldenmatch/tui/engine.py` — expose explanations in `EngineResult`
- Tests: `tests/test_explain.py`

**Interface contract:**
```python
def explain_pair_nl(
    row_a: dict, row_b: dict,
    field_scores: list[dict],   # from existing lineage field data
    overall_score: float,
) -> str: ...
# "Matched (score 0.92): names are phonetically identical (John Smith ~ Jon Smyth,
#  Soundex S530), zip codes match exactly (90210), phone numbers differ by 2 digits
#  (555-1234 vs 555-1236, score 0.83). Weakest signal: phone."

def explain_cluster_nl(
    cluster: dict,              # cluster info dict from build_clusters
    df: pl.DataFrame,
    matchkeys: list[MatchkeyConfig],
) -> str: ...
# Template-based cluster summary

def explain_cluster_llm(
    cluster: dict,
    df: pl.DataFrame,
    matchkeys: list[MatchkeyConfig],
    budget: BudgetTracker | None = None,
) -> str | None: ...
# LLM-generated narrative; returns None if budget exhausted
```

---

### 3C. DuckDB Backend (User-Maintained)

**Scope:**
- New `backend: duckdb` config option
- User provides DuckDB database path (or in-memory)
- GoldenMatch reads from DuckDB tables/views instead of files
- Blocking via DuckDB SQL (spill to disk for large datasets)
- Scoring still in Polars (pull blocks into memory via `pl.from_arrow`)
- Write results back to DuckDB tables
- User manages their own DuckDB — we don't create/migrate schemas

**Config contract:**
```yaml
backend:
  type: duckdb
  path: ./my_data.duckdb        # or ":memory:"

sources:
  - table: customers             # DuckDB table/view name
    source_name: customers
    query: "SELECT * FROM customers WHERE active"  # optional override
```

**New files:**
- `goldenmatch/backends/__init__.py`
- `goldenmatch/backends/duckdb_backend.py` — read/write via DuckDB → Polars bridge
- Modify: `goldenmatch/core/ingest.py` — backend dispatch
- Modify: `goldenmatch/output/writer.py` — DuckDB write-back
- Modify: `pyproject.toml` — `pip install goldenmatch[duckdb]`
- Tests: `tests/test_duckdb_backend.py`

---

## Phase 4: Real-Time + Relational Entity Resolution

**Depends on Phase 3:** Streaming uses connectors for sources. Graph ER uses all preceding features.

### 4A. Streaming / CDC Mode

**Scope:**
- Watch a source (connector, file directory, Postgres LISTEN/NOTIFY) for new records
- Each new record → block → score → cluster update → write-back / notify
- LLM scoring for borderline pairs (async, budget-controlled via Phase 1A)
- Configurable processing: immediate (per-record) or micro-batch (every N seconds)
- Health endpoint, PID file (extend existing `watch_daemon` pattern)
- Dashboard metrics: records/sec, match rate, LLM cost, latency

**Config contract:**
```yaml
streaming:
  enabled: true
  mode: micro_batch             # immediate | micro_batch
  batch_interval_sec: 30
  source:
    connector: snowflake        # or: watch_dir, postgres_notify, webhook
    poll_interval_sec: 60
    query: "SELECT * FROM customers WHERE updated_at > :last_seen"
  write_back:
    connector: snowflake
    table: customer_golden
    mode: upsert
  budget:
    max_cost_usd_per_hour: 1.00
```

**New files:**
- `goldenmatch/core/streaming.py` — stream processor, micro-batch logic
- `goldenmatch/core/cdc.py` — change detection for each source type
- Modify: `goldenmatch/db/watch.py` — generalize to use streaming processor
- Modify: `goldenmatch/cli/main.py` — `goldenmatch stream` command
- Tests: `tests/test_streaming.py`

---

### 4B. Multi-Table / Graph Entity Resolution

**Feasibility note:** This is a research-level feature. Iterative convergence across entity types requires running the pipeline N times per entity type per iteration. **Recommend a feasibility spike** (1-2 days) before committing to full implementation: prototype evidence propagation on a 2-entity toy dataset, measure convergence behavior, validate that the architecture doesn't require fundamental pipeline restructuring. If the spike reveals issues, scope can be reduced to single-iteration evidence propagation (no convergence loop).

**Scope:**
- Define entity types and relationships in config
- Match within entity types (standard ER) then propagate evidence across relationships
- Graph-based clustering: if customer A's orders match customer B's orders, boost A-B customer match score
- Iterative convergence: match → propagate → re-match until stable (subject to feasibility spike)
- Output: unified entity graph with cross-type links

**Config contract:**
```yaml
entities:
  - name: customers
    sources:
      - path: crm_customers.csv
        source_name: crm
    matchkeys:
      - name: cust_fuzzy
        type: weighted
        fields: [...]

  - name: orders
    sources:
      - path: orders.csv
        source_name: orders
    matchkeys:
      - name: order_exact
        type: exact
        fields: [...]

relationships:
  - from: orders
    to: customers
    join_key: customer_id          # foreign key in orders table
    evidence_weight: 0.3           # how much order matches boost customer matches

graph:
  max_iterations: 5
  convergence_threshold: 0.01
  propagation_mode: additive       # additive | multiplicative
```

**New files:**
- `goldenmatch/core/graph_er.py` — multi-entity orchestrator, evidence propagation
- `goldenmatch/core/entity_graph.py` — graph data structure, cross-entity links
- Modify: `goldenmatch/config/schemas.py` — `EntityConfig`, `RelationshipConfig`, `GraphConfig`
- Modify: `goldenmatch/cli/main.py` — `goldenmatch graph-dedupe` command
- Tests: `tests/test_graph_er.py`

---

## Cross-Cutting Concerns

### Testing Strategy (All Phases)
- Every new module gets unit tests FIRST (TDD)
- Integration tests use existing fixtures in `tests/fixtures/` + new synthetic data
- Benchmark tests added for performance-critical features (F-S, learned blocking)
- Total test count must only increase — no existing tests broken
- `pytest --tb=short` must pass after every commit

### Backward Compatibility
- All new config fields have defaults — existing YAML configs work unchanged
- New matchkey type `probabilistic` is opt-in
- Plugin system is additive — built-in features don't change behavior
- Connectors are optional deps — base install stays lightweight
- **Existing Postgres integration (`goldenmatch/db/`)**: Remains as-is. It is a purpose-built sync/reconcile system, not a generic connector. The new connector system (Phase 3A) handles data ingestion from external sources; the existing `db/` package handles persistent state management. They serve different purposes and coexist. If a Postgres *connector* (read-only data source) is needed, it can be added as a connector plugin separately.

### Documentation
- Each phase updates `CLAUDE.md` with new architecture notes
- README updated with new features at each release
- Config examples in `examples/` directory

---

## Execution Protocol

For each phase:

1. **Write detailed plan** — Full implementation plan with TDD steps, exact file paths, code snippets
2. **Execute plan** — Using superpowers:subagent-driven-development or superpowers:executing-plans
3. **Review** — Code review via superpowers:requesting-code-review
4. **Test** — Full `pytest --tb=short` pass
5. **Commit** — Feature branch per phase
6. **Retrospective** — Update master roadmap with learnings, adjust subsequent phases if needed
7. **Write next phase plan** — Informed by what was actually built, not just what was planned

---

## Current Status

- [ ] **Phase 1A: LLM Budget Controller** — Not started
- [ ] **Phase 1B: Fellegi-Sunter Probabilistic Model** — Not started
- [ ] **Phase 2A: Plugin Architecture** — Not started
- [ ] **Phase 2B: Learned Blocking** — Not started
- [ ] **Phase 2C: Auto-Config / Zero-Config** — Not started
- [ ] **Phase 3A: Connectors** — Not started
- [ ] **Phase 3B: Explainability** — Not started
- [ ] **Phase 3C: DuckDB Backend** — Not started
- [ ] **Phase 4A: Streaming / CDC** — Not started
- [ ] **Phase 4B: Multi-Table Graph ER** — Not started
