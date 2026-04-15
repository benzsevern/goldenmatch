---
layout: default
title: Python API
nav_order: 4
---

# Python API

GoldenMatch exports 106 symbols from a single import. Every feature is accessible via `import goldenmatch as gm`.

```python
import goldenmatch as gm
```

---

## High-level convenience functions

These are the primary entry points for most users.

### dedupe

```python
gm.dedupe(
    *files: str,
    config: str | GoldenMatchConfig | None = None,
    exact: list[str] | None = None,
    fuzzy: dict[str, float] | None = None,
    blocking: list[str] | None = None,
    threshold: float | None = None,
    llm_scorer: bool = False,
    backend: str | None = None,
) -> DedupeResult
```

Deduplicate one or more files. Pass file paths as positional arguments.

```python
result = gm.dedupe("customers.csv")
result = gm.dedupe("a.csv", "b.csv", exact=["email"], fuzzy={"name": 0.85})
result = gm.dedupe("products.csv", config="config.yaml", llm_scorer=True)
result = gm.dedupe("huge.parquet", exact=["email"], backend="ray")
```

### dedupe_df

```python
gm.dedupe_df(
    df: pl.DataFrame,
    *,
    config: GoldenMatchConfig | None = None,
    exact: list[str] | None = None,
    fuzzy: dict[str, float] | None = None,
    blocking: list[str] | None = None,
    threshold: float | None = None,
    llm_scorer: bool = False,
    llm_auto: bool = False,
    backend: str | None = None,
    source_name: str = "dataframe",
) -> DedupeResult
```

Deduplicate a Polars DataFrame directly (no file I/O). Also used by SQL extensions.

```python
df = pl.read_csv("customers.csv")
result = gm.dedupe_df(df, exact=["email"], fuzzy={"name": 0.85})
```

### match

```python
gm.match(
    target: str,
    reference: str,
    *,
    config: str | GoldenMatchConfig | None = None,
    exact: list[str] | None = None,
    fuzzy: dict[str, float] | None = None,
    blocking: list[str] | None = None,
    threshold: float | None = None,
    backend: str | None = None,
) -> MatchResult
```

Match a target file against a reference file.

```python
result = gm.match("new.csv", "master.csv", fuzzy={"name": 0.85})
result.matched.write_csv("matches.csv")
```

### match_df

```python
gm.match_df(
    target: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    config: GoldenMatchConfig | None = None,
    exact: list[str] | None = None,
    fuzzy: dict[str, float] | None = None,
    blocking: list[str] | None = None,
    threshold: float | None = None,
    backend: str | None = None,
) -> MatchResult
```

Match DataFrames directly without file I/O.

```python
result = gm.match_df(target_df, reference_df, fuzzy={"name": 0.85})
```

---

## Scoring functions

### score_strings

```python
gm.score_strings(value_a: str, value_b: str, scorer: str = "jaro_winkler") -> float
```

Score two strings with a named similarity scorer. Returns 0.0--1.0.

```python
gm.score_strings("John Smith", "Jon Smyth", "jaro_winkler")  # 0.884
gm.score_strings("hello", "helo", "levenshtein")              # 0.8
gm.score_strings("Smith", "Smyth", "soundex_match")           # 1.0
```

### score_pair_df

```python
gm.score_pair_df(
    record_a: dict, record_b: dict,
    *, fuzzy: dict[str, float] | None = None,
    exact: list[str] | None = None,
    scorer: str = "jaro_winkler",
) -> float
```

Score a pair of records. Returns weighted overall score.

```python
score = gm.score_pair_df(
    {"name": "John Smith", "zip": "10001"},
    {"name": "Jon Smyth", "zip": "10001"},
    fuzzy={"name": 0.7, "zip": 0.3},
)
```

### explain_pair_df

```python
gm.explain_pair_df(
    record_a: dict, record_b: dict,
    *, fuzzy: dict[str, float] | None = None,
    exact: list[str] | None = None,
    scorer: str = "jaro_winkler",
) -> str
```

Generate a natural-language explanation for why two records match (or don't).

```python
explanation = gm.explain_pair_df(
    {"name": "John Smith", "email": "john@example.com"},
    {"name": "Jon Smyth", "email": "john@example.com"},
    fuzzy={"name": 0.5}, exact=["email"],
)
print(explanation)
```

---

## PPRL (Privacy-Preserving Record Linkage)

### pprl_link

```python
gm.pprl_link(
    file_a: str, file_b: str,
    *, fields: list[str] | None = None,
    threshold: float | None = None,
    security_level: str = "high",
    protocol: str = "trusted_third_party",
    auto_config: bool = True,
) -> dict
```

Link records across organizations without sharing raw data.

```python
result = gm.pprl_link("hospital_a.csv", "hospital_b.csv")
result = gm.pprl_link("a.csv", "b.csv", fields=["name", "dob", "zip"], security_level="paranoid")
```

### pprl_auto_config / auto_configure_pprl

```python
gm.pprl_auto_config(df: pl.DataFrame, security_level: str = "high") -> AutoConfigResult
```

Profile a DataFrame and recommend optimal PPRL fields, bloom filter parameters, and threshold.

```python
config = gm.pprl_auto_config(df)
print(config.recommended_fields)  # ['first_name', 'last_name', 'zip_code', 'birth_year']
```

### run_pprl

```python
gm.run_pprl(df_a: pl.DataFrame, df_b: pl.DataFrame, config: PPRLConfig) -> LinkageResult
```

Low-level PPRL execution with full control over parameters.

### PPRLConfig

```python
gm.PPRLConfig(
    fields: list[str],
    threshold: float = 0.85,
    security_level: str = "high",
    ngram_size: int = 2,
    hash_functions: int = 30,
    bloom_filter_size: int = 1024,
    protocol: str = "trusted_third_party",
)
```

### Other PPRL exports

| Symbol | Description |
|--------|-------------|
| `compute_bloom_filters` | Convert field values to bloom filter bit arrays |
| `link_trusted_third_party` | TTP protocol implementation |
| `link_smc` | Secure multi-party computation protocol |
| `PartyData` | Dataclass for party data + bloom filters |
| `LinkageResult` | Dataclass with clusters, match_count, total_comparisons |
| `profile_for_pprl` | Profile data for PPRL field selection |
| `auto_configure_pprl_llm` | LLM-assisted PPRL configuration |

---

## Evaluation

### evaluate

```python
gm.evaluate(
    *files: str,
    config: str | GoldenMatchConfig,
    ground_truth: str,
    col_a: str = "id_a",
    col_b: str = "id_b",
) -> dict
```

Run the full pipeline and evaluate against ground truth. Returns dict with `precision`, `recall`, `f1`, `tp`, `fp`, `fn`.

```python
metrics = gm.evaluate("data.csv", config="config.yaml", ground_truth="gt.csv")
```

### evaluate_pairs

```python
gm.evaluate_pairs(predicted: set[tuple], ground_truth: set[tuple]) -> EvalResult
```

Evaluate predicted pairs against ground truth pairs directly.

### evaluate_clusters

```python
gm.evaluate_clusters(clusters: dict, ground_truth: set[tuple]) -> EvalResult
```

Evaluate cluster dict (from `build_clusters`) against ground truth.

### EvalResult

```python
@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def summary(self) -> dict  # Returns dict representation
```

### load_ground_truth_csv

```python
gm.load_ground_truth_csv(path: str, col_a: str = "id_a", col_b: str = "id_b") -> set[tuple]
```

### compare_clusters

```python
gm.compare_clusters(
    clusters_a: dict[int, dict],
    clusters_b: dict[int, dict],
) -> CompareResult
```

Compare two clustering outcomes on the same dataset using CCMS. Classifies each cluster from A as unchanged, merged, partitioned, or overlapping relative to B.

```python
result = gm.compare_clusters(clusters_run1, clusters_run2)
print(result.summary())  # {"unchanged": 42, "merged": 3, "twi": 0.92, ...}
```

### CompareResult

```python
@dataclass
class CompareResult:
    unchanged: int          # Clusters identical in both runs
    merged: int             # Clusters absorbed into larger clusters
    partitioned: int        # Clusters split into smaller clusters
    overlapping: int        # Complex reorganization
    rc: int                 # Total reference count
    cc1: int                # Cluster count in run A
    cc2: int                # Cluster count in run B
    sc1: int                # Singleton count in run A
    sc2: int                # Singleton count in run B
    twi: float              # Talburt-Wang Index (1.0 = identical)
    cases: list[ClusterCase]  # Per-cluster details

    def summary(self) -> dict
```

### run_sensitivity

```python
gm.run_sensitivity(
    file_specs: list,
    config: GoldenMatchConfig,
    sweep_params: list[SweepParam],
    sample_size: int | None = None,
) -> list[SensitivityResult]
```

Sweep parameters and compare each run against a baseline using CCMS.

```python
results = gm.run_sensitivity(
    file_specs=[("data.csv", "src")],
    config=config,
    sweep_params=[gm.SweepParam("threshold", 0.70, 0.95, 0.05)],
    sample_size=5000,
)
print(results[0].stability_report())
```

### SweepParam

```python
gm.SweepParam(field: str, start: float, stop: float, step: float)
```

Fields: `"threshold"`, `"matchkey.<name>.threshold"`, `"blocking.max_block_size"`.

---

## Configuration

### load_config

```python
gm.load_config(path: str) -> GoldenMatchConfig
```

Load a YAML config file into a Pydantic model.

```python
config = gm.load_config("config.yaml")
matchkeys = config.get_matchkeys()
```

### GoldenMatchConfig

Top-level config model. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `matchkeys` | `list[MatchkeyConfig]` | Match rules |
| `blocking` | `BlockingConfig` | Blocking strategy |
| `golden_rules` | `GoldenRulesConfig` | Merge strategies |
| `standardization` | `StandardizationConfig` | Column transforms |
| `validation` | `ValidationConfig` | Data quality rules |
| `llm_scorer` | `LLMScorerConfig` | LLM scoring options |
| `domain` | `DomainConfig` | Domain extraction |
| `output` | `OutputConfig` | Output format and directory |
| `backend` | `str` | `None`, `"ray"`, or `"duckdb"` |

### MatchkeyConfig

```python
gm.MatchkeyConfig(
    name: str,
    type: Literal["exact", "weighted", "probabilistic"],
    fields: list[MatchkeyField],
    threshold: float | None = None,      # required for weighted
    rerank: bool = False,                 # cross-encoder reranking
    rerank_band: float = 0.1,
    em_iterations: int = 20,             # for probabilistic
)
```

### MatchkeyField

```python
gm.MatchkeyField(
    field: str,
    scorer: str | None = None,           # jaro_winkler, levenshtein, exact, etc.
    weight: float | None = None,         # required for weighted matchkeys
    transforms: list[str] = [],          # lowercase, strip, soundex, etc.
    columns: list[str] | None = None,    # for record_embedding scorer
    column_weights: dict[str, float] | None = None,
    levels: int = 2,                     # for probabilistic: 2 or 3
)
```

### Other config classes

| Class | Description |
|-------|-------------|
| `BlockingConfig` | Strategy, keys, ANN settings, learned blocking params |
| `BlockingKeyConfig` | Fields + transforms for one blocking key |
| `GoldenRulesConfig` | Default strategy + per-field merge rules |
| `GoldenFieldRule` | Strategy for a single field |
| `LLMScorerConfig` | Provider, model, thresholds, budget, mode |
| `BudgetConfig` | max_cost_usd, max_calls, escalation settings |
| `DomainConfig` | Domain extraction settings |
| `StandardizationConfig` | Column-level standardization rules |
| `ValidationConfig` | Data validation rules |
| `OutputConfig` | Output path, format, directory |

---

## Result classes

### DedupeResult

```python
@dataclass
class DedupeResult:
    golden: pl.DataFrame | None       # Canonical merged records
    clusters: dict[int, dict]         # Cluster info (members, pair_scores, confidence)
    dupes: pl.DataFrame | None        # Duplicate records
    unique: pl.DataFrame | None       # Non-duplicate records
    stats: dict                       # total_records, total_clusters, match_rate
    scored_pairs: list[tuple]         # (id_a, id_b, score) tuples
    config: GoldenMatchConfig
    postflight_report: PostflightReport | None  # v1.5.0: signals + advisories + adjustments

    def to_csv(path, which="golden")  # Write results to CSV
    match_rate: float                 # Property: percentage of dupes
    total_records: int                # Property
    total_clusters: int               # Property
```

Both `DedupeResult` and `MatchResult` have `_repr_html_()` for rich Jupyter display.

### MatchResult

```python
@dataclass
class MatchResult:
    matched: pl.DataFrame | None      # Matched target records with scores
    unmatched: pl.DataFrame | None    # Unmatched target records
    stats: dict
    postflight_report: PostflightReport | None  # v1.5.0: signals + advisories + adjustments

    def to_csv(path)
```

---

## Streaming and incremental

### match_one

```python
gm.match_one(record: dict, df: pl.DataFrame, matchkey: MatchkeyConfig) -> list[tuple]
```

Match a single record against an existing DataFrame. Returns list of `(row_id, score)` tuples.

```python
matches = gm.match_one({"name": "John Smith", "zip": "10001"}, existing_df, matchkey)
```

Note: returns empty list for exact matchkeys (threshold=None). Use `find_exact_matches` for exact matching.

### StreamProcessor

```python
gm.StreamProcessor(df: pl.DataFrame, config: GoldenMatchConfig)
```

Incremental record matching with immediate or micro-batch processing.

```python
processor = gm.StreamProcessor(df, config)
matches = processor.process_record(new_record)
```

### run_stream

```python
gm.run_stream(df, config, records) -> dict
```

---

## Pipeline functions

### run_dedupe / run_match

```python
gm.run_dedupe(file_specs: list[tuple], config: GoldenMatchConfig) -> dict
gm.run_match(target_spec: tuple, ref_specs: list[tuple], config: GoldenMatchConfig) -> dict
```

Low-level pipeline entry points. File specs are `(path, source_name)` tuples.

### Scorer functions

| Function | Description |
|----------|-------------|
| `find_exact_matches(df, fields)` | Polars self-join for exact matching |
| `find_fuzzy_matches(block_df, matchkey, ...)` | Vectorized fuzzy scoring via `rapidfuzz.cdist` |
| `score_pair(record_a, record_b, fields)` | Score a single pair |
| `score_blocks_parallel(blocks, ...)` | Parallel block scoring via ThreadPoolExecutor |
| `rerank_top_pairs(pairs, df, matchkey)` | Cross-encoder reranking |

### Cluster functions

| Function | Description |
|----------|-------------|
| `build_clusters(pairs)` | Iterative Union-Find clustering |
| `add_to_cluster(record_id, matches, clusters)` | Incremental cluster update |
| `unmerge_record(record_id, clusters)` | Remove record, re-cluster remaining |
| `unmerge_cluster(cluster_id, clusters)` | Shatter cluster to singletons |
| `compute_cluster_confidence(cluster)` | Confidence = 0.4*min + 0.3*avg + 0.3*connectivity |

### Other pipeline functions

| Function | Description |
|----------|-------------|
| `build_blocks(df, config)` | Apply blocking strategy |
| `build_golden_record(cluster, df, rules)` | Merge cluster into one record |
| `load_file(path)` | Load CSV/Excel/Parquet as LazyFrame |
| `load_files(file_specs)` | Load and concat multiple files |
| `apply_standardization(df, config)` | Apply column standardization |
| `compute_matchkeys(df, matchkeys)` | Compute matchkey columns |

---

## LLM scoring

### llm_score_pairs

```python
gm.llm_score_pairs(pairs, df, config: LLMScorerConfig) -> list[tuple]
```

Score borderline pairs with GPT/Claude. Accepts `LLMScorerConfig` with optional `BudgetConfig`.

### llm_cluster_pairs

```python
gm.llm_cluster_pairs(pairs, df, config: LLMScorerConfig) -> list[tuple]
```

In-context block clustering as alternative to pairwise scoring. Set `config.mode = "cluster"`.

### BudgetTracker

```python
gm.BudgetTracker(max_cost_usd=0.05, max_calls=100)
```

Tracks token usage, cost, and enforces limits. Degrades gracefully: cluster mode falls back to pairwise, then stops.

### Other LLM exports

| Symbol | Description |
|--------|-------------|
| `llm_label_pairs` | LLM-labeled training pairs |
| `llm_extract_features` | LLM-based feature extraction for low-confidence records |

---

## Probabilistic matching (Fellegi-Sunter)

### train_em

```python
gm.train_em(df, matchkey, n_sample_pairs=10000, blocking_fields=None) -> EMResult
```

Train m/u probabilities via Expectation-Maximization. Pass `blocking_fields` to exclude them from training.

### score_probabilistic

```python
gm.score_probabilistic(block_df, matchkey, em_result) -> list[tuple]
```

Score pairs using trained Fellegi-Sunter weights.

---

## Learned blocking

```python
gm.learn_blocking_rules(df, matchkey, ...) -> LearnedBlockingResult
gm.apply_learned_blocks(df, rules) -> list[pl.DataFrame]
```

Data-driven predicate selection. Achieves 96.9% F1 matching hand-tuned blocking.

---

## Explainability

```python
gm.explain_pair(record_a, record_b, field_scores, overall) -> str
gm.explain_cluster(cluster, df) -> str
```

Template-based natural-language explanations at zero LLM cost. `explain_pair` is an alias for `explain_pair_nl`.

---

## Domain extraction

```python
gm.discover_rulebooks() -> dict[str, DomainRulebook]  # 7 built-in packs
gm.load_rulebook(name) -> DomainRulebook
gm.save_rulebook(name, rulebook)
gm.match_domain(df, column) -> str | None
gm.extract_with_rulebook(df, column, rulebook) -> tuple[pl.DataFrame, pl.DataFrame]
```

---

## Data quality

```python
gm.auto_fix_dataframe(df) -> tuple[pl.DataFrame, list]
gm.validate_dataframe(df, config) -> list[dict]
gm.detect_anomalies(df) -> list[dict]
gm.auto_map_columns(df_a, df_b) -> dict[str, str]  # Schema matching
```

---

## Lineage and profiling

```python
gm.build_lineage(clusters, df, matchkeys) -> dict
gm.save_lineage(lineage, path)
gm.profile_dataframe(df) -> dict
```

---

## Auto-configuration

```python
gm.auto_configure(file_specs) -> GoldenMatchConfig
gm.auto_configure_df(
    df,
    llm_provider=None,
    domain_config=None,
    llm_auto=False,
    strict=False,              # v1.5.0
    allow_remote_assets=False, # v1.5.0
) -> GoldenMatchConfig
gm.suggest_threshold(df, matchkey) -> float
```

New v1.5.0 kwargs on `auto_configure_df`:

- `strict` — compute postflight signals and emit advisories, but suppress auto-adjustments (threshold nudges, etc.). Use for DQBench / regression / reproducibility runs.
- `allow_remote_assets` — permit `embedding`, `record_embedding`, and cross-encoder rerank scorers. Default `False` demotes them so auto-config is offline-safe and never triggers a surprise HuggingFace download.

The returned config carries `config._preflight_report: PreflightReport` (underscore — private-by-convention but stable across v1.5.x).

---

## Verification (v1.5.0)

The preflight + postflight layer validates an auto-generated config against the data it was built for. `auto_configure_df` runs preflight automatically at the end; the pipeline runs postflight automatically after scoring. Both are also callable directly.

### preflight

```python
gm.preflight(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    *,
    profiles: list[ColumnProfile] | None = None,
    allow_remote_assets: bool = False,
) -> PreflightReport
```

Runs 6 checks on `(df, config)`:

1. **Column resolution** — every column referenced by blocking/matchkeys exists, or is a pipeline-synthesized `__mk_*`, or is a domain-extracted column recoverable by enabling `config.domain` (auto-repaired when a domain profile was stashed during auto-config).
2. **Exact-matchkey cardinality** — drops keys with ratio >= 0.99 (near-unique, no pair ever agrees) or < 0.01 (near-constant, produces giant blocks).
3. **Block-size sanity** — samples blocking keys and flags blocks that would stall the scorer.
4. **Remote-asset demotion** — `embedding` / `record_embedding` / cross-encoder rerank scorers are demoted unless `allow_remote_assets=True`.
5. **Weight confidence capping** — matchkey fields with profile confidence < 0.5 cap at weight 0.3 (requires `profiles` kwarg).
6. **Domain auto-repair** — when a column like `__title_key__` is missing but a domain profile is available, enables `config.domain` so the pipeline produces the column at runtime.

Auto-repairs what it can (setting `finding.repaired = True`) and records unrepairable issues as `severity="error"` findings. `auto_configure_df` raises `ConfigValidationError` if `report.has_errors`.

```python
report = gm.preflight(df, config)
for f in report.findings:
    print(f"[{f.severity}] {f.check}: {f.message} (repaired={f.repaired})")
```

### postflight

```python
gm.postflight(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    *,
    pair_scores: list[tuple[int, int, float]],
    current_threshold: float | None = None,
) -> PostflightReport
```

Runs 4 signals on scored pairs:

- **Score histogram + bimodality** — if the score distribution is clearly bimodal (valley depth ratio < 0.5) and the valley is > 0.05 away from the current threshold, emits a `PostflightAdjustment` nudging the threshold to the valley. Suppressed under `strict=True`.
- **Blocking recall estimate** — gated at >= 10K rows; returns `"deferred"` below that.
- **Preliminary cluster sizes + oversized-cluster bottleneck pair** — p50/p95/p99/max plus a list of oversized clusters with their weakest edge.
- **Threshold-band overlap** — fraction of pairs within 0.02 of the threshold. Advises `--llm-auto` when > 20% and LLM scorer is off.

```python
report = gm.postflight(df, config, pair_scores=scored, current_threshold=0.85)
print(report.signals["threshold_overlap_pct"])
for adj in report.adjustments:
    print(f"{adj.field}: {adj.from_value} -> {adj.to_value} ({adj.reason})")
```

### Report shapes

```python
@dataclass
class PreflightFinding:
    check: str                    # "missing_column" | "cardinality" | "block_size" |
                                  # "remote_asset" | "weight_confidence"
    severity: str                 # "error" | "warning" | "info"
    subject: str                  # column / matchkey name
    message: str
    repaired: bool
    repair_note: str | None

@dataclass
class PreflightReport:
    findings: list[PreflightFinding]
    config_was_modified: bool
    has_errors: bool              # property: True if any unrepaired error

class ConfigValidationError(Exception):
    report: PreflightReport       # full report attached for programmatic inspection

@dataclass
class PostflightAdjustment:
    field: str                    # e.g. "threshold"
    from_value: Any
    to_value: Any
    reason: str
    signal: str                   # which signal motivated the change

@dataclass
class PostflightReport:
    signals: PostflightSignals    # TypedDict, schema below
    adjustments: list[PostflightAdjustment]
    advisories: list[str]
```

### PostflightSignals schema

The `signals` dict is a stable TypedDict contract (defined in `goldenmatch/core/autoconfig_verify.py`):

```python
class PostflightSignals(TypedDict):
    score_histogram: ScoreHistogram           # {"bins": list[float], "counts": list[int]}
    blocking_recall: float | Literal["deferred"]  # "deferred" when <10K rows
    block_size_percentiles: BlockSizePercentiles  # {"p50", "p95", "p99", "max"}
    threshold_overlap_pct: float              # fraction of pairs within 0.02 of threshold
    total_pairs_scored: int
    current_threshold: float
    preliminary_cluster_sizes: ClusterSizePercentiles
        # {"p50", "p95", "p99", "max", "count"}
    oversized_clusters: list[OversizedCluster]
        # each: {"cluster_id": int, "size": int, "bottleneck_pair": [int, int]}
```

`ScoreHistogram`, `BlockSizePercentiles`, `ClusterSizePercentiles`, `OversizedCluster` are all TypedDicts — import them from `goldenmatch.core.autoconfig_verify` if you want to type-check consumer code.

### Where the reports live

- `config._preflight_report: PreflightReport | None` — set by `auto_configure_df`. Underscore-prefixed, documented as private-by-convention; stable contract.
- `DedupeResult.postflight_report: PostflightReport | None` — set by the pipeline after scoring.
- `MatchResult.postflight_report: PostflightReport | None` — same for match flows.

---

## Active learning

```python
gm.boost_accuracy(pairs, df, config) -> dict
```

Label borderline pairs, train LogisticRegression, reclassify.

---

## Diff and rollback

```python
gm.generate_diff(before, after) -> dict
gm.rollback_run(run_id) -> dict
```

---

## Graph ER

```python
gm.run_graph_er(entities, relationships) -> dict
```

Multi-table entity resolution with cross-relationship evidence propagation.

---

## Output

```python
gm.write_output(result, config) -> dict
gm.generate_dedupe_report(result) -> str  # HTML report
```

---

## REST API client

```python
client = gm.Client("http://localhost:8080")
client.match(record)
client.list_clusters()
client.explain(id_a, id_b)
client.reviews()
```

Uses stdlib `urllib` only -- no extra dependencies.
