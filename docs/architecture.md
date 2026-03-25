---
layout: default
title: Architecture
nav_order: 20
---

# Architecture

GoldenMatch is organized into 11 top-level packages. Core pipeline logic lives in `goldenmatch/core/` with no UI dependencies.

---

## Module map

```
goldenmatch/
  __init__.py          # 101 public exports
  _api.py              # High-level convenience functions (dedupe, match, etc.)
  client.py            # REST API client (stdlib urllib only)

  cli/                 # 21 Typer CLI commands
    main.py            # App definition + command registration
    dedupe.py          # dedupe command
    match.py           # match command
    serve.py           # REST API server command
    mcp_serve.py       # MCP server command
    evaluate.py        # evaluate command
    incremental.py     # incremental matching
    pprl.py            # pprl link / pprl auto-config
    label.py           # ground truth labeling
    rollback.py        # rollback, runs, unmerge commands
    ...

  config/              # Configuration
    schemas.py         # Pydantic models (GoldenMatchConfig, MatchkeyConfig, etc.)
    loader.py          # YAML loading + normalization

  core/                # Pipeline modules (no Textual dependency)
    pipeline.py        # run_dedupe, run_match, _run_dedupe_pipeline, _run_match_pipeline
    ingest.py          # load_file, load_files
    standardize.py     # apply_standardization (native Polars fast path)
    matchkey.py        # compute_matchkeys (_try_native_chain fast path)
    blocker.py         # build_blocks, ANNBlocker
    scorer.py          # find_exact_matches, find_fuzzy_matches, score_blocks_parallel
    cluster.py         # build_clusters, add_to_cluster, unmerge_record, unmerge_cluster
    golden.py          # build_golden_record
    autoconfig.py      # auto_configure
    autofix.py         # auto_fix_dataframe
    validate.py        # validate_dataframe
    anomaly.py         # detect_anomalies
    explain.py         # explain_pair_nl, explain_cluster_nl
    explainer.py       # Legacy explainer
    evaluate.py        # evaluate_pairs, evaluate_clusters, EvalResult
    profiler.py        # profile_dataframe
    lineage.py         # build_lineage, save_lineage, save_lineage_streaming
    match_one.py       # Single-record matching primitive
    streaming.py       # StreamProcessor, run_stream
    probabilistic.py   # Fellegi-Sunter EM training + scoring
    learned_blocking.py  # Data-driven predicate selection
    llm_scorer.py      # llm_score_pairs (pairwise LLM scoring)
    llm_cluster.py     # llm_cluster_pairs (in-context block clustering)
    llm_budget.py      # BudgetTracker
    llm_labeler.py     # LLM-labeled training pairs
    llm_extract.py     # LLM-based feature extraction
    domain.py          # Domain extraction (brand/model/SKU)
    domain_registry.py # YAML-based custom domain registry
    boost.py           # Active learning accuracy boost
    threshold.py       # suggest_threshold
    schema_match.py    # auto_map_columns
    graph_er.py        # Multi-table entity resolution
    diff.py            # generate_diff
    rollback.py        # rollback_run
    block_analyzer.py  # analyze_blocking (blocking strategy suggestions)
    chunked.py         # Chunked processing for large files
    cloud_ingest.py    # S3, GCS, Azure Blob ingestion
    api_connector.py   # REST/GraphQL API connector
    scheduler.py       # Cron-like scheduling
    gpu.py             # GPU detection (detect_gpu_mode)
    vertex_embedder.py # Vertex AI managed embeddings
    report.py          # HTML report generation
    dashboard.py       # Before/after dashboard

  tui/                 # Textual TUI (gold-themed)
    app.py             # Main TUI application
    engine.py          # MatchEngine (no Textual dependency)
    ...

  db/                  # PostgreSQL integration
    connector.py       # Database connection
    sync.py            # Incremental sync
    reconcile.py       # Conflict reconciliation
    clusters.py        # Persistent cluster management
    ann_index.py       # Persistent ANN index
    watch.py           # Watch mode + daemon (watch_daemon)

  api/                 # REST API
    server.py          # HTTP server (MatchServer, endpoints)

  mcp/                 # MCP server
    server.py          # MCP tools for Claude Desktop

  plugins/             # Plugin system
    registry.py        # PluginRegistry singleton (entry point discovery)
    base.py            # Protocol classes for scorer/transform/connector/golden_strategy

  connectors/          # Enterprise data connectors
    base.py            # BaseConnector ABC + load_connector()
    snowflake.py       # Snowflake
    databricks.py      # Databricks
    bigquery.py        # BigQuery
    hubspot.py         # HubSpot
    salesforce.py      # Salesforce

  backends/            # Processing backends
    duckdb_backend.py  # DuckDB out-of-core processing
    ray_backend.py     # Ray distributed block scoring

  pprl/                # Privacy-preserving record linkage
    protocol.py        # PPRLConfig, run_pprl, TTP + SMC protocols
    autoconfig.py      # auto_configure_pprl, profile_for_pprl

  domains/             # Built-in YAML domain packs
    electronics.yaml
    software.yaml
    healthcare.yaml
    financial.yaml
    real_estate.yaml
    people.yaml
    retail.yaml

  prefs/               # Settings persistence
    store.py           # PresetStore

  output/              # Output formatting
    writer.py          # write_output
    report.py          # generate_dedupe_report

  utils/               # Shared utilities
    transforms.py      # apply_transforms, bloom_filter transforms
```

---

## Code patterns

### Internal columns

All internal columns are prefixed with `__`:

| Column | Purpose |
|--------|---------|
| `__row_id__` | Unique record identifier (int64) |
| `__source__` | Source file/table name |
| `__mk_*__` | Computed matchkey values |
| `__cluster_id__` | Cluster assignment |
| `__brand__`, `__model__`, etc. | Domain extraction outputs |

### File specs

File specifications are tuples:

```python
(path, source_name)                    # Basic
(path, source_name, column_map)        # With column mapping
```

### Scorer return type

All scorers return `list[tuple[int, int, float]]` -- `(row_id_a, row_id_b, score)`.

### Cluster dict structure

`build_clusters` returns `dict[int, dict]` where each value has:

```python
{
    "members": [1, 42, 108],           # List of row_ids
    "size": 3,
    "oversized": False,
    "pair_scores": {(1, 42): 0.92, (1, 108): 0.87, (42, 108): 0.83},
    "confidence": 0.87,               # 0.4*min + 0.3*avg + 0.3*connectivity
    "bottleneck_pair": (42, 108),      # Weakest link
}
```

### Config access

```python
config.get_matchkeys()  # Returns matchkeys from top-level or match_settings
mk.type                 # Use .type (not .comparison) after validation
```

---

## Plugin system

Extend GoldenMatch with custom scorers, transforms, connectors, and golden strategies via pip-installable plugins.

### Plugin protocols

```python
from goldenmatch.plugins.base import ScorerPlugin, TransformPlugin

class MyScorer(ScorerPlugin):
    name = "my_scorer"

    def score(self, value_a: str, value_b: str) -> float:
        ...
```

### Registration

Plugins are discovered via Python entry points:

```toml
# pyproject.toml
[project.entry-points."goldenmatch.scorers"]
my_scorer = "my_package:MyScorer"
```

The `PluginRegistry` singleton discovers plugins at startup. Schema validators fall through to plugins for unknown scorer/transform names.

---

## Pipeline architecture

### Backend selection

```python
# pipeline.py
def _get_block_scorer(config):
    if config.backend == "ray":
        from goldenmatch.backends.ray_backend import score_blocks_ray
        return score_blocks_ray
    return score_blocks_parallel  # ThreadPoolExecutor default
```

### Shared internal functions

Both file-based and DataFrame-based entry points call the same internal pipeline:

```
gm.dedupe("file.csv")    -->  run_dedupe()     -->  _run_dedupe_pipeline()
gm.dedupe_df(df)          -->  run_dedupe_df()  -->  _run_dedupe_pipeline()
```

### Performance fast paths

| Module | Fast Path | Fallback |
|--------|-----------|----------|
| `standardize.py` | `_NATIVE_STANDARDIZERS` (Polars expressions) | Python UDFs |
| `matchkey.py` | `_try_native_chain` (Polars expressions) | Python apply |
| `scorer.py` | `rapidfuzz.cdist` (vectorized C) | Python loop |
| `cluster.py` | Iterative Union-Find | N/A |

---

## Testing

- 944 tests (+ 6 skipped), ~40s on Windows
- `pytest --tb=short` from project root
- CI: `.github/workflows/ci.yml` (Python 3.11/3.12/3.13, ruff lint)
- DB tests (`test_db.py`, `test_reconcile.py`) need PostgreSQL
- TUI tests use `pytest-asyncio` with `app.run_test()` pilot
- Ray tests use `pytest.mark.skipif(not HAS_RAY)` pattern
