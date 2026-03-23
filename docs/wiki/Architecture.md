# Architecture

## Project Structure

```
goldenmatch/
├── cli/                    # CLI layer (Typer)
│   ├── main.py             # App entry point, command registration
│   ├── dedupe.py           # goldenmatch dedupe command
│   ├── match.py            # goldenmatch match command
│   ├── sync.py             # goldenmatch sync command (database)
│   └── label.py            # goldenmatch label command (ground truth builder)
│
├── config/                 # Configuration
│   ├── schemas.py          # Pydantic models (MatchkeyConfig, BlockingConfig, etc.)
│   ├── loader.py           # YAML config loading + normalization
│   ├── settings.py         # User preferences persistence (global + project)
│   └── wizard.py           # Interactive config generator
│
├── core/                   # Pipeline modules (no UI dependency)
│   ├── pipeline.py         # Orchestrator: run_dedupe(), run_match()
│   ├── ingest.py           # File loading (CSV, Excel, Parquet)
│   ├── smart_ingest.py     # Auto-detection (encoding, delimiter, format)
│   ├── autofix.py          # Auto data quality fixes
│   ├── standardize.py      # Per-column standardization
│   ├── matchkey.py         # Matchkey computation
│   ├── blocker.py          # 8+ blocking strategies
│   ├── scorer.py           # 10 scoring methods (incl. dice, jaccard)
│   ├── cluster.py          # Union-Find clustering
│   ├── golden.py           # Golden record merging
│   ├── autoconfig.py       # Zero-config column profiling
│   ├── profiler.py         # Data profiling + type detection
│   ├── embedder.py         # Sentence-transformer wrapper
│   ├── ann_blocker.py      # FAISS ANN blocking
│   ├── canopy.py           # TF-IDF canopy clustering
│   ├── threshold.py        # Otsu's method auto-threshold
│   ├── boost.py            # LLM boost orchestrator
│   ├── cross_encoder.py    # Ditto-style cross-encoder
│   ├── llm_labeler.py      # LLM pair labeling (Claude/GPT-4)
│   ├── llm_scorer.py       # LLM scorer (GPT-4o-mini/Claude for borderline pairs)
│   ├── llm_budget.py       # Budget tracking for LLM calls (cost caps, model tiering)
│   ├── lineage.py          # Lineage persistence (per-field explanations, JSON sidecar)
│   ├── match_one.py        # Single-record matching primitive for streaming
│   ├── evaluate.py         # Evaluation engine (precision, recall, F1 from ground truth)
│   ├── probabilistic.py    # Fellegi-Sunter EM-trained probabilistic matching
│   ├── learned_blocking.py # Data-driven blocking predicate selection
│   ├── streaming.py        # StreamProcessor for incremental/CDC matching
│   ├── graph_er.py         # Multi-table graph entity resolution
│   ├── domain.py           # Product domain extraction (brand, model, specs)
│   ├── domain_registry.py  # Custom YAML domain rulebooks
│   ├── llm_extract.py      # LLM-based feature extraction for low-confidence records
│   ├── explain.py          # Template-based NL explanations (zero LLM cost)
│   └── anomaly.py          # Anomaly detection (fake emails, placeholder data)
│
├── db/                     # Database integration
│   ├── connector.py        # Abstract interface + PostgresConnector
│   ├── blocking.py         # SQL WHERE query builder
│   ├── metadata.py         # gm_* table management
│   ├── sync.py             # Incremental matching orchestrator
│   ├── writer.py           # Result write-back
│   ├── ann_index.py        # Persistent FAISS index
│   ├── hybrid_blocking.py  # SQL + ANN union blocking
│   ├── clusters.py         # Persistent cluster management
│   ├── reconcile.py        # Merge-back + conflict resolution
│   └── watch.py            # Daemon mode (health endpoint, PID file, SIGTERM)
│
├── domains/                # Built-in YAML domain packs
│   ├── electronics.yaml    # Electronics: model numbers, SKUs, specs
│   ├── software.yaml       # Software: versions, editions, platforms
│   ├── healthcare.yaml     # Healthcare: NDC, NPI, ICD-10, pharma brands
│   ├── financial.yaml      # Financial: CUSIP, ISIN, LEI, institutions
│   ├── real_estate.yaml    # Real estate: ZIP, APN, MLS, property attributes
│   ├── people.yaml         # People: SSN, DOB, phone, email
│   └── retail.yaml         # Retail: UPC, EAN, GTIN, CPG brands
│
├── plugins/                # Plugin system
│   ├── registry.py         # PluginRegistry singleton (entry point discovery)
│   └── base.py             # Protocol classes (ScorerPlugin, TransformPlugin, etc.)
│
├── connectors/             # Enterprise data source connectors
│   ├── base.py             # BaseConnector ABC + load_connector()
│   ├── snowflake.py        # Snowflake connector
│   ├── databricks.py       # Databricks connector
│   ├── bigquery.py         # BigQuery connector
│   ├── hubspot.py          # HubSpot connector
│   └── salesforce.py       # Salesforce connector
│
├── backends/               # Storage backends
│   ├── duckdb_backend.py   # DuckDB for out-of-core processing
│   └── ray_backend.py      # Ray distributed block scoring
│
├── tui/                    # Interactive TUI (Textual)
│   ├── app.py              # GoldenMatchApp (gold theme, bindings, routing)
│   ├── sidebar.py          # Persistent stats sidebar
│   ├── engine.py           # MatchEngine (no Textual dependency)
│   ├── screens/
│   │   └── autoconfig_screen.py  # Zero-config summary screen
│   ├── widgets/
│   │   ├── progress_overlay.py   # Full-screen pipeline progress
│   │   └── threshold_slider.py   # Live threshold with arrow keys
│   └── tabs/               # Data, Config, Matches, Golden, Boost, Export
│
└── utils/
    └── transforms.py       # Transform implementations
```

## Key Design Principles

### Separation of Concerns

- **core/**: Pure pipeline logic, no UI or DB dependency
- **tui/**: Textual UI wraps core pipeline via MatchEngine
- **db/**: Database operations, uses core pipeline for scoring
- **cli/**: Thin layer connecting CLI args to pipeline/TUI/DB

### Internal Column Convention

Internal columns are prefixed with `__`:
- `__row_id__`: Unique row identifier
- `__source__`: Source file/table name
- `__mk_*__`: Computed matchkey values
- `__cluster_id__`: Cluster assignment
- `__is_golden__`: Golden record flag

### Configuration Flow

```
YAML file → loader.py → GoldenMatchConfig (Pydantic) → pipeline
CLI flags → override fields on GoldenMatchConfig
Auto-config → generate GoldenMatchConfig from data profiling
```

### Scorer Architecture

Scoring uses NxN matrix computation for vectorized performance:

```
Block DataFrame → _get_transformed_values (per field)
                → _fuzzy_score_matrix (cdist for RapidFuzz, embedding for ST)
                → weighted combination → threshold filter → pairs
```

**Parallel block scoring:** All block-scoring call sites (pipeline.py, engine.py, chunked.py) use `score_blocks_parallel()` from scorer.py. This scores independent blocks concurrently via `ThreadPoolExecutor` (4 threads default). RapidFuzz's `cdist` releases the GIL, so threads provide real parallelism. A frozen snapshot of `exclude_pairs` avoids race conditions.

**Intra-field early termination:** After each expensive field (fuzzy/embedding), the scorer checks whether any pair in the upper triangle can still reach the threshold even with perfect scores on all remaining fields. If not, it breaks out of the field loop, skipping unnecessary `cdist` calls.

**Cross-encoder reranking:** When `rerank: true` is set on a weighted matchkey, pairs within `threshold +/- rerank_band` are re-scored using a pre-trained cross-encoder (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`). No training needed -- zero-shot reranking for improved precision on borderline pairs.

**Histogram-based auto-select:** When `auto_select: true` is set on blocking config with multiple keys, `select_best_blocking_key()` evaluates each key's group-size distribution and picks the one with smallest max_block_size while maintaining >= 50% coverage.

**Dynamic block splitting:** When adaptive blocking encounters oversized blocks and no `sub_block_keys` are configured, `_auto_split_block()` picks the column with the most useful groups (groups with >= 2 records) and splits by it. Prefers columns that create right-sized groups over maximum cardinality.

**Weighted multi-field embedding:** `column_weights` on `record_embedding` fields allows biasing the embedding toward important fields. High-weight fields have their text repeated in the concatenation (weight 2.0 = text included twice), while weight 0 excludes a field entirely.

### Database Sync Flow

```
New records → Hybrid blocking (SQL + ANN) → Score → Reconcile → Write
                  ↑                                      ↓
           Persistent FAISS index              Versioned golden records
           Progressive embedding               Persistent clusters
```

### Active Learning (Boost Tab)

The TUI includes a "Boost" tab for human-in-the-loop active learning:

1. Active sampling selects the 10 hardest borderline pairs (combined strategy: uncertainty + boundary + disagreement + diversity)
2. User labels pairs with `y` (match), `n` (non-match), or `s` (skip)
3. Logistic regression trains on the labeled pairs' feature vectors (Jaro-Winkler, token sort, Levenshtein, exact, length ratio per column)
4. All pairs are re-scored with classifier probabilities and re-clustered
5. Matches + Golden tabs refresh with boosted results

### Per-Entity Unmerge

Two operations in `cluster.py` for fine-grained undo:

- **`unmerge_record(record_id, clusters, threshold)`** — removes a record from its cluster, re-clusters remaining members using stored `pair_scores`. The removed record becomes a singleton; remaining members that are still connected stay clustered.
- **`unmerge_cluster(cluster_id, clusters)`** — shatters a cluster into individual singletons, discarding all pair connections.

Available via CLI (`goldenmatch unmerge RECORD_ID`) and programmatically via `MatchEngine.unmerge_record()` / `MatchEngine.unmerge_cluster()`.

### LLM Scorer

`core/llm_scorer.py` sends borderline pairs to GPT-4o-mini (or Claude) for yes/no match decisions. Three-tier approach:

1. **Auto-accept** (score >= 0.95): high-confidence pairs skip the LLM
2. **LLM score** (0.75-0.95): borderline pairs batched and sent to the LLM
3. **Auto-reject** (< 0.75): low-confidence pairs keep original scores

The LLM sees both records' fields and decides "same entity?" — it understands model number abbreviations, naming conventions, and semantic equivalences that embedding similarity alone cannot capture. On Abt-Buy, this approach jumps from 62.8% (embedding-only) to **81.7% F1** at ~$0.74 cost.

Configured via `llm_scorer` in the config YAML. Supports OpenAI and Anthropic APIs, auto-detected from environment variables.

### Streaming Foundation (match_one)

The `core/match_one.py` module provides a single-record matching primitive -- the building block for streaming entity resolution:

- **`match_one(record, df, mk)`** — embeds the new record, queries top-K candidates from the FAISS index, scores each candidate pair using the matchkey's fields/weights, returns matches above threshold. Falls back to brute-force when no ANN index is available.
- **`ANNBlocker.add_to_index(embedding)`** — incrementally adds a vector to the FAISS index without rebuilding.
- **`ANNBlocker.query_one(embedding)`** — queries top-K neighbors for a single vector.
- **`add_to_cluster(record_id, matches, clusters)`** — incrementally updates the cluster graph. If the new record matches members of one cluster, it joins. If it bridges two clusters, they merge. Confidence is recomputed automatically.

These primitives enable the `watch` command and incremental DB sync to process single records without re-running the full pipeline.

### MCP Server (12 Tools)

The MCP server (`goldenmatch mcp-serve`) exposes 12 tools for Claude Desktop integration:

1. **match_record** -- match a single record against the dataset
2. **unmerge_record** -- remove a record from its cluster
3. **shatter_cluster** -- split a cluster into singletons
4. **suggest_config** -- analyze bad merges and suggest threshold/weight changes
5. **explain_match** -- per-field score breakdown for a pair
6. **list_clusters** -- paginated cluster listing
7. **get_cluster** -- cluster detail with members and golden record
8. **search_records** -- full-text search across dataset
9. **get_stats** -- pipeline statistics
10. **run_dedupe** -- trigger deduplication
11. **get_golden_record** -- retrieve a golden record by cluster
12. **get_config** -- current configuration

### REST API (10 Endpoints)

The REST API server (`goldenmatch serve`) provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/match` | POST | Match a record against the dataset |
| `/clusters` | GET | List clusters (paginated) |
| `/clusters/{id}` | GET | Cluster detail |
| `/golden/{id}` | GET | Golden record for a cluster |
| `/explain/{id_a}/{id_b}` | GET | Per-field match explanation |
| `/stats` | GET | Pipeline statistics |
| `/config` | GET | Current configuration |
| `/reviews` | GET | Borderline pairs for steward review |
| `/reviews/decide` | POST | Record approve/reject decisions |
| `/health` | GET | Health check |

### Lineage Persistence

`core/lineage.py` provides `build_lineage()` and `save_lineage()` which save per-pair field-level explanations to a `{run_name}_lineage.json` sidecar file. Auto-generated when the pipeline writes output. Each lineage entry includes field-by-field scores and the overall match decision rationale.

### Daemon Mode

`db/watch.py` provides `watch_daemon()` which extends the `watch` command with:
- **Health endpoint** -- HTTP server on configurable port responding to `/health`
- **PID file** -- written on start, cleaned up on exit
- **SIGTERM handling** -- graceful shutdown on signal

Activated via `goldenmatch watch --daemon`.

## Python API

All features are accessible via `import goldenmatch as gm` (95 public exports). The API has three tiers:

**Tier 1 -- Convenience functions** (most users):
- `gm.dedupe()`, `gm.match()`, `gm.pprl_link()`, `gm.evaluate()`
- Simple kwargs, typed result objects, docstrings with examples

**Tier 2 -- Config + pipeline** (power users):
- `gm.GoldenMatchConfig`, `gm.MatchkeyConfig`, `gm.BlockingConfig`
- `gm.run_dedupe()`, `gm.build_clusters()`, `gm.build_blocks()`

**Tier 3 -- Full toolkit** (library developers):
- Every scorer, blocker, clusterer, explainer, domain tool, LLM function
- `gm.train_em()`, `gm.llm_cluster_pairs()`, `gm.run_graph_er()`, etc.

## Test Structure

911 tests across:
- `tests/test_*.py` — unit tests for core modules
- `tests/test_db.py` — Postgres integration tests
- `tests/test_reconcile.py` — reconciliation + versioning tests
- `tests/test_incremental.py` — ANN index + hybrid blocking tests
- `tests/benchmarks/` — Leipzig benchmark scripts
