# Architecture

## Project Structure

```
goldenmatch/
├── cli/                    # CLI layer (Typer)
│   ├── main.py             # App entry point, command registration
│   ├── dedupe.py           # goldenmatch dedupe command
│   ├── match.py            # goldenmatch match command
│   └── sync.py             # goldenmatch sync command (database)
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
│   ├── blocker.py          # 7 blocking strategies
│   ├── scorer.py           # 8 scoring methods
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
│   └── llm_labeler.py      # LLM pair labeling (Claude/GPT-4)
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
│   └── reconcile.py        # Merge-back + conflict resolution
│
├── tui/                    # Interactive TUI (Textual)
│   ├── app.py              # GoldenMatchApp (5-tab interface)
│   └── engine.py           # MatchEngine (no Textual dependency)
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

### Database Sync Flow

```
New records → Hybrid blocking (SQL + ANN) → Score → Reconcile → Write
                  ↑                                      ↓
           Persistent FAISS index              Versioned golden records
           Progressive embedding               Persistent clusters
```

## Test Structure

605+ tests across:
- `tests/test_*.py` — unit tests for core modules
- `tests/test_db.py` — Postgres integration tests
- `tests/test_reconcile.py` — reconciliation + versioning tests
- `tests/test_incremental.py` — ANN index + hybrid blocking tests
- `tests/benchmarks/` — Leipzig benchmark scripts
