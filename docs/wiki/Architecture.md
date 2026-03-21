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
│   ├── app.py              # GoldenMatchApp (gold theme, bindings, routing)
│   ├── sidebar.py          # Persistent stats sidebar
│   ├── engine.py           # MatchEngine (no Textual dependency)
│   ├── screens/
│   │   └── autoconfig_screen.py  # Zero-config summary screen
│   ├── widgets/
│   │   ├── progress_overlay.py   # Full-screen pipeline progress
│   │   └── threshold_slider.py   # Live threshold with arrow keys
│   └── tabs/               # Data, Config, Matches, Golden, Export
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

## Test Structure

688 tests across:
- `tests/test_*.py` — unit tests for core modules
- `tests/test_db.py` — Postgres integration tests
- `tests/test_reconcile.py` — reconciliation + versioning tests
- `tests/test_incremental.py` — ANN index + hybrid blocking tests
- `tests/benchmarks/` — Leipzig benchmark scripts
