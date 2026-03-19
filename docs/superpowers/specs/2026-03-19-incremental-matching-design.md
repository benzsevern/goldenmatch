# Incremental Matching Engine — Design Specification

## Overview

Add persistent ANN indexing, hybrid blocking, and progressive embedding computation to GoldenMatch's database integration. New records are matched against existing database records using both SQL-side blocking (exact/phonetic fields) and FAISS ANN blocking (semantic fields) in parallel, with results unioned for maximum recall. Embeddings are computed progressively across runs so the system is useful immediately and gets better over time.

## Problem Statement

The current `goldenmatch sync` command does database-side SQL blocking for exact fields but has no semantic matching capability against existing records. For large tables (10M+), computing embeddings for all records up front takes hours. The system needs to work immediately with SQL blocking and progressively add ANN capability as embeddings are computed.

## Goals

- Persistent FAISS index with hybrid disk/DB storage
- Hybrid blocking: SQL WHERE + ANN in parallel, union results
- Progressive embedding: compute in chunks across runs, ANN available incrementally
- Scale to 10M+ existing records
- No degradation of existing sync functionality

## Non-Goals

- GPU-accelerated embedding (CPU is the baseline)
- Distributed FAISS (single-machine index)
- Real-time index updates (batch updates per sync run)

---

## Feature 1: Persistent ANN Index Manager

### Design

New module `goldenmatch/db/ann_index.py` — manages the FAISS index lifecycle across sync runs.

### API

```python
class PersistentANNIndex:
    def __init__(self, index_dir: Path, connector: DatabaseConnector, source_table: str)

    def load_or_build(self) -> None
        """Load from disk if fresh, rebuild from gm_embeddings if stale."""

    def query(self, embeddings: np.ndarray, top_k: int = 20) -> list[tuple[int, int, float]]
        """Find neighbors. Returns (query_idx, db_record_id, score)."""

    def add(self, record_ids: list[int], embeddings: np.ndarray) -> None
        """Add new embeddings to index + save to gm_embeddings."""

    def save(self) -> None
        """Persist index to disk."""

    @property
    def is_available(self) -> bool
        """True if index has enough embeddings for useful queries."""
```

### Disk Storage

```
.goldenmatch_faiss/
├── index.faiss          # FAISS IndexFlatIP binary
├── index_meta.json      # {"record_count": 5000000, "last_updated": "...", "model": "all-MiniLM-L6-v2"}
└── id_map.npy           # Maps FAISS positional index → database record ID
```

### Staleness Detection

On startup:
1. Load `index_meta.json` from disk
2. Query `SELECT COUNT(*) FROM gm_embeddings WHERE source_table = ?`
3. If DB count > disk count: load disk index, then append delta embeddings from DB
4. If disk cache missing: full rebuild from `gm_embeddings`
5. If DB count == disk count: load from disk (fast path)

### Incremental Update

After scoring new records, their embeddings are added to the in-memory FAISS index via `index.add()` and stored in `gm_embeddings`. The updated index is saved to disk at the end of the sync run.

---

## Feature 2: Hybrid Blocking Orchestrator

### Design

New module `goldenmatch/db/hybrid_blocking.py` — runs SQL and ANN blocking in parallel, unions candidate sets.

### Flow

```
New Record
    ├─ SQL Blocking
    │   └─ Build WHERE clause from exact/phonetic fields
    │   └─ Query DB → candidate IDs {101, 203, 507}
    │
    ├─ ANN Blocking
    │   └─ Embed new record
    │   └─ Query FAISS index → candidate IDs {203, 412, 889}
    │
    └─ Union → {101, 203, 412, 507, 889}
        └─ Pull full records by ID from DB
        └─ Score all candidates
```

### API

```python
def find_candidates(
    new_record: dict,
    connector: DatabaseConnector,
    ann_index: PersistentANNIndex | None,
    blocking_config: BlockingConfig,
    source_table: str,
    columns: list[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> pl.DataFrame:
    """Find candidate matches using SQL + ANN blocking, return union."""
```

### Behaviors

- If ANN index not yet available (too few embeddings), falls back to SQL-only. No error, just lower recall.
- Deduplicates candidate IDs before pulling records from DB.
- Single DB query to fetch all candidates: `SELECT * FROM table WHERE id IN (...)`.
- ANN blocking threshold: index is considered available when it has >= 10% of total table records embedded (configurable via `ann_min_coverage: 0.1`).

---

## Feature 3: Progressive Embedding Computation

### Design

Embeddings for existing records are computed in chunks across sync runs. The system is useful immediately with SQL-only blocking and progressively gains ANN capability.

### Progress Tracking

Stored in `gm_state` table using a JSON field:

```json
{"embeddings_computed": 500000, "total_records": 10000000, "last_embed_id": 500000}
```

### Per-Run Behavior

| Run | Blocking | Embedding Work |
|---|---|---|
| 1st | SQL-only | Embed first 100K existing records |
| 2nd | SQL-only (< 10% coverage) | Embed next 100K, build FAISS index |
| 3rd+ | SQL + ANN (if >= 10% coverage) | Continue embedding chunks |
| Fully indexed | SQL + ANN (full recall) | Only embed new records |

### Chunk Size

Configurable via `--embed-chunk-size` (default 100K). At ~1000 rec/s on CPU with MiniLM, 100K takes ~100 seconds — acceptable as background work during sync.

### New Record Embeddings

When a new record is processed:
1. Embed it for ANN query (required for find_candidates)
2. Store embedding in `gm_embeddings`
3. Add to in-memory FAISS index

No separate embedding step — it happens as part of the matching flow.

---

## Feature 4: Updated Sync Pipeline

### Modified Flow in `sync.py`

```python
def _incremental_pipeline(...):
    # 1. Load or build ANN index
    ann_index = PersistentANNIndex(index_dir, connector, source_table)
    ann_index.load_or_build()

    # 2. Process new records in batches
    for batch in chunk(new_records, size=1000):
        for record in batch:
            # 3. Hybrid blocking: SQL + ANN union
            candidates = find_candidates(
                record, connector, ann_index,
                blocking_config, source_table, columns,
            )

            # 4. Score candidates
            matches = score_and_filter(record, candidates, matchkeys)

            # 5. Log actions
            for match in matches:
                log_match(...)

        # 6. Add new record embeddings to index
        ann_index.add(batch_ids, batch_embeddings)

    # 7. Progress background embedding of existing records
    _embed_next_chunk(connector, ann_index, source_table, embed_chunk_size)

    # 8. Save index + update state
    ann_index.save()
    update_state(...)
```

### Batching

New records processed in batches of 1,000:
- Reduces DB round-trips (one `WHERE id IN (...)` per batch)
- Controls memory usage
- Allows incremental index updates between batches

### No Changes to Scoring

Once candidates are in a Polars DataFrame, existing `score_pair` / `find_fuzzy_matches` handles scoring unchanged.

---

## File Structure

```
goldenmatch/db/
├── __init__.py           # existing
├── connector.py          # existing
├── blocking.py           # existing (SQL blocking)
├── metadata.py           # existing
├── sync.py               # modified (wire hybrid blocking + progressive embedding)
├── writer.py             # existing
├── ann_index.py          # NEW: persistent FAISS index manager
└── hybrid_blocking.py    # NEW: SQL + ANN union orchestrator
```

## Rollout Plan

1. **Phase 1: Persistent ANN Index**
   - `ann_index.py` with load/save/query/add
   - Staleness detection + incremental update
   - Tests with small FAISS indices

2. **Phase 2: Hybrid Blocking**
   - `hybrid_blocking.py` with find_candidates
   - SQL + ANN union logic
   - Fallback to SQL-only when ANN unavailable
   - Tests for union dedup, fallback behavior

3. **Phase 3: Progressive Embedding**
   - Background chunk embedding in sync.py
   - Progress tracking in gm_state
   - Coverage threshold for ANN activation
   - Tests for multi-run progression

4. **Phase 4: Integration + Performance**
   - Wire into _incremental_pipeline
   - Test with 100K+ records
   - Measure query latency and memory usage

## Testing Strategy

### Unit Tests

- `test_ann_index.py` — build/save/load round-trip, staleness detection, incremental add, query returns valid neighbors, id_map correctness
- `test_hybrid_blocking.py` — SQL-only fallback, ANN-only, union dedup, empty results, candidate fetch by ID
- `test_progressive_embedding.py` — chunk tracking, coverage threshold, multi-run progression

### Integration Tests

- Full sync with hybrid blocking against test Postgres
- Progressive: run 1 (SQL-only), run 2 (SQL-only, embedding), run 3 (SQL+ANN)
- Index persistence: sync, restart, verify index loads from disk

## Dependencies

Existing:
- `faiss-cpu` (already in `[embeddings]` extra)
- `sentence-transformers` (already in `[embeddings]` extra)
- `numpy` for embedding serialization
- `psycopg2-binary` (already in `[postgres]` extra)

No new dependencies.
