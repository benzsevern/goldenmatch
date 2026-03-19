# Merge-Back & Reconciliation — Design Specification

## Overview

Add reconciliation logic to GoldenMatch's database sync so that matched records are properly merged into golden records, clusters are managed persistently, and golden record history is preserved with append-only versioning. Handles single-cluster matches, multi-cluster merges (with size cap safety valve), and conflict detection.

## Problem Statement

The current `goldenmatch sync` logs matches but doesn't maintain persistent clusters or update golden records when new records join existing clusters. When a new record matches an existing entity, the system needs to: add it to the cluster, re-compute the golden record, version the old golden record, and optionally update the source table. When a new record bridges two previously separate clusters, the system needs to merge them safely.

## Goals

- Persistent cluster membership across sync runs
- Golden record re-computation when clusters change
- Append-only golden record versioning (is_current flag)
- Multi-cluster merge with max_cluster_size safety cap
- Conflict detection when merge would exceed size cap
- Configurable merge mode (recompute vs incremental)
- Full audit trail via gm_match_log and golden record versions

## Non-Goals

- Manual conflict resolution UI (log conflicts, user resolves externally)
- Undo/rollback of merges (use golden record history for auditing)
- Real-time merge triggers (batch per sync run)

---

## Feature 1: Reconciliation Engine

### Design

New module `goldenmatch/db/reconcile.py` — handles merging new records into existing clusters.

### API

```python
@dataclass
class ReconcileResult:
    action: str              # "merged", "new", "conflict"
    cluster_id: int          # assigned cluster
    golden_record: dict      # new golden record field values
    merged_clusters: list[int]  # cluster IDs combined (if multi-merge)
    previous_version_id: int | None  # superseded golden record ID

def reconcile_match(
    new_record: dict,
    matched_cluster_ids: list[int],
    connector: DatabaseConnector,
    source_table: str,
    golden_rules: GoldenRulesConfig,
    max_cluster_size: int = 100,
    merge_mode: str = "recompute",
    run_id: str = "",
) -> ReconcileResult:
```

### Logic

1. **No matches** → action="new", create single-record cluster, golden record = the record itself
2. **One cluster match** → add to existing cluster, recompute (or incrementally merge) golden record
3. **Multiple cluster matches** →
   - Calculate merged size (sum of all cluster sizes + 1)
   - If merged_size ≤ max_cluster_size → merge all clusters + new record, recompute golden record
   - If merged_size > max_cluster_size → assign to best-scoring cluster, log "conflict" for others

### Merge Modes

- **recompute** (default): Pull all cluster members from DB, run `build_golden_record` on the full set. Most accurate but requires fetching all member records.
- **incremental**: Take existing golden record values, merge just the new record's values using golden rules. Faster for large clusters but may drift from full recompute.

---

## Feature 2: Golden Record Versioning

### Updated Schema

`gm_golden_records` gains versioning columns:

| Column | Type | Purpose |
|---|---|---|
| id | serial | Primary key |
| cluster_id | bigint | Which cluster |
| source_table | text | Source table name |
| source_ids | bigint[] | All member record IDs |
| record_data | jsonb | Merged field values |
| merged_at | timestamp | When this version was created |
| is_current | boolean | TRUE for latest version |
| version | int | Version number (1, 2, 3...) |
| run_id | uuid | Which sync run produced this |

### Operations

**On re-compute (single cluster update):**
1. `UPDATE gm_golden_records SET is_current = FALSE WHERE cluster_id = ? AND is_current = TRUE`
2. `INSERT ... (is_current = TRUE, version = prev + 1)`

**On cluster merge (A + B → combined):**
1. Mark both old golden records as `is_current = FALSE`
2. Insert new golden record with combined `source_ids`
3. Log merge event in `gm_match_log`

### Queries

- Current golden records: `SELECT * FROM gm_golden_records WHERE is_current = TRUE`
- Cluster history: `SELECT * FROM gm_golden_records WHERE cluster_id = ? ORDER BY version`
- Audit: join `gm_match_log` with `gm_golden_records` on `run_id`

### Migration

Add columns to existing table:
```sql
ALTER TABLE gm_golden_records ADD COLUMN IF NOT EXISTS is_current BOOLEAN DEFAULT TRUE;
ALTER TABLE gm_golden_records ADD COLUMN IF NOT EXISTS version INT DEFAULT 1;
ALTER TABLE gm_golden_records ADD COLUMN IF NOT EXISTS run_id UUID;
```

---

## Feature 3: Persistent Cluster Management

### Design

New module `goldenmatch/db/clusters.py` — persistent cluster membership.

### New Table

**`gm_clusters`:**

| Column | Type | Purpose |
|---|---|---|
| cluster_id | bigint | Cluster identifier |
| record_id | bigint | Member record ID |
| source_table | text | Source table |
| added_at | timestamp | When record joined cluster |
| run_id | uuid | Which sync run |

Primary key: `(cluster_id, record_id, source_table)`

### API

```python
def get_cluster_members(connector, cluster_id) -> list[int]
def add_to_cluster(connector, cluster_id, record_ids, run_id) -> None
def merge_clusters(connector, cluster_ids, run_id) -> int  # returns new cluster_id
def create_cluster(connector, record_ids, source_table, run_id) -> int
def get_cluster_for_record(connector, record_id, source_table) -> int | None
def get_cluster_size(connector, cluster_id) -> int
def next_cluster_id(connector) -> int
```

### Cluster ID Assignment

Uses `COALESCE(MAX(cluster_id), 0) + 1` from `gm_clusters` table. Simple, works for single-writer model.

### Table Creation

Added to `ensure_metadata_tables()` in `metadata.py`:

```sql
CREATE TABLE IF NOT EXISTS gm_clusters (
    cluster_id BIGINT NOT NULL,
    record_id BIGINT NOT NULL,
    source_table TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT NOW(),
    run_id UUID,
    PRIMARY KEY (cluster_id, record_id, source_table)
);
```

---

## Feature 4: Sync Pipeline Integration

### Updated Incremental Flow

```python
for record in new_records:
    # 1. Hybrid blocking → candidates
    candidates = find_candidates(...)

    # 2. Score → best matches
    matches = score_candidates(record, candidates, matchkeys)

    # 3. Look up clusters for matched records
    matched_cluster_ids = []
    for match_id, score in matches:
        cid = get_cluster_for_record(connector, match_id, source_table)
        if cid is not None:
            matched_cluster_ids.append(cid)

    # 4. Reconcile
    result = reconcile_match(
        record, matched_cluster_ids, connector,
        source_table, golden_rules, max_cluster_size,
        merge_mode, run_id,
    )

    # 5. Log
    log_match(connector, record_id, result.cluster_id, score, result.action, run_id)
```

### New CLI Flag

`--merge-mode recompute|incremental` — controls golden record rebuild strategy. Default: `recompute`.

### No Changes Needed

- `connector.py` — read/write already works
- `hybrid_blocking.py` — candidate finding unchanged
- `ann_index.py` — embedding management unchanged

---

## File Structure

```
goldenmatch/db/
├── __init__.py
├── connector.py          # existing
├── blocking.py           # existing
├── metadata.py           # modified (add gm_clusters DDL, update gm_golden_records)
├── sync.py               # modified (use reconcile_match)
├── writer.py             # modified (versioned golden record writes)
├── ann_index.py          # existing
├── hybrid_blocking.py    # existing
├── reconcile.py          # NEW: reconciliation engine
└── clusters.py           # NEW: persistent cluster management
```

## Rollout Plan

1. **Phase 1: Cluster Management**
   - `clusters.py` with create/add/merge/query
   - `gm_clusters` table DDL in metadata.py
   - Tests for cluster operations

2. **Phase 2: Golden Record Versioning**
   - Update `gm_golden_records` schema
   - Versioned write in `writer.py`
   - Tests for version chain, is_current flag

3. **Phase 3: Reconciliation Engine**
   - `reconcile.py` with reconcile_match
   - Single match, multi-match, conflict handling
   - Recompute vs incremental merge modes
   - Tests for each scenario

4. **Phase 4: Sync Integration**
   - Wire reconcile_match into _incremental_pipeline
   - Add --merge-mode CLI flag
   - Integration tests

## Testing Strategy

### Unit Tests

- `test_clusters.py` — create, add, merge, query, cluster-for-record lookup, next_id
- `test_reconcile.py` — no match (new entity), single match (add to cluster), multi-match under cap (merge), multi-match over cap (conflict), recompute vs incremental mode
- `test_versioning.py` — insert first version, update creates new version, old marked not current, version numbers increment

### Integration Tests

- Full sync with reconciliation against test Postgres
- Multi-run: create clusters in run 1, new record matches in run 2, verify cluster grows and golden record updates
- Cluster merge: records that bridge two clusters in run 3

## Dependencies

No new dependencies. Uses existing:
- `psycopg2-binary` for Postgres
- `goldenmatch.core.golden.build_golden_record` for golden record computation
- `goldenmatch.config.schemas.GoldenRulesConfig` for merge strategies
