# Database Connector — Design Specification

## Overview

Add Postgres database integration to GoldenMatch, enabling incremental entity resolution against large existing tables. New records are matched against the database using database-side blocking (SQL WHERE) for exact fields and persistent ANN indexing for semantic fields. Results are written back to the database as golden records with full audit logging. Turns GoldenMatch from a batch CLI tool into a persistent entity resolution service.

This is Phase 1 of database integration. Snowflake support will follow in Phase 2 once the interface is proven with Postgres.

## Problem Statement

GoldenMatch currently operates on files — load CSV, match, output results. Real-world entity resolution needs to work against live databases where records are continuously added. A user with 10M customer records in Postgres needs to check each new record against the existing database, merge duplicates, and maintain a clean golden record set — without reprocessing the entire table each time.

## Goals

- Read from and write to Postgres tables
- Incremental matching: only process new/changed records since last run
- Database-side blocking for exact fields (push filtering to SQL)
- Persistent ANN index for semantic field matching
- GoldenMatch metadata tables for state tracking, embeddings, match history
- Scale to 10M+ existing records
- CLI command: `goldenmatch sync` for database operations
- Safe by default (separate result tables), in-place updates opt-in
- Configurable via YAML with CLI overrides

## Non-Goals

- Snowflake support (Phase 2)
- Real-time / streaming / webhook triggers (future)
- Multi-tenant / concurrent write safety (single-writer model)
- Database schema migrations (idempotent CREATE IF NOT EXISTS only)
- Replacing file-based workflows — database is an additional source type

---

## Feature 1: Database Connector Interface

### Design

New module `goldenmatch/db/connector.py` with abstract interface and Postgres implementation.

```python
class DatabaseConnector:
    def connect(config: dict) -> Connection
    def read_table(table: str, chunk_size: int = 10000) -> Iterator[pl.DataFrame]
    def read_blocked(table: str, blocking_query: str) -> pl.DataFrame
    def write_results(df: pl.DataFrame, table: str, mode: str = "append")
    def execute(sql: str) -> None
    def table_exists(table: str) -> bool
    def get_row_count(table: str) -> int

class PostgresConnector(DatabaseConnector):
    """Postgres implementation using psycopg2 + Polars read_database."""
```

### Configuration

YAML config:

```yaml
source:
  type: postgres
  connection: postgresql://user:pass@host:5432/mydb
  table: customers
  incremental_column: updated_at  # optional, for incremental runs

output:
  mode: separate  # separate (default) | in_place
  tables:
    golden: gm_golden_records
    clusters: gm_clusters
    matches: gm_match_log
```

CLI overrides:

```bash
goldenmatch sync --source-type postgres \
  --connection-string "$DATABASE_URL" \
  --table customers \
  --config config.yaml
```

Connection strings can also come from environment variables: `GOLDENMATCH_DATABASE_URL`.

### Chunked Reading

For large tables, `read_table` uses server-side cursors with configurable chunk size:

```python
def read_table(self, table: str, chunk_size: int = 10000) -> Iterator[pl.DataFrame]:
    with self.conn.cursor(name="gm_cursor") as cursor:
        cursor.execute(f"SELECT * FROM {table}")
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
            yield pl.DataFrame(rows, schema=...)
```

### Dependencies

New optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
postgres = ["psycopg2-binary>=2.9"]
```

Install via: `pip install goldenmatch[postgres]`

---

## Feature 2: GoldenMatch Metadata Tables

### Tables

All prefixed with `gm_` to avoid collisions. Created on first run with `CREATE TABLE IF NOT EXISTS`.

**`gm_state`** — tracks processing state:

| Column | Type | Purpose |
|---|---|---|
| id | serial | Primary key |
| source_table | text | Which table was processed |
| last_processed_at | timestamp | When last run completed |
| last_row_id | bigint | Highest row ID processed |
| last_incremental_value | text | Value of incremental_column at last run |
| config_hash | text | Hash of matching config (detect config changes) |
| record_count | bigint | Total records processed |

**`gm_embeddings`** — persistent embedding cache:

| Column | Type | Purpose |
|---|---|---|
| record_id | bigint | Foreign key to source table |
| source_table | text | Which table |
| embedding | bytea | Serialized numpy array |
| model_name | text | Which model produced this |
| created_at | timestamp | When computed |

**`gm_match_log`** — audit trail:

| Column | Type | Purpose |
|---|---|---|
| id | serial | Primary key |
| record_id_a | bigint | First record |
| record_id_b | bigint | Second record |
| score | float | Match score |
| action | text | 'merged', 'replaced', 'new', 'skipped' |
| run_id | uuid | Groups actions from same run |
| created_at | timestamp | When matched |

### Initialization

```python
def ensure_metadata_tables(connector: DatabaseConnector) -> None:
    """Create gm_* tables if they don't exist."""
```

Called automatically on first `goldenmatch sync`. No migration framework — schema is simple enough to be idempotent.

---

## Feature 3: Incremental Matching Flow

### Detection of New Records

Query `gm_state` for `last_incremental_value`:

```sql
SELECT * FROM customers
WHERE updated_at > '2026-03-17T10:00:00'
ORDER BY updated_at
```

If no `incremental_column` configured, fall back to row count comparison. If config hash changed since last run, force full rescan.

### Database-Side Blocking

For exact/phonetic fields, translate blocking keys into SQL WHERE clauses:

```sql
SELECT * FROM customers
WHERE soundex(last_name) = soundex('Smith')
  AND zip = '10001'
  AND id != 42
LIMIT 1000
```

This pulls only candidate matches into memory. For a 10M row table, typically returns 50-200 candidates per new record.

Implementation in new module `goldenmatch/db/blocking.py`:

```python
def build_blocking_query(
    table: str,
    new_record: dict,
    blocking_config: BlockingConfig,
    exclude_id: int,
) -> str:
    """Translate blocking keys into SQL WHERE clause."""
```

Supports: exact match, soundex (Postgres has `soundex()` function), substring, prefix match.

### ANN Blocking for Semantic Fields

For embedding-scored columns, use persistent ANN index:

1. On first run: compute embeddings for all records, build FAISS index, store embeddings in `gm_embeddings`
2. On incremental run: embed new record, query FAISS index for top-K, pull matched rows by ID from DB
3. After matching: add new embedding to FAISS index and `gm_embeddings`

FAISS index stored on disk (`.goldenmatch_faiss_index/`) and rebuilt from `gm_embeddings` if missing.

### Scoring and Clustering

Feed candidates into existing `find_fuzzy_matches` — the pipeline from here is unchanged. New records matched against pulled candidates produce `(id_a, id_b, score)` pairs.

### State Update

After each run:
- Update `gm_state` with new watermark
- Insert new embeddings into `gm_embeddings`
- Log all match decisions to `gm_match_log`
- Update FAISS index

---

## Feature 4: Result Write-Back

### Separate Mode (Default)

Results written to GoldenMatch-managed tables:

**`gm_golden_records`** — mirrors source schema plus:

| Extra Column | Type | Purpose |
|---|---|---|
| __cluster_id__ | bigint | Cluster this golden record represents |
| __is_golden__ | boolean | Always true in this table |
| __source_ids__ | bigint[] | Array of source record IDs merged |
| __merged_at__ | timestamp | When golden record was computed |

### In-Place Mode (Opt-In)

```yaml
output:
  mode: in_place
  add_columns:
    - __cluster_id__
    - __is_golden__
```

Adds columns to source table via `ALTER TABLE ADD COLUMN IF NOT EXISTS`. Updates matched rows. Original data never modified.

### Actions Per New Record

| Action | Meaning | Write |
|---|---|---|
| `new` | No match found | Insert as new entity |
| `merged` | Matched existing cluster | Re-compute golden record |
| `skipped` | Below threshold | Log only, no data change |

All actions logged to `gm_match_log` for audit.

---

## Feature 5: CLI Command

New `goldenmatch sync` command:

```bash
# First run: full scan, build index, create metadata tables
goldenmatch sync --source-type postgres \
  --connection-string "$DATABASE_URL" \
  --table customers \
  --config config.yaml

# Subsequent runs: incremental (automatic)
goldenmatch sync --source-type postgres \
  --connection-string "$DATABASE_URL" \
  --table customers

# Force full rescan
goldenmatch sync --full-rescan ...

# Dry run: show what would be matched without writing
goldenmatch sync --dry-run ...
```

Flags:
- `--source-type`: postgres (required)
- `--connection-string`: DB URL (or use env var / YAML config)
- `--table`: source table name
- `--config`: matching config YAML (optional if `.goldenmatch.yaml` exists)
- `--output-mode`: separate | in_place
- `--full-rescan`: ignore state, reprocess all records
- `--dry-run`: match but don't write results
- `--chunk-size`: records per chunk (default 10000)

---

## File Structure

```
goldenmatch/db/
├── __init__.py
├── connector.py          # Abstract interface + PostgresConnector
├── blocking.py           # SQL blocking query builder
├── metadata.py           # gm_* table management
├── sync.py               # Incremental matching orchestrator
└── writer.py             # Result write-back (separate / in-place)

goldenmatch/cli/
└── sync.py               # New CLI command
```

## Rollout Plan

1. **Phase 1: Connector + Read**
   - `DatabaseConnector` interface + `PostgresConnector`
   - Chunked table reading with Polars
   - Config parsing for database sources
   - Tests with test Postgres instance

2. **Phase 2: Metadata Tables**
   - `gm_state`, `gm_embeddings`, `gm_match_log` creation
   - State tracking (last processed, config hash)
   - Tests for idempotent table creation

3. **Phase 3: Database-Side Blocking**
   - SQL query builder from blocking config
   - Soundex, exact, substring, prefix support
   - Tests with known blocking queries

4. **Phase 4: Incremental Sync**
   - `goldenmatch sync` CLI command
   - New record detection
   - Block → Score → Cluster → Write flow
   - Integration tests

5. **Phase 5: Result Write-Back**
   - Separate mode (gm_golden_records)
   - In-place mode (ALTER TABLE + UPDATE)
   - Match log writing

6. **Phase 6: Persistent ANN Index**
   - Store/load embeddings from gm_embeddings
   - FAISS index persistence
   - Incremental index updates

## Testing Strategy

### Unit Tests

- `test_connector.py` — connection, chunked reads, writes, table existence
- `test_blocking.py` — SQL query generation for each blocking type
- `test_metadata.py` — table creation, state read/write, config hash
- `test_writer.py` — separate mode, in-place mode, action logging

### Integration Tests

- Full sync flow against a test Postgres (via Docker or `testing.postgresql`)
- Incremental sync: insert records, run sync, verify only new records processed
- Golden record merge: match → merge → verify golden record updated

### Performance Tests

- 1M row table: verify chunked reading doesn't OOM
- Blocking query efficiency: measure query time for soundex + zip blocking
- Incremental sync: 100 new records against 1M existing in < 30s

## Dependencies

New optional dependency group:

```toml
[project.optional-dependencies]
postgres = ["psycopg2-binary>=2.9"]
```

Existing dependencies used:
- `polars` — `read_database` for reading, DataFrame for processing
- `faiss-cpu` — persistent ANN index (already in `[embeddings]`)
- `numpy` — embedding serialization
