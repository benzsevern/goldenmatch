# Database Integration

GoldenMatch can run entity resolution against live Postgres databases, with incremental matching so only new records are processed on each run.

## Setup

```bash
pip install goldenmatch[postgres]
```

## Basic Usage

```bash
# First run: full scan, creates metadata tables
goldenmatch sync \
  --table customers \
  --connection-string "postgresql://user:pass@localhost:5432/mydb" \
  --config config.yaml

# Subsequent runs: incremental (only new records)
goldenmatch sync \
  --table customers \
  --connection-string "$DATABASE_URL"
```

## Connection Configuration

### Via CLI

```bash
goldenmatch sync --source-type postgres --connection-string "$DATABASE_URL" --table customers
```

### Via YAML Config

```yaml
source:
  type: postgres
  connection: postgresql://user:pass@host:5432/mydb
  table: customers
  incremental_column: updated_at
```

### Via Environment Variable

```bash
export GOLDENMATCH_DATABASE_URL=postgresql://user:pass@host/db
goldenmatch sync --table customers
```

## How Incremental Sync Works

1. **First run**: Full table scan → match all records → build clusters → create golden records
2. **Second run**: Read only records added since last run → match against existing clusters → update golden records
3. **Progressive embedding**: Each run embeds 100K existing records in the background. ANN blocking becomes available once 10% of records are embedded.

### Hybrid Blocking

For each new record, GoldenMatch uses two blocking strategies in parallel:

- **SQL blocking**: Translates blocking keys into `WHERE` clauses (soundex, substring, exact)
- **ANN blocking**: Queries persistent FAISS index for semantically similar records

Results are unioned for maximum recall.

## Metadata Tables

GoldenMatch creates and manages these tables (all prefixed with `gm_`):

### `gm_state`

Tracks processing state for incremental sync.

| Column | Purpose |
|--------|---------|
| source_table | Which table was processed |
| last_processed_at | When last run completed |
| last_incremental_value | Watermark for incremental detection |
| config_hash | Detects config changes |

### `gm_clusters`

Persistent cluster membership.

| Column | Purpose |
|--------|---------|
| cluster_id | Cluster identifier |
| record_id | Member record ID |
| source_table | Source table |
| run_id | Which sync run added this |

### `gm_golden_records`

Versioned golden records with append-only history.

| Column | Purpose |
|--------|---------|
| cluster_id | Which cluster |
| source_ids | All member record IDs |
| record_data | Merged field values (JSONB) |
| is_current | TRUE for latest version |
| version | Version number |

Query current golden records:
```sql
SELECT * FROM gm_golden_records WHERE is_current = TRUE;
```

Query cluster history:
```sql
SELECT * FROM gm_golden_records WHERE cluster_id = 42 ORDER BY version;
```

### `gm_embeddings`

Cached embeddings for ANN blocking.

### `gm_match_log`

Audit trail of all match decisions.

| Column | Purpose |
|--------|---------|
| record_id_a, record_id_b | Matched pair |
| score | Match score |
| action | merged, new, conflict, skipped |
| run_id | Which sync run |

## Output Modes

### Separate (Default)

Results written to `gm_golden_records`. Source table is never modified.

```yaml
output:
  mode: separate
```

### In-Place (Opt-In)

Adds `__cluster_id__` and `__is_golden__` columns to the source table.

```yaml
output:
  mode: in_place
```

## Reconciliation

When a new record matches an existing cluster:

1. **Single match**: Record is added to the cluster, golden record re-computed
2. **Multiple cluster match**: If merged size ≤ max_cluster_size, clusters are merged. Otherwise, assigned to best-scoring cluster and conflict is logged.
3. **No match**: New single-record cluster created

## CLI Flags

| Flag | Description |
|------|-------------|
| `--source-type` | Database type (postgres) |
| `--connection-string` | Database URL |
| `--table` | Source table name |
| `--config` | Matching config YAML |
| `--output-mode` | separate or in_place |
| `--full-rescan` | Force reprocess all records |
| `--dry-run` | Match without writing |
| `--incremental-column` | Column for incremental detection |
| `--chunk-size` | Records per chunk (default 10000) |
