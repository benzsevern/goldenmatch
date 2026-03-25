---
layout: default
title: Streaming & Incremental
nav_order: 14
---

# Streaming & Incremental

Match new records against existing data in real time. GoldenMatch supports single-record matching, micro-batch streaming, and CLI-based incremental matching.

---

## match_one

The core primitive for streaming. Matches a single record against an existing DataFrame.

```python
import goldenmatch as gm

matches = gm.match_one(
    {"name": "John Smith", "zip": "10001"},
    existing_df,
    matchkey,
)
# Returns list of (row_id, score) tuples
```

`match_one` works with fuzzy (weighted) matchkeys. For exact matchkeys (threshold=None), use `find_exact_matches` with a Polars join instead.

---

## StreamProcessor

Incremental record matching with immediate or micro-batch processing. Wraps `match_one` and `add_to_cluster` for continuous operation.

```python
import goldenmatch as gm

processor = gm.StreamProcessor(existing_df, config)

# Process a single record immediately
matches = processor.process_record({"name": "John Smith", "zip": "10001"})

# Micro-batch mode: buffer records and process in batches
for record in incoming_records:
    processor.add_record(record)

# Flush the buffer
results = processor.flush()
```

### Immediate mode

Each record is matched and clustered as it arrives:

```python
processor = gm.StreamProcessor(df, config)
result = processor.process_record(new_record)
# result includes matches and updated cluster assignments
```

### Micro-batch mode

Buffer records and process them together for better throughput:

```python
processor = gm.StreamProcessor(df, config)
for record in batch:
    processor.add_record(record)
batch_results = processor.flush()
```

---

## Incremental cluster updates

When a new record matches existing records, update the cluster structure:

```python
import goldenmatch as gm

# Add a record to a cluster (join or merge clusters)
gm.add_to_cluster(record_id, matches, clusters)

# Remove a record from its cluster and re-cluster
gm.unmerge_record(record_id, clusters)

# Shatter a cluster into singletons
gm.unmerge_cluster(cluster_id, clusters)
```

`add_to_cluster` handles three cases:
1. New record matches records in one cluster -- joins that cluster
2. New record matches records in multiple clusters -- merges those clusters
3. New record has no matches -- creates a singleton cluster

---

## Incremental CLI

Match new CSV records against an existing base dataset:

```bash
goldenmatch incremental base.csv --new new_records.csv --config config.yaml
```

The incremental CLI handles exact and fuzzy matchkeys separately:
- **Exact matchkeys**: Polars join between new and base records (fast)
- **Fuzzy matchkeys**: `match_one` brute-force against the base (thorough)

---

## ANN incremental operations

For embedding-based matching, the ANN index supports incremental updates:

```python
from goldenmatch.core.blocker import ANNBlocker

blocker = ANNBlocker(model="all-MiniLM-L6-v2")
blocker.add_to_index(embedding)        # Add a single embedding
neighbors = blocker.query_one(embedding)  # Find nearest neighbors
```

The ANN index is persistent when using database sync mode. Embeddings are computed progressively across runs (100K per run).

---

## Database watch mode

Continuously monitor a database table for new records and match them incrementally:

```bash
goldenmatch watch --table customers --connection-string "$DATABASE_URL" --interval 30
```

### Daemon mode

Run watch as a background service with health endpoint and PID file:

```bash
goldenmatch watch --table customers --connection-string "$DATABASE_URL" --daemon
```

Daemon mode adds:
- HTTP health endpoint at `/health`
- PID file for process management
- SIGTERM handling for graceful shutdown

---

## Database sync

Full incremental matching against a live Postgres database:

```bash
# First run: full scan
goldenmatch sync --table customers --connection-string "$DATABASE_URL" --config config.yaml

# Subsequent runs: incremental (only new records)
goldenmatch sync --table customers --connection-string "$DATABASE_URL"
```

Features:
- **Incremental sync** -- only processes records added since last run
- **Hybrid blocking** -- SQL WHERE clauses for exact fields + FAISS ANN for semantic fields
- **Persistent ANN index** -- disk cache + DB source of truth
- **Golden record versioning** -- append-only with `is_current` flag

---

## run_stream

Run a streaming pipeline programmatically:

```python
import goldenmatch as gm

result = gm.run_stream(existing_df, config, new_records)
```

---

## Architecture

```
New Record
    |
    v
match_one() --> (row_id, score) pairs
    |
    v
add_to_cluster() --> updated cluster dict
    |
    v
build_golden_record() --> updated canonical record
```

For database mode:

```
Postgres table (new rows)
    |
    v
watch/sync --> load new records
    |
    v
Hybrid blocking (SQL + ANN)
    |
    v
Score + cluster + golden
    |
    v
Write back to gm_clusters, gm_golden_records
```
