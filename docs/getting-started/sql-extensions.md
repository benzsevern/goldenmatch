# SQL Extensions

Run entity resolution directly from PostgreSQL or DuckDB.

## PostgreSQL

### Setup

```sql
CREATE EXTENSION goldenmatch_pg;
```

### Deduplicate a Table

```sql
-- Returns JSON with golden records
SELECT goldenmatch.goldenmatch_dedupe_table('customers', '{"exact": ["email"]}');

-- Returns matched pairs as rows
SELECT * FROM goldenmatch.goldenmatch_dedupe_pairs('customers', '{"exact": ["email"]}');
-- id_a | id_b | score
--    0 |    1 | 1.0

-- Returns cluster assignments as rows
SELECT * FROM goldenmatch.goldenmatch_dedupe_clusters('customers', '{"exact": ["email"]}');
-- cluster_id | record_id | cluster_size
--          0 |         0 |            2
--          0 |         1 |            2
```

### Match Two Tables

```sql
SELECT goldenmatch.goldenmatch_match_tables(
    'prospects', 'customers',
    '{"fuzzy": {"name": 0.85}}'
);
```

### Score Strings

```sql
SELECT goldenmatch.goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler');
-- 0.91
```

### Pipeline (Job Management)

```sql
-- Configure a job
SELECT goldenmatch.gm_configure('my_job', '{"exact": ["email"]}');

-- Run against a table
SELECT goldenmatch.gm_run('my_job', 'customers');

-- Query results
SELECT * FROM goldenmatch.gm_pairs('my_job');
SELECT * FROM goldenmatch.gm_clusters('my_job');
SELECT goldenmatch.gm_golden('my_job');

-- List and manage jobs
SELECT goldenmatch.gm_jobs();
SELECT goldenmatch.gm_drop('my_job');
```

### All 18 Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `goldenmatch_dedupe_table(table, config)` | TEXT (JSON) | Deduplicate a table |
| `goldenmatch_match_tables(target, ref, config)` | TEXT (JSON) | Match two tables |
| `goldenmatch_dedupe_pairs(table, config)` | TABLE(id_a, id_b, score) | Pairs as rows |
| `goldenmatch_dedupe_clusters(table, config)` | TABLE(cluster_id, record_id, size) | Clusters as rows |
| `goldenmatch_score(a, b, scorer)` | FLOAT | Score two strings |
| `goldenmatch_score_pair(rec_a, rec_b, config)` | FLOAT | Score two records |
| `goldenmatch_explain(rec_a, rec_b, config)` | TEXT | Explain a match |
| `goldenmatch_dedupe(json, config)` | TEXT (JSON) | Deduplicate JSON |
| `goldenmatch_match(target, ref, config)` | TEXT (JSON) | Match JSON |
| `gm_configure(name, config)` | TEXT | Configure a job |
| `gm_run(name, table)` | TEXT (JSON) | Run a job |
| `gm_jobs()` | TEXT (JSON) | List jobs |
| `gm_golden(name)` | TEXT (JSON) | Get golden records |
| `gm_pairs(name)` | TABLE(id_a, id_b, score) | Query stored pairs |
| `gm_clusters(name)` | TABLE(cluster_id, record_id) | Query stored clusters |
| `gm_drop(name)` | TEXT | Drop a job |

## DuckDB

### Setup

```python
import duckdb
import goldenmatch_duckdb

con = duckdb.connect()
goldenmatch_duckdb.register(con)
```

### Usage

```python
# Score strings
con.sql("SELECT goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler')").show()

# Deduplicate a table
con.sql("""
    CREATE TABLE customers AS SELECT * FROM read_csv('customers.csv');
    SELECT goldenmatch_dedupe_table('customers', '{"exact": ["email"]}');
""").show()

# Pipeline
con.sql("SELECT gm_configure('job1', '{\"exact\": [\"email\"]}')").fetchone()
con.sql("SELECT gm_run('job1', 'customers')").fetchone()
con.sql("SELECT gm_golden('job1')").fetchone()
```

### 12 DuckDB Functions

Same interface as PostgreSQL -- `goldenmatch_score`, `goldenmatch_dedupe_table`, `gm_configure`, `gm_run`, etc.
