---
layout: default
title: DuckDB Extension
nav_order: 16
---

# DuckDB Extension

Use GoldenMatch as Python UDFs inside DuckDB. Score strings, deduplicate tables, and run pipelines from SQL.

---

## Installation

```bash
pip install goldenmatch-duckdb
```

Requires `goldenmatch` and `duckdb` as dependencies. Also needs `pyarrow` for Polars conversion (`.pl()`).

---

## Quick start

```python
import duckdb
import goldenmatch_duckdb

con = duckdb.connect()
goldenmatch_duckdb.register(con)

# Score two strings
con.sql("SELECT goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler')").show()
# 0.884
```

---

## Functions

### goldenmatch_score

Score two strings with a named scorer.

```sql
SELECT goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler');
-- 0.884

SELECT goldenmatch_score('hello', 'helo', 'levenshtein');
-- 0.8

SELECT goldenmatch_score('Smith', 'Smyth', 'soundex_match');
-- 1.0
```

Available scorers: `jaro_winkler`, `levenshtein`, `exact`, `token_sort`, `soundex_match`.

### goldenmatch_dedupe

Deduplicate a DuckDB table.

```sql
SELECT * FROM goldenmatch_dedupe('customers', '{"exact": ["email"]}');
```

```python
con.sql("""
    SELECT * FROM goldenmatch_dedupe('customers', '{
        "matchkeys": [{
            "name": "fuzzy",
            "type": "weighted",
            "threshold": 0.85,
            "fields": [
                {"field": "name", "scorer": "jaro_winkler", "weight": 0.7},
                {"field": "zip", "scorer": "exact", "weight": 0.3}
            ]
        }]
    }')
""").show()
```

### goldenmatch_match

Match a target table against a reference table.

```sql
SELECT * FROM goldenmatch_match('new_customers', 'master', '{"fuzzy": {"name": 0.85}}');
```

### goldenmatch_score_pair

Score a pair of JSON records.

```sql
SELECT goldenmatch_score_pair(
    '{"name": "John Smith", "zip": "10001"}',
    '{"name": "Jon Smyth", "zip": "10001"}',
    '{"fuzzy": {"name": 0.7, "zip": 0.3}}'
);
```

### goldenmatch_explain

Explain a match decision in natural language.

```sql
SELECT goldenmatch_explain(
    '{"name": "John Smith"}',
    '{"name": "Jon Smyth"}',
    '{"fuzzy": {"name": 1.0}}'
);
```

### Pipeline management

```sql
-- List completed jobs
SELECT * FROM goldenmatch_list_jobs();

-- Get clusters from a job
SELECT * FROM goldenmatch_get_clusters('job_id');

-- Get golden records
SELECT * FROM goldenmatch_get_golden('job_id');

-- Get scored pairs
SELECT * FROM goldenmatch_get_pairs('job_id');

-- Get job status
SELECT goldenmatch_job_status('job_id');
```

### Utility functions

```sql
-- Version
SELECT goldenmatch_version();

-- Available scorers
SELECT * FROM goldenmatch_list_scorers();
```

---

## Examples

### Score columns in a query

```python
import duckdb
import goldenmatch_duckdb

con = duckdb.connect()
goldenmatch_duckdb.register(con)

con.sql("""
    CREATE TABLE customers AS
    SELECT * FROM read_csv_auto('customers.csv');

    SELECT
        a.name AS name_a,
        b.name AS name_b,
        goldenmatch_score(a.name, b.name, 'jaro_winkler') AS similarity
    FROM customers a, customers b
    WHERE a.rowid < b.rowid
      AND goldenmatch_score(a.name, b.name, 'jaro_winkler') > 0.85;
""").show()
```

### Deduplicate and export

```python
con.sql("""
    CREATE TABLE customers AS SELECT * FROM read_csv_auto('customers.csv');
""")

result = con.sql("SELECT * FROM goldenmatch_dedupe('customers', '{\"exact\": [\"email\"]}')").pl()
result.write_csv("deduped.csv")
```

### Use with dbt

The `dbt-goldenmatch` package provides macros for DuckDB-based entity resolution:

```bash
pip install dbt-goldenmatch
```

{% raw %}
```sql
-- In a dbt model
{{ run_goldenmatch_dedupe('customers', '{"exact": ["email"]}') }}
```
{% endraw %}

---

## DuckDB backend

GoldenMatch also supports DuckDB as an out-of-core processing backend for datasets that don't fit in memory:

```python
import goldenmatch as gm

result = gm.dedupe("huge.csv", exact=["email"], backend="duckdb")
```

The DuckDB backend provides:

```python
from goldenmatch.backends.duckdb_backend import DuckDBBackend

backend = DuckDBBackend("data.duckdb")
backend.write_table("customers", df)
df = backend.read_table("customers")
tables = backend.list_tables()
```

---

## Notes

- DuckDB UDFs cannot query the same connection they are called on (deadlock) -- the extension uses `con.cursor()` for internal table reads
- `.pl()` (Polars conversion) requires `pyarrow` as a dependency
- See [goldenmatch-extensions](https://github.com/benzsevern/goldenmatch-extensions) for source code and CI details
