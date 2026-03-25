# SQL Functions Reference

## PostgreSQL

18 functions in the `goldenmatch` schema. Install: [SQL Extensions Guide](../getting-started/sql-extensions.md)

### Table Operations

```sql
-- Deduplicate a table (returns JSON)
SELECT goldenmatch.goldenmatch_dedupe_table('customers'::TEXT, '{"exact": ["email"]}'::TEXT);

-- Match two tables (returns JSON)
SELECT goldenmatch.goldenmatch_match_tables('prospects'::TEXT, 'customers'::TEXT, '{"fuzzy": {"name": 0.85}}'::TEXT);

-- Deduplicate and return pairs as rows
SELECT * FROM goldenmatch.goldenmatch_dedupe_pairs('customers'::TEXT, '{"exact": ["email"]}'::TEXT);

-- Deduplicate and return clusters as rows
SELECT * FROM goldenmatch.goldenmatch_dedupe_clusters('customers'::TEXT, '{"exact": ["email"]}'::TEXT);
```

### Scalar Functions

```sql
-- Score two strings
SELECT goldenmatch.goldenmatch_score('John Smith'::TEXT, 'Jon Smyth'::TEXT, 'jaro_winkler'::TEXT);
-- 0.91

-- Score two JSON records
SELECT goldenmatch.goldenmatch_score_pair(
    '{"name": "John Smith"}'::TEXT,
    '{"name": "Jon Smyth"}'::TEXT,
    '{"fuzzy": {"name": 0.85}}'::TEXT
);

-- Explain a match
SELECT goldenmatch.goldenmatch_explain(
    '{"name": "John Smith", "email": "j@x.com"}'::TEXT,
    '{"name": "Jon Smyth", "email": "j@x.com"}'::TEXT,
    '{"fuzzy": {"name": 0.85}, "exact": ["email"]}'::TEXT
);
```

### Pipeline (Job Management)

```sql
-- Configure
SELECT goldenmatch.gm_configure('my_job'::TEXT, '{"exact": ["email"]}'::TEXT);

-- Run
SELECT goldenmatch.gm_run('my_job'::TEXT, 'customers'::TEXT);

-- Query results
SELECT * FROM goldenmatch.gm_pairs('my_job'::TEXT);
SELECT * FROM goldenmatch.gm_clusters('my_job'::TEXT);
SELECT goldenmatch.gm_golden('my_job'::TEXT);

-- Manage
SELECT goldenmatch.gm_jobs();
SELECT goldenmatch.gm_drop('my_job'::TEXT);
```

### Config JSON Format

```json
{
    "exact": ["email", "phone"],
    "fuzzy": {"name": 0.85, "address": 0.90},
    "blocking": ["zip"],
    "threshold": 0.85
}
```

### Supported Scorers

`jaro_winkler` (default), `levenshtein`, `exact`, `token_sort`, `soundex_match`

## DuckDB

12 functions with the same interface. Install: `pip install goldenmatch-duckdb`

```python
import duckdb, goldenmatch_duckdb
con = duckdb.connect()
goldenmatch_duckdb.register(con)
```

Same function names, no schema prefix needed:

```sql
SELECT goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler');
SELECT goldenmatch_dedupe_table('customers', '{"exact": ["email"]}');
SELECT gm_configure('my_job', '{"exact": ["email"]}');
SELECT gm_run('my_job', 'customers');
```
