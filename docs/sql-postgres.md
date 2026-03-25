---
layout: default
title: PostgreSQL Extension
nav_order: 15
---

# PostgreSQL Extension

Use GoldenMatch directly from SQL. The PostgreSQL extension provides 18 functions for scoring, deduplication, and pipeline management.

---

## Installation

### Pre-built packages

Download from [goldenmatch-extensions releases](https://github.com/benzsevern/goldenmatch-extensions/releases):

```bash
# Debian/Ubuntu
sudo dpkg -i goldenmatch-pg-0.1.0-pg16-amd64.deb
sudo systemctl restart postgresql

# RHEL/Fedora
sudo rpm -i goldenmatch-pg-0.1.0-pg16.x86_64.rpm
sudo systemctl restart postgresql
```

### Docker

```bash
docker pull ghcr.io/benzsevern/goldenmatch-extensions:latest

docker run -d \
  -e POSTGRES_PASSWORD=secret \
  -p 5432:5432 \
  ghcr.io/benzsevern/goldenmatch-extensions:latest
```

### From source

Requires Rust and pgrx. Build on Linux (pgrx needs libclang/LLVM):

```bash
cd goldenmatch-extensions/postgres
cargo pgrx install --release
```

### Enable the extension

```sql
CREATE EXTENSION goldenmatch_pg;
```

All functions live in the `goldenmatch` schema.

---

## Functions

### String scoring

```sql
-- Score two strings
SELECT goldenmatch.goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler');
-- 0.884

-- Available scorers: jaro_winkler, levenshtein, exact, token_sort, soundex_match
SELECT goldenmatch.goldenmatch_score('hello', 'helo', 'levenshtein');
-- 0.8
```

### Table deduplication

```sql
-- Dedupe a table with JSON config
SELECT goldenmatch.goldenmatch_dedupe_table(
    'customers',
    '{"exact": ["email"], "fuzzy": {"name": 0.85, "zip": 0.95}}'
);

-- With full YAML-style config
SELECT goldenmatch.goldenmatch_dedupe_table(
    'customers',
    '{
        "matchkeys": [
            {"name": "email", "type": "exact", "fields": [{"field": "email"}]},
            {"name": "fuzzy", "type": "weighted", "threshold": 0.85,
             "fields": [
                {"field": "name", "scorer": "jaro_winkler", "weight": 0.7},
                {"field": "zip", "scorer": "exact", "weight": 0.3}
             ]}
        ]
    }'
);
```

### Table matching

```sql
-- Match target table against reference
SELECT goldenmatch.goldenmatch_match_tables(
    'new_customers',
    'master_customers',
    '{"fuzzy": {"name": 0.85}}'
);
```

### Pair scoring and explanation

```sql
-- Score a pair of records
SELECT goldenmatch.goldenmatch_score_pair(
    '{"name": "John Smith", "zip": "10001"}'::jsonb,
    '{"name": "Jon Smyth", "zip": "10001"}'::jsonb,
    '{"fuzzy": {"name": 0.7, "zip": 0.3}}'
);

-- Explain a match decision
SELECT goldenmatch.goldenmatch_explain_pair(
    '{"name": "John Smith"}'::jsonb,
    '{"name": "Jon Smyth"}'::jsonb,
    '{"fuzzy": {"name": 1.0}}'
);
```

### Pipeline management

```sql
-- List pipeline runs
SELECT * FROM goldenmatch.goldenmatch_list_jobs();

-- Get job status
SELECT goldenmatch.goldenmatch_job_status('job_id');

-- Get clusters from a run
SELECT * FROM goldenmatch.goldenmatch_get_clusters('job_id');

-- Get golden records from a run
SELECT * FROM goldenmatch.goldenmatch_get_golden('job_id');

-- Get scored pairs
SELECT * FROM goldenmatch.goldenmatch_get_pairs('job_id');
```

---

## Pipeline schema

The extension creates metadata tables for pipeline tracking:

| Table | Purpose |
|-------|---------|
| `goldenmatch._jobs` | Pipeline run tracking (job_id, status, config, timestamps) |
| `goldenmatch._pairs` | Scored pairs (job_id, id_a, id_b, score, matchkey) |
| `goldenmatch._clusters` | Cluster membership (job_id, cluster_id, record_id) |
| `goldenmatch._golden` | Golden records (job_id, cluster_id, record_data) |

---

## Database sync tables

When using `goldenmatch sync` from the CLI, additional metadata tables are created:

| Table | Purpose |
|-------|---------|
| `gm_state` | Processing state, watermarks |
| `gm_clusters` | Persistent cluster membership |
| `gm_golden_records` | Versioned golden records (append-only, `is_current` flag) |
| `gm_embeddings` | Cached embeddings for ANN blocking |
| `gm_match_log` | Audit trail of all match decisions |

---

## Examples

### Deduplicate a customer table

```sql
-- Create and populate
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name TEXT,
    email TEXT,
    phone TEXT,
    zip TEXT
);

INSERT INTO customers (name, email, phone, zip) VALUES
    ('John Smith', 'john@example.com', '555-1234', '10001'),
    ('Jon Smyth', 'john@example.com', '5551234', '10001'),
    ('Jane Doe', 'jane@example.com', '555-5678', '90210');

-- Dedupe
SELECT goldenmatch.goldenmatch_dedupe_table(
    'customers',
    '{"exact": ["email"]}'
);

-- View clusters
SELECT * FROM goldenmatch._clusters
WHERE job_id = (SELECT job_id FROM goldenmatch._jobs ORDER BY created_at DESC LIMIT 1);
```

### Score columns in a query

```sql
SELECT
    a.name AS name_a,
    b.name AS name_b,
    goldenmatch.goldenmatch_score(a.name, b.name, 'jaro_winkler') AS similarity
FROM customers a
CROSS JOIN customers b
WHERE a.id < b.id
  AND goldenmatch.goldenmatch_score(a.name, b.name, 'jaro_winkler') > 0.85;
```

---

## Notes

- Extension functions live in the `goldenmatch` schema -- use `goldenmatch.function_name()` or set `search_path`
- Explicit `::TEXT` casts may be needed for some argument types in psql
- pgrx does not auto-generate SQL files -- the extension uses handwritten SQL at `sql/goldenmatch_pg--0.1.0.sql`
- See [goldenmatch-extensions](https://github.com/benzsevern/goldenmatch-extensions) for full documentation and CI details
