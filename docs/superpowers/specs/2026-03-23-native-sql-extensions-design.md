# Native SQL Extensions for GoldenMatch

## Overview

Ship GoldenMatch as native database extensions for Postgres and DuckDB, so users can run entity resolution directly from SQL with `CREATE EXTENSION goldenmatch` or `INSTALL goldenmatch`.

**Goal:** A data engineer can deduplicate a table without leaving their SQL client.

**Architecture:** Rust workspace with three crates -- a shared Python bridge layer (`pyo3`), a Postgres extension (`pgrx`), and a DuckDB extension. The bridge embeds CPython, calls the existing GoldenMatch Python package via Arrow-based interchange, and returns results as database tuples.

**Tech Stack:** Rust, pgrx, pyo3, Apache Arrow, Postgres 15/16/17, DuckDB

**Prerequisites:** Requires a new `dedupe_df()` / `match_df()` entry point in the GoldenMatch Python package that accepts a Polars DataFrame directly (the current `gm.dedupe()` only accepts file paths). This is a small addition to `_api.py` since the internal pipeline already operates on DataFrames after the ingest step.

---

## Repo Structure

New repository: `benzsevern/goldenmatch-extensions`

```
goldenmatch-extensions/
├── Cargo.toml                    # Rust workspace root
├── README.md
├── LICENSE
│
├── bridge/                       # Shared crate: goldenmatch-bridge
│   ├── Cargo.toml               # pyo3 dependency
│   └── src/
│       ├── lib.rs               # Python interpreter lifecycle
│       ├── convert.rs           # Arrow <-> database type conversion
│       ├── api.rs               # Wrappers: dedupe(), match(), score(), configure()
│       └── error.rs             # Python exception -> Rust error mapping
│
├── postgres/                     # Crate: goldenmatch-pg (pgrx extension)
│   ├── Cargo.toml               # pgrx + goldenmatch-bridge deps
│   ├── goldenmatch.control      # Postgres extension metadata
│   ├── sql/                     # SQL migration files
│   │   └── goldenmatch--0.1.0.sql
│   └── src/
│       ├── lib.rs               # pgrx extension entry, function registration
│       ├── quick.rs             # Quick-start functions (goldenmatch_dedupe, etc.)
│       └── pipeline.rs          # Pipeline schema (goldenmatch.configure, .run, etc.)
│
├── duckdb/                       # Crate: goldenmatch-duckdb (Phase 3)
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
│
└── .github/
    └── workflows/
        ├── ci.yml               # Lint + test on push
        └── release.yml          # Build binaries for platforms x PG versions
```

Three crates in a Rust workspace:
- **`bridge`** -- shared Python embedding + Arrow type conversion (used by both extensions)
- **`postgres`** -- Postgres-specific extension via `pgrx`
- **`duckdb`** -- DuckDB-specific extension (stub initially, built in Phase 3)

---

## SQL Interface

### Quick-Start Functions

Registered in `public` schema (no prefix needed). One function call, get results:

```sql
-- Deduplicate a table
SELECT * FROM goldenmatch_dedupe(
    'customers',
    '{"fuzzy": {"name": 0.85}, "exact": ["email"]}'
);
-- Returns: all columns + __cluster_id__, __is_golden__

-- Match across two tables
SELECT * FROM goldenmatch_match(
    'prospects', 'customers',
    '{"fuzzy": {"name": 0.85}}'
);
-- Returns: id_a, id_b, score, matchkey

-- Score a single pair of strings
SELECT goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler');
-- Returns: 0.91

-- Score two full records
SELECT goldenmatch_score_pair(
    '{"name": "John Smith", "email": "j@x.com"}',
    '{"name": "Jon Smyth", "email": "j@x.com"}',
    '{"fuzzy": {"name": 0.85}, "exact": ["email"]}'
);
-- Returns: 0.95

-- Explain why two records matched
SELECT goldenmatch_explain(42, 87, 'my_job');
-- Returns: text explanation
```

### Pipeline Schema

For production workflows. Functions registered in `goldenmatch` schema:

```sql
-- Configure a job
CALL goldenmatch.configure('customer_dedupe', '{
    "matchkeys": [{"name": "email_exact", "type": "exact", "fields": [{"field": "email"}]}],
    "fuzzy": {"name": 0.85},
    "blocking": [{"key": "zip"}]
}');

-- Run the job
CALL goldenmatch.run('customer_dedupe', 'customers');

-- Inspect results
SELECT * FROM goldenmatch.clusters('customer_dedupe');
SELECT * FROM goldenmatch.golden('customer_dedupe');
SELECT * FROM goldenmatch.pairs('customer_dedupe');

-- Explain decisions
SELECT * FROM goldenmatch.explain_pair('customer_dedupe', 42, 87);
SELECT * FROM goldenmatch.explain_cluster('customer_dedupe', 5);

-- Evaluate against ground truth
SELECT * FROM goldenmatch.evaluate('customer_dedupe', 'ground_truth_pairs');

-- Manage jobs
SELECT * FROM goldenmatch.jobs();
CALL goldenmatch.drop('customer_dedupe');
```

Quick-start functions are thin wrappers: `goldenmatch_dedupe('t', config)` internally creates a temp job, runs it, returns results, and drops the job.

---

## Bridge Layer Architecture

### Python Interpreter Lifecycle

```
First SQL call -> bridge::init()
  -> pyo3::prepare_freethreaded_python()   // Boot CPython once
  -> Python::with_gil(|py| {
      py.import("goldenmatch")             // Verify goldenmatch is installed
    })
  -> Cache interpreter for all subsequent calls
```

Interpreter stays alive for the database backend process lifetime. No per-query startup after the first call.

### Data Flow

```
SQL query
  -> Postgres SPI reads table -> Arrow RecordBatch
  -> bridge::api::dedupe(arrow_batch, config_json)
    -> Python GIL acquire
    -> polars.from_arrow(batch)           // Arrow -> Polars DataFrame
    -> gm.dedupe_df(df, **config)         // Full GoldenMatch pipeline (DataFrame entry point)
    -> result.golden.to_arrow()           // Polars -> Arrow
    -> Python GIL release
  -> Arrow RecordBatch -> Postgres tuples
  -> Return to user as table
```

Key decisions:
- **Arrow-based interchange** -- Polars and both databases speak Arrow natively. Data is copied twice (SQL tuples -> Arrow on input, Arrow -> SQL tuples on output), but the middle section (Arrow -> Polars -> Arrow) is near-zero-copy. This is efficient for batch ER workloads.
- **GIL held only during batch call** -- ER is a batch operation (one call per table, not per row). Note: concurrent `goldenmatch_dedupe()` calls from different Postgres backends will serialize on the GIL. This is acceptable for typical usage but should be documented.
- **Config as JSON** -- Every GoldenMatch config option is automatically available in SQL without the extension needing to enumerate them.
- **DataFrame entry point** -- The bridge calls new `dedupe_df(df, config)` / `match_df(df_a, df_b, config)` functions in `_api.py` that accept Polars DataFrames directly, bypassing the file-based ingest step.

### Error Handling

Python exceptions caught by `pyo3` -> Rust `Result::Err` -> database error reporting (`ereport(ERROR, ...)` for Postgres, `throw InternalException` for DuckDB).

---

## Postgres Extension Details

### Extension Control File

```
comment = 'Entity resolution toolkit -- deduplicate, match, and maintain golden records'
default_version = '0.1.0'
module_pathname = '$libdir/goldenmatch'
relocatable = false
schema = goldenmatch
```

Note: No `requires = 'plpython3u'` -- the extension embeds its own CPython via pyo3, independent of PL/Python. This is critical for compatibility with managed platforms like Supabase where plpython3u may not be available.

### Installation

```sql
CREATE EXTENSION goldenmatch;
-- Auto-creates: goldenmatch schema, _jobs table, _results table, all functions
```

### Python Dependency Validation

On `CREATE EXTENSION`, the bridge verifies Python + goldenmatch are available:

```
ERROR: goldenmatch requires Python package 'goldenmatch>=1.0.0'
HINT: Run 'pip install goldenmatch' in the Python environment used by PostgreSQL
```

### Postgres Version Support

PG 15, 16, 17. `pgrx` builds against each version's headers from the same codebase.

### Internal Tables

Created automatically in `goldenmatch` schema:

**`goldenmatch._jobs`**
| Column | Type | Description |
|--------|------|-------------|
| name | TEXT PRIMARY KEY | Job identifier |
| config_json | JSONB | Full GoldenMatch config |
| created_at | TIMESTAMPTZ | Job creation time |
| last_run_at | TIMESTAMPTZ | Last execution time |
| status | TEXT | pending/running/completed/failed |

**`goldenmatch._pairs`**
| Column | Type | Description |
|--------|------|-------------|
| job_name | TEXT | FK to _jobs |
| id_a | BIGINT | Row ID of first record |
| id_b | BIGINT | Row ID of second record |
| score | DOUBLE PRECISION | Overall match score |
| matchkey | TEXT | Which matchkey matched |
| field_scores | JSONB | Per-field scores (for explain) |

Index: `(job_name, id_a, id_b)` for `explain_pair()` lookups.

**`goldenmatch._clusters`**
| Column | Type | Description |
|--------|------|-------------|
| job_name | TEXT | FK to _jobs |
| cluster_id | BIGINT | Cluster identifier |
| record_id | BIGINT | Member record ID |
| is_golden | BOOLEAN | Whether this record is the golden representative |

Index: `(job_name, cluster_id)` for cluster queries.

**`goldenmatch._golden`**
| Column | Type | Description |
|--------|------|-------------|
| job_name | TEXT | FK to _jobs |
| cluster_id | BIGINT | Cluster identifier |
| record_data | JSONB | Merged golden record fields |

---

## Distribution

### Primary: Trunk

```bash
trunk install goldenmatch
```

Trunk is the pip-equivalent for Postgres. Targets Supabase, Tembo, and self-hosted. Submission is straightforward once the extension builds.

### Install Script

```bash
curl -sSL https://get.goldenmatch.dev/pg | bash
```

Auto-detects `pg_config`, copies `.so` + `.control` + `.sql` to the right directories.

### Docker Image

```bash
docker run -p 5432:5432 goldenmatch/postgres:17
# Extension pre-installed, just connect and CREATE EXTENSION
```

### Pre-built Binaries (GitHub Releases)

Build matrix:

| Platform | Postgres Versions |
|----------|-------------------|
| Linux x86_64 | 15, 16, 17 |
| Linux aarch64 | 15, 16, 17 |
| macOS arm64 | 15, 16, 17 |

9 builds per release. Artifact naming: `goldenmatch-pg17-linux-x86_64.tar.gz`

Windows excluded initially (rare for production Postgres).

### Future: apt/yum

```bash
apt install postgresql-17-goldenmatch
yum install goldenmatch_17
```

Only if demand warrants the packaging effort.

### Platform Compatibility

| Platform | Extension Support | Install Method |
|----------|------------------|----------------|
| Self-hosted Postgres | Full | Trunk, script, binary, Docker |
| Supabase | Full | Trunk (native) |
| Tembo | Full | Trunk (native) |
| Neon | Curated list | Needs their approval |
| AWS RDS | Curated list | Needs AWS to add |
| Azure Flexible Server | Curated list | Needs Azure to add |
| Google Cloud SQL | Curated list | Needs GCP to add |
| DuckDB | Built-in | `INSTALL goldenmatch` via community repo |

---

## Python Environment

### Build-time Binding

`pyo3` links against a specific `libpython` at compile time (controlled by `PYO3_PYTHON` env var or auto-detected). This means:

- Pre-built binaries target a specific Python minor version (e.g., 3.12)
- Users must install `goldenmatch` into that same Python, not an arbitrary venv
- The release matrix adds a Python version dimension: PG 15/16/17 x platform x Python 3.12

### Discovery

The install script validates:
1. `pg_config` exists and returns valid paths
2. Python 3.12 is available at a known location
3. `goldenmatch` package is importable from that Python
4. Prints clear instructions if any check fails

### Docker Sidestep

The Docker image bundles everything: Postgres + Python 3.12 + goldenmatch + extension. Zero environment issues. This is the recommended path for evaluation and development.

### Release Binaries

Each release artifact specifies its Python target in the filename:
```
goldenmatch-pg17-py312-linux-x86_64.tar.gz
```

---

## Limitations & Constraints

### Memory

GoldenMatch loads tables fully into memory. The existing codebase OOMs at ~1M records in-memory. Since the extension reads the entire table into Arrow, the same limits apply. An OOM in a Postgres backend process is worse than in a standalone application (triggers Postgres recovery).

**Mitigations:**
- Document recommended max table size (~500K records for fuzzy, ~2M for exact-only)
- For larger tables, users should use the Python CLI with chunked processing or the DuckDB backend
- Future: integrate chunked processing into the bridge layer

### Concurrency

Multiple concurrent `goldenmatch_dedupe()` calls from different Postgres connections serialize on the Python GIL. The second caller blocks until the first completes. This is acceptable for typical batch ER usage but should be documented.

### Security

- Extension requires superuser to `CREATE EXTENSION` (standard for compiled extensions)
- The embedded Python environment has file system access -- same as any plpython3u function
- Config JSON is passed through to GoldenMatch's existing config parser which validates schema
- The extension does not execute arbitrary Python from user input -- only calls known GoldenMatch API functions
- Note: some GoldenMatch features (LLM scoring, cloud connectors) make outbound network calls. These are only triggered by explicit config options, not by default

---

## Migration Strategy

Each phase introduces new SQL objects. Postgres extensions use versioned migration files:

```
sql/
├── goldenmatch--0.1.0.sql          # Phase 1: quick-start functions
├── goldenmatch--0.1.0--0.2.0.sql   # Phase 2: add pipeline schema + tables
└── goldenmatch--0.2.0--0.3.0.sql   # Phase 2 patches if needed
```

Users upgrade with:
```sql
ALTER EXTENSION goldenmatch UPDATE TO '0.2.0';
```

---

## DuckDB Extension Notes

DuckDB's extension ecosystem primarily uses C++. The Rust extension API (`duckdb-rs` with loadable extension support) is less mature than pgrx. Phase 3 options:

1. **Rust via `duckdb-rs`** -- if loadable extension support is stable by then. Reuses the bridge crate directly.
2. **C++ wrapper** -- thin C++ extension that links to the bridge crate via C FFI (`extern "C"` exports from the bridge). More work but guaranteed compatibility with DuckDB's extension loader.
3. **Python extension** -- DuckDB supports Python UDFs natively. Simplest path but doesn't give the `INSTALL` experience.

Decision deferred to Phase 3. The bridge crate is designed to expose a C-compatible API regardless, so both options 1 and 2 are viable.

---

## Build & CI

### Local Development

```bash
rustup install stable
cargo install cargo-pgrx
cargo pgrx init               # Downloads PG 15/16/17 headers

cd goldenmatch-extensions
cargo build                    # Build all crates
cargo test                     # Unit tests (bridge, no DB needed)
cargo pgrx test pg17           # Integration tests against real PG 17
cargo pgrx run pg17            # Launch test PG with extension loaded
```

### CI Workflow (ci.yml)

On push/PR:
1. `cargo fmt --check` + `cargo clippy`
2. `cargo test` (bridge crate unit tests)
3. `cargo pgrx test pg17` (integration tests)
4. Ubuntu-latest with Python 3.12 + `pip install goldenmatch`

### Release Workflow (release.yml)

On GitHub Release tag:
- Build matrix: 3 platforms x 3 PG versions = 9 artifacts
- Upload to GitHub Releases
- Publish to Trunk registry
- Build + push Docker image

---

## Implementation Phases

### Phase 0: Python API Prerequisite

Add DataFrame entry points to the GoldenMatch Python package (in the existing `goldenmatch` repo):

1. Add `dedupe_df(df, **config)` and `match_df(df_a, df_b, **config)` to `_api.py`
2. These bypass file ingest and feed DataFrames directly into the pipeline
3. Release as goldenmatch v1.1.0

Note: The Python package (v1.x) and the extension (v0.x) follow independent version tracks. The extension declares its minimum Python package dependency (e.g., `goldenmatch>=1.1.0`).

### Phase 1: Postgres Quick-Start (v0.1.0)

`CREATE EXTENSION goldenmatch` works. Quick-start functions return results.

1. Scaffold Rust workspace + bridge crate with pyo3 Python embedding
2. Implement `bridge::api::dedupe()` -- embed Python, call `gm.dedupe_df()`, return Arrow
3. Scaffold postgres crate with pgrx, wire up `goldenmatch_dedupe()` function
4. Add `goldenmatch_match()`, `goldenmatch_score()`, `goldenmatch_score_pair()`
5. Add `goldenmatch_explain()`
6. Integration tests with `cargo pgrx test`
7. Install script + Docker image
8. README with quickstart

### Phase 2: Pipeline Schema (v0.2.0)

Full production workflow with job management.

1. Create `goldenmatch` schema + `_jobs` / `_results` tables on CREATE EXTENSION
2. Implement `CALL goldenmatch.configure(name, config_json)`
3. Implement `CALL goldenmatch.run(name, table)`
4. Implement result queries: `clusters()`, `golden()`, `pairs()`
5. Implement `explain_pair()`, `explain_cluster()`, `evaluate()`
6. Implement `goldenmatch.jobs()` and `goldenmatch.drop()`
7. Make quick-start functions use pipeline internally (temp job pattern)

### Phase 3: DuckDB Extension (v0.3.0)

Same functions in DuckDB via community extensions.

1. Implement duckdb crate using DuckDB Rust extension API
2. Reuse bridge crate -- same Python embedding, same Arrow conversion
3. Adapt SQL function signatures to DuckDB conventions
4. Submit to DuckDB community extensions repo

### Phase 4: Distribution Polish (v0.4.0)

One-line install on all major platforms.

1. Release workflow builds 9 platform/version combos
2. Submit to Trunk registry (gets Supabase + Tembo for free)
3. Publish Docker image `goldenmatch/postgres:17`
4. apt/yum packages if demand warrants
