---
layout: default
title: Installation
nav_order: 2
---

# Installation

Install GoldenMatch from PyPI with pip. Optional extras add embeddings, LLM scoring, database sync, and more.

---

## pip (recommended)

```bash
pip install goldenmatch
```

Requires Python 3.11 or later. Core dependencies: Polars, RapidFuzz, Typer, Pydantic, Textual.

### Optional extras

```bash
pip install goldenmatch[embeddings]     # sentence-transformers + FAISS
pip install goldenmatch[llm]            # Claude/OpenAI for LLM scoring
pip install goldenmatch[postgres]       # PostgreSQL database sync
pip install goldenmatch[snowflake]      # Snowflake connector
pip install goldenmatch[bigquery]       # BigQuery connector
pip install goldenmatch[databricks]     # Databricks connector
pip install goldenmatch[salesforce]     # Salesforce connector
pip install goldenmatch[duckdb]         # DuckDB out-of-core backend
pip install goldenmatch[quality]        # GoldenCheck data quality scanning
pip install goldenmatch[ray]            # Ray distributed backend
```

Install multiple extras at once:

```bash
pip install goldenmatch[embeddings,llm,postgres]
```

---

## Docker

```bash
docker pull ghcr.io/benzsevern/goldenmatch:latest

# Run a dedupe
docker run --rm -v $(pwd):/data ghcr.io/benzsevern/goldenmatch:latest \
    dedupe /data/customers.csv --output-dir /data/results

# Start the REST API
docker run --rm -p 8080:8080 -v $(pwd):/data ghcr.io/benzsevern/goldenmatch:latest \
    serve --file /data/customers.csv --port 8080
```

---

## PostgreSQL Extension

Pre-built packages for the SQL extension (separate from the Python package):

```bash
# Debian/Ubuntu
sudo dpkg -i goldenmatch-pg-0.1.0-pg16-amd64.deb
sudo systemctl restart postgresql

# RHEL/Fedora
sudo rpm -i goldenmatch-pg-0.1.0-pg16.x86_64.rpm
sudo systemctl restart postgresql
```

Download `.deb` and `.rpm` from the [goldenmatch-extensions releases](https://github.com/benzsevern/goldenmatch-extensions/releases) page.

---

## DuckDB UDFs

```bash
pip install goldenmatch-duckdb
```

```python
import duckdb, goldenmatch_duckdb

con = duckdb.connect()
goldenmatch_duckdb.register(con)
con.sql("SELECT goldenmatch_score('John', 'Jon', 'jaro_winkler')")
```

---

## dbt Integration

```bash
pip install dbt-goldenmatch
```

The `dbt-goldenmatch` package provides macros for running entity resolution inside dbt pipelines using DuckDB.

---

## Verify installation

```bash
goldenmatch --version
# goldenmatch 1.1.1

goldenmatch demo
# Runs a built-in demo with sample data
```

```python
import goldenmatch as gm
print(gm.__version__)   # "1.1.1"
```

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM scorer and LLM boost (OpenAI) |
| `ANTHROPIC_API_KEY` | LLM scorer (Claude) |
| `DATABASE_URL` | PostgreSQL connection string for `sync` / `watch` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Vertex AI embeddings (GCP service account) |

---

## Setup wizard

Run the interactive wizard to configure GPU mode, API keys, and database connections:

```bash
goldenmatch setup
```

The wizard guides you through:
- GPU mode selection (CPU, CUDA, MPS, Vertex AI, Colab)
- LLM API key configuration
- PostgreSQL connection setup
- Saved preferences at `~/.goldenmatch/settings.yaml`
