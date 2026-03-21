# GoldenMatch Wiki

**Deduplicate records, match across sources, and maintain golden records.**

```bash
pip install goldenmatch
goldenmatch demo
goldenmatch setup
```

## Getting Started

| Page | What you'll learn |
|------|-------------------|
| [Installation](Installation.md) | pip install with optional extras |
| [Quick Start](Quick-Start.md) | 5 usage scenarios from zero-config to database sync |
| [Interactive TUI](Interactive-TUI.md) | Gold-themed terminal UI with keyboard shortcuts |

## Core Concepts

| Page | What it covers |
|------|----------------|
| [Pipeline Overview](Pipeline-Overview.md) | Ingest, Block, Score, Cluster, Golden Records |
| [Blocking Strategies](Blocking-Strategies.md) | 7 strategies: static, adaptive, multi-pass, ANN, canopy |

## Advanced Features

| Page | What it does |
|------|-------------|
| [GPU Routing & Vertex AI](GPU-Routing.md) | Managed embeddings via Google Cloud |
| [Database Integration](Database-Integration.md) | Incremental Postgres sync with golden record versioning |
| [LLM Boost](LLM-Boost.md) | Claude/GPT-4 labeling + fine-tuning |
| [dbt Integration](dbt-Integration.md) | Post-hooks, Postgres sync, Snowflake/BigQuery recipes |

## Reference

| Page | Details |
|------|---------|
| [Benchmarks](Benchmarks.md) | 97.2% DBLP-ACM, 81.7% Abt-Buy, 8,200 rec/s |
| [Comparison](Comparison.md) | vs dedupe, Splink, Zingg, Ditto |
| [Architecture](Architecture.md) | Project structure and module map |

## CLI Commands

| Command | Description |
|---------|-------------|
| `goldenmatch demo` | Built-in demo with sample data |
| `goldenmatch setup` | Interactive setup wizard |
| `goldenmatch dedupe FILE [...]` | Deduplicate files |
| `goldenmatch match TARGET --against REF` | Match across files |
| `goldenmatch sync --table TABLE` | Sync against Postgres |
| `goldenmatch watch --table TABLE` | Live stream mode |
| `goldenmatch schedule --every 1h FILE` | Scheduled runs |
| `goldenmatch serve FILE [...]` | REST API server |
| `goldenmatch mcp-serve FILE [...]` | MCP server (Claude Desktop) |
| `goldenmatch rollback RUN_ID` | Undo a previous run |
| `goldenmatch unmerge RECORD_ID` | Remove a record from its cluster |
| `goldenmatch runs` | List run history |

## Key Dedupe Flags

| Flag | Description |
|------|-------------|
| `--anomalies` | Detect fake/suspicious records |
| `--preview` | Show what will change before writing |
| `--diff` / `--diff-html` | Before/after change report |
| `--dashboard` | Data quality dashboard (HTML) |
| `--html-report` | Detailed match report with charts |
| `--chunked` | Large dataset mode |
| `--llm-boost` | LLM-powered accuracy boost |
| `s3://` / `gs://` / `az://` | Cloud storage ingest |
