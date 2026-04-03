---
layout: default
title: Home
nav_order: 1
---

# GoldenMatch

**Entity resolution that finds duplicates in your data so you don't have to define the rules yourself.**

[![PyPI](https://img.shields.io/pypi/v/goldenmatch?color=d4a017)](https://pypi.org/project/goldenmatch/)
[![Downloads](https://static.pepy.tech/badge/goldenmatch/month)](https://pepy.tech/projects/goldenmatch)
[![Tests](https://github.com/benzsevern/goldenmatch/actions/workflows/ci.yml/badge.svg)](https://github.com/benzsevern/goldenmatch/actions)
[![Python](https://img.shields.io/pypi/pyversions/goldenmatch)](https://pypi.org/project/goldenmatch/)
[![Coverage](https://img.shields.io/badge/coverage-72%25-yellow)](https://github.com/benzsevern/goldenmatch)
[![Tests](https://img.shields.io/badge/tests-1070%20passing-brightgreen)](https://github.com/benzsevern/goldenmatch)

---

## What It Does

GoldenMatch takes messy records and figures out which ones refer to the same entity — without requiring you to hand-write matching rules.

```
INGEST → STANDARDIZE → BLOCK → SCORE → CLUSTER → GOLDEN RECORD
```

| Step | What Happens |
|------|-------------|
| **Ingest** | Load CSV, Excel, Parquet, or a DataFrame |
| **Standardize** | Normalize casing, whitespace, phonetic encoding |
| **Block** | Group candidates to avoid N^2 comparisons |
| **Score** | Fuzzy match (jaro-winkler, levenshtein, token sort) |
| **Cluster** | Union-Find with confidence scoring |
| **Golden** | Merge clusters into canonical records |

---

## Quick Install

```bash
pip install goldenmatch
```

```python
import goldenmatch as gm

result = gm.dedupe("customers.csv", exact=["email"], fuzzy={"name": 0.85})
print(f"{result.total_clusters} clusters, {result.match_rate:.0%} match rate")
result.golden.write_csv("golden_records.csv")
```

---

## Benchmarks

| Dataset | Records | Method | F1 | Time |
|---------|---------|--------|-----|------|
| DBLP-ACM (academic) | 4,910 | Fuzzy matching | **97.2%** | 2.1s |
| Abt-Buy (electronics) | 2,162 | Domain + LLM | **72.2%** | 4.2s |
| FEBRL4 (PPRL) | 10,000 | Auto-config bloom filters | **92.4%** | 14s |
| Synthetic | 100K | Fuzzy (name+zip) | -- | 12.8s |
| Synthetic | 1M | Exact dedupe | -- | 7.8s |

Scale: 7,823 records/sec on a laptop (fuzzy + exact + golden).

---

## 7 Ways to Use It

| Interface | Install | Best For |
|-----------|---------|----------|
| [Python API](python-api) | `pip install goldenmatch` | Notebooks, scripts, AI agents |
| [CLI](cli) | Same package, 21 commands | Terminal workflows |
| [Interactive TUI](tui) | `goldenmatch tui` | Visual exploration |
| [PostgreSQL](sql-postgres) | [Pre-built .deb/.rpm](https://github.com/benzsevern/goldenmatch-extensions/releases) | Production databases |
| [DuckDB](sql-duckdb) | `pip install goldenmatch-duckdb` | Analytics |
| [REST API / MCP](rest-api) | `goldenmatch serve` / `mcp-serve` | Microservices, AI assistants |
| [ER Agent (A2A)](agent) | `goldenmatch agent-serve` | AI-to-AI discovery, autonomous ER |

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Installation](installation) | pip, apt, rpm, Docker, build from source |
| [Quick Start](quick-start) | First dedupe in 30 seconds |
| [Python API](python-api) | 101 exports: dedupe, match, score, explain, PPRL |
| [CLI Reference](cli) | 23 commands with examples |
| [Interactive TUI](tui) | 6-tab visual interface |
| [Configuration](configuration) | YAML config with matchkeys, blocking, golden rules |
| [Pipeline](pipeline) | 10-step pipeline architecture |
| [Blocking Strategies](blocking) | Static, learned, ANN blocking |
| [Scoring](scoring) | Fuzzy, exact, probabilistic, LLM scoring |
| [Domain Packs](domain-packs) | 7 built-in YAML rulebooks |
| [PPRL](pprl) | Privacy-preserving record linkage |
| [LLM Integration](llm) | LLM scorer, LLM clustering, budget tracking |
| [Streaming & Incremental](streaming) | Real-time matching, append-only mode |
| [PostgreSQL Extension](sql-postgres) | 18 SQL functions, pipeline schema |
| [DuckDB Extension](sql-duckdb) | 12 Python UDFs |
| [REST API](rest-api) | HTTP endpoints, review queue |
| [MCP Server](mcp) | Claude Desktop integration |
| [Evaluation](evaluation) | Benchmarks, CI/CD quality gates, cluster comparison |
| [ER Agent](agent) | A2A + MCP autonomous agent, confidence gating |
| [Architecture](architecture) | Module map, code patterns |
| [Benchmarks](benchmarks) | Performance and accuracy numbers |

---

## Part of the Golden Suite

| Package | What It Does |
|---------|-------------|
| **[GoldenMatch](https://github.com/benzsevern/goldenmatch)** | Entity resolution (this project) |
| **[GoldenCheck](https://github.com/benzsevern/goldencheck)** | Data validation that discovers rules |
| **[goldenmatch-extensions](https://github.com/benzsevern/goldenmatch-extensions)** | SQL extensions for Postgres + DuckDB |
| **[goldenmatch-duckdb](https://pypi.org/project/goldenmatch-duckdb/)** | DuckDB UDFs for entity resolution |
