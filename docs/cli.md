---
layout: default
title: CLI Reference
nav_order: 5
---

# CLI Reference

GoldenMatch provides 21 CLI commands via `goldenmatch <command>`. All commands support `--help`.

```bash
pip install goldenmatch
goldenmatch --version
```

---

## dedupe

Deduplicate one or more files.

```bash
# Zero-config (auto-detects columns, scorers, blocking)
goldenmatch dedupe customers.csv

# With config
goldenmatch dedupe customers.csv --config config.yaml --output-all --output-dir results/

# Multiple files
goldenmatch dedupe crm.csv marketing.csv --config config.yaml

# With LLM scorer
goldenmatch dedupe products.csv --config config.yaml --llm-scorer

# With anomaly detection
goldenmatch dedupe customers.csv --anomalies

# Preview changes before writing
goldenmatch dedupe customers.csv --preview

# Generate HTML report
goldenmatch dedupe customers.csv --html-report

# Before/after dashboard
goldenmatch dedupe customers.csv --dashboard

# Diff report
goldenmatch dedupe customers.csv --diff --diff-html

# Chunked processing for large files
goldenmatch dedupe huge.csv --chunked

# Ray distributed backend
goldenmatch dedupe huge.parquet --backend ray

# Cloud storage
goldenmatch dedupe s3://bucket/customers.csv
```

---

## match

Match a target file against reference files.

```bash
goldenmatch match targets.csv --against reference.csv --config config.yaml --output-all
```

---

## demo

Run a built-in demo with sample data. No files needed.

```bash
goldenmatch demo
```

---

## tui / interactive

Launch the interactive terminal UI.

```bash
goldenmatch interactive customers.csv
goldenmatch interactive customers.csv --config config.yaml
```

---

## evaluate

Measure matching quality against ground truth pairs.

```bash
goldenmatch evaluate data.csv --config config.yaml --gt ground_truth.csv

# CI/CD quality gates
goldenmatch evaluate data.csv --config config.yaml --gt gt.csv \
    --min-f1 0.90 --min-precision 0.80 --min-recall 0.70
```

Exits with code 1 if thresholds are not met. Ground truth CSV must have `id_a` and `id_b` columns (configurable).

---

## incremental

Match new CSV records against an existing base dataset.

```bash
goldenmatch incremental base.csv --new new_records.csv --config config.yaml
```

Handles exact matchkeys via Polars join and fuzzy matchkeys via `match_one` brute-force.

---

## pprl link

Privacy-preserving record linkage between two files.

```bash
goldenmatch pprl link party_a.csv party_b.csv --security-level high
goldenmatch pprl link a.csv b.csv --fields first_name last_name dob zip --threshold 0.85
```

### pprl auto-config

Analyze data and recommend PPRL parameters.

```bash
goldenmatch pprl auto-config data.csv
```

---

## label

Build ground truth by labeling record pairs interactively. Type `y` (match), `n` (no match), or `s` (skip).

```bash
goldenmatch label customers.csv --config config.yaml --gt ground_truth.csv
```

---

## serve

Start the REST API server for real-time matching.

```bash
goldenmatch serve --file customers.csv --config config.yaml --port 8080
```

See [REST API](rest-api) for endpoint details.

---

## mcp-serve

Start the MCP server for Claude Desktop integration.

```bash
goldenmatch mcp-serve --file customers.csv --config config.yaml
```

See [MCP](mcp) for tool details.

---

## unmerge

Remove a record from its cluster (per-entity unmerge).

```bash
goldenmatch unmerge RECORD_ID --run-dir results/
```

---

## explain

Explain why two records matched.

```bash
goldenmatch explain ID_A ID_B --run-dir results/
```

---

## diff

Generate a before/after change report.

```bash
goldenmatch diff --run-dir results/ --html
```

---

## rollback

Undo a previous merge run.

```bash
goldenmatch rollback RUN_ID --run-dir results/
```

### runs

List previous runs available for rollback.

```bash
goldenmatch runs --run-dir results/
```

---

## graph

Multi-table entity resolution with cross-relationship evidence propagation.

```bash
goldenmatch graph --entities people.csv companies.csv --relationships edges.csv --config config.yaml
```

---

## anomaly

Detect fake emails, placeholder data, and suspicious records.

```bash
goldenmatch anomaly customers.csv
```

---

## report

Generate a detailed HTML match report.

```bash
goldenmatch report --run-dir results/ --output report.html
```

---

## dashboard

Generate a before/after data quality dashboard.

```bash
goldenmatch dashboard --run-dir results/ --output dashboard.html
```

---

## schema-match

Auto-map columns between different schemas.

```bash
goldenmatch schema-match file_a.csv file_b.csv
```

---

## watch

Watch a database table and match new records continuously.

```bash
goldenmatch watch --table customers --connection-string "$DATABASE_URL" --interval 30

# Daemon mode with health endpoint and PID file
goldenmatch watch --table customers --connection-string "$DATABASE_URL" --daemon
```

---

## Other commands

| Command | Description |
|---------|-------------|
| `goldenmatch setup` | Interactive setup wizard (GPU, API keys, database) |
| `goldenmatch init` | Interactive config wizard |
| `goldenmatch profile FILE` | Profile data quality |
| `goldenmatch sync --table TABLE` | Sync database table |
| `goldenmatch schedule --every 1h FILE` | Run on a schedule |
| `goldenmatch config save/load/list/show` | Manage config presets |
| `goldenmatch analyze-blocking FILE -c config.yaml` | Suggest blocking strategies |

---

## Common flags

| Flag | Available On | Description |
|------|-------------|-------------|
| `--config`, `-c` | dedupe, match | Path to YAML config file |
| `--output-all` | dedupe, match | Write golden, dupes, unique, lineage |
| `--output-dir` | dedupe, match | Output directory |
| `--llm-scorer` | dedupe | Enable LLM scoring for borderline pairs |
| `--llm-boost` | dedupe | LLM-labeled training + fine-tuning |
| `--backend ray` | dedupe, match | Use Ray distributed backend |
| `--preview` | dedupe | Show merge preview before writing |
| `--anomalies` | dedupe | Run anomaly detection |
| `--dashboard` | dedupe | Generate HTML dashboard |
| `--html-report` | dedupe | Generate HTML match report |
| `--diff` | dedupe | Generate diff report |
| `--chunked` | dedupe | Process in chunks for large files |
