# CLI Reference

```bash
goldenmatch --help
```

## Core Commands

| Command | Description |
|---------|-------------|
| `goldenmatch dedupe <files>` | Deduplicate one or more files |
| `goldenmatch match <target> <reference>` | Match target against reference |
| `goldenmatch demo` | Run a demo with sample data |
| `goldenmatch tui` | Launch interactive TUI |

## Options (dedupe/match)

```bash
goldenmatch dedupe data.csv \
  --config config.yaml \
  --exact email phone \
  --fuzzy name:0.85 address:0.90 \
  --blocking zip \
  --output-golden \
  --output-clusters \
  --output-dupes \
  --format csv
```

## Evaluation

```bash
goldenmatch evaluate \
  --config config.yaml \
  --ground-truth labels.csv \
  --min-f1 0.90 \
  --min-precision 0.80
```

Exit code 1 if thresholds not met (for CI/CD quality gates).

## Incremental Matching

```bash
goldenmatch incremental \
  --base existing_data.csv \
  --new new_records.csv \
  --config config.yaml
```

## PPRL

```bash
# Link two parties
goldenmatch pprl link \
  --file-a hospital_a.csv \
  --file-b hospital_b.csv \
  --fields name dob zip \
  --security-level high

# Auto-configure PPRL parameters
goldenmatch pprl auto-config \
  --file data.csv
```

## Interactive Labeling

```bash
goldenmatch label \
  --config config.yaml \
  --output labels.csv \
  --strategy borderline \
  --count 100
```

## Server

```bash
# REST API
goldenmatch serve --port 8000

# MCP server for AI assistants
goldenmatch mcp-serve
```

## All 21 Commands

`dedupe`, `match`, `demo`, `tui`, `evaluate`, `incremental`, `pprl link`, `pprl auto-config`, `label`, `serve`, `mcp-serve`, `unmerge`, `explain`, `diff`, `rollback`, `graph`, `anomaly`, `report`, `dashboard`, `schema-match`, `watch`
