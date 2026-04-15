---
layout: default
title: Quick Start
nav_order: 3
---

# Quick Start

Go from raw CSV to deduplicated golden records in under a minute.

---

## Deduplicate a CSV (zero-config)

```bash
goldenmatch dedupe customers.csv
```

GoldenMatch auto-detects column types (name, email, phone, zip, address), assigns appropriate scorers, picks a blocking strategy, and launches the TUI for review.

---

## Deduplicate with Python

```python
import goldenmatch as gm

# Zero-config: auto-detects everything
result = gm.dedupe("customers.csv")

# Exact + fuzzy matching
result = gm.dedupe("customers.csv", exact=["email"], fuzzy={"name": 0.85, "zip": 0.95})
result.golden.write_csv("deduped.csv")
print(result)  # DedupeResult(records=5000, clusters=847, match_rate=12.0%)
```

---

## Deduplicate a DataFrame

```python
import goldenmatch as gm
import polars as pl

df = pl.read_csv("customers.csv")
result = gm.dedupe_df(df, exact=["email"], fuzzy={"name": 0.85})
result.golden  # Polars DataFrame of canonical records
```

---

## Inspecting the verification report (v1.5.0)

Zero-config runs attach a `PostflightReport` to the result — score-distribution signals, cluster-size percentiles, threshold-band overlap, plus any auto-applied adjustments and human-readable advisories.

```python
result = gm.dedupe_df(df)
if result.postflight_report:
    for adv in result.postflight_report.advisories:
        print(f"advisory: {adv}")
    for adj in result.postflight_report.adjustments:
        print(f"adjusted {adj.field}: {adj.from_value} -> {adj.to_value} ({adj.reason})")
```

The auto-generated config also carries a `PreflightReport` for the checks that ran during `auto_configure_df`:

```python
for finding in result.config._preflight_report.findings:
    print(f"[{finding.severity}] {finding.check}: {finding.message}")
```

See [Verification](python-api.html#verification-v150) in the Python API docs for the full signatures and signal schema.

---

## Match two files

```python
import goldenmatch as gm

result = gm.match("new_customers.csv", "master.csv", fuzzy={"name": 0.85})
result.matched.write_csv("matches.csv")
print(result)  # MatchResult(matched=412, unmatched=88)
```

CLI equivalent:

```bash
goldenmatch match new_customers.csv --against master.csv --config config.yaml
```

---

## Score two strings

```python
import goldenmatch as gm

score = gm.score_strings("John Smith", "Jon Smyth", "jaro_winkler")
print(score)  # 0.884
```

---

## Use a YAML config

```yaml
# config.yaml
matchkeys:
  - name: exact_email
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]

  - name: fuzzy_name
    type: weighted
    threshold: 0.85
    fields:
      - field: first_name
        scorer: jaro_winkler
        weight: 0.5
        transforms: [lowercase, strip]
      - field: last_name
        scorer: jaro_winkler
        weight: 0.3
        transforms: [lowercase, strip]
      - field: zip
        scorer: exact
        weight: 0.2

blocking:
  strategy: adaptive
  keys:
    - fields: [zip]

golden_rules:
  default_strategy: most_complete
```

```python
import goldenmatch as gm

result = gm.dedupe("customers.csv", config="config.yaml")
```

---

## Privacy-preserving linkage (PPRL)

Match across organizations without sharing raw data:

```python
import goldenmatch as gm

# Auto-configured: picks fields and threshold from your data
result = gm.pprl_link("hospital_a.csv", "hospital_b.csv")
print(f"Found {result['match_count']} matches")

# Manual field selection
result = gm.pprl_link(
    "party_a.csv", "party_b.csv",
    fields=["first_name", "last_name", "dob", "zip"],
    threshold=0.85,
    security_level="high",
)
```

CLI equivalent:

```bash
goldenmatch pprl link party_a.csv party_b.csv --security-level high
```

---

## LLM scoring for hard datasets

For product matching or other domains where fuzzy matching alone falls short:

```python
import goldenmatch as gm

result = gm.dedupe("products.csv", fuzzy={"title": 0.80}, llm_scorer=True)
```

The LLM scorer sends borderline pairs (score 0.75--0.95) to GPT-4o-mini and auto-accepts pairs above 0.95. Budget cap defaults to $0.05.

---

## Evaluate accuracy

```python
import goldenmatch as gm

metrics = gm.evaluate("data.csv", config="config.yaml", ground_truth="gt.csv")
print(f"F1: {metrics['f1']:.1%}, Precision: {metrics['precision']:.1%}")
```

```bash
# CI/CD quality gate: fail if F1 drops below 90%
goldenmatch evaluate data.csv --config config.yaml --gt gt.csv --min-f1 0.90
```

---

## Next steps

| Topic | Link |
|-------|------|
| Full Python API (101 exports) | [Python API](python-api) |
| All 21 CLI commands | [CLI Reference](cli) |
| Interactive TUI walkthrough | [TUI](tui) |
| Complete YAML config reference | [Configuration](configuration) |
| Use from Claude Desktop (30 MCP tools) | [MCP Server](mcp) |
| Build AI agents that deduplicate (A2A) | [ER Agent](agent) |
