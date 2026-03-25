# Quick Start

## Deduplicate a CSV

```python
import goldenmatch as gm

result = gm.dedupe("customers.csv", exact=["email"], fuzzy={"name": 0.85})

print(result)
# DedupeResult(records=1000, clusters=42, match_rate=8.4%)

result.golden.write_csv("golden_records.csv")
```

## Deduplicate a DataFrame

```python
import polars as pl
import goldenmatch as gm

df = pl.DataFrame({
    "name": ["John Smith", "Jon Smith", "Jane Doe", "JOHN SMITH"],
    "email": ["john@x.com", "john@x.com", "jane@y.com", "john@x.com"],
})

result = gm.dedupe_df(df, exact=["email"])
print(result.total_clusters)  # 1 cluster (3 records with john@x.com)
```

## Match Two Files

```python
result = gm.match("new_customers.csv", "master.csv", fuzzy={"name": 0.85})
result.matched.write_csv("matches.csv")
```

## Score Two Strings

```python
score = gm.score_strings("John Smith", "Jon Smyth", "jaro_winkler")
print(score)  # 0.91
```

## Use a YAML Config

```yaml
# config.yaml
matchkeys:
  - name: email_exact
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]

  - name: name_fuzzy
    type: weighted
    threshold: 0.85
    fields:
      - field: name
        scorer: jaro_winkler
        weight: 0.85
        transforms: [lowercase, strip]

blocking:
  keys:
    - fields: [zip]
```

```python
result = gm.dedupe("customers.csv", config="config.yaml")
```

## Privacy-Preserving Linkage

```python
result = gm.pprl_link(
    "hospital_a.csv", "hospital_b.csv",
    fields=["name", "dob", "zip"],
    security_level="high",
)
print(f"Found {result['match_count']} matches across parties")
```

## CLI

```bash
# Deduplicate
goldenmatch dedupe data.csv --exact email --fuzzy name:0.85

# Interactive TUI
goldenmatch tui

# Evaluate accuracy
goldenmatch evaluate --config config.yaml --ground-truth labels.csv --min-f1 0.90
```
