# dbt-goldenmatch

dbt integration for [GoldenMatch](https://github.com/benzsevern/goldenmatch) entity resolution.

## Installation

```bash
pip install dbt-goldenmatch
```

## Usage

Run GoldenMatch deduplication on a DuckDB table:

```python
from dbt_goldenmatch.materialize import run_goldenmatch_dedupe

result = run_goldenmatch_dedupe(
    input_table="raw_customers",
    config_path="match.yaml",
    output_table="deduped_customers",
    database="warehouse.duckdb",
)
print(f"Deduped {result['input_rows']} -> {result['clusters']} clusters")
```

## Status

Early stage -- API may change. Full dbt materialization plugin coming soon.
