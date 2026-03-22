"""GoldenMatch materialization for dbt.

Usage in dbt model:
    {{ config(materialized='goldenmatch_dedupe', match_config='match.yaml') }}
    SELECT * FROM {{ ref('raw_customers') }}
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from goldenmatch.config.loader import load_config
from goldenmatch.core.pipeline import run_dedupe


def run_goldenmatch_dedupe(
    input_table: str,
    config_path: str,
    output_table: str,
    database: str = ":memory:",
) -> dict:
    """Run GoldenMatch dedupe on a DuckDB table and write results back.

    Args:
        input_table: Source table name in DuckDB
        config_path: Path to GoldenMatch YAML config
        output_table: Destination table name
        database: DuckDB database path

    Returns:
        Summary dict with record counts and match rate
    """
    conn = duckdb.connect(database)

    # Read input
    df = conn.execute(f"SELECT * FROM {input_table}").pl()

    # Write to temp CSV for GoldenMatch ingest
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name
        df.write_csv(tmp_path)

    cfg = load_config(config_path)
    result = run_dedupe([(tmp_path, "source")], cfg)

    # Write results to DuckDB
    output_df = result.get("golden") or result.get("output")
    if output_df is not None:
        conn.execute(f"DROP TABLE IF EXISTS {output_table}")
        conn.execute(f"CREATE TABLE {output_table} AS SELECT * FROM output_df")

    Path(tmp_path).unlink(missing_ok=True)

    stats = result.get("stats", {})
    conn.close()
    return {
        "input_rows": df.height,
        "output_rows": output_df.height if output_df is not None else 0,
        "clusters": stats.get("total_clusters", 0),
    }
