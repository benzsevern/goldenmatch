"""DuckDB backend for GoldenMatch -- out-of-core processing for large datasets.

User maintains their own DuckDB database. GoldenMatch reads tables/views
and writes results back. No schema creation or migration.

Requires: pip install goldenmatch[duckdb]
Dependencies: duckdb

Usage in config:
    backend:
      type: duckdb
      path: ./my_data.duckdb  # or ":memory:"

    sources:
      - table: customers
        source_name: customers
        query: "SELECT * FROM customers WHERE active"  # optional override
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class DuckDBBackend:
    """Read/write data from a user-maintained DuckDB database."""

    def __init__(self, path: str = ":memory:") -> None:
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "DuckDB backend requires duckdb. "
                "Install with: pip install goldenmatch[duckdb]"
            )
        self._path = path
        self._conn = duckdb.connect(path)
        logger.info("Connected to DuckDB: %s", path)

    def read_table(self, table: str, query: str | None = None) -> pl.LazyFrame:
        """Read a table or execute a query, returning a Polars LazyFrame.

        Args:
            table: Table or view name. Ignored if query is provided.
            query: Optional SQL query. If None, reads the full table.

        Returns:
            Polars LazyFrame with the data.
        """
        sql = query or f"SELECT * FROM {table}"
        arrow_table = self._conn.execute(sql).fetch_arrow_table()
        df = pl.from_arrow(arrow_table)
        logger.info("DuckDB: read %d rows from %s", df.height, table if not query else "query")
        return df.lazy()

    def write_table(
        self,
        df: pl.DataFrame,
        table: str,
        mode: str = "append",
    ) -> None:
        """Write a DataFrame to a DuckDB table.

        Args:
            df: DataFrame to write.
            table: Target table name.
            mode: Write mode -- "append", "replace", or "upsert".
                  User must ensure the table exists.
        """
        arrow_table = df.to_arrow()

        if mode == "replace":
            self._conn.execute(f"DELETE FROM {table}")

        # Register arrow table and insert
        self._conn.register("_gm_temp", arrow_table)
        self._conn.execute(f"INSERT INTO {table} SELECT * FROM _gm_temp")
        self._conn.unregister("_gm_temp")

        logger.info("DuckDB: wrote %d rows to %s (mode=%s)", df.height, table, mode)

    def execute(self, sql: str):
        """Execute arbitrary SQL."""
        return self._conn.execute(sql)

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        result = self._conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in result]

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
