"""Snowflake connector for GoldenMatch.

Requires: pip install goldenmatch[snowflake]
Dependencies: snowflake-connector-python
"""
from __future__ import annotations

import logging

import polars as pl

from goldenmatch.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)


class SnowflakeConnector(BaseConnector):
    """Read/write data from Snowflake."""

    name = "snowflake"

    def read(self, config: dict) -> pl.LazyFrame:
        try:
            import snowflake.connector
        except ImportError:
            raise ConnectorError(
                "Snowflake connector requires snowflake-connector-python. "
                "Install with: pip install goldenmatch[snowflake]"
            )

        query = config.get("query")
        if not query:
            table = config.get("table")
            if not table:
                raise ConnectorError("Snowflake connector requires 'query' or 'table'.")
            query = f"SELECT * FROM {table}"

        conn = snowflake.connector.connect(
            account=self._credentials.get("key") or self._credentials.get("account", ""),
            user=self._credentials.get("user", ""),
            password=self._credentials.get("password", ""),
            database=self._credentials.get("database", ""),
            schema=self._credentials.get("schema", ""),
            warehouse=self._credentials.get("warehouse", ""),
        )

        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            df = pl.DataFrame(
                {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            )
            logger.info("Snowflake: read %d rows, %d columns", df.height, df.width)
            return df.lazy()
        finally:
            conn.close()

    def write(self, df: pl.DataFrame, config: dict) -> None:
        try:
            import snowflake.connector
            from snowflake.connector.pandas_tools import write_pandas
        except ImportError:
            raise ConnectorError(
                "Snowflake write requires snowflake-connector-python. "
                "Install with: pip install goldenmatch[snowflake]"
            )

        table = config.get("table")
        if not table:
            raise ConnectorError("Snowflake write requires 'table'.")

        mode = config.get("mode", "append")
        conn = snowflake.connector.connect(
            account=self._credentials.get("key") or self._credentials.get("account", ""),
            user=self._credentials.get("user", ""),
            password=self._credentials.get("password", ""),
            database=self._credentials.get("database", ""),
            schema=self._credentials.get("schema", ""),
            warehouse=self._credentials.get("warehouse", ""),
        )

        try:
            pdf = df.to_pandas()
            if mode == "replace":
                conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {table}")
            write_pandas(conn, pdf, table, auto_create_table=(mode == "replace"))
            logger.info("Snowflake: wrote %d rows to %s", df.height, table)
        finally:
            conn.close()
