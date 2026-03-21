"""Databricks connector for GoldenMatch.

Requires: pip install goldenmatch[databricks]
Dependencies: databricks-sql-connector
"""
from __future__ import annotations

import logging

import polars as pl

from goldenmatch.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)


class DatabricksConnector(BaseConnector):
    """Read data from Databricks SQL warehouses."""

    name = "databricks"

    def read(self, config: dict) -> pl.LazyFrame:
        try:
            from databricks import sql
        except ImportError:
            raise ConnectorError(
                "Databricks connector requires databricks-sql-connector. "
                "Install with: pip install goldenmatch[databricks]"
            )

        query = config.get("query")
        if not query:
            table = config.get("table")
            if not table:
                raise ConnectorError("Databricks connector requires 'query' or 'table'.")
            query = f"SELECT * FROM {table}"

        conn = sql.connect(
            server_hostname=self._credentials.get("key", ""),
            http_path=self._credentials.get("http_path", config.get("http_path", "")),
            access_token=self._credentials.get("password", ""),
        )

        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            df = pl.DataFrame(
                {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            )
            logger.info("Databricks: read %d rows, %d columns", df.height, df.width)
            return df.lazy()
        finally:
            conn.close()
