"""BigQuery connector for GoldenMatch.

Requires: pip install goldenmatch[bigquery]
Dependencies: google-cloud-bigquery
"""
from __future__ import annotations

import logging

import polars as pl

from goldenmatch.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)


class BigQueryConnector(BaseConnector):
    """Read/write data from Google BigQuery."""

    name = "bigquery"

    def read(self, config: dict) -> pl.LazyFrame:
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ConnectorError(
                "BigQuery connector requires google-cloud-bigquery. "
                "Install with: pip install goldenmatch[bigquery]"
            )

        query = config.get("query")
        if not query:
            table = config.get("table")
            if not table:
                raise ConnectorError("BigQuery connector requires 'query' or 'table'.")
            query = f"SELECT * FROM `{table}`"

        project = config.get("project") or self._credentials.get("key")
        client = bigquery.Client(project=project)
        result = client.query(query).to_dataframe()
        df = pl.from_pandas(result)
        logger.info("BigQuery: read %d rows, %d columns", df.height, df.width)
        return df.lazy()

    def write(self, df: pl.DataFrame, config: dict) -> None:
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ConnectorError(
                "BigQuery write requires google-cloud-bigquery. "
                "Install with: pip install goldenmatch[bigquery]"
            )

        table = config.get("table")
        if not table:
            raise ConnectorError("BigQuery write requires 'table'.")

        mode = config.get("mode", "append")
        project = config.get("project") or self._credentials.get("key")
        client = bigquery.Client(project=project)

        disposition = {
            "append": bigquery.WriteDisposition.WRITE_APPEND,
            "replace": bigquery.WriteDisposition.WRITE_TRUNCATE,
        }.get(mode, bigquery.WriteDisposition.WRITE_APPEND)

        job_config = bigquery.LoadJobConfig(write_disposition=disposition)
        pdf = df.to_pandas()
        client.load_table_from_dataframe(pdf, table, job_config=job_config).result()
        logger.info("BigQuery: wrote %d rows to %s", df.height, table)
