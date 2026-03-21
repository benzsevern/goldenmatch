"""Salesforce connector for GoldenMatch.

Requires: pip install goldenmatch[salesforce]
Dependencies: simple-salesforce
"""
from __future__ import annotations

import logging

import polars as pl

from goldenmatch.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)


class SalesforceConnector(BaseConnector):
    """Read data from Salesforce via SOQL queries."""

    name = "salesforce"

    def read(self, config: dict) -> pl.LazyFrame:
        try:
            from simple_salesforce import Salesforce
        except ImportError:
            raise ConnectorError(
                "Salesforce connector requires simple-salesforce. "
                "Install with: pip install goldenmatch[salesforce]"
            )

        query = config.get("query")
        if not query:
            obj = config.get("object_type", "Contact")
            fields = config.get("fields", ["Id", "Name", "Email"])
            query = f"SELECT {', '.join(fields)} FROM {obj}"

        sf = Salesforce(
            username=self._credentials.get("user", ""),
            password=self._credentials.get("password", ""),
            security_token=self._credentials.get("key", ""),
        )

        result = sf.query_all(query)
        records = result.get("records", [])

        # Remove Salesforce metadata
        cleaned = []
        for rec in records:
            row = {k: v for k, v in rec.items() if k != "attributes"}
            cleaned.append(row)

        if not cleaned:
            raise ConnectorError("No records returned from Salesforce.")

        df = pl.DataFrame(cleaned)
        logger.info("Salesforce: read %d rows", df.height)
        return df.lazy()
