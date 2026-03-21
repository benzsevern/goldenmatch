"""HubSpot connector for GoldenMatch.

Uses HubSpot REST API v3 directly (no extra SDK required).
Requires: HUBSPOT_API_KEY environment variable.
"""
from __future__ import annotations

import json
import logging
import urllib.request

import polars as pl

from goldenmatch.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.hubapi.com"


class HubSpotConnector(BaseConnector):
    """Read contacts, companies, or deals from HubSpot CRM."""

    name = "hubspot"

    def read(self, config: dict) -> pl.LazyFrame:
        api_key = self._credentials.get("key")
        if not api_key:
            raise ConnectorError(
                "HubSpot connector requires credentials_env pointing to an env var "
                "with your HubSpot API key."
            )

        object_type = config.get("object_type", "contacts")
        properties = config.get("properties", [])
        limit = config.get("limit", 100)

        all_records = []
        after = None

        while True:
            url = f"{_BASE_URL}/crm/v3/objects/{object_type}"
            params = [f"limit={min(limit, 100)}"]
            if properties:
                params.append(f"properties={','.join(properties)}")
            if after:
                params.append(f"after={after}")
            url += "?" + "&".join(params)

            req = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            })

            try:
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read())
            except Exception as e:
                raise ConnectorError(f"HubSpot API error: {e}") from e

            for item in data.get("results", []):
                record = {"hubspot_id": item["id"]}
                record.update(item.get("properties", {}))
                all_records.append(record)

            paging = data.get("paging", {}).get("next")
            if paging and len(all_records) < config.get("max_records", 100000):
                after = paging.get("after")
            else:
                break

        if not all_records:
            raise ConnectorError(f"No {object_type} returned from HubSpot.")

        df = pl.DataFrame(all_records)
        logger.info("HubSpot: read %d %s", df.height, object_type)
        return df.lazy()
