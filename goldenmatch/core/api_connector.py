"""API connector -- pull records from CRMs and APIs for deduplication.

Supports:
  goldenmatch dedupe --source salesforce --query "SELECT Id, Name, Email FROM Contact"
  goldenmatch dedupe --source hubspot --query "contacts"
  goldenmatch dedupe --source api --url "https://api.example.com/customers" --headers '{"Authorization": "Bearer TOKEN"}'

Each connector returns a Polars DataFrame ready for the pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def fetch_from_api(
    source: str,
    query: str | None = None,
    url: str | None = None,
    headers: dict | None = None,
    api_key: str | None = None,
    limit: int | None = None,
) -> pl.DataFrame:
    """Fetch records from an API source.

    Args:
        source: "salesforce", "hubspot", "api" (generic REST), or "graphql".
        query: Source-specific query (SOQL for Salesforce, object name for HubSpot).
        url: API URL (for generic REST/GraphQL).
        headers: HTTP headers.
        api_key: API key (added as Authorization header if provided).
        limit: Max records to fetch.

    Returns:
        Polars DataFrame with fetched records.
    """
    if source == "salesforce":
        return _fetch_salesforce(query or "", api_key, limit)
    elif source == "hubspot":
        return _fetch_hubspot(query or "contacts", api_key, limit)
    elif source == "api":
        return _fetch_rest(url or "", headers or {}, api_key, limit)
    elif source == "graphql":
        return _fetch_graphql(url or "", query or "", headers or {}, api_key)
    else:
        raise ValueError(f"Unknown API source: {source}. Supported: salesforce, hubspot, api, graphql")


def _fetch_salesforce(query: str, api_key: str | None, limit: int | None) -> pl.DataFrame:
    """Fetch from Salesforce via simple_salesforce or REST API."""
    try:
        from simple_salesforce import Salesforce
    except ImportError:
        raise ImportError(
            "Salesforce support requires simple-salesforce. "
            "Install with: pip install simple-salesforce"
        )

    import os
    sf = Salesforce(
        username=os.environ.get("SF_USERNAME", ""),
        password=os.environ.get("SF_PASSWORD", ""),
        security_token=os.environ.get("SF_TOKEN", ""),
        domain=os.environ.get("SF_DOMAIN", "login"),
    )

    if limit:
        query = query.rstrip(";").rstrip()
        if "LIMIT" not in query.upper():
            query += f" LIMIT {limit}"

    logger.info("Salesforce query: %s", query)
    result = sf.query_all(query)
    records = result.get("records", [])

    # Clean Salesforce metadata
    clean = []
    for rec in records:
        row = {k: v for k, v in rec.items() if k != "attributes"}
        clean.append(row)

    logger.info("Fetched %d records from Salesforce", len(clean))
    return pl.DataFrame(clean) if clean else pl.DataFrame()


def _fetch_hubspot(object_type: str, api_key: str | None, limit: int | None) -> pl.DataFrame:
    """Fetch from HubSpot CRM API."""
    import os
    import urllib.request

    token = api_key or os.environ.get("HUBSPOT_API_KEY", "")
    if not token:
        raise ValueError("Set HUBSPOT_API_KEY environment variable")

    base_url = f"https://api.hubapi.com/crm/v3/objects/{object_type}"
    all_records = []
    after = None
    page_limit = min(limit or 100, 100)

    while True:
        url = f"{base_url}?limit={page_limit}"
        if after:
            url += f"&after={after}"

        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())

        for result in data.get("results", []):
            row = {"id": result.get("id")}
            row.update(result.get("properties", {}))
            all_records.append(row)

        paging = data.get("paging", {}).get("next")
        if paging and (limit is None or len(all_records) < limit):
            after = paging.get("after")
        else:
            break

    if limit:
        all_records = all_records[:limit]

    logger.info("Fetched %d records from HubSpot (%s)", len(all_records), object_type)
    return pl.DataFrame(all_records) if all_records else pl.DataFrame()


def _fetch_rest(url: str, headers: dict, api_key: str | None, limit: int | None) -> pl.DataFrame:
    """Fetch from a generic REST API endpoint."""
    import urllib.request

    if not url:
        raise ValueError("URL required for API source")

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())

    # Handle common response shapes
    records = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ["data", "results", "records", "items", "rows", "entries"]:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        if not records and all(isinstance(v, (str, int, float, bool, type(None))) for v in data.values()):
            records = [data]

    if limit:
        records = records[:limit]

    logger.info("Fetched %d records from %s", len(records), url)
    return pl.DataFrame(records) if records else pl.DataFrame()


def _fetch_graphql(url: str, query: str, headers: dict, api_key: str | None) -> pl.DataFrame:
    """Fetch from a GraphQL endpoint."""
    import urllib.request

    if not url:
        raise ValueError("URL required for GraphQL source")

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps({"query": query}).encode()
    headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers)
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())

    # Extract records from GraphQL response
    records = []
    if "data" in data:
        for key, value in data["data"].items():
            if isinstance(value, list):
                records = value
                break
            elif isinstance(value, dict) and "edges" in value:
                records = [edge.get("node", {}) for edge in value["edges"]]
                break

    logger.info("Fetched %d records via GraphQL from %s", len(records), url)
    return pl.DataFrame(records) if records else pl.DataFrame()
