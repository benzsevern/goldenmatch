"""GoldenMatch REST API client.

Usage:
    import goldenmatch as gm

    client = gm.Client("http://localhost:8000")
    result = client.match({"name": "John Smith", "email": "john@x.com"})
    clusters = client.list_clusters()
    explanation = client.explain(record_a, record_b)
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError


class Client:
    """Python client for the GoldenMatch REST API.

    Connects to a running `goldenmatch serve` instance.

    Args:
        base_url: Server URL (e.g., "http://localhost:8000").
        timeout: Request timeout in seconds.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = Request(url, method="GET")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e}")

    def _post(self, path: str, data: dict) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode()
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e}")

    def health(self) -> dict:
        """Check server health."""
        return self._get("/health")

    def stats(self) -> dict:
        """Get pipeline statistics."""
        return self._get("/stats")

    def match(self, record: dict, top_k: int = 5) -> list[dict]:
        """Match a single record against the loaded dataset.

        Args:
            record: Record fields to match (e.g., {"name": "John", "email": "j@x.com"}).
            top_k: Max matches to return.

        Returns:
            List of matches with scores.
        """
        result = self._post("/match", {"record": record, "top_k": top_k})
        return result.get("matches", [])

    def list_clusters(self, min_size: int = 2, limit: int = 20) -> list[dict]:
        """List clusters, optionally filtered by size."""
        return self._get(f"/clusters?min_size={min_size}&limit={limit}")

    def get_cluster(self, cluster_id: int) -> dict:
        """Get cluster details including members and golden record."""
        return self._get(f"/clusters/{cluster_id}")

    def get_golden(self, cluster_id: int) -> dict:
        """Get the golden (canonical) record for a cluster."""
        return self._get(f"/golden/{cluster_id}")

    def explain(self, record_a: dict, record_b: dict) -> dict:
        """Get per-field match explanation for two records.

        Returns field-by-field scores and overall match decision.
        """
        # Use positional IDs if provided, else use the explain endpoint with records
        id_a = record_a.get("__row_id__", 0)
        id_b = record_b.get("__row_id__", 1)
        return self._get(f"/explain/{id_a}/{id_b}")

    def reviews(self) -> list[dict]:
        """Get borderline pairs pending human review."""
        return self._get("/reviews")

    def decide(self, pair_id: str, decision: str) -> dict:
        """Submit a review decision for a borderline pair.

        Args:
            pair_id: Pair identifier.
            decision: "approve" or "reject".
        """
        return self._post("/reviews/decide", {"pair_id": pair_id, "decision": decision})

    def config(self) -> dict:
        """Get current server configuration."""
        return self._get("/config")

    def __repr__(self) -> str:
        return f"GoldenMatch.Client({self.base_url!r})"
