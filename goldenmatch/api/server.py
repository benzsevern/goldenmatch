"""GoldenMatch REST API server — local HTTP server for real-time matching.

Usage:
    goldenmatch serve --file customers.csv --config config.yaml --port 8080

Endpoints:
    GET  /health              Health check
    GET  /stats               Current dataset stats
    POST /match               Match a single record against loaded data
    POST /match/batch         Match multiple records
    POST /explain             Explain why two records match
    GET  /clusters            List all clusters
    GET  /clusters/<id>       Get cluster detail
    GET  /reviews             Review queue (borderline pairs for steward review)
    GET  /reviews/decisions   List completed review decisions
    POST /reviews/decide      Approve or reject a pair (steward action)
"""

from __future__ import annotations

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse, parse_qs

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig

logger = logging.getLogger(__name__)


class MatchServer:
    """Holds the loaded data and engine for the API."""

    def __init__(self, engine, config: GoldenMatchConfig):
        self.engine = engine
        self.config = config
        self.result = None
        self._rows: list[dict] = []
        self._id_to_idx: dict[int, int] = {}
        self._review_queue: list[dict] = []  # pending reviews
        self._review_decisions: list[dict] = []  # completed reviews

    def initialize(self) -> None:
        """Run initial matching and cache results."""
        self.result = self.engine.run_full(self.config)
        self._rows = self.engine.data.to_dicts()
        self._id_to_idx = {
            row["__row_id__"]: i for i, row in enumerate(self._rows)
        }
        logger.info(
            "Server initialized: %d records, %d clusters",
            len(self._rows),
            len([c for c in self.result.clusters.values() if c["size"] > 1]),
        )

    def get_stats(self) -> dict:
        if not self.result:
            return {"status": "not_initialized"}
        s = self.result.stats
        return {
            "total_records": s.total_records,
            "total_clusters": s.total_clusters,
            "singleton_count": s.singleton_count,
            "match_rate": round(s.match_rate, 2),
            "avg_cluster_size": round(s.avg_cluster_size, 2),
            "max_cluster_size": s.max_cluster_size,
        }

    def match_record(self, record: dict, top_k: int = 5) -> list[dict]:
        """Match a single record against loaded data."""
        from goldenmatch.core.explainer import explain_pair

        matchkeys = self.config.get_matchkeys()
        results = []

        for mk in matchkeys:
            if mk.type != "weighted":
                continue
            for row in self._rows:
                exp = explain_pair(record, row, mk.fields, mk.threshold or 0.80)
                if exp.is_match:
                    clean_row = {k: v for k, v in row.items() if not k.startswith("__")}
                    results.append({
                        "record": clean_row,
                        "score": round(exp.total_score, 4),
                        "row_id": row.get("__row_id__"),
                    })

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def explain_pair(self, record_a: dict, record_b: dict) -> dict:
        """Explain why two records match."""
        from goldenmatch.core.explainer import explain_pair as _explain

        matchkeys = self.config.get_matchkeys()
        fields = []
        threshold = 0.80
        for mk in matchkeys:
            if mk.type == "weighted":
                fields = mk.fields
                threshold = mk.threshold or 0.80
                break

        exp = _explain(record_a, record_b, fields, threshold)
        return {
            "total_score": round(exp.total_score, 4),
            "threshold": exp.threshold,
            "is_match": exp.is_match,
            "top_contributor": exp.top_contributor,
            "weakest_field": exp.weakest_field,
            "fields": [
                {
                    "field": f.field_name,
                    "scorer": f.scorer,
                    "value_a": f.value_a,
                    "value_b": f.value_b,
                    "score": round(f.score, 4),
                    "weight": f.weight,
                    "contribution": round(f.contribution, 4),
                    "diff_type": f.diff_type,
                }
                for f in exp.fields
            ],
        }

    def get_clusters(self, min_size: int = 2) -> list[dict]:
        """List clusters."""
        if not self.result:
            return []
        clusters = []
        for cid, info in self.result.clusters.items():
            if info["size"] >= min_size:
                clusters.append({
                    "cluster_id": cid,
                    "size": info["size"],
                    "oversized": info.get("oversized", False),
                })
        clusters.sort(key=lambda x: -x["size"])
        return clusters

    def get_cluster_detail(self, cluster_id: int) -> dict | None:
        """Get cluster members."""
        if not self.result:
            return None
        info = self.result.clusters.get(cluster_id)
        if not info:
            return None

        members = []
        for mid in info["members"]:
            idx = self._id_to_idx.get(mid)
            if idx is not None:
                clean = {k: v for k, v in self._rows[idx].items() if not k.startswith("__")}
                members.append({"row_id": mid, **clean})

        return {
            "cluster_id": cluster_id,
            "size": info["size"],
            "members": members,
        }

    def build_review_queue(self, band_lo: float = 0.70, band_hi: float = 0.90, limit: int = 50) -> list[dict]:
        """Build review queue from borderline pairs."""
        from goldenmatch.core.explainer import explain_pair

        if not self.result:
            return []

        matchkeys = self.config.get_matchkeys()
        fields = []
        threshold = 0.80
        for mk in matchkeys:
            if mk.type == "weighted":
                fields = mk.fields
                threshold = mk.threshold or 0.80
                break

        queue = []
        for a, b, score in self.result.scored_pairs:
            if band_lo <= score <= band_hi:
                idx_a = self._id_to_idx.get(a)
                idx_b = self._id_to_idx.get(b)
                if idx_a is None or idx_b is None:
                    continue

                row_a = {k: v for k, v in self._rows[idx_a].items() if not k.startswith("__")}
                row_b = {k: v for k, v in self._rows[idx_b].items() if not k.startswith("__")}

                exp = explain_pair(self._rows[idx_a], self._rows[idx_b], fields, threshold)

                queue.append({
                    "pair_id": f"{a}_{b}",
                    "row_id_a": a,
                    "row_id_b": b,
                    "score": round(score, 4),
                    "is_match": exp.is_match,
                    "record_a": row_a,
                    "record_b": row_b,
                    "top_contributor": exp.top_contributor,
                    "weakest_field": exp.weakest_field,
                    "status": "pending",
                })

        queue.sort(key=lambda x: abs(x["score"] - threshold))
        self._review_queue = queue[:limit]
        return self._review_queue

    def review_decision(self, pair_id: str, decision: str, reviewer: str = "api") -> dict:
        """Record a review decision (approve/reject)."""
        from datetime import datetime

        for item in self._review_queue:
            if item["pair_id"] == pair_id:
                item["status"] = decision
                record = {
                    **item,
                    "reviewed_by": reviewer,
                    "reviewed_at": datetime.now().isoformat(),
                }
                self._review_decisions.append(record)

                # Apply decision: unmerge if rejected
                if decision == "reject":
                    self.engine.unmerge_record(item["row_id_a"])
                    self.result = self.engine._last_result

                return {"status": "recorded", "pair_id": pair_id, "decision": decision}

        return {"error": f"Pair {pair_id} not found in review queue"}


# Global server instance
_server_instance: MatchServer | None = None


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the GoldenMatch API."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self._json_response({"status": "ok", "service": "goldenmatch"})
        elif path == "/stats":
            self._json_response(_server_instance.get_stats())
        elif path == "/clusters":
            params = parse_qs(parsed.query)
            min_size = int(params.get("min_size", ["2"])[0])
            self._json_response(_server_instance.get_clusters(min_size))
        elif path.startswith("/clusters/"):
            try:
                cid = int(path.split("/")[-1])
                detail = _server_instance.get_cluster_detail(cid)
                if detail:
                    self._json_response(detail)
                else:
                    self._json_response({"error": "Cluster not found"}, 404)
            except ValueError:
                self._json_response({"error": "Invalid cluster ID"}, 400)
        elif path == "/reviews":
            params = parse_qs(parsed.query)
            lo = float(params.get("lo", ["0.70"])[0])
            hi = float(params.get("hi", ["0.90"])[0])
            limit = int(params.get("limit", ["50"])[0])
            queue = _server_instance.build_review_queue(lo, hi, limit)
            self._json_response({"queue": queue, "count": len(queue)})
        elif path == "/reviews/decisions":
            self._json_response({
                "decisions": _server_instance._review_decisions,
                "count": len(_server_instance._review_decisions),
            })
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        if path == "/match":
            record = data.get("record", data)
            top_k = data.get("top_k", 5)
            matches = _server_instance.match_record(record, top_k)
            self._json_response({"matches": matches, "count": len(matches)})
        elif path == "/match/batch":
            records = data.get("records", [])
            results = []
            for record in records:
                matches = _server_instance.match_record(record, top_k=3)
                results.append({"record": record, "matches": matches})
            self._json_response({"results": results})
        elif path == "/explain":
            record_a = data.get("record_a", {})
            record_b = data.get("record_b", {})
            explanation = _server_instance.explain_pair(record_a, record_b)
            self._json_response(explanation)
        elif path == "/reviews/decide":
            pair_id = data.get("pair_id", "")
            decision = data.get("decision", "")
            reviewer = data.get("reviewer", "api")
            if decision not in ("approve", "reject"):
                self._json_response({"error": "Decision must be 'approve' or 'reject'"}, 400)
            else:
                result = _server_instance.review_decision(pair_id, decision, reviewer)
                self._json_response(result)
        else:
            self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args) -> None:
        logger.info("%s %s", self.address_string(), format % args)


def start_server(
    engine,
    config: GoldenMatchConfig,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """Start the GoldenMatch API server."""
    global _server_instance
    _server_instance = MatchServer(engine, config)

    print(f"Initializing GoldenMatch API...")
    _server_instance.initialize()

    server = HTTPServer((host, port), APIHandler)
    stats = _server_instance.get_stats()

    print(f"\n⚡ GoldenMatch API running at http://{host}:{port}")
    print(f"   Records: {stats.get('total_records', 0):,}")
    print(f"   Clusters: {stats.get('total_clusters', 0):,}")
    print(f"\n   Endpoints:")
    print(f"   GET  /health              Health check")
    print(f"   GET  /stats               Dataset statistics")
    print(f"   POST /match               Match a record")
    print(f"   POST /match/batch         Match multiple records")
    print(f"   POST /explain             Explain a match")
    print(f"   GET  /clusters            List clusters")
    print(f"   GET  /clusters/<id>       Cluster detail")
    print(f"   GET  /reviews             Review queue (steward)")
    print(f"   POST /reviews/decide      Approve/reject a pair")
    print(f"\n   Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
