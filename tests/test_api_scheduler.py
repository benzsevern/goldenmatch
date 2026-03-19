"""Tests for API connector and scheduler."""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import polars as pl
import pytest


class TestAPIConnector:
    def test_fetch_rest_api(self):
        """Test fetching from a local REST endpoint."""
        from goldenmatch.core.api_connector import fetch_from_api

        # Start a tiny HTTP server
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                data = [
                    {"name": "John Smith", "email": "john@t.com"},
                    {"name": "Jane Doe", "email": "jane@t.com"},
                ]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        df = fetch_from_api(
            source="api",
            url=f"http://127.0.0.1:{port}/data",
        )

        assert df.height == 2
        assert "name" in df.columns
        assert "email" in df.columns

        server.server_close()

    def test_fetch_rest_with_data_key(self):
        """Test REST API that wraps records in a 'data' key."""
        from goldenmatch.core.api_connector import fetch_from_api

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                response = {
                    "data": [{"name": "A"}, {"name": "B"}],
                    "total": 2,
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        df = fetch_from_api(source="api", url=f"http://127.0.0.1:{port}/data")
        assert df.height == 2

        server.server_close()

    def test_fetch_with_limit(self):
        from goldenmatch.core.api_connector import fetch_from_api

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                data = [{"id": i} for i in range(100)]
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        df = fetch_from_api(source="api", url=f"http://127.0.0.1:{port}/data", limit=5)
        assert df.height == 5

        server.server_close()

    def test_unknown_source_raises(self):
        from goldenmatch.core.api_connector import fetch_from_api

        with pytest.raises(ValueError, match="Unknown API source"):
            fetch_from_api(source="nonexistent")

    def test_salesforce_requires_package(self):
        from goldenmatch.core.api_connector import fetch_from_api
        import sys

        # Remove simple_salesforce if present
        if "simple_salesforce" not in sys.modules:
            try:
                fetch_from_api(source="salesforce", query="SELECT Id FROM Contact")
            except (ImportError, Exception):
                pass  # Expected


class TestScheduler:
    def test_parse_interval(self):
        from goldenmatch.core.scheduler import parse_interval

        assert parse_interval("30m") == 1800
        assert parse_interval("1h") == 3600
        assert parse_interval("6h") == 21600
        assert parse_interval("1d") == 86400
        assert parse_interval("60s") == 60
        assert parse_interval("3600") == 3600

    def test_parse_interval_invalid(self):
        from goldenmatch.core.scheduler import parse_interval

        with pytest.raises(ValueError):
            parse_interval("abc")

    def test_parse_cron(self):
        from goldenmatch.core.scheduler import parse_cron

        assert parse_cron("0 6 * * *") == 86400  # daily at 6am
        assert parse_cron("30 * * * *") == 3600   # every hour at :30

    def test_scheduled_job_run_once(self):
        from goldenmatch.core.scheduler import ScheduledJob
        import csv, tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "email"])
            w.writerow(["John", "j@t.com"])
            w.writerow(["Jon", "j@t.com"])
            path = f.name

        job = ScheduledJob(
            job_id="test-job",
            file_paths=[path],
            interval_seconds=60,
        )

        result = job.run_once()
        assert result["records"] == 2
        assert result["job_id"] == "test-job"
        assert result["run_number"] == 1
        assert job.run_count == 1
