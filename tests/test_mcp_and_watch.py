"""Tests for MCP server and watch mode."""

from __future__ import annotations

import csv
import json
import tempfile

import pytest


class TestMCPServer:
    @pytest.fixture
    def demo_file(self, tmp_path):
        f = tmp_path / "demo.csv"
        with open(f, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["name", "email", "zip"])
            w.writerow(["John Smith", "john@test.com", "10001"])
            w.writerow(["Jon Smith", "jon@test.com", "10001"])
            w.writerow(["Jane Doe", "jane@test.com", "90210"])
        return str(f)

    def test_mcp_server_creates(self, demo_file):
        from goldenmatch.mcp.server import create_server
        server = create_server([demo_file])
        assert server is not None

    def test_mcp_tool_get_stats(self, demo_file):
        from goldenmatch.mcp.server import create_server, _handle_tool
        create_server([demo_file])

        result = _handle_tool("get_stats", {})
        assert result["total_records"] == 3
        assert "total_clusters" in result
        assert "match_rate" in result

    def test_mcp_tool_find_duplicates(self, demo_file):
        from goldenmatch.mcp.server import create_server, _handle_tool
        create_server([demo_file])

        result = _handle_tool("find_duplicates", {
            "record": {"name": "John Smith", "zip": "10001"},
            "top_k": 3,
        })
        assert "matches" in result
        assert result["count"] >= 0

    def test_mcp_tool_explain_match(self, demo_file):
        from goldenmatch.mcp.server import create_server, _handle_tool
        create_server([demo_file])

        result = _handle_tool("explain_match", {
            "record_a": {"name": "John Smith", "zip": "10001"},
            "record_b": {"name": "Jon Smith", "zip": "10001"},
        })
        assert "total_score" in result
        assert "is_match" in result
        assert len(result["fields"]) > 0

    def test_mcp_tool_list_clusters(self, demo_file):
        from goldenmatch.mcp.server import create_server, _handle_tool
        create_server([demo_file])

        result = _handle_tool("list_clusters", {"min_size": 2, "limit": 10})
        assert "clusters" in result
        assert "total" in result

    def test_mcp_tool_profile_data(self, demo_file):
        from goldenmatch.mcp.server import create_server, _handle_tool
        create_server([demo_file])

        result = _handle_tool("profile_data", {})
        assert "columns" in result
        assert result["total_records"] == 3
        col_names = [c["column"] for c in result["columns"]]
        assert "name" in col_names

    def test_mcp_tool_export_results(self, demo_file, tmp_path):
        from goldenmatch.mcp.server import create_server, _handle_tool
        create_server([demo_file])

        output = str(tmp_path / "export.json")
        result = _handle_tool("export_results", {
            "output_path": output,
            "format": "json",
        })
        assert result["exported"] == output
        assert (tmp_path / "export.json").exists()

    def test_mcp_tool_unknown(self, demo_file):
        from goldenmatch.mcp.server import create_server, _handle_tool
        create_server([demo_file])

        result = _handle_tool("nonexistent_tool", {})
        assert "error" in result


class TestWatchMode:
    def test_watch_imports(self):
        from goldenmatch.db.watch import watch, _print_header, _log_event, _print_summary
        assert callable(watch)

    def test_log_event_formats(self, capsys):
        from goldenmatch.db.watch import _log_event

        _log_event("test normal")
        captured = capsys.readouterr()
        assert "test normal" in captured.out

        _log_event("test dim", dim=True)
        captured = capsys.readouterr()
        assert "test dim" in captured.out

        _log_event("test error", error=True)
        captured = capsys.readouterr()
        assert "test error" in captured.out

    def test_print_summary(self, capsys):
        from goldenmatch.db.watch import _print_summary

        _print_summary(total_syncs=5, total_merged=10, total_new=3, elapsed=120.5)
        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "10" in captured.out
        assert "120" in captured.out
