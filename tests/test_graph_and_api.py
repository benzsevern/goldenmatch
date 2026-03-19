"""Tests for cluster graph and REST API."""

from __future__ import annotations

import json

import polars as pl
import pytest


class TestClusterGraph:
    def test_generate_graph(self, tmp_path):
        from goldenmatch.core.graph import generate_cluster_graph

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3, 4],
            "name": ["John Smith", "Jon Smith", "Jane Doe", "Janet Doe", "Bob"],
            "email": ["j@t.com", "j@t.com", "jane@t.com", "janet@t.com", "bob@t.com"],
        })
        clusters = {
            1: {"size": 2, "members": [0, 1], "oversized": False},
            2: {"size": 2, "members": [2, 3], "oversized": False},
            3: {"size": 1, "members": [4], "oversized": False},
        }
        pairs = [(0, 1, 0.95), (2, 3, 0.88)]

        output = tmp_path / "graph.html"
        result = generate_cluster_graph(df, clusters, pairs, output_path=output)

        assert result.exists()
        content = result.read_text()
        assert "GoldenMatch" in content
        assert "John Smith" in content
        assert "cluster" in content.lower()
        assert "<svg" in content or "svg" in content

    def test_graph_auto_label(self, tmp_path):
        from goldenmatch.core.graph import generate_cluster_graph

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "title": ["Product A", "Product B"],
        })
        clusters = {1: {"size": 2, "members": [0, 1], "oversized": False}}
        pairs = [(0, 1, 0.9)]

        output = tmp_path / "graph2.html"
        generate_cluster_graph(df, clusters, pairs, output_path=output)

        content = output.read_text()
        assert "Product A" in content

    def test_graph_empty_clusters(self, tmp_path):
        from goldenmatch.core.graph import generate_cluster_graph

        df = pl.DataFrame({"__row_id__": [0], "name": ["Solo"]})
        output = tmp_path / "empty.html"
        generate_cluster_graph(df, {}, [], output_path=output)

        assert output.exists()

    def test_graph_max_clusters(self, tmp_path):
        from goldenmatch.core.graph import generate_cluster_graph

        df = pl.DataFrame({
            "__row_id__": list(range(20)),
            "name": [f"Record {i}" for i in range(20)],
        })
        clusters = {
            i: {"size": 2, "members": [i * 2, i * 2 + 1], "oversized": False}
            for i in range(10)
        }
        pairs = [(i * 2, i * 2 + 1, 0.9) for i in range(10)]

        output = tmp_path / "limited.html"
        generate_cluster_graph(
            df, clusters, pairs, output_path=output, max_clusters=3,
        )

        content = output.read_text()
        assert "3" in content  # 3 clusters in header


class TestRESTAPI:
    def test_match_server_init(self):
        from goldenmatch.tui.engine import MatchEngine
        from goldenmatch.api.server import MatchServer
        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
            BlockingConfig, BlockingKeyConfig, GoldenRulesConfig, OutputConfig,
        )

        import tempfile, csv
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "email", "zip"])
            w.writerow(["John Smith", "john@test.com", "10001"])
            w.writerow(["Jon Smith", "jon@test.com", "10001"])
            w.writerow(["Jane Doe", "jane@test.com", "90210"])
            path = f.name

        engine = MatchEngine([path])
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(
                name="test", type="weighted", threshold=0.80,
                fields=[
                    MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7, transforms=["lowercase"]),
                    MatchkeyField(field="zip", scorer="exact", weight=0.3),
                ],
            )],
            blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])]),
            golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
            output=OutputConfig(),
        )

        server = MatchServer(engine, config)
        server.initialize()

        # Test stats
        stats = server.get_stats()
        assert stats["total_records"] == 3

        # Test matching
        matches = server.match_record({"name": "John Smith", "zip": "10001"})
        assert len(matches) > 0
        assert matches[0]["score"] > 0.8

        # Test explain
        exp = server.explain_pair(
            {"name": "John Smith", "zip": "10001"},
            {"name": "Jon Smith", "zip": "10001"},
        )
        assert exp["is_match"] is True
        assert len(exp["fields"]) > 0

        # Test clusters
        clusters = server.get_clusters()
        assert len(clusters) >= 1

    def test_match_no_results(self):
        from goldenmatch.tui.engine import MatchEngine
        from goldenmatch.api.server import MatchServer
        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
            BlockingConfig, BlockingKeyConfig, OutputConfig,
        )

        import tempfile, csv
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "zip"])
            w.writerow(["John", "10001"])
            path = f.name

        engine = MatchEngine([path])
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(
                name="t", type="weighted", threshold=0.95,
                fields=[MatchkeyField(field="name", scorer="exact", weight=1.0)],
            )],
            blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])]),
            output=OutputConfig(),
        )

        server = MatchServer(engine, config)
        server.initialize()

        # Query with completely different name
        matches = server.match_record({"name": "ZZZZZ", "zip": "99999"})
        assert len(matches) == 0
