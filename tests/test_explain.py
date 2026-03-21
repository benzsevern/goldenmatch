"""Tests for natural language explanations and enhanced lineage."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from goldenmatch.core.explain import explain_cluster_nl, explain_pair_nl


class TestExplainPairNL:
    def test_identical_fields(self):
        field_scores = [
            {"field": "name", "scorer": "exact", "value_a": "John", "value_b": "John", "score": 1.0, "weight": 1.0, "diff_type": "identical"},
            {"field": "zip", "scorer": "exact", "value_a": "90210", "value_b": "90210", "score": 1.0, "weight": 0.5, "diff_type": "identical"},
        ]
        result = explain_pair_nl({}, {}, field_scores, 1.0)
        assert "Match" in result
        assert "exactly" in result

    def test_partial_match(self):
        field_scores = [
            {"field": "name", "scorer": "jaro_winkler", "value_a": "John", "value_b": "Jon", "score": 0.93, "weight": 1.0, "diff_type": "similar"},
            {"field": "zip", "scorer": "exact", "value_a": "90210", "value_b": "90211", "score": 0.0, "weight": 0.5, "diff_type": "different"},
        ]
        result = explain_pair_nl({}, {}, field_scores, 0.62)
        assert "name" in result
        assert "zip" in result
        assert "Weakest signal" in result

    def test_empty_fields(self):
        result = explain_pair_nl({}, {}, [], 0.5)
        assert "Match" in result

    def test_null_values(self):
        field_scores = [
            {"field": "email", "scorer": "exact", "value_a": None, "value_b": "a@b.com", "score": 0.0, "weight": 1.0, "diff_type": "missing"},
        ]
        result = explain_pair_nl({}, {}, field_scores, 0.0)
        assert "[null]" in result


class TestExplainClusterNL:
    def test_multi_member_cluster(self):
        cluster = {
            "members": [1, 2, 3],
            "size": 3,
            "confidence": 0.85,
            "pair_scores": {(1, 2): 0.92, (1, 3): 0.78, (2, 3): 0.85},
            "bottleneck_pair": (1, 3),
            "oversized": False,
        }
        result = explain_cluster_nl(cluster, None, [])
        assert "3 records" in result
        assert "confidence" in result
        assert "Weakest link" in result

    def test_singleton_cluster(self):
        cluster = {"members": [1], "size": 1}
        result = explain_cluster_nl(cluster, None, [])
        assert "Singleton" in result

    def test_oversized_warning(self):
        cluster = {
            "members": list(range(200)),
            "size": 200,
            "confidence": 0.3,
            "pair_scores": {},
            "bottleneck_pair": None,
            "oversized": True,
        }
        result = explain_cluster_nl(cluster, None, [])
        assert "WARNING" in result


class TestLineageNL:
    def test_build_lineage_with_nl(self):
        from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
        from goldenmatch.core.lineage import build_lineage

        df = pl.DataFrame({
            "__row_id__": [1, 2],
            "name": ["John Smith", "Jon Smith"],
            "zip": ["90210", "90210"],
        })
        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.7,
            fields=[
                MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0),
                MatchkeyField(field="zip", scorer="exact", weight=0.5),
            ],
        )
        clusters = {1: {"members": [1, 2], "size": 2, "oversized": False, "pair_scores": {(1, 2): 0.9}, "confidence": 0.9, "bottleneck_pair": (1, 2)}}

        lineage = build_lineage(
            [(1, 2, 0.90)], df, [mk], clusters,
            natural_language=True,
        )
        assert len(lineage) == 1
        assert "explanation" in lineage[0]
        assert "Match" in lineage[0]["explanation"]

    def test_build_lineage_no_nl_by_default(self):
        from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
        from goldenmatch.core.lineage import build_lineage

        df = pl.DataFrame({
            "__row_id__": [1, 2],
            "name": ["John", "Jon"],
        })
        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.7,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)],
        )
        clusters = {1: {"members": [1, 2], "size": 2, "oversized": False, "pair_scores": {}, "confidence": 0.5, "bottleneck_pair": None}}

        lineage = build_lineage([(1, 2, 0.85)], df, [mk], clusters)
        assert "explanation" not in lineage[0]


class TestLineageStreaming:
    def test_streaming_writes_valid_json(self, tmp_path):
        from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
        from goldenmatch.core.lineage import save_lineage_streaming

        df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "name": ["Alice", "Alce", "Bob"],
        })
        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.7,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)],
        )
        clusters = {1: {"members": [1, 2], "size": 2, "oversized": False, "pair_scores": {}, "confidence": 0.5, "bottleneck_pair": None}}

        path = save_lineage_streaming(
            [(1, 2, 0.88)], df, [mk], clusters,
            output_dir=tmp_path, run_name="test",
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_pairs"] == 1
        assert len(data["pairs"]) == 1

    def test_streaming_with_nl(self, tmp_path):
        from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
        from goldenmatch.core.lineage import save_lineage_streaming

        df = pl.DataFrame({
            "__row_id__": [1, 2],
            "name": ["John", "Jon"],
        })
        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.7,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)],
        )
        clusters = {1: {"members": [1, 2], "size": 2, "oversized": False, "pair_scores": {}, "confidence": 0.5, "bottleneck_pair": None}}

        path = save_lineage_streaming(
            [(1, 2, 0.90)], df, [mk], clusters,
            output_dir=tmp_path, run_name="test_nl",
            natural_language=True,
        )

        data = json.loads(path.read_text())
        assert "explanation" in data["pairs"][0]


class TestLineageNoCap:
    def test_max_pairs_zero_means_no_cap(self):
        from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
        from goldenmatch.core.lineage import build_lineage

        df = pl.DataFrame({
            "__row_id__": list(range(1, 6)),
            "name": ["A", "B", "C", "D", "E"],
        })
        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.5,
            fields=[MatchkeyField(field="name", scorer="exact", weight=1.0)],
        )
        clusters = {}
        pairs = [(i, j, 0.8) for i in range(1, 6) for j in range(i+1, 6)]

        lineage = build_lineage(pairs, df, [mk], clusters, max_pairs=0)
        assert len(lineage) == len(pairs)
