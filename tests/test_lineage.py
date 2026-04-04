"""Tests for lineage module — golden_records section."""

import json

import polars as pl
import pytest

from goldenmatch.core.golden import ClusterProvenance, FieldProvenance


def test_lineage_includes_golden_records(tmp_path):
    """save_lineage includes golden_records when provenance is passed."""
    from goldenmatch.core.lineage import save_lineage

    provenance = [
        ClusterProvenance(
            cluster_id=1,
            cluster_quality="strong",
            cluster_confidence=0.9,
            fields={
                "name": FieldProvenance(
                    value="Alice",
                    source_row_id=1,
                    strategy="most_complete",
                    confidence=1.0,
                    candidates=[],
                )
            },
        )
    ]
    path = save_lineage([], tmp_path, "test_run", golden_provenance=provenance)
    data = json.loads(path.read_text())
    assert "golden_records" in data
    assert len(data["golden_records"]) == 1
    assert data["golden_records"][0]["cluster_id"] == 1


def test_lineage_without_golden_records_backward_compat(tmp_path):
    """save_lineage without golden_provenance produces no golden_records key."""
    from goldenmatch.core.lineage import save_lineage

    path = save_lineage([], tmp_path, "test_run")
    data = json.loads(path.read_text())
    assert "golden_records" not in data


def test_streaming_lineage_includes_golden_records(tmp_path):
    """save_lineage_streaming includes golden_records section."""
    from goldenmatch.core.lineage import load_lineage, save_lineage_streaming

    df = pl.DataFrame({"__row_id__": [1, 2], "name": ["Alice", "Bob"]})
    clusters = {
        1: {
            "members": [1, 2],
            "size": 2,
            "pair_scores": {},
            "cluster_quality": "strong",
        }
    }
    provenance = [
        ClusterProvenance(
            cluster_id=1,
            cluster_quality="strong",
            cluster_confidence=0.9,
            fields={
                "name": FieldProvenance(
                    value="Alice",
                    source_row_id=1,
                    strategy="most_complete",
                    confidence=1.0,
                    candidates=[],
                )
            },
        )
    ]
    path = save_lineage_streaming(
        [], df, [], clusters, tmp_path, "test", golden_provenance=provenance
    )
    data = load_lineage(path)
    assert "golden_records" in data
    assert data["golden_records"][0]["cluster_quality"] == "strong"
