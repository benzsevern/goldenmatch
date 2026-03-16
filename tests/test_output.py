"""Tests for goldenmatch output writer and report generators."""

import polars as pl
import pytest

from goldenmatch.output.writer import write_output
from goldenmatch.output.report import generate_dedupe_report, generate_match_report


class TestWriteOutput:
    """Tests for write_output."""

    def test_write_csv(self, tmp_path):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = write_output(df, tmp_path, "test_run", "results", "csv")
        assert path.exists()
        assert path.name == "test_run_results.csv"
        loaded = pl.read_csv(path)
        assert loaded.shape == (2, 2)

    def test_write_parquet(self, tmp_path):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = write_output(df, tmp_path, "test_run", "results", "parquet")
        assert path.exists()
        assert path.name == "test_run_results.parquet"
        loaded = pl.read_parquet(path)
        assert loaded.shape == (2, 2)

    def test_write_xlsx(self, tmp_path):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = write_output(df, tmp_path, "test_run", "results", "xlsx")
        assert path.exists()
        assert path.name == "test_run_results.xlsx"

    def test_creates_directory(self, tmp_path):
        subdir = tmp_path / "nested" / "dir"
        df = pl.DataFrame({"a": [1]})
        path = write_output(df, subdir, "run", "out", "csv")
        assert path.exists()
        assert subdir.exists()


class TestDedupeReport:
    """Tests for generate_dedupe_report."""

    def test_report_contents(self):
        report = generate_dedupe_report(
            total_records=100,
            total_clusters=80,
            cluster_sizes=[1, 1, 1, 2, 2, 3],
            oversized_clusters=1,
            matchkeys_used=["mk1", "mk2"],
        )
        assert report["total_records"] == 100
        assert report["total_clusters"] == 80
        assert report["match_rate"] == pytest.approx(80 / 100)
        assert report["avg_cluster_size"] == pytest.approx(10 / 6)
        assert report["max_cluster_size"] == 3
        assert report["oversized_count"] == 1
        assert report["matchkeys_used"] == ["mk1", "mk2"]
        # cluster_size_distribution should be a Counter-like dict
        assert report["cluster_size_distribution"][1] == 3
        assert report["cluster_size_distribution"][2] == 2
        assert report["cluster_size_distribution"][3] == 1


class TestMatchReport:
    """Tests for generate_match_report."""

    def test_report_contents(self):
        report = generate_match_report(
            total_targets=50,
            matched=30,
            unmatched=20,
            scores=[0.8, 0.9, 0.7, 1.0, 0.6],
        )
        assert report["total_targets"] == 50
        assert report["matched"] == 30
        assert report["unmatched"] == 20
        assert report["hit_rate"] == pytest.approx(30 / 50)
        assert report["avg_score"] == pytest.approx(0.8)
        assert report["min_score"] == pytest.approx(0.6)
        assert report["max_score"] == pytest.approx(1.0)
