"""Tests for large dataset chunked processing mode."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    BlockingConfig, BlockingKeyConfig, GoldenRulesConfig, OutputConfig,
)


@pytest.fixture
def demo_csv(tmp_path):
    """Create a test CSV with known duplicates."""
    f = tmp_path / "test_large.csv"
    with open(f, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "email", "zip"])
        # 500 records with ~15% duplicates
        for i in range(425):
            w.writerow([f"Person {i}", f"person{i}@test.com", f"{10000 + i % 100}"])
        for i in range(75):
            w.writerow([f"Person {i}", f"person{i}@test.com", f"{10000 + i % 100}"])
    return str(f)


@pytest.fixture
def config():
    return GoldenMatchConfig(
        matchkeys=[
            MatchkeyConfig(
                name="email_exact",
                type="exact",
                fields=[MatchkeyField(field="email", transforms=["lowercase"])],
            ),
        ],
        blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])]),
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
        output=OutputConfig(),
    )


class TestChunkedMatcher:
    def test_process_file(self, demo_csv, config):
        from goldenmatch.core.chunked import ChunkedMatcher

        matcher = ChunkedMatcher(config=config, chunk_size=100)
        result = matcher.process_file(demo_csv)

        assert result["total_records"] == 500
        assert result["chunks_processed"] == 5
        assert result["total_pairs"] > 0
        assert result["total_clusters"] > 0
        assert result["records_per_second"] > 0

    def test_single_chunk(self, demo_csv, config):
        from goldenmatch.core.chunked import ChunkedMatcher

        matcher = ChunkedMatcher(config=config, chunk_size=1000)
        result = matcher.process_file(demo_csv)

        # Should process in 1 chunk since file is 500 records
        assert result["chunks_processed"] <= 2
        assert result["total_records"] == 500

    def test_callback(self, demo_csv, config):
        from goldenmatch.core.chunked import ChunkedMatcher

        chunks_seen = []

        def on_chunk(chunk_num, total, pairs):
            chunks_seen.append((chunk_num, total, pairs))

        matcher = ChunkedMatcher(config=config, chunk_size=100)
        matcher.process_file(demo_csv, on_chunk=on_chunk)

        assert len(chunks_seen) > 0
        assert chunks_seen[-1][1] == 500  # total processed

    def test_cross_chunk_matching(self, tmp_path, config):
        """Verify duplicates across chunk boundaries are found."""
        from goldenmatch.core.chunked import ChunkedMatcher

        f = tmp_path / "cross_chunk.csv"
        with open(f, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["name", "email", "zip"])
            # Put duplicate pair across chunk boundary
            # Chunk 1 (rows 0-4)
            for i in range(5):
                w.writerow([f"Unique {i}", f"unique{i}@test.com", "10001"])
            # Chunk 2 (rows 5-9) — row 5 is a dupe of row 0
            w.writerow(["Unique 0", "unique0@test.com", "10001"])
            for i in range(4):
                w.writerow([f"Other {i}", f"other{i}@test.com", "10001"])

        matcher = ChunkedMatcher(config=config, chunk_size=5)
        result = matcher.process_file(str(f))

        # Should find at least the cross-chunk duplicate
        assert result["total_pairs"] >= 1
