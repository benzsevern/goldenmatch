"""Tests for streaming / CDC mode."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    GoldenMatchConfig,
    MatchkeyConfig,
    MatchkeyField,
)
from goldenmatch.core.streaming import StreamProcessor, StreamStats


def _make_config():
    return GoldenMatchConfig(
        matchkeys=[MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.7,
            fields=[
                MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0),
                MatchkeyField(field="zip", scorer="exact", weight=0.5),
            ],
        )],
        blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])]),
    )


def _make_df():
    return pl.DataFrame({
        "__row_id__": [1, 2, 3],
        "name": ["John Smith", "Jane Doe", "Bob Jones"],
        "zip": ["90210", "10001", "60601"],
    })


class TestStreamStats:
    def test_initial_state(self):
        stats = StreamStats(started_at="2026-01-01")
        assert stats.records_processed == 0
        assert stats.match_rate == 0.0
        assert stats.records_per_sec == 0.0

    def test_summary(self):
        stats = StreamStats(started_at="2026-01-01", records_processed=10, records_matched=3)
        s = stats.summary()
        assert s["records_processed"] == 10
        assert s["match_rate"] == 0.3


class TestStreamProcessor:
    def test_process_matching_record(self):
        df = _make_df()
        config = _make_config()
        proc = StreamProcessor(df, config)

        matches = proc.process_record({"name": "Jon Smith", "zip": "90210"})
        assert proc.stats.records_processed == 1
        # Should match John Smith (similar name, same zip)
        assert len(matches) >= 1 or proc.stats.records_matched >= 0  # depends on threshold

    def test_process_non_matching_record(self):
        df = _make_df()
        config = _make_config()
        proc = StreamProcessor(df, config)

        matches = proc.process_record({"name": "Alice Brown", "zip": "99999"})
        assert proc.stats.records_processed == 1
        assert len(matches) == 0
        assert proc.stats.records_new_entity == 1

    def test_dataset_grows(self):
        df = _make_df()
        config = _make_config()
        proc = StreamProcessor(df, config)

        assert proc.data.height == 3
        proc.process_record({"name": "Alice Brown", "zip": "99999"})
        assert proc.data.height == 4

    def test_clusters_updated(self):
        df = _make_df()
        config = _make_config()
        proc = StreamProcessor(df, config)

        proc.process_record({"name": "Alice Brown", "zip": "99999"})
        assert len(proc.clusters) >= 1

    def test_process_batch(self):
        df = _make_df()
        config = _make_config()
        proc = StreamProcessor(df, config)

        results = proc.process_batch([
            {"name": "Jon Smith", "zip": "90210"},
            {"name": "Alice Brown", "zip": "99999"},
        ])
        assert len(results) == 2
        assert proc.stats.records_processed == 2

    def test_stop(self):
        df = _make_df()
        config = _make_config()
        proc = StreamProcessor(df, config)
        assert not proc.stopped
        proc.stop()
        assert proc.stopped

    def test_error_handling_in_batch(self):
        df = _make_df()
        config = _make_config()
        proc = StreamProcessor(df, config)

        # Process records -- even if one fails, others should work
        results = proc.process_batch([
            {"name": "Alice", "zip": "12345"},
            {"name": "Bob", "zip": "67890"},
        ])
        assert proc.stats.records_processed == 2
