import pytest
import polars as pl
from goldenmatch.tui.engine import EngineResult, EngineStats


class TestEngineStats:
    def test_create(self):
        stats = EngineStats(
            total_records=1000,
            total_clusters=50,
            singleton_count=900,
            match_rate=0.05,
            cluster_sizes=[2, 3, 2, 5],
            avg_cluster_size=3.0,
            max_cluster_size=5,
            oversized_count=0,
        )
        assert stats.total_records == 1000
        assert stats.hit_rate is None

    def test_match_mode_stats(self):
        stats = EngineStats(
            total_records=500,
            total_clusters=0,
            singleton_count=0,
            match_rate=0.0,
            cluster_sizes=[],
            avg_cluster_size=0.0,
            max_cluster_size=0,
            oversized_count=0,
            hit_rate=0.7,
            avg_score=0.88,
        )
        assert stats.hit_rate == 0.7


class TestEngineResult:
    def test_create_dedupe(self):
        result = EngineResult(
            clusters={1: {"members": [0, 1], "size": 2, "oversized": False, "pair_scores": {}}},
            golden=None,
            unique=None,
            dupes=None,
            quarantine=None,
            matched=None,
            unmatched=None,
            scored_pairs=[(0, 1, 0.95)],
            stats=EngineStats(
                total_records=5, total_clusters=1, singleton_count=3,
                match_rate=0.2, cluster_sizes=[2], avg_cluster_size=2.0,
                max_cluster_size=2, oversized_count=0,
            ),
        )
        assert len(result.scored_pairs) == 1
