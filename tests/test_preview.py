import pytest
import polars as pl
from goldenmatch.core.preview import (
    format_preview_stats, format_preview_clusters,
    format_preview_golden, format_score_histogram,
)
from goldenmatch.tui.engine import EngineResult, EngineStats

@pytest.fixture
def sample_engine_result():
    clusters = {
        1: {"members": [0, 1], "size": 2, "oversized": False, "pair_scores": {(0, 1): 1.0}},
        2: {"members": [2, 3, 4], "size": 3, "oversized": False, "pair_scores": {}},
        3: {"members": [5], "size": 1, "oversized": False, "pair_scores": {}},
    }
    golden = pl.DataFrame({
        "__cluster_id__": [1, 2],
        "__golden_confidence__": [1.0, 0.85],
        "name": ["John Smith", "Jane Doe"],
        "email": ["john@test.com", "jane@test.com"],
    })
    stats = EngineStats(
        total_records=6, total_clusters=2, singleton_count=1,
        match_rate=0.33, cluster_sizes=[2, 3],
        avg_cluster_size=2.5, max_cluster_size=3, oversized_count=0,
    )
    return EngineResult(
        clusters=clusters, golden=golden, unique=None, dupes=None,
        quarantine=None, matched=None, unmatched=None,
        scored_pairs=[(0, 1, 1.0), (2, 3, 0.9), (3, 4, 0.85)],
        stats=stats,
    )

class TestFormatPreviewStats:
    def test_returns_string(self, sample_engine_result):
        output = format_preview_stats(sample_engine_result.stats)
        assert isinstance(output, str)
        assert "6" in output

    def test_match_mode_stats(self):
        stats = EngineStats(
            total_records=100, total_clusters=0, singleton_count=0,
            match_rate=0.0, cluster_sizes=[], avg_cluster_size=0,
            max_cluster_size=0, oversized_count=0, hit_rate=0.7, avg_score=0.88,
        )
        output = format_preview_stats(stats)
        assert "70" in output or "0.7" in output  # hit rate somewhere

class TestFormatPreviewClusters:
    def test_returns_string(self, sample_engine_result):
        data = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3, 4, 5],
            "name": ["John", "John", "Jane", "Jane", "J Doe", "Bob"],
            "email": ["j@t.com", "j@t.com", "ja@t.com", "ja@t.com", "jd@t.com", "b@t.com"],
            "__source__": ["a"] * 6,
        })
        output = format_preview_clusters(sample_engine_result.clusters, data)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_no_clusters(self):
        output = format_preview_clusters({1: {"members": [0], "size": 1, "oversized": False, "pair_scores": {}}}, pl.DataFrame({"__row_id__": [0], "a": [1]}))
        assert "No clusters" in output or "no clusters" in output.lower()

class TestFormatPreviewGolden:
    def test_returns_string(self, sample_engine_result):
        output = format_preview_golden(sample_engine_result.golden)
        assert isinstance(output, str)
        assert "John Smith" in output or "john" in output.lower()

    def test_none_golden(self):
        output = format_preview_golden(None)
        assert "No golden" in output or "no golden" in output.lower()

class TestFormatScoreHistogram:
    def test_returns_string(self):
        output = format_score_histogram([0.85, 0.9, 0.92, 0.95, 1.0, 1.0])
        assert isinstance(output, str)
        assert len(output) > 0

    def test_empty_scores(self):
        output = format_score_histogram([])
        assert isinstance(output, str)

    def test_single_score(self):
        output = format_score_histogram([1.0, 1.0, 1.0])
        assert isinstance(output, str)
