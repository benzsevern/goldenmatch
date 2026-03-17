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


from goldenmatch.tui.engine import MatchEngine


class TestMatchEngineLoad:
    def test_load_single_file(self, sample_csv):
        engine = MatchEngine([sample_csv])
        assert engine.row_count == 5
        assert "email" in engine.columns
        assert engine.profile is not None
        assert engine.profile["total_rows"] == 5

    def test_load_multiple_files(self, sample_csv, sample_csv_b):
        engine = MatchEngine([sample_csv, sample_csv_b])
        assert engine.row_count == 8

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MatchEngine([tmp_path / "missing.csv"])

    def test_columns_property(self, sample_csv):
        engine = MatchEngine([sample_csv])
        cols = engine.columns
        assert "first_name" in cols
        assert "last_name" in cols
        # Internal columns should not appear
        assert "__source__" not in cols
        assert "__row_id__" not in cols

    def test_sample_extraction(self, sample_csv):
        engine = MatchEngine([sample_csv])
        sample = engine.get_sample(3)
        assert isinstance(sample, pl.DataFrame)
        assert sample.height == 3


from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    OutputConfig, GoldenRulesConfig, GoldenFieldRule,
)


@pytest.fixture
def exact_email_config(tmp_path):
    return GoldenMatchConfig(
        matchkeys=[
            MatchkeyConfig(
                name="email_key",
                fields=[MatchkeyField(column="email", transforms=["lowercase"])],
                comparison="exact",
            )
        ],
        output=OutputConfig(format="csv", directory=str(tmp_path), run_name="test"),
        golden_rules=GoldenRulesConfig(
            default=GoldenFieldRule(strategy="most_complete"),
        ),
    )


class TestMatchEngineRunSample:
    def test_run_sample_dedupe(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_sample(exact_email_config, sample_size=5)
        assert isinstance(result, EngineResult)
        assert result.stats.total_records == 5
        assert result.stats.total_clusters >= 1
        assert len(result.scored_pairs) >= 1

    def test_run_sample_small(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_sample(exact_email_config, sample_size=3)
        assert result.stats.total_records == 3

    def test_scored_pairs_cached(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_sample(exact_email_config)
        assert engine._last_result is not None
        assert engine._last_result.scored_pairs == result.scored_pairs

    def test_golden_records_created(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_sample(exact_email_config, sample_size=5)
        # sample_csv has john@example.com twice, so golden should exist
        if result.stats.total_clusters > 0:
            assert result.golden is not None


class TestMatchEngineRecluster:
    def test_recluster_at_threshold(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        engine.run_sample(exact_email_config)
        stats = engine.recluster_at_threshold(1.0)
        assert isinstance(stats, EngineStats)
        assert stats.total_records > 0

    def test_recluster_without_run_raises(self, sample_csv):
        engine = MatchEngine([sample_csv])
        with pytest.raises(RuntimeError):
            engine.recluster_at_threshold(0.8)


class TestMatchEngineRunFull:
    def test_run_full(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_full(exact_email_config)
        assert result.stats.total_records == 5
