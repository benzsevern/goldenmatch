"""Tests for Fellegi-Sunter probabilistic matching."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    GoldenMatchConfig,
    MatchkeyConfig,
    MatchkeyField,
)


# ── Schema Tests ──────────────────────────────────────────────────────────


class TestProbabilisticSchema:
    def test_probabilistic_type_accepted(self):
        mk = MatchkeyConfig(
            name="fs_test",
            type="probabilistic",
            fields=[
                MatchkeyField(field="name", scorer="jaro_winkler", levels=3, partial_threshold=0.8),
                MatchkeyField(field="zip", scorer="exact", levels=2),
            ],
            em_iterations=20,
        )
        assert mk.type == "probabilistic"
        assert mk.em_iterations == 20
        assert mk.fields[0].levels == 3
        assert mk.fields[0].partial_threshold == 0.8

    def test_probabilistic_no_threshold_required(self):
        """Probabilistic matchkeys don't need a threshold upfront -- EM computes it."""
        mk = MatchkeyConfig(
            name="fs_test",
            type="probabilistic",
            fields=[
                MatchkeyField(field="name", scorer="jaro_winkler", levels=3, partial_threshold=0.8),
            ],
        )
        assert mk.threshold is None

    def test_probabilistic_fields_need_scorer(self):
        """Each field in a probabilistic matchkey must have a scorer."""
        with pytest.raises(ValueError, match="scorer"):
            MatchkeyConfig(
                name="fs_test",
                type="probabilistic",
                fields=[MatchkeyField(field="name", levels=3)],
            )

    def test_probabilistic_default_levels(self):
        """Fields default to 2 levels (agree/disagree) if not specified."""
        mk = MatchkeyConfig(
            name="fs_test",
            type="probabilistic",
            fields=[MatchkeyField(field="name", scorer="exact")],
        )
        assert mk.fields[0].levels == 2

    def test_probabilistic_comparison_alias(self):
        """The 'comparison' alias works for probabilistic type."""
        mk = MatchkeyConfig(
            name="fs_test",
            comparison="probabilistic",
            fields=[MatchkeyField(field="name", scorer="exact")],
        )
        assert mk.type == "probabilistic"

    def test_probabilistic_in_full_config(self):
        """Probabilistic matchkey works in a full GoldenMatchConfig."""
        cfg = GoldenMatchConfig(
            matchkeys=[{
                "name": "fs",
                "type": "probabilistic",
                "fields": [{"field": "name", "scorer": "jaro_winkler"}],
            }],
            blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["name"])]),
        )
        mks = cfg.get_matchkeys()
        assert mks[0].type == "probabilistic"

    def test_em_fields_have_defaults(self):
        mk = MatchkeyConfig(
            name="fs_test",
            type="probabilistic",
            fields=[MatchkeyField(field="name", scorer="exact")],
        )
        assert mk.em_iterations == 20
        assert mk.convergence_threshold == 0.001
        assert mk.link_threshold is None
        assert mk.review_threshold is None


# ── EM Core Tests ─────────────────────────────────────────────────────────

from goldenmatch.core.probabilistic import (
    EMResult,
    comparison_vector,
    compute_thresholds,
    score_pair_probabilistic,
    score_probabilistic,
    train_em,
)


def _make_dedupe_df():
    """DataFrame with obvious duplicates for EM training."""
    return pl.DataFrame({
        "__row_id__": list(range(1, 11)),
        "first_name": [
            "John", "Jon", "Jane", "Janet", "Bob",
            "Robert", "Alice", "Alicia", "Tom", "Thomas",
        ],
        "last_name": [
            "Smith", "Smith", "Doe", "Doe", "Jones",
            "Jones", "Brown", "Brown", "Wilson", "Wilson",
        ],
        "zip": [
            "90210", "90210", "10001", "10001", "60601",
            "60601", "30301", "30301", "20001", "20002",
        ],
    })


def _make_probabilistic_mk(**kwargs):
    defaults = dict(
        name="fs",
        type="probabilistic",
        fields=[
            MatchkeyField(field="first_name", scorer="jaro_winkler", levels=3, partial_threshold=0.8),
            MatchkeyField(field="last_name", scorer="jaro_winkler", levels=2, partial_threshold=0.85),
            MatchkeyField(field="zip", scorer="exact", levels=2),
        ],
    )
    defaults.update(kwargs)
    return MatchkeyConfig(**defaults)


class TestComparisonVector:
    def test_exact_agree(self):
        mk = _make_probabilistic_mk()
        vec = comparison_vector(
            {"first_name": "John", "last_name": "Smith", "zip": "90210"},
            {"first_name": "John", "last_name": "Smith", "zip": "90210"},
            mk,
        )
        # first_name: 3-level, exact match -> 2 (agree)
        # last_name: 2-level, exact match -> 1 (agree)
        # zip: 2-level, exact match -> 1 (agree)
        assert vec == [2, 1, 1]

    def test_partial_agree(self):
        mk = _make_probabilistic_mk()
        vec = comparison_vector(
            {"first_name": "John", "last_name": "Smith", "zip": "90210"},
            {"first_name": "Jon", "last_name": "Smyth", "zip": "90211"},
            mk,
        )
        # first_name: JW("John","Jon") ~ 0.93 -> partial (>0.8, <0.95)
        # last_name: JW("Smith","Smyth") ~ 0.87 -> agree (>0.85 for 2-level)
        # zip: exact("90210","90211") = 0 -> disagree
        assert vec[0] == 1  # partial
        assert vec[2] == 0  # disagree

    def test_full_disagree(self):
        mk = _make_probabilistic_mk()
        vec = comparison_vector(
            {"first_name": "Alice", "last_name": "Brown", "zip": "30301"},
            {"first_name": "Tom", "last_name": "Wilson", "zip": "20001"},
            mk,
        )
        assert vec[0] == 0  # disagree
        assert vec[1] == 0  # disagree
        assert vec[2] == 0  # disagree

    def test_null_values_disagree(self):
        mk = _make_probabilistic_mk()
        vec = comparison_vector(
            {"first_name": None, "last_name": "Smith", "zip": "90210"},
            {"first_name": "John", "last_name": "Smith", "zip": "90210"},
            mk,
        )
        assert vec[0] == 0  # null -> disagree


class TestEMTraining:
    def test_em_converges(self):
        df = _make_dedupe_df()
        mk = _make_probabilistic_mk()
        result = train_em(df, mk, n_sample_pairs=100, max_iterations=50)
        assert result.converged or result.iterations <= 50
        assert 0 < result.proportion_matched < 1

    def test_em_produces_valid_probabilities(self):
        df = _make_dedupe_df()
        mk = _make_probabilistic_mk()
        result = train_em(df, mk, n_sample_pairs=100)
        for field_name, probs in result.m_probs.items():
            assert abs(sum(probs) - 1.0) < 0.01, f"m_probs for {field_name} don't sum to 1"
        for field_name, probs in result.u_probs.items():
            assert abs(sum(probs) - 1.0) < 0.01, f"u_probs for {field_name} don't sum to 1"

    def test_em_match_weights_direction(self):
        """Match weights should be positive for agree, negative for disagree."""
        df = _make_dedupe_df()
        mk = _make_probabilistic_mk()
        result = train_em(df, mk, n_sample_pairs=100)
        for field_name, weights in result.match_weights.items():
            # Highest level (agree) should have positive weight
            assert weights[-1] > 0, f"Agree weight for {field_name} should be positive"
            # Lowest level (disagree) should have negative weight
            assert weights[0] < 0, f"Disagree weight for {field_name} should be negative"

    def test_em_with_small_data(self):
        """EM should handle datasets too small for proper training."""
        df = pl.DataFrame({
            "__row_id__": [1, 2],
            "first_name": ["John", "Jon"],
            "last_name": ["Smith", "Smith"],
            "zip": ["90210", "90210"],
        })
        mk = _make_probabilistic_mk()
        result = train_em(df, mk, n_sample_pairs=10)
        assert result is not None
        assert len(result.m_probs) == 3


class TestComputeThresholds:
    def test_thresholds_in_range(self):
        df = _make_dedupe_df()
        mk = _make_probabilistic_mk()
        result = train_em(df, mk, n_sample_pairs=100)
        link, review = compute_thresholds(result)
        assert 0 < review < link < 1


class TestScoreProbabilistic:
    def test_scores_obvious_matches(self):
        """Obvious duplicates should score high."""
        df = _make_dedupe_df()
        mk = _make_probabilistic_mk()
        em = train_em(df, mk, n_sample_pairs=100)

        # Block with just the first two records (John/Jon Smith, same zip)
        block = df.head(2)
        pairs = score_probabilistic(block, mk, em)
        # Should find a match
        assert len(pairs) >= 1
        # Score should be high
        assert pairs[0][2] > 0.5

    def test_excludes_pairs(self):
        df = _make_dedupe_df()
        mk = _make_probabilistic_mk()
        em = train_em(df, mk, n_sample_pairs=100)

        block = df.head(2)
        pairs_all = score_probabilistic(block, mk, em)
        pairs_excluded = score_probabilistic(block, mk, em, exclude_pairs={(1, 2)})
        assert len(pairs_excluded) < len(pairs_all) or len(pairs_all) == 0

    def test_returns_standard_pair_format(self):
        df = _make_dedupe_df()
        mk = _make_probabilistic_mk()
        em = train_em(df, mk, n_sample_pairs=100)

        block = df.head(4)
        pairs = score_probabilistic(block, mk, em)
        for p in pairs:
            assert len(p) == 3
            assert isinstance(p[0], int)
            assert isinstance(p[1], int)
            assert isinstance(p[2], float)
            assert 0 <= p[2] <= 1


class TestPipelineIntegration:
    def test_full_pipeline_with_probabilistic(self, tmp_path):
        """End-to-end: write CSV, run_dedupe with probabilistic matchkey."""
        import csv
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["first_name", "last_name", "zip"])
            w.writerow(["John", "Smith", "90210"])
            w.writerow(["Jon", "Smith", "90210"])
            w.writerow(["Jane", "Doe", "10001"])
            w.writerow(["Janet", "Doe", "10001"])
            w.writerow(["Alice", "Brown", "30301"])

        from goldenmatch.core.pipeline import run_dedupe
        cfg = GoldenMatchConfig(
            matchkeys=[{
                "name": "fs",
                "type": "probabilistic",
                "fields": [
                    {"field": "first_name", "scorer": "jaro_winkler", "levels": 3, "partial_threshold": 0.8},
                    {"field": "last_name", "scorer": "jaro_winkler", "levels": 2},
                    {"field": "zip", "scorer": "exact", "levels": 2},
                ],
            }],
            blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])]),
        )

        result = run_dedupe(
            [(str(csv_path), "test")], cfg,
            output_clusters=True,
        )
        clusters = result["clusters"]
        # Should find at least 1 cluster (John/Jon Smith share zip)
        multi_clusters = {cid: c for cid, c in clusters.items() if c["size"] > 1}
        assert len(multi_clusters) >= 1

    def test_engine_with_probabilistic(self, tmp_path):
        """MatchEngine works with probabilistic matchkeys."""
        import csv
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["first_name", "last_name", "zip"])
            w.writerow(["John", "Smith", "90210"])
            w.writerow(["Jon", "Smith", "90210"])
            w.writerow(["Bob", "Jones", "60601"])

        from goldenmatch.tui.engine import MatchEngine
        engine = MatchEngine([str(csv_path)])
        cfg = GoldenMatchConfig(
            matchkeys=[{
                "name": "fs",
                "type": "probabilistic",
                "fields": [
                    {"field": "first_name", "scorer": "jaro_winkler", "levels": 3, "partial_threshold": 0.8},
                    {"field": "last_name", "scorer": "exact", "levels": 2},
                    {"field": "zip", "scorer": "exact", "levels": 2},
                ],
            }],
            blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])]),
        )
        result = engine.run_full(cfg)
        assert result.stats.total_records == 3


class TestScorePairProbabilistic:
    def test_single_pair_scoring(self):
        mk = _make_probabilistic_mk()
        df = _make_dedupe_df()
        em = train_em(df, mk, n_sample_pairs=100)

        score = score_pair_probabilistic(
            {"first_name": "John", "last_name": "Smith", "zip": "90210"},
            {"first_name": "John", "last_name": "Smith", "zip": "90210"},
            mk, em,
        )
        assert score > 0.8  # identical records should score very high

    def test_non_match_scores_low(self):
        mk = _make_probabilistic_mk()
        df = _make_dedupe_df()
        em = train_em(df, mk, n_sample_pairs=100)

        score = score_pair_probabilistic(
            {"first_name": "Alice", "last_name": "Brown", "zip": "30301"},
            {"first_name": "Tom", "last_name": "Wilson", "zip": "20001"},
            mk, em,
        )
        assert score < 0.5
