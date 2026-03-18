"""Tests for threshold auto-tuning via Otsu's method."""

from __future__ import annotations

import numpy as np
import pytest


class TestSuggestThreshold:
    def test_bimodal_distribution(self):
        from goldenmatch.core.threshold import suggest_threshold

        rng = np.random.default_rng(42)
        non_matches = rng.normal(0.3, 0.05, 500).clip(0, 1).tolist()
        matches = rng.normal(0.9, 0.05, 100).clip(0, 1).tolist()
        scores = non_matches + matches

        result = suggest_threshold(scores)
        assert result is not None
        assert 0.5 < result < 0.8

    def test_unimodal_returns_none(self):
        from goldenmatch.core.threshold import suggest_threshold

        rng = np.random.default_rng(42)
        scores = rng.normal(0.5, 0.05, 500).clip(0, 1).tolist()

        result = suggest_threshold(scores)
        assert result is None

    def test_empty_scores(self):
        from goldenmatch.core.threshold import suggest_threshold

        result = suggest_threshold([])
        assert result is None

    def test_single_score(self):
        from goldenmatch.core.threshold import suggest_threshold

        result = suggest_threshold([0.5])
        assert result is None

    def test_all_identical_scores(self):
        from goldenmatch.core.threshold import suggest_threshold

        result = suggest_threshold([0.8] * 100)
        assert result is None

    def test_two_distinct_clusters(self):
        from goldenmatch.core.threshold import suggest_threshold

        scores = [0.1] * 50 + [0.9] * 50
        result = suggest_threshold(scores)
        assert result is not None
        assert 0.3 < result < 0.7

    def test_result_within_valid_range(self):
        from goldenmatch.core.threshold import suggest_threshold

        rng = np.random.default_rng(42)
        non_matches = rng.normal(0.2, 0.1, 300).clip(0, 1).tolist()
        matches = rng.normal(0.85, 0.05, 100).clip(0, 1).tolist()

        result = suggest_threshold(non_matches + matches)
        if result is not None:
            assert 0.0 < result < 1.0
