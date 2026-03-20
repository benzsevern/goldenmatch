"""Tests for active sampling strategies."""

from __future__ import annotations

import numpy as np
import pytest

from goldenmatch.core.active_sampling import (
    select_active_pairs,
    estimate_label_savings,
)


class TestActiveSampling:
    @pytest.fixture
    def pairs_and_scores(self):
        """100 pairs with varied scores."""
        rng = np.random.default_rng(42)
        pairs = [(i, i + 100, rng.random()) for i in range(100)]
        return pairs

    def test_uncertainty_selects_near_threshold(self, pairs_and_scores):
        pairs = pairs_and_scores
        scores = np.array([s for _, _, s in pairs])

        selected = select_active_pairs(pairs, strategy="uncertainty", n=20)
        assert len(selected) == 20

        # Selected pairs should have scores closer to 0.5 than random
        selected_scores = scores[selected]
        all_distance = np.mean(np.abs(scores - 0.5))
        selected_distance = np.mean(np.abs(selected_scores - 0.5))
        assert selected_distance < all_distance

    def test_boundary_selects_threshold_range(self, pairs_and_scores):
        pairs = pairs_and_scores
        scores = np.array([s for _, _, s in pairs])

        selected = select_active_pairs(pairs, strategy="boundary", n=20)
        assert len(selected) <= 20

        # Most selected should be in threshold range
        selected_scores = scores[selected]
        in_range = sum(1 for s in selected_scores if 0.6 <= s <= 0.9)
        assert in_range >= len(selected) * 0.5  # at least half in range

    def test_disagreement_needs_features(self, pairs_and_scores):
        pairs = pairs_and_scores

        # Without features, should return empty
        selected = select_active_pairs(pairs, strategy="disagreement", n=20)
        assert len(selected) == 0

        # With features, should select high-variance pairs
        rng = np.random.default_rng(42)
        features = rng.random((100, 5))
        # Make some pairs have high disagreement
        features[10] = [0.95, 0.1, 0.9, 0.05, 0.8]  # high variance
        features[20] = [0.5, 0.5, 0.5, 0.5, 0.5]     # low variance

        selected = select_active_pairs(pairs, features=features, strategy="disagreement", n=10)
        assert 10 in selected  # high disagreement should be selected

    def test_diversity_covers_score_range(self, pairs_and_scores):
        pairs = pairs_and_scores
        scores = np.array([s for _, _, s in pairs])

        selected = select_active_pairs(pairs, strategy="diversity", n=20)
        assert len(selected) == 20

        # Selected should span the score range
        selected_scores = scores[selected]
        score_range = selected_scores.max() - selected_scores.min()
        assert score_range > 0.5  # should cover most of [0, 1]

    def test_combined_mixes_strategies(self, pairs_and_scores):
        pairs = pairs_and_scores
        rng = np.random.default_rng(42)
        features = rng.random((100, 5))

        selected = select_active_pairs(
            pairs, features=features, strategy="combined", n=30,
        )
        assert len(selected) == 30
        assert len(set(selected)) == 30  # no duplicates

    def test_respects_labeled_indices(self, pairs_and_scores):
        pairs = pairs_and_scores
        labeled = {0, 1, 2, 3, 4}

        selected = select_active_pairs(
            pairs, labeled_indices=labeled, strategy="uncertainty", n=10,
        )
        assert len(selected) == 10
        assert not set(selected) & labeled  # no overlap

    def test_with_model_probabilities(self, pairs_and_scores):
        pairs = pairs_and_scores
        probs = np.array([s for _, _, s in pairs])

        # Make some probs very uncertain
        probs[50] = 0.50  # most uncertain
        probs[51] = 0.51

        selected = select_active_pairs(
            pairs, current_probs=probs, strategy="uncertainty", n=5,
        )
        assert 50 in selected or 51 in selected

    def test_empty_pairs(self):
        selected = select_active_pairs([], strategy="combined", n=10)
        assert selected == []

    def test_n_larger_than_available(self):
        pairs = [(0, 1, 0.5), (2, 3, 0.8)]
        selected = select_active_pairs(pairs, strategy="diversity", n=100)
        assert len(selected) == 2


class TestLabelSavings:
    def test_estimate_savings(self):
        result = estimate_label_savings(10000, strategy="combined")
        assert result["random_labels_needed"] == 300
        assert result["active_labels_needed"] < 300
        assert result["labels_saved"] > 0
        assert "45%" in result["saving_percentage"]

    def test_savings_all_strategies(self):
        for strategy in ["uncertainty", "disagreement", "boundary", "diversity", "combined"]:
            result = estimate_label_savings(5000, strategy=strategy)
            assert result["active_labels_needed"] < result["random_labels_needed"]
