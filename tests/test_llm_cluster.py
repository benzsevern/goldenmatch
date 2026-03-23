"""Tests for in-context LLM clustering."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import json

import polars as pl
import pytest

from goldenmatch.core.llm_cluster import (
    _build_components,
    _split_component,
    _parse_cluster_response,
    _apply_cluster_results,
    llm_cluster_pairs,
)
from goldenmatch.config.schemas import LLMScorerConfig, BudgetConfig


class TestBuildComponents:
    def test_single_component(self):
        pairs = [(1, 2, 0.8), (2, 3, 0.85), (4, 5, 0.9)]
        candidates = [0, 1]  # pairs 0 and 1 are borderline
        components = _build_components(pairs, candidates)
        assert len(components) == 1
        records, pair_indices = components[0]
        assert set(records) == {1, 2, 3}
        assert set(pair_indices) == {0, 1}

    def test_multiple_components(self):
        pairs = [(1, 2, 0.8), (3, 4, 0.85), (5, 6, 0.9)]
        candidates = [0, 1, 2]
        components = _build_components(pairs, candidates)
        assert len(components) == 3

    def test_empty_candidates(self):
        pairs = [(1, 2, 0.8)]
        components = _build_components(pairs, [])
        assert len(components) == 0

    def test_bridging_pair(self):
        """A pair connecting two otherwise separate groups."""
        pairs = [(1, 2, 0.8), (3, 4, 0.85), (2, 3, 0.82)]
        candidates = [0, 1, 2]
        components = _build_components(pairs, candidates)
        assert len(components) == 1
        records, _ = components[0]
        assert set(records) == {1, 2, 3, 4}


class TestSplitComponent:
    def test_small_component_no_split(self):
        pairs = [(1, 2, 0.8), (2, 3, 0.85)]
        blocks = _split_component([1, 2, 3], [0, 1], pairs, max_size=100)
        assert len(blocks) == 1

    def test_oversized_splits(self):
        # 5 records in a chain, max_size=3
        pairs = [(1, 2, 0.8), (2, 3, 0.7), (3, 4, 0.9), (4, 5, 0.6)]
        blocks = _split_component([1, 2, 3, 4, 5], [0, 1, 2, 3], pairs, max_size=3)
        # Should split into blocks of 3 or fewer
        for records, _ in blocks:
            assert len(records) <= 3


class TestParseClusterResponse:
    def test_valid_json(self):
        response = json.dumps({
            "clusters": [
                {"members": [1, 2, 3], "confidence": 0.92},
                {"members": [4, 5], "confidence": 0.71},
            ],
            "singletons": [6],
        })
        result = _parse_cluster_response(response, [1, 2, 3, 4, 5, 6])
        assert len(result["clusters"]) == 2
        assert result["clusters"][0]["members"] == [1, 2, 3]
        assert result["clusters"][0]["confidence"] == 0.92
        assert 6 in result["singletons"]

    def test_markdown_code_block(self):
        response = '```json\n{"clusters": [{"members": [1, 2], "confidence": 0.8}], "singletons": []}\n```'
        result = _parse_cluster_response(response, [1, 2])
        assert len(result["clusters"]) == 1

    def test_invalid_json_returns_all_singletons(self):
        result = _parse_cluster_response("not json at all", [1, 2, 3])
        assert result["clusters"] == []
        assert set(result["singletons"]) == {1, 2, 3}

    def test_filters_invalid_ids(self):
        response = json.dumps({
            "clusters": [{"members": [1, 2, 999], "confidence": 0.8}],
            "singletons": [],
        })
        result = _parse_cluster_response(response, [1, 2, 3])
        # 999 is not valid, so only [1, 2] in cluster
        assert result["clusters"][0]["members"] == [1, 2]
        assert 3 in result["singletons"]

    def test_confidence_clamped(self):
        response = json.dumps({
            "clusters": [{"members": [1, 2], "confidence": 1.5}],
            "singletons": [],
        })
        result = _parse_cluster_response(response, [1, 2])
        assert result["clusters"][0]["confidence"] == 1.0

    def test_single_member_cluster_dropped(self):
        """Clusters with <2 members should be treated as singletons."""
        response = json.dumps({
            "clusters": [{"members": [1], "confidence": 0.9}],
            "singletons": [],
        })
        result = _parse_cluster_response(response, [1, 2])
        assert result["clusters"] == []
        assert set(result["singletons"]) == {1, 2}


class TestApplyClusterResults:
    def test_same_cluster_gets_confidence(self):
        cluster_result = {
            "clusters": [{"members": [1, 2, 3], "confidence": 0.92}],
            "singletons": [4],
        }
        pairs = [(1, 2, 0.8), (1, 3, 0.82), (2, 4, 0.78)]
        result_pairs = list(pairs)
        _apply_cluster_results(cluster_result, [0, 1, 2], pairs, result_pairs)

        # Pairs within same cluster get LLM confidence
        assert result_pairs[0] == (1, 2, 0.92)
        assert result_pairs[1] == (1, 3, 0.92)
        # Pair across clusters gets 0.0
        assert result_pairs[2] == (2, 4, 0.0)

    def test_singletons_get_zero(self):
        cluster_result = {
            "clusters": [],
            "singletons": [1, 2],
        }
        pairs = [(1, 2, 0.8)]
        result_pairs = list(pairs)
        _apply_cluster_results(cluster_result, [0], pairs, result_pairs)
        assert result_pairs[0] == (1, 2, 0.0)


class TestLLMClusterPairsIntegration:
    @pytest.fixture
    def sample_df(self):
        return pl.DataFrame({
            "__row_id__": [1, 2, 3, 4, 5],
            "name": ["John Smith", "john smith", "Jane Doe", "J. Smith", "Bob Jones"],
            "email": ["j@x.com", "j@x.com", "jane@y.com", "j@x.com", "bob@z.com"],
        })

    def test_auto_accept_pairs_unchanged(self, sample_df):
        """Pairs above auto_threshold should get score 1.0 regardless of mode."""
        config = LLMScorerConfig(enabled=True, mode="cluster")
        pairs = [(1, 2, 0.96), (3, 4, 0.80)]

        with patch("goldenmatch.core.llm_cluster._call_llm_cluster") as mock_llm:
            mock_llm.return_value = {
                "clusters": [{"members": [3, 4], "confidence": 0.75}],
                "singletons": [],
            }
            result = llm_cluster_pairs(pairs, sample_df, config, api_key="test")

        # Auto-accept pair gets 1.0
        assert result[0] == (1, 2, 1.0)

    def test_no_candidates_skips_llm(self, sample_df):
        """If all pairs are auto-accept or below, no LLM call."""
        config = LLMScorerConfig(enabled=True, mode="cluster")
        pairs = [(1, 2, 0.96), (3, 4, 0.50)]

        with patch("goldenmatch.core.llm_cluster._call_llm_cluster") as mock_llm:
            result = llm_cluster_pairs(pairs, sample_df, config, api_key="test")
            mock_llm.assert_not_called()

    def test_small_component_falls_back_to_pairwise(self, sample_df):
        """Components under cluster_min_size fall back to pairwise."""
        config = LLMScorerConfig(enabled=True, mode="cluster", cluster_min_size=10)
        pairs = [(1, 2, 0.80)]

        with patch("goldenmatch.core.llm_cluster._pairwise_fallback") as mock_pw:
            result = llm_cluster_pairs(pairs, sample_df, config, api_key="test")
            mock_pw.assert_called_once()

    def test_budget_tracking(self, sample_df):
        config = LLMScorerConfig(
            enabled=True, mode="cluster", cluster_min_size=2,
            budget=BudgetConfig(max_cost_usd=1.0),
        )
        pairs = [(1, 2, 0.80), (2, 3, 0.82)]

        with patch("goldenmatch.core.llm_cluster._call_llm_cluster") as mock_llm:
            mock_llm.return_value = {
                "clusters": [{"members": [1, 2, 3], "confidence": 0.88}],
                "singletons": [],
            }
            result, budget = llm_cluster_pairs(
                pairs, sample_df, config, api_key="test", return_budget=True,
            )
            assert budget is not None
            assert "total_cost_usd" in budget


class TestLLMScorerConfigModes:
    def test_default_mode_is_pairwise(self):
        config = LLMScorerConfig(enabled=True)
        assert config.mode == "pairwise"

    def test_cluster_mode(self):
        config = LLMScorerConfig(enabled=True, mode="cluster")
        assert config.mode == "cluster"
        assert config.cluster_max_size == 100
        assert config.cluster_min_size == 5
