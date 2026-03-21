"""Tests for LLM budget tracking."""
from __future__ import annotations

import pytest

from goldenmatch.config.schemas import LLMScorerConfig, BudgetConfig


class TestBudgetConfig:
    def test_defaults(self):
        b = BudgetConfig()
        assert b.max_cost_usd is None
        assert b.max_calls is None
        assert b.escalation_model is None
        assert b.escalation_band == [0.80, 0.90]
        assert b.escalation_budget_pct == 20
        assert b.warn_at_pct == 80

    def test_embedded_in_llm_scorer(self):
        cfg = LLMScorerConfig(
            enabled=True,
            budget=BudgetConfig(max_cost_usd=5.0, max_calls=500),
        )
        assert cfg.budget.max_cost_usd == 5.0
        assert cfg.budget.max_calls == 500

    def test_llm_scorer_default_no_budget(self):
        cfg = LLMScorerConfig(enabled=True)
        assert cfg.budget is None


from goldenmatch.core.llm_budget import BudgetTracker


class TestBudgetTrackerAccounting:
    def test_initial_state(self):
        bt = BudgetTracker(BudgetConfig(max_cost_usd=10.0))
        assert bt.total_cost_usd == 0.0
        assert bt.total_calls == 0
        assert bt.budget_exhausted is False
        assert bt.budget_remaining_pct == 100.0

    def test_record_usage(self):
        bt = BudgetTracker(BudgetConfig(max_cost_usd=1.0))
        bt.record_usage(input_tokens=1000, output_tokens=100, model="gpt-4o-mini")
        assert bt.total_calls == 1
        assert bt.total_cost_usd > 0

    def test_budget_exhausted_by_cost(self):
        bt = BudgetTracker(BudgetConfig(max_cost_usd=0.001))
        bt.record_usage(input_tokens=100000, output_tokens=10000, model="gpt-4o-mini")
        assert bt.budget_exhausted is True

    def test_budget_exhausted_by_calls(self):
        bt = BudgetTracker(BudgetConfig(max_calls=2))
        bt.record_usage(input_tokens=10, output_tokens=5, model="gpt-4o-mini")
        bt.record_usage(input_tokens=10, output_tokens=5, model="gpt-4o-mini")
        assert bt.budget_exhausted is True

    def test_can_send_respects_budget(self):
        bt = BudgetTracker(BudgetConfig(max_cost_usd=0.001))
        bt.record_usage(input_tokens=100000, output_tokens=10000, model="gpt-4o-mini")
        assert bt.can_send(estimated_tokens=100) is False

    def test_no_budget_means_unlimited(self):
        bt = BudgetTracker(BudgetConfig())
        bt.record_usage(input_tokens=999999, output_tokens=999999, model="gpt-4o-mini")
        assert bt.budget_exhausted is False
        assert bt.can_send(estimated_tokens=100) is True


class TestBudgetTrackerTiering:
    def test_select_model_default(self):
        bt = BudgetTracker(BudgetConfig())
        assert bt.select_model(pair_score=0.85, default_model="gpt-4o-mini") == "gpt-4o-mini"

    def test_select_model_escalation(self):
        bt = BudgetTracker(BudgetConfig(
            escalation_model="gpt-4o",
            escalation_band=[0.80, 0.90],
            escalation_budget_pct=50,
        ))
        assert bt.select_model(pair_score=0.85, default_model="gpt-4o-mini") == "gpt-4o"

    def test_select_model_outside_band(self):
        bt = BudgetTracker(BudgetConfig(
            escalation_model="gpt-4o",
            escalation_band=[0.80, 0.90],
        ))
        assert bt.select_model(pair_score=0.75, default_model="gpt-4o-mini") == "gpt-4o-mini"
        assert bt.select_model(pair_score=0.95, default_model="gpt-4o-mini") == "gpt-4o-mini"

    def test_escalation_respects_budget_pct(self):
        bt = BudgetTracker(BudgetConfig(
            max_cost_usd=1.0,
            escalation_model="gpt-4o",
            escalation_band=[0.80, 0.90],
            escalation_budget_pct=20,
        ))
        bt._escalation_cost = 0.20
        assert bt.select_model(pair_score=0.85, default_model="gpt-4o-mini") == "gpt-4o-mini"


class TestBudgetTrackerSummary:
    def test_summary_keys(self):
        bt = BudgetTracker(BudgetConfig(max_cost_usd=5.0))
        bt.record_usage(input_tokens=100, output_tokens=50, model="gpt-4o-mini")
        s = bt.summary()
        assert "total_cost_usd" in s
        assert "total_calls" in s
        assert "budget_remaining_pct" in s
        assert "models_used" in s


import polars as pl
from unittest.mock import patch


def _make_test_df():
    return pl.DataFrame({
        "__row_id__": [1, 2, 3, 4],
        "name": ["Alice Smith", "Alce Smith", "Bob Jones", "Robert Jones"],
        "email": ["a@b.com", "a@b.com", "b@c.com", "bob@c.com"],
    })


class TestLLMScorerWithBudget:
    def test_budget_stops_llm_calls(self):
        """When budget is exhausted, remaining candidates keep fuzzy scores."""
        from goldenmatch.core.llm_scorer import llm_score_pairs
        cfg = LLMScorerConfig(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            auto_threshold=0.95,
            candidate_lo=0.75,
            candidate_hi=0.95,
            budget=BudgetConfig(max_calls=0),
        )
        pairs = [(1, 2, 0.85), (3, 4, 0.80)]
        df = _make_test_df()

        result = llm_score_pairs(pairs, df, config=cfg, api_key="fake-key")
        # With zero budget, no LLM calls happen; candidates keep original scores
        assert result[0][2] == 0.85
        assert result[1][2] == 0.80

    def test_budget_summary_in_result(self):
        """llm_score_pairs returns budget summary when return_budget=True."""
        from goldenmatch.core.llm_scorer import llm_score_pairs
        cfg = LLMScorerConfig(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            budget=BudgetConfig(max_calls=0),
        )
        pairs = [(1, 2, 0.85)]
        df = _make_test_df()

        result, budget_summary = llm_score_pairs(
            pairs, df, config=cfg, api_key="fake-key", return_budget=True,
        )
        assert budget_summary is not None
        assert "total_cost_usd" in budget_summary

    @patch("goldenmatch.core.llm_scorer._call_openai")
    def test_budget_tracks_real_calls(self, mock_call):
        """Budget tracker records usage from actual LLM calls."""
        mock_call.return_value = ("1. YES\n2. NO", 50, 20)

        from goldenmatch.core.llm_scorer import llm_score_pairs
        cfg = LLMScorerConfig(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            budget=BudgetConfig(max_cost_usd=10.0),
        )
        pairs = [(1, 2, 0.85), (3, 4, 0.80)]
        df = _make_test_df()

        result, budget_summary = llm_score_pairs(
            pairs, df, config=cfg, api_key="fake-key", return_budget=True,
        )
        assert budget_summary["total_calls"] == 1
        assert budget_summary["total_cost_usd"] > 0

    def test_backward_compat_flat_kwargs(self):
        """Old-style flat kwargs still work without config param."""
        from goldenmatch.core.llm_scorer import llm_score_pairs
        pairs = [(1, 2, 0.96)]  # above auto_threshold, no LLM needed
        df = _make_test_df()

        # Pass provider/api_key explicitly so auto-detect doesn't bail out
        result = llm_score_pairs(
            pairs, df, auto_threshold=0.95,
            provider="openai", api_key="fake",
        )
        assert result[0][2] == 1.0  # auto-accepted

    def test_no_budget_returns_plain_list(self):
        """Without return_budget, returns plain list (not tuple)."""
        from goldenmatch.core.llm_scorer import llm_score_pairs
        pairs = [(1, 2, 0.96)]
        df = _make_test_df()

        result = llm_score_pairs(pairs, df, auto_threshold=0.95)
        assert isinstance(result, list)
        assert not isinstance(result, tuple)


class TestEngineStatsLLMCost:
    def test_llm_cost_field_exists(self):
        from goldenmatch.tui.engine import EngineStats
        stats = EngineStats(
            total_records=100,
            total_clusters=10,
            singleton_count=5,
            match_rate=0.9,
            cluster_sizes=[2, 3, 5],
            avg_cluster_size=3.3,
            max_cluster_size=5,
            oversized_count=0,
            llm_cost={"total_cost_usd": 0.42, "total_calls": 10},
        )
        assert stats.llm_cost["total_cost_usd"] == 0.42
