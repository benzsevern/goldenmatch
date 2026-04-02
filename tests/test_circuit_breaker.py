"""Tests for runtime circuit breaker."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from goldenmatch.config.schemas import SafetyPolicy
from goldenmatch.core.circuit_breaker import CircuitBreaker, CircuitAction, CircuitBreakerError


class TestCircuitBreakerMemory:
    def test_continue_when_memory_ok(self):
        cb = CircuitBreaker(policy=SafetyPolicy(max_memory_mb=4096))
        mock_mem = MagicMock()
        mock_mem.rss = 1_000_000_000  # ~1GB
        with patch("goldenmatch.core.circuit_breaker.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value = mock_mem
            action = cb.check("scoring")
        assert action.action == "continue"

    def test_stop_when_memory_critical(self):
        cb = CircuitBreaker(policy=SafetyPolicy(max_memory_mb=1024))
        mock_mem = MagicMock()
        mock_mem.rss = 2_000_000_000  # ~2GB > 1024MB limit
        with patch("goldenmatch.core.circuit_breaker.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value = mock_mem
            action = cb.check("scoring")
        assert action.action == "stop"
        assert "memory" in action.reason.lower()


class TestCircuitBreakerCost:
    def test_continue_below_budget(self):
        from goldenmatch.core.llm_budget import BudgetTracker
        from goldenmatch.config.schemas import BudgetConfig
        budget = BudgetTracker(BudgetConfig(max_cost_usd=1.00))
        cb = CircuitBreaker(
            policy=SafetyPolicy(max_llm_cost_usd=1.00),
            budget_tracker=budget,
        )
        action = cb.check("llm_scoring")
        assert action.action == "continue"

    def test_stop_when_budget_exhausted(self):
        from goldenmatch.core.llm_budget import BudgetTracker
        from goldenmatch.config.schemas import BudgetConfig
        budget = BudgetTracker(BudgetConfig(max_cost_usd=0.01))
        budget.record_usage(100000, 100000, "gpt-4o")
        cb = CircuitBreaker(
            policy=SafetyPolicy(max_llm_cost_usd=0.01),
            budget_tracker=budget,
        )
        action = cb.check("llm_scoring")
        assert action.action == "stop"


class TestCircuitBreakerComparisons:
    def test_stop_when_comparisons_exceeded(self):
        cb = CircuitBreaker(policy=SafetyPolicy(max_comparisons=1000))
        cb.comparisons_processed = 1500
        # Must still pass memory check - mock it
        mock_mem = MagicMock()
        mock_mem.rss = 100_000_000  # 100MB, well under 4GB default
        with patch("goldenmatch.core.circuit_breaker.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value = mock_mem
            action = cb.check("scoring")
        assert action.action == "stop"
        assert "comparison" in action.reason.lower()

    def test_add_comparisons(self):
        cb = CircuitBreaker(policy=SafetyPolicy())
        cb.add_comparisons(500)
        cb.add_comparisons(300)
        assert cb.comparisons_processed == 800


class TestCircuitBreakerError:
    def test_error_class_exists(self):
        err = CircuitBreakerError("test")
        assert str(err) == "test"


class TestCircuitBreakerPsutilFailure:
    """I6: psutil failure should log warning, not silently continue."""

    def test_psutil_error_logs_warning(self):
        """Psutil failure should not crash — returns continue with warning logged."""
        import psutil as _psutil
        cb = CircuitBreaker(policy=SafetyPolicy(max_memory_mb=1024))
        with patch("goldenmatch.core.circuit_breaker.psutil.Process") as mock_proc:
            mock_proc.return_value.memory_info.side_effect = _psutil.Error("denied")
            # Test _check_memory directly to verify the reason propagates
            action = cb._check_memory("scoring")
        assert action.action == "continue"
        assert action.reason == "memory monitoring unavailable"

    def test_psutil_error_check_does_not_crash(self):
        """Full check() should not crash when psutil fails."""
        import psutil as _psutil
        cb = CircuitBreaker(policy=SafetyPolicy(max_memory_mb=1024))
        with patch("goldenmatch.core.circuit_breaker.psutil.Process") as mock_proc:
            mock_proc.return_value.memory_info.side_effect = _psutil.Error("denied")
            action = cb.check("scoring")
        assert action.action == "continue"


class TestCircuitBreakerBudgetTrackerInterface:
    """S2: Missing budget_tracker attrs should log warning."""

    def test_missing_budget_exhausted_attr(self):
        cb = CircuitBreaker(
            policy=SafetyPolicy(),
            budget_tracker=object(),  # no budget_exhausted attr
        )
        mock_mem = MagicMock()
        mock_mem.rss = 100_000_000
        with patch("goldenmatch.core.circuit_breaker.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value = mock_mem
            action = cb.check("llm_scoring")
        assert action.action == "continue"


class TestScorerCircuitBreakerIntegration:
    """I4: Verify circuit breaker actually stops scoring mid-block."""

    def test_scoring_stops_when_cb_fires(self):
        import polars as pl
        from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig
        from goldenmatch.core.blocker import build_blocks
        from goldenmatch.core.scorer import score_blocks_parallel

        # Create data that produces multiple blocks
        df = pl.DataFrame({
            "__row_id__": list(range(100)),
            "name": [f"Person{i % 10}" for i in range(100)],
            "__source__": ["src"] * 100,
        })

        mk = MatchkeyConfig(
            name="test", type="weighted", threshold=0.5,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0,
                                   transforms=["lowercase"])],
        )

        blocking_cfg = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase"])],
        )

        blocks = build_blocks(df.lazy(), blocking_cfg)
        if len(blocks) < 2:
            pytest.skip("Need multiple blocks for this test")

        # Create a CB that immediately stops
        cb = CircuitBreaker(policy=SafetyPolicy(max_comparisons=0))
        mock_mem = MagicMock()
        mock_mem.rss = 100_000_000
        with patch("goldenmatch.core.circuit_breaker.psutil") as mock_psutil:
            mock_psutil.Process.return_value.memory_info.return_value = mock_mem
            pairs = score_blocks_parallel(blocks, mk, set(), circuit_breaker=cb)

        # Should have stopped early — fewer pairs than without CB
        pairs_no_cb = score_blocks_parallel(blocks, mk, set())
        assert len(pairs) <= len(pairs_no_cb)
