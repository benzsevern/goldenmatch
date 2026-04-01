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
