"""LLM budget tracking -- cost accounting, model tiering, and graceful degradation."""
from __future__ import annotations

import logging

from goldenmatch.config.schemas import BudgetConfig

logger = logging.getLogger(__name__)

# Approximate costs per 1K tokens (USD)
_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # (input_per_1k, output_per_1k)
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4-turbo": (0.01, 0.03),
    "claude-haiku-4-5-20251001": (0.0008, 0.004),
    "claude-sonnet-4-20250514": (0.003, 0.015),
}

_DEFAULT_COST = (0.001, 0.004)


class BudgetTracker:
    """Tracks LLM token usage and cost against a budget."""

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0
        self._total_cost = 0.0
        self._escalation_cost = 0.0
        self._models_used: dict[str, int] = {}

    def can_send(self, estimated_tokens: int) -> bool:
        """Check if we can send a batch without exceeding budget."""
        if self.budget_exhausted:
            return False
        if self._config.max_cost_usd is not None:
            est_cost = self._estimate_cost(estimated_tokens, 0, "gpt-4o-mini")
            if self._total_cost + est_cost > self._config.max_cost_usd:
                return False
        return True

    def record_usage(
        self, input_tokens: int, output_tokens: int, model: str,
    ) -> None:
        """Record token usage from an API call."""
        cost = self._estimate_cost(input_tokens, output_tokens, model)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_calls += 1
        self._total_cost += cost
        self._models_used[model] = self._models_used.get(model, 0) + 1

        if self._config.escalation_model and model == self._config.escalation_model:
            self._escalation_cost += cost

        if (
            self._config.max_cost_usd
            and self._config.warn_at_pct
            and self.budget_remaining_pct <= (100 - self._config.warn_at_pct)
        ):
            logger.warning(
                "LLM budget %.0f%% consumed ($%.4f / $%.2f)",
                100 - self.budget_remaining_pct,
                self._total_cost,
                self._config.max_cost_usd,
            )

    def select_model(self, pair_score: float, default_model: str) -> str:
        """Select model based on pair score and escalation config."""
        if not self._config.escalation_model:
            return default_model

        lo, hi = self._config.escalation_band
        if not (lo <= pair_score <= hi):
            return default_model

        if self._config.max_cost_usd and self._config.escalation_budget_pct:
            max_escalation = (
                self._config.max_cost_usd * (self._config.escalation_budget_pct / 100)
            )
            if self._escalation_cost >= max_escalation:
                return default_model

        return self._config.escalation_model

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost, 6)

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def budget_exhausted(self) -> bool:
        if (
            self._config.max_cost_usd is not None
            and self._total_cost >= self._config.max_cost_usd
        ):
            return True
        if (
            self._config.max_calls is not None
            and self._total_calls >= self._config.max_calls
        ):
            return True
        return False

    @property
    def budget_remaining_pct(self) -> float:
        if self._config.max_cost_usd is not None and self._config.max_cost_usd > 0:
            return max(0.0, 100.0 * (1 - self._total_cost / self._config.max_cost_usd))
        if self._config.max_calls is not None and self._config.max_calls > 0:
            return max(0.0, 100.0 * (1 - self._total_calls / self._config.max_calls))
        return 100.0

    def summary(self) -> dict:
        """Return summary dict for EngineStats / lineage."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "budget_remaining_pct": round(self.budget_remaining_pct, 1),
            "budget_exhausted": self.budget_exhausted,
            "models_used": dict(self._models_used),
        }

    @staticmethod
    def _estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        costs = _MODEL_COSTS.get(model, _DEFAULT_COST)
        return (input_tokens / 1000) * costs[0] + (output_tokens / 1000) * costs[1]
