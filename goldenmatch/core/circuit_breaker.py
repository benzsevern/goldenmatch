"""Runtime circuit breaker — monitors resource usage at pipeline checkpoints."""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

import psutil

from goldenmatch.config.schemas import SafetyPolicy

logger = logging.getLogger(__name__)


@dataclass
class CircuitAction:
    """Result of a circuit breaker check."""
    action: str  # "continue", "downgrade", "stop"
    reason: str | None = None
    component: str | None = None


class CircuitBreakerError(Exception):
    """Raised when circuit breaker decides to stop the pipeline."""
    pass


@dataclass
class CircuitBreaker:
    """Lightweight resource monitor checked at pipeline natural boundaries."""

    policy: SafetyPolicy
    budget_tracker: object | None = None
    comparisons_processed: int = 0
    _warned_memory: bool = field(default=False, repr=False)
    _warned_cost: bool = field(default=False, repr=False)

    def check(self, stage: str) -> CircuitAction:
        """Check resource usage at a pipeline checkpoint."""
        mem_action = self._check_memory(stage)
        if mem_action.action != "continue":
            return mem_action

        if self.comparisons_processed > self.policy.max_comparisons:
            return CircuitAction(
                action="stop",
                reason=f"Comparison count {self.comparisons_processed:,} exceeds limit {self.policy.max_comparisons:,}",
                component="scoring",
            )

        if self.budget_tracker is not None:
            cost_action = self._check_cost(stage)
            if cost_action.action != "continue":
                return cost_action

        return CircuitAction(action="continue")

    def _check_memory(self, stage: str) -> CircuitAction:
        try:
            rss_bytes = psutil.Process().memory_info().rss
        except psutil.Error as exc:
            logger.warning(
                "Circuit breaker: cannot read memory at '%s': %s. "
                "Memory safety checks disabled for this checkpoint.",
                stage, exc,
            )
            return CircuitAction(action="continue", reason="memory monitoring unavailable")

        rss_mb = rss_bytes / (1024 * 1024)

        if rss_mb > self.policy.max_memory_mb:
            gc.collect()
            rss_bytes = psutil.Process().memory_info().rss
            rss_mb = rss_bytes / (1024 * 1024)

            if rss_mb > self.policy.max_memory_mb:
                logger.error(
                    "Circuit breaker STOP at '%s': memory %.0fMB exceeds %.0fMB limit",
                    stage, rss_mb, self.policy.max_memory_mb,
                )
                return CircuitAction(
                    action="stop",
                    reason=f"Memory {rss_mb:.0f}MB exceeds {self.policy.max_memory_mb:.0f}MB limit",
                    component="memory",
                )

        warn_threshold = self.policy.max_memory_mb * 0.8
        if rss_mb > warn_threshold and not self._warned_memory:
            self._warned_memory = True
            logger.warning(
                "Circuit breaker WARNING at '%s': memory %.0fMB (%.0f%% of %.0fMB limit)",
                stage, rss_mb, (rss_mb / self.policy.max_memory_mb) * 100, self.policy.max_memory_mb,
            )

        return CircuitAction(action="continue")

    def _check_cost(self, stage: str) -> CircuitAction:
        tracker = self.budget_tracker
        if tracker is None:
            return CircuitAction(action="continue")

        if not hasattr(tracker, "budget_exhausted"):
            logger.warning(
                "Circuit breaker: budget_tracker at '%s' missing 'budget_exhausted'; "
                "cost monitoring disabled",
                stage,
            )
            return CircuitAction(action="continue")

        if tracker.budget_exhausted:
            logger.error("Circuit breaker STOP at '%s': LLM budget exhausted", stage)
            return CircuitAction(
                action="stop",
                reason="LLM budget exhausted",
                component="llm_scorer",
            )

        if hasattr(tracker, "budget_remaining_pct") and not self._warned_cost:
            remaining = tracker.budget_remaining_pct
            if remaining < 20:
                self._warned_cost = True
                logger.warning(
                    "Circuit breaker WARNING at '%s': LLM budget %.0f%% remaining",
                    stage, remaining,
                )

        return CircuitAction(action="continue")

    def add_comparisons(self, count: int) -> None:
        self.comparisons_processed += count
