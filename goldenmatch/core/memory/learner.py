"""MemoryLearner -- threshold tuning and field weight adjustment."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from goldenmatch.core.memory.store import Correction, LearnedAdjustment

if TYPE_CHECKING:
    from goldenmatch.core.memory.store import MemoryStore


class MemoryLearner:
    """Analyzes accumulated corrections and produces adjustments."""

    def __init__(
        self,
        store: MemoryStore,
        threshold_min: int = 10,
        weights_min: int = 50,
    ) -> None:
        self._store = store
        self._threshold_min = threshold_min
        self._weights_min = weights_min

    def has_new_corrections(self) -> bool:
        """True if corrections exist since the last learning pass."""
        last = self._store.last_learn_time()
        if last is None:
            return self._store.count_corrections() > 0
        since = self._store.corrections_since(last)
        return len(since) > 0

    def learn(self, matchkey_name: str | None = None) -> list[LearnedAdjustment]:
        """Run learning pass. Returns list of learned adjustments."""
        all_corrections = self._store.get_corrections()
        if not all_corrections:
            return []

        # Group corrections by matchkey_name (fall back to dataset if not set)
        by_matchkey: dict[str, list[Correction]] = {}
        for c in all_corrections:
            key = c.matchkey_name or c.dataset or "_default"
            if matchkey_name and key != matchkey_name:
                continue
            by_matchkey.setdefault(key, []).append(c)

        results = []
        for mk_name, corrections in by_matchkey.items():
            if len(corrections) < self._threshold_min:
                continue

            approved = [c.original_score for c in corrections if c.decision == "approve"]
            rejected = [c.original_score for c in corrections if c.decision == "reject"]

            if not approved or not rejected:
                continue

            threshold = self._compute_threshold(approved, rejected, corrections)

            # Field weight adjustment (if enough data)
            field_weights = None
            if len(corrections) >= self._weights_min:
                field_weights = self._compute_weights(corrections)

            adj = LearnedAdjustment(
                matchkey_name=mk_name,
                threshold=threshold,
                field_weights=field_weights,
                sample_size=len(corrections),
                learned_at=datetime.now(),
            )
            self._store.save_adjustment(adj)
            results.append(adj)

        return results

    def _compute_threshold(
        self,
        approved_scores: list[float],
        rejected_scores: list[float],
        corrections: list[Correction],
    ) -> float:
        """Find optimal threshold separating approved from rejected."""
        max_rejected = max(rejected_scores)
        min_approved = min(approved_scores)

        if max_rejected < min_approved:
            return (max_rejected + min_approved) / 2

        # Overlapping: grid search over candidate thresholds
        all_scores = sorted(set(approved_scores + rejected_scores))
        best_threshold = (max_rejected + min_approved) / 2
        best_cost = float("inf")

        for i in range(len(all_scores) - 1):
            candidate = (all_scores[i] + all_scores[i + 1]) / 2
            cost = 0.0
            for c in corrections:
                if c.decision == "approve" and c.original_score < candidate:
                    cost += c.trust
                elif c.decision == "reject" and c.original_score >= candidate:
                    cost += c.trust
            if cost < best_cost:
                best_cost = cost
                best_threshold = candidate

        return best_threshold

    def _compute_weights(self, corrections: list[Correction]) -> dict[str, float] | None:
        """Compute field weights from correction patterns.

        Note: Full field-level scoring requires per-field scores which are
        not stored in corrections. Returns None until per-field scores are
        available. Threshold tuning is the primary learning mechanism.
        """
        return None
