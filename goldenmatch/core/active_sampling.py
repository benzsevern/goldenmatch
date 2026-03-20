"""Active sampling -- select the most informative pairs for LLM labeling.

Strategies:
1. Uncertainty sampling: pairs where the model is least confident (near 0.5)
2. Disagreement sampling: pairs where different scorers disagree
3. Boundary sampling: pairs near cluster boundaries (would change clustering)
4. Diversity sampling: ensure labeled pairs cover different blocking regions

Active learning loop:
  1. Score all pairs with current model
  2. Select most informative unlabeled pairs
  3. Label with LLM
  4. Retrain model
  5. Repeat until budget exhausted or accuracy plateaus
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def select_active_pairs(
    pairs: list[tuple[int, int, float]],
    features: np.ndarray | None = None,
    current_probs: np.ndarray | None = None,
    labeled_indices: set[int] | None = None,
    n: int = 100,
    strategy: str = "combined",
) -> list[int]:
    """Select the most informative pairs for labeling.

    Args:
        pairs: All candidate pairs (id_a, id_b, score).
        features: Feature matrix (n_pairs, n_features) if available.
        current_probs: Current model's predicted probabilities.
        labeled_indices: Indices already labeled.
        n: Number of pairs to select.
        strategy: "uncertainty", "disagreement", "boundary", "diversity", "combined".

    Returns:
        List of pair indices to label.
    """
    labeled = labeled_indices or set()
    total = len(pairs)

    if total == 0:
        return []

    scores = np.array([s for _, _, s in pairs])

    if strategy == "uncertainty":
        return _uncertainty_sampling(scores, current_probs, labeled, n)
    elif strategy == "disagreement":
        return _disagreement_sampling(features, labeled, n)
    elif strategy == "boundary":
        return _boundary_sampling(scores, labeled, n)
    elif strategy == "diversity":
        return _diversity_sampling(scores, features, labeled, n)
    elif strategy == "combined":
        return _combined_sampling(scores, features, current_probs, labeled, n)
    else:
        # Fallback to stratified random
        return _stratified_random(scores, labeled, n)


def _uncertainty_sampling(
    scores: np.ndarray,
    probs: np.ndarray | None,
    labeled: set[int],
    n: int,
) -> list[int]:
    """Select pairs where the model is least confident.

    If we have model probabilities, use those.
    Otherwise, use raw scores as a proxy (pairs near threshold are most uncertain).
    """
    if probs is not None:
        # Uncertainty = distance from 0.5 (lower = more uncertain)
        uncertainty = np.abs(probs - 0.5)
    else:
        # Use raw score distance from 0.5 as proxy
        uncertainty = np.abs(scores - 0.5)

    # Mask already labeled
    for idx in labeled:
        if idx < len(uncertainty):
            uncertainty[idx] = 999.0

    # Select lowest uncertainty (most informative)
    selected = np.argsort(uncertainty)
    result = []
    for idx in selected:
        if idx not in labeled and len(result) < n:
            result.append(int(idx))
    return result


def _disagreement_sampling(
    features: np.ndarray | None,
    labeled: set[int],
    n: int,
) -> list[int]:
    """Select pairs where different scoring features disagree.

    A pair where jaro_winkler=0.95 but token_sort=0.3 is more informative
    than one where all scorers agree.
    """
    if features is None or features.shape[1] < 2:
        return []

    # Compute variance across features for each pair
    # Higher variance = more disagreement between scorers
    feature_std = np.std(features, axis=1)

    for idx in labeled:
        if idx < len(feature_std):
            feature_std[idx] = -1.0

    selected = np.argsort(-feature_std)  # highest disagreement first
    result = []
    for idx in selected:
        if idx not in labeled and len(result) < n:
            result.append(int(idx))
    return result


def _boundary_sampling(
    scores: np.ndarray,
    labeled: set[int],
    n: int,
    threshold_range: tuple[float, float] = (0.6, 0.9),
) -> list[int]:
    """Select pairs near typical matching thresholds.

    These are the pairs that would change the outcome if the threshold shifted.
    """
    low, high = threshold_range
    mid = (low + high) / 2

    # Distance from the middle of the threshold range
    distance = np.abs(scores - mid)

    for idx in labeled:
        if idx < len(distance):
            distance[idx] = 999.0

    # Prefer pairs IN the threshold range
    in_range = (scores >= low) & (scores <= high)
    for idx in labeled:
        if idx < len(in_range):
            in_range[idx] = False

    # Select in-range pairs first, then closest to range
    in_range_indices = np.where(in_range)[0]
    rng = np.random.default_rng(42)
    if len(in_range_indices) >= n:
        return rng.choice(in_range_indices, n, replace=False).tolist()

    result = list(in_range_indices)
    remaining = n - len(result)

    sorted_by_distance = np.argsort(distance)
    for idx in sorted_by_distance:
        if idx not in labeled and int(idx) not in result and len(result) < n:
            result.append(int(idx))

    return result[:n]


def _diversity_sampling(
    scores: np.ndarray,
    features: np.ndarray | None,
    labeled: set[int],
    n: int,
) -> list[int]:
    """Select diverse pairs that cover different regions of the feature space.

    Uses score-stratified sampling with feature-space diversity.
    """
    rng = np.random.default_rng(42)
    unlabeled = [i for i in range(len(scores)) if i not in labeled]

    if not unlabeled:
        return []

    if len(unlabeled) <= n:
        return unlabeled

    # Stratify by score buckets
    bucket_size = n // 5
    buckets = [
        (0.0, 0.3),   # clear non-matches
        (0.3, 0.5),   # weak non-matches
        (0.5, 0.7),   # uncertain
        (0.7, 0.85),  # likely matches (near threshold)
        (0.85, 1.0),  # strong matches
    ]

    result = []
    for low, high in buckets:
        bucket_indices = [i for i in unlabeled if low <= scores[i] < high]
        if bucket_indices:
            sample = rng.choice(
                bucket_indices,
                min(bucket_size, len(bucket_indices)),
                replace=False,
            ).tolist()
            result.extend(sample)

    # Fill remaining with random unlabeled
    remaining = n - len(result)
    if remaining > 0:
        leftover = [i for i in unlabeled if i not in set(result)]
        if leftover:
            extra = rng.choice(leftover, min(remaining, len(leftover)), replace=False).tolist()
            result.extend(extra)

    return result[:n]


def _combined_sampling(
    scores: np.ndarray,
    features: np.ndarray | None,
    probs: np.ndarray | None,
    labeled: set[int],
    n: int,
) -> list[int]:
    """Combined strategy: mix of uncertainty, disagreement, boundary, and diversity.

    Allocates budget across strategies:
    - 40% uncertainty (most impactful)
    - 20% boundary (threshold-critical)
    - 20% disagreement (conflicting signals)
    - 20% diversity (coverage)
    """
    n_uncertainty = int(n * 0.4)
    n_boundary = int(n * 0.2)
    n_disagreement = int(n * 0.2)
    n_diversity = n - n_uncertainty - n_boundary - n_disagreement

    selected = set()

    # Uncertainty
    uncertain = _uncertainty_sampling(scores, probs, labeled, n_uncertainty)
    selected.update(uncertain)

    # Boundary
    boundary = _boundary_sampling(scores, labeled | selected, n_boundary)
    selected.update(boundary)

    # Disagreement
    if features is not None:
        disagree = _disagreement_sampling(features, labeled | selected, n_disagreement)
        selected.update(disagree)
    else:
        n_diversity += n_disagreement

    # Diversity
    diverse = _diversity_sampling(scores, features, labeled | selected, n_diversity)
    selected.update(diverse)

    return list(selected)[:n]


def _stratified_random(
    scores: np.ndarray,
    labeled: set[int],
    n: int,
) -> list[int]:
    """Fallback: stratified random (30% high, 30% low, 40% middle)."""
    rng = np.random.default_rng(42)
    unlabeled = [i for i in range(len(scores)) if i not in labeled]

    if len(unlabeled) <= n:
        return unlabeled

    high = [i for i in unlabeled if scores[i] >= 0.8]
    low = [i for i in unlabeled if scores[i] < 0.3]
    mid = [i for i in unlabeled if 0.3 <= scores[i] < 0.8]

    n_high = min(int(n * 0.3), len(high))
    n_low = min(int(n * 0.3), len(low))
    n_mid = min(n - n_high - n_low, len(mid))

    result = []
    if high:
        result.extend(rng.choice(high, n_high, replace=False).tolist())
    if low:
        result.extend(rng.choice(low, n_low, replace=False).tolist())
    if mid:
        result.extend(rng.choice(mid, n_mid, replace=False).tolist())

    return result[:n]


def estimate_label_savings(
    total_pairs: int,
    strategy: str = "combined",
) -> dict:
    """Estimate how many labels active sampling saves vs random.

    Returns estimated labels needed and cost savings.
    """
    # Based on literature: active learning typically needs 30-50% fewer labels
    random_labels = min(300, total_pairs)

    savings = {
        "uncertainty": 0.40,
        "disagreement": 0.30,
        "boundary": 0.35,
        "diversity": 0.25,
        "combined": 0.45,
    }

    saving_pct = savings.get(strategy, 0.30)
    active_labels = int(random_labels * (1 - saving_pct))

    return {
        "random_labels_needed": random_labels,
        "active_labels_needed": active_labels,
        "labels_saved": random_labels - active_labels,
        "saving_percentage": f"{saving_pct:.0%}",
        "estimated_cost_random": f"${random_labels * 0.001:.2f}",
        "estimated_cost_active": f"${active_labels * 0.001:.2f}",
    }
