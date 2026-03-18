"""Threshold auto-tuning for GoldenMatch using Otsu's method."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def suggest_threshold(
    scores: list[float], n_bins: int = 100,
) -> float | None:
    """Find optimal threshold using Otsu's method.

    Finds the threshold that best separates a bimodal score distribution
    into "match" and "non-match" populations by minimizing intra-class
    variance.

    Returns None if the distribution is unimodal (no clear separation)
    or if there are too few scores.
    """
    if len(scores) < 2:
        return None

    arr = np.array(scores, dtype=np.float64)

    if arr.std() < 1e-10:
        return None

    counts, bin_edges = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()

    best_between = 0.0
    best_thresholds: list[float] = []

    cum_count = 0
    cum_sum = 0.0
    total_sum = sum(counts[i] * (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins))

    for i in range(n_bins):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        cum_count += counts[i]
        cum_sum += counts[i] * bin_center

        if cum_count == 0 or cum_count == total:
            continue

        w0 = cum_count / total
        w1 = 1.0 - w0
        mean0 = cum_sum / cum_count
        mean1 = (total_sum - cum_sum) / (total - cum_count)

        between_variance = w0 * w1 * (mean0 - mean1) ** 2

        if between_variance > best_between + 1e-12:
            best_between = between_variance
            best_thresholds = [bin_edges[i + 1]]
        elif abs(between_variance - best_between) < 1e-12 and best_between > 0:
            best_thresholds.append(bin_edges[i + 1])

    if not best_thresholds:
        return None

    # Pick median threshold when there's a plateau
    best_threshold = best_thresholds[len(best_thresholds) // 2]

    # Check if distribution is truly bimodal
    total_variance = arr.var()
    if total_variance < 1e-10:
        return None

    variance_ratio = best_between / total_variance

    if variance_ratio < 0.70:
        logger.info(
            "Score distribution is unimodal (variance ratio %.2f). "
            "Using configured threshold as fallback.",
            variance_ratio,
        )
        return None

    logger.info(
        "Auto-threshold: %.3f (variance ratio: %.2f, %d scores)",
        best_threshold,
        variance_ratio,
        len(scores),
    )
    return float(best_threshold)
