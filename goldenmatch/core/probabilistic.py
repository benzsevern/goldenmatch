"""Fellegi-Sunter probabilistic matching with EM-trained parameters.

Implements the classic Fellegi-Sunter model for record linkage:
- Comparison vectors classify field agreements into levels (agree/partial/disagree)
- Expectation-Maximization estimates m-probabilities (P(level|match)) and
  u-probabilities (P(level|non-match)) from unlabeled data
- Match weights are log-likelihood ratios: log2(m/u)
- Thresholds computed from the weight distribution

References:
    Fellegi & Sunter (1969). "A Theory for Record Linkage"
    Winkler (2006). "Overview of Record Linkage and Current Research Directions"
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import polars as pl

from goldenmatch.config.schemas import MatchkeyConfig
from goldenmatch.core.scorer import score_field

logger = logging.getLogger(__name__)


@dataclass
class EMResult:
    """Result of EM training for Fellegi-Sunter model."""

    m_probs: dict[str, list[float]]  # field -> P(level_i | match)
    u_probs: dict[str, list[float]]  # field -> P(level_i | non-match)
    match_weights: dict[str, list[float]]  # field -> log2(m/u) per level
    converged: bool
    iterations: int
    proportion_matched: float  # estimated match rate in the data


def comparison_vector(
    row_a: dict,
    row_b: dict,
    mk: MatchkeyConfig,
) -> list[int]:
    """Compute comparison vector for a pair of records.

    Returns a list of level indices, one per field.
    For 2-level fields: 0=disagree, 1=agree
    For 3-level fields: 0=disagree, 1=partial, 2=agree
    """
    levels = []
    for f in mk.fields:
        val_a = str(row_a.get(f.field, "")) if row_a.get(f.field) is not None else None
        val_b = str(row_b.get(f.field, "")) if row_b.get(f.field) is not None else None
        s = score_field(val_a, val_b, f.scorer)

        if s is None:
            levels.append(0)  # treat nulls as disagree
        elif f.levels == 2:
            levels.append(1 if s >= f.partial_threshold else 0)
        else:  # 3 levels
            if s >= 0.95:
                levels.append(2)  # agree
            elif s >= f.partial_threshold:
                levels.append(1)  # partial
            else:
                levels.append(0)  # disagree
    return levels


def _sample_pairs(
    df: pl.DataFrame,
    n_pairs: int = 10000,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Sample random pairs for EM training."""
    row_ids = df["__row_id__"].to_list()
    rng = random.Random(seed)

    if len(row_ids) < 2:
        return []

    # For small datasets, use all pairs
    max_possible = len(row_ids) * (len(row_ids) - 1) // 2
    if max_possible <= n_pairs:
        return list(combinations(row_ids, 2))

    # Reservoir sampling of random pairs
    pairs = set()
    attempts = 0
    max_attempts = n_pairs * 10
    while len(pairs) < n_pairs and attempts < max_attempts:
        i, j = rng.sample(row_ids, 2)
        pair = (min(i, j), max(i, j))
        pairs.add(pair)
        attempts += 1

    return list(pairs)


def _build_comparison_matrix(
    pairs: list[tuple[int, int]],
    row_lookup: dict[int, dict],
    mk: MatchkeyConfig,
) -> np.ndarray:
    """Build NxF comparison matrix where N=pairs, F=fields."""
    n_pairs = len(pairs)
    n_fields = len(mk.fields)
    matrix = np.zeros((n_pairs, n_fields), dtype=np.int8)

    for i, (a, b) in enumerate(pairs):
        row_a = row_lookup.get(a, {})
        row_b = row_lookup.get(b, {})
        vec = comparison_vector(row_a, row_b, mk)
        matrix[i] = vec

    return matrix


def train_em(
    df: pl.DataFrame,
    mk: MatchkeyConfig,
    n_sample_pairs: int = 10000,
    max_iterations: int = 20,
    convergence: float = 0.001,
    seed: int = 42,
) -> EMResult:
    """Train Fellegi-Sunter model using Expectation-Maximization.

    Args:
        df: DataFrame with __row_id__ and field columns.
        mk: Probabilistic matchkey config.
        n_sample_pairs: Number of random pairs to sample for training.
        max_iterations: Maximum EM iterations.
        convergence: Stop when max change in any probability < this.
        seed: Random seed for pair sampling.

    Returns:
        EMResult with trained m/u probabilities and match weights.
    """
    cols = [f.field for f in mk.fields if f.field != "__record__"]
    row_lookup: dict[int, dict] = {}
    for row in df.select(["__row_id__"] + cols).to_dicts():
        row_lookup[row["__row_id__"]] = row

    # Sample pairs
    pairs = _sample_pairs(df, n_sample_pairs, seed)
    if len(pairs) < 10:
        logger.warning("Too few pairs (%d) for EM training", len(pairs))
        return _fallback_result(mk)

    # Build comparison matrix
    comp_matrix = _build_comparison_matrix(pairs, row_lookup, mk)
    n_pairs = len(pairs)
    n_fields = len(mk.fields)

    # Initialize parameters
    p_match = 0.05  # prior: 5% of pairs are matches
    # Initialize m/u with reasonable priors
    m_probs = {}
    u_probs = {}
    for j, f in enumerate(mk.fields):
        n_levels = f.levels
        if n_levels == 2:
            m_probs[f.field] = [0.1, 0.9]  # matches mostly agree
            u_probs[f.field] = [0.9, 0.1]  # non-matches mostly disagree
        else:
            m_probs[f.field] = [0.05, 0.15, 0.80]  # matches: mostly full agree
            u_probs[f.field] = [0.80, 0.15, 0.05]  # non-matches: mostly disagree

    # EM iterations
    converged = False
    for iteration in range(max_iterations):
        old_m = {k: list(v) for k, v in m_probs.items()}
        old_u = {k: list(v) for k, v in u_probs.items()}

        # E-step: compute posterior P(match | comparison vector)
        posteriors = np.zeros(n_pairs)
        for i in range(n_pairs):
            log_m = 0.0
            log_u = 0.0
            for j, f in enumerate(mk.fields):
                level = comp_matrix[i, j]
                # Clamp to avoid log(0)
                m_val = max(m_probs[f.field][level], 1e-10)
                u_val = max(u_probs[f.field][level], 1e-10)
                log_m += math.log(m_val)
                log_u += math.log(u_val)

            log_match = math.log(max(p_match, 1e-10)) + log_m
            log_nonmatch = math.log(max(1 - p_match, 1e-10)) + log_u

            # Numerically stable softmax
            max_log = max(log_match, log_nonmatch)
            posteriors[i] = math.exp(log_match - max_log) / (
                math.exp(log_match - max_log) + math.exp(log_nonmatch - max_log)
            )

        # M-step: update parameters
        total_match = posteriors.sum()
        total_nonmatch = n_pairs - total_match
        p_match = max(total_match / n_pairs, 1e-6)

        for j, f in enumerate(mk.fields):
            n_levels = f.levels
            new_m = [0.0] * n_levels
            new_u = [0.0] * n_levels

            for level in range(n_levels):
                mask = comp_matrix[:, j] == level
                new_m[level] = (posteriors[mask].sum() + 1e-6) / (total_match + n_levels * 1e-6)
                new_u[level] = ((1 - posteriors[mask]).sum() + 1e-6) / (total_nonmatch + n_levels * 1e-6)

            m_probs[f.field] = new_m
            u_probs[f.field] = new_u

        # Check convergence
        max_delta = 0.0
        for f in mk.fields:
            for k in range(f.levels):
                max_delta = max(max_delta, abs(m_probs[f.field][k] - old_m[f.field][k]))
                max_delta = max(max_delta, abs(u_probs[f.field][k] - old_u[f.field][k]))

        if max_delta < convergence:
            converged = True
            logger.info("EM converged after %d iterations (delta=%.6f)", iteration + 1, max_delta)
            break

    if not converged:
        logger.warning("EM did not converge after %d iterations (delta=%.6f)", max_iterations, max_delta)

    # Compute match weights: log2(m/u)
    match_weights = {}
    for f in mk.fields:
        weights = []
        for k in range(f.levels):
            m_val = max(m_probs[f.field][k], 1e-10)
            u_val = max(u_probs[f.field][k], 1e-10)
            weights.append(math.log2(m_val / u_val))
        match_weights[f.field] = weights

    return EMResult(
        m_probs=m_probs,
        u_probs=u_probs,
        match_weights=match_weights,
        converged=converged,
        iterations=min(iteration + 1, max_iterations) if not converged else iteration + 1,
        proportion_matched=p_match,
    )


def _fallback_result(mk: MatchkeyConfig) -> EMResult:
    """Return a fallback EMResult when EM can't be trained."""
    m_probs = {}
    u_probs = {}
    match_weights = {}
    for f in mk.fields:
        if f.levels == 2:
            m_probs[f.field] = [0.1, 0.9]
            u_probs[f.field] = [0.9, 0.1]
            match_weights[f.field] = [math.log2(0.1 / 0.9), math.log2(0.9 / 0.1)]
        else:
            m_probs[f.field] = [0.05, 0.15, 0.80]
            u_probs[f.field] = [0.80, 0.15, 0.05]
            match_weights[f.field] = [
                math.log2(0.05 / 0.80),
                math.log2(0.15 / 0.15),
                math.log2(0.80 / 0.05),
            ]
    return EMResult(
        m_probs=m_probs, u_probs=u_probs, match_weights=match_weights,
        converged=False, iterations=0, proportion_matched=0.05,
    )


def compute_thresholds(em_result: EMResult) -> tuple[float, float]:
    """Compute link and review thresholds from EM result.

    Returns (link_threshold, review_threshold) as normalized 0-1 scores.
    link_threshold: pairs above this are matches
    review_threshold: pairs between review and link are uncertain
    """
    # Compute max possible weight (all fields agree at highest level)
    max_weight = 0.0
    min_weight = 0.0
    for field_name, weights in em_result.match_weights.items():
        max_weight += max(weights)
        min_weight += min(weights)

    weight_range = max_weight - min_weight
    if weight_range == 0:
        return 0.85, 0.60

    # Link threshold at ~85th percentile of weight range
    link_raw = min_weight + 0.85 * weight_range
    # Review threshold at ~60th percentile
    review_raw = min_weight + 0.60 * weight_range

    # Normalize to 0-1
    link_norm = (link_raw - min_weight) / weight_range
    review_norm = (review_raw - min_weight) / weight_range

    return round(link_norm, 4), round(review_norm, 4)


def score_probabilistic(
    block_df: pl.DataFrame,
    mk: MatchkeyConfig,
    em_result: EMResult,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, float]]:
    """Score pairs in a block using Fellegi-Sunter match weights.

    Returns pairs above the link threshold as (row_id_a, row_id_b, normalized_score).
    Score is normalized to 0-1 range for compatibility with the rest of the pipeline.
    """
    if exclude_pairs is None:
        exclude_pairs = set()

    # Build row lookup
    cols = [f.field for f in mk.fields if f.field != "__record__"]
    row_lookup: dict[int, dict] = {}
    for row in block_df.select(["__row_id__"] + cols).to_dicts():
        row_lookup[row["__row_id__"]] = row

    row_ids = block_df["__row_id__"].to_list()

    # Compute weight range for normalization
    max_weight = sum(max(em_result.match_weights[f.field]) for f in mk.fields)
    min_weight = sum(min(em_result.match_weights[f.field]) for f in mk.fields)
    weight_range = max_weight - min_weight

    # Determine threshold
    if mk.link_threshold is not None:
        link_threshold = mk.link_threshold
    else:
        link_threshold, _ = compute_thresholds(em_result)

    results = []
    for i in range(len(row_ids)):
        for j in range(i + 1, len(row_ids)):
            a, b = row_ids[i], row_ids[j]
            pair_key = (min(a, b), max(a, b))
            if pair_key in exclude_pairs:
                continue

            row_a = row_lookup.get(a, {})
            row_b = row_lookup.get(b, {})
            vec = comparison_vector(row_a, row_b, mk)

            # Sum match weights
            total_weight = 0.0
            for k, f in enumerate(mk.fields):
                total_weight += em_result.match_weights[f.field][vec[k]]

            # Normalize to 0-1
            if weight_range > 0:
                normalized = (total_weight - min_weight) / weight_range
            else:
                normalized = 0.5

            if normalized >= link_threshold:
                results.append((a, b, round(normalized, 4)))

    return results


def score_pair_probabilistic(
    row_a: dict,
    row_b: dict,
    mk: MatchkeyConfig,
    em_result: EMResult,
) -> float:
    """Score a single pair using Fellegi-Sunter weights. For match_one."""
    vec = comparison_vector(row_a, row_b, mk)

    max_weight = sum(max(em_result.match_weights[f.field]) for f in mk.fields)
    min_weight = sum(min(em_result.match_weights[f.field]) for f in mk.fields)
    weight_range = max_weight - min_weight

    total_weight = 0.0
    for k, f in enumerate(mk.fields):
        total_weight += em_result.match_weights[f.field][vec[k]]

    if weight_range > 0:
        return (total_weight - min_weight) / weight_range
    return 0.5
