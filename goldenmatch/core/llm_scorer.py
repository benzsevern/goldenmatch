"""LLM scorer -- use GPT/Claude to score borderline record pairs.

Sends pairs to an LLM with "Are these the same entity?" and uses the
yes/no response as the match decision. Dramatically outperforms embedding
similarity and cross-encoder approaches on product matching.

Usage in config:
    llm_scorer:
      enabled: true
      provider: openai          # openai or anthropic
      model: gpt-4o-mini        # or claude-haiku
      auto_threshold: 0.95      # auto-accept pairs above this
      candidate_range: [0.75, 0.95]  # score range to send to LLM
      budget:
        max_cost_usd: 5.00
        max_calls: 500
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

logger = logging.getLogger(__name__)


def llm_score_pairs(
    pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    auto_threshold: float = 0.95,
    candidate_lo: float = 0.75,
    candidate_hi: float = 0.95,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    batch_size: int = 20,
    max_workers: int = 5,
    display_columns: list[str] | None = None,
    config: "LLMScorerConfig | None" = None,
    return_budget: bool = False,
    circuit_breaker: object | None = None,
) -> "list[tuple[int, int, float]] | tuple[list[tuple[int, int, float]], dict | None]":
    """Score borderline pairs with an LLM.

    Three-tier approach:
    1. Pairs above auto_threshold: auto-accept (score = 1.0)
    2. Pairs in [candidate_lo, candidate_hi]: send to LLM for yes/no
    3. Pairs below candidate_lo: auto-reject (keep original score)

    Args:
        pairs: Scored pairs (row_id_a, row_id_b, score).
        df: DataFrame with record data.
        auto_threshold: Score above which pairs are auto-accepted.
        candidate_lo: Lower bound of LLM scoring range.
        candidate_hi: Upper bound (same as auto_threshold by default).
        provider: "openai" or "anthropic". Auto-detected from env vars.
        api_key: API key. Auto-detected from env vars.
        model: Model name. Defaults to gpt-4o-mini or claude-haiku.
        batch_size: Pairs per LLM request.
        display_columns: Columns to show the LLM. Defaults to all non-internal.
        config: LLMScorerConfig object. When provided, overrides individual kwargs.
        return_budget: If True, return (pairs, budget_summary) tuple.

    Returns:
        Updated pairs list (or tuple with budget summary if return_budget=True).
        LLM-approved pairs get score=1.0. LLM-rejected or unconfirmed pairs
        keep their original fuzzy score (never demoted).
    """
    from goldenmatch.config.schemas import LLMScorerConfig

    # Resolve config -> individual params
    calibration_sample_size = 100
    calibration_max_rounds = 5
    calibration_convergence_delta = 0.01
    if config is not None:
        auto_threshold = config.auto_threshold
        candidate_lo = config.candidate_lo
        candidate_hi = config.candidate_hi
        batch_size = config.batch_size
        max_workers = config.max_workers
        calibration_sample_size = config.calibration_sample_size
        calibration_max_rounds = config.calibration_max_rounds
        calibration_convergence_delta = config.calibration_convergence_delta
        if config.provider:
            provider = config.provider
        if config.model:
            model = config.model

    # Set up budget tracker
    budget = None
    if config is not None and config.budget is not None:
        from goldenmatch.core.llm_budget import BudgetTracker
        budget = BudgetTracker(config.budget)

    def _return(result):
        if return_budget:
            return result, budget.summary() if budget else None
        return result

    if not pairs:
        return _return(pairs)

    # Auto-detect provider
    if not provider or not api_key:
        detected_provider, detected_key = _detect_provider()
        if not provider:
            provider = detected_provider
        if not api_key:
            api_key = detected_key
        if not provider:
            logger.warning("No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
            return _return(pairs)

    if not model:
        model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5-20251001"

    # Build row lookup
    cols = display_columns or [c for c in df.columns if not c.startswith("__")]
    row_lookup: dict[int, dict] = {}
    for row in df.select(["__row_id__"] + cols).to_dicts():
        row_lookup[row["__row_id__"]] = row

    # Classify pairs into tiers
    auto_accept = []
    candidates = []
    below = []

    for i, (a, b, s) in enumerate(pairs):
        if s >= auto_threshold:
            auto_accept.append(i)
        elif s >= candidate_lo:
            candidates.append(i)
        else:
            below.append(i)

    logger.info(
        "LLM scorer: %d auto-accept (>%.2f), %d candidates (%.2f-%.2f), %d below",
        len(auto_accept), auto_threshold, len(candidates), candidate_lo, candidate_hi, len(below),
    )

    if not candidates:
        result = list(pairs)
        for i in auto_accept:
            a, b, _ = result[i]
            result[i] = (a, b, 1.0)
        return _return(result)

    # Check budget before starting
    if budget and budget.budget_exhausted:
        logger.info("LLM budget exhausted before scoring. Candidates keep fuzzy scores.")
        result = list(pairs)
        for i in auto_accept:
            a, b, _ = result[i]
            result[i] = (a, b, 1.0)
        return _return(result)

    # Score candidates with LLM
    t0 = time.perf_counter()

    if len(candidates) > calibration_sample_size:
        # Iterative calibration path
        learned_threshold, llm_results = _iterative_calibrate(
            candidates, pairs, row_lookup, cols,
            provider, api_key, model, batch_size,
            budget=budget, max_workers=max_workers,
            sample_size=calibration_sample_size,
            max_rounds=calibration_max_rounds,
            convergence_delta=calibration_convergence_delta,
            candidate_lo=candidate_lo,
            candidate_hi=candidate_hi,
        )

        elapsed = time.perf_counter() - t0

        # Build result: auto-accept
        result = list(pairs)
        for i in auto_accept:
            a, b, _ = result[i]
            result[i] = (a, b, 1.0)

        # Apply LLM labels to sampled pairs
        for i, is_match in llm_results.items():
            if is_match:
                a, b, _ = result[i]
                result[i] = (a, b, 1.0)
            # else: keep original fuzzy score (never demote)

        # Apply learned threshold to unsampled pairs
        n_promoted = 0
        n_unchanged = 0
        for i in candidates:
            if i not in llm_results:
                if pairs[i][2] >= learned_threshold:
                    a, b, _ = result[i]
                    result[i] = (a, b, 1.0)
                    n_promoted += 1
                else:
                    n_unchanged += 1

        logger.info(
            "LLM calibration applied in %.1fs: %d promoted, %d unchanged",
            elapsed, n_promoted + sum(1 for m in llm_results.values() if m), n_unchanged,
        )
    else:
        # Direct scoring path: few candidates, score them all
        llm_results = _batch_score(
            candidates, pairs, row_lookup, cols,
            provider, api_key, model, batch_size,
            budget=budget, max_workers=max_workers,
            circuit_breaker=circuit_breaker,
        )
        elapsed = time.perf_counter() - t0

        n_match = sum(1 for v in llm_results.values() if v)
        logger.info(
            "LLM scored %d pairs in %.1fs: %d matches, %d non-matches",
            len(llm_results), elapsed, n_match, len(llm_results) - n_match,
        )

        # Build result
        result = list(pairs)
        for i in auto_accept:
            a, b, _ = result[i]
            result[i] = (a, b, 1.0)
        for i in candidates:
            if i in llm_results and llm_results[i]:
                a, b, _ = result[i]
                result[i] = (a, b, 1.0)
            # else: keep original fuzzy score (never demote)

    return _return(result)


def _detect_provider() -> tuple[str | None, str | None]:
    """Auto-detect LLM provider from environment variables."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return "openai", key
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return "anthropic", key
    return None, None


def _batch_score(
    candidate_indices: list[int],
    pairs: list[tuple[int, int, float]],
    row_lookup: dict[int, dict],
    cols: list[str],
    provider: str,
    api_key: str,
    model: str,
    batch_size: int,
    budget: "BudgetTracker | None" = None,
    max_workers: int = 5,
    circuit_breaker: object | None = None,
) -> dict[int, bool]:
    """Score candidate pairs in batches with concurrent requests.

    Returns {pair_index: is_match}.
    """
    results: dict[int, bool] = {}

    # Pre-build all batch slices
    all_batches: list[list[int]] = []
    for bi in range(0, len(candidate_indices), batch_size):
        all_batches.append(candidate_indices[bi:bi + batch_size])

    total_pairs = len(candidate_indices)

    def _score_one_batch(batch_idx: list[int]) -> dict[int, bool]:
        """Score a single batch via LLM. Called from worker threads."""
        # Check budget (thread-safe via lock in BudgetTracker)
        if budget:
            estimated_tokens = len(batch_idx) * 80
            if not budget.can_send(estimated_tokens):
                logger.debug("Budget exhausted, skipping batch of %d pairs.", len(batch_idx))
                return {}

        # Build prompt
        prompt_parts = [
            "For each numbered pair, answer YES if they are the same entity/product, "
            "NO if they are different. Answer with just the number and YES/NO, one per line.\n"
        ]
        for k, idx in enumerate(batch_idx):
            a, b, s = pairs[idx]
            row_a = row_lookup.get(a, {})
            row_b = row_lookup.get(b, {})
            text_a = " | ".join(f"{c}: {row_a.get(c, '')}" for c in cols if row_a.get(c))[:200]
            text_b = " | ".join(f"{c}: {row_b.get(c, '')}" for c in cols if row_b.get(c))[:200]
            prompt_parts.append(f"{k+1}. A: {text_a}\n   B: {text_b}")

        prompt = "\n".join(prompt_parts)

        # Call LLM with retries
        for attempt in range(3):
            try:
                if provider == "openai":
                    answer, in_tok, out_tok = _call_openai(
                        prompt, api_key, model, max_tokens=len(batch_idx) * 10,
                    )
                else:
                    answer, in_tok, out_tok = _call_anthropic(
                        prompt, api_key, model, max_tokens=len(batch_idx) * 10,
                    )

                if budget:
                    budget.record_usage(
                        input_tokens=in_tok, output_tokens=out_tok, model=model,
                    )

                # Parse responses
                batch_results = []
                for line in answer.split("\n"):
                    line = line.strip().upper()
                    if "YES" in line:
                        batch_results.append(True)
                    elif "NO" in line:
                        batch_results.append(False)

                while len(batch_results) < len(batch_idx):
                    batch_results.append(False)

                return {
                    idx: batch_results[k]
                    for k, idx in enumerate(batch_idx)
                    if k < len(batch_results)
                }
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = (attempt + 1) * 5
                    logger.warning("Rate limited, waiting %ds...", wait)
                    time.sleep(wait)
                    continue
                logger.error("LLM API error: %s", e)
                return {}
            except (urllib.error.URLError, OSError) as e:
                if attempt < 2:
                    wait = (attempt + 1) * 5
                    logger.warning("LLM network error: %s. Retrying in %ds...", e, wait)
                    time.sleep(wait)
                    continue
                logger.error("LLM network error (unrecoverable): %s", e)
                return {}

        return {}

    # Sequential fast path for small workloads
    if len(all_batches) <= 2 or max_workers <= 1:
        for bi, batch_idx in enumerate(all_batches):
            batch_result = _score_one_batch(batch_idx)
            if not batch_result and budget and budget.budget_exhausted:
                logger.info("LLM budget exhausted after %d/%d pairs.",
                            bi * batch_size, total_pairs)
                break
            results.update(batch_result)
            if (bi + 1) % 10 == 0:
                logger.info("  LLM progress: %d/%d pairs",
                            min((bi + 1) * batch_size, total_pairs), total_pairs)
        return results

    # Concurrent path
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for batch_idx in all_batches:
            # Pre-check budget before submitting (avoid wasting threads)
            if budget and budget.budget_exhausted:
                logger.info("LLM budget exhausted, skipping remaining %d batches.",
                            len(all_batches) - len(futures))
                break
            fut = pool.submit(_score_one_batch, batch_idx)
            futures[fut] = batch_idx

        for fut in as_completed(futures):
            try:
                batch_result = fut.result()
            except Exception:
                batch_idx = futures[fut]
                logger.error(
                    "LLM batch scoring failed for %d pairs. Skipping batch.",
                    len(batch_idx), exc_info=True,
                )
                continue
            results.update(batch_result)
            completed += 1
            if completed % 10 == 0 or completed == len(futures):
                logger.info("  LLM progress: %d/%d batches (%d pairs scored)",
                            completed, len(futures), len(results))

    if circuit_breaker is not None:
        action = circuit_breaker.check("llm_scoring")
        if action.action == "stop":
            logger.warning("Circuit breaker stopped LLM scoring: %s", action.reason)
            return results

    return results


def _compute_threshold(labels: dict[int, tuple[float, bool]]) -> float:
    """Find optimal threshold separating YES from NO labels via grid search.

    Args:
        labels: {pair_index: (fuzzy_score, is_match)} from LLM responses.

    Returns:
        Optimal threshold score. Pairs at or above this are matches.
    """
    yes_scores = [s for s, m in labels.values() if m]
    no_scores = [s for s, m in labels.values() if not m]

    if not yes_scores:
        return max(s for s, _ in labels.values()) + 0.001
    if not no_scores:
        return min(s for s, _ in labels.values()) - 0.001

    if max(no_scores) < min(yes_scores):
        return (max(no_scores) + min(yes_scores)) / 2

    all_scores = sorted(set(yes_scores + no_scores))
    best_threshold = (max(no_scores) + min(yes_scores)) / 2
    best_cost = float("inf")

    for i in range(len(all_scores) - 1):
        candidate = (all_scores[i] + all_scores[i + 1]) / 2
        cost = 0
        for score, is_match in labels.values():
            if is_match and score < candidate:
                cost += 1
            elif not is_match and score >= candidate:
                cost += 1
        if cost < best_cost:
            best_cost = cost
            best_threshold = candidate

    return best_threshold


def _stratified_sample(
    candidate_indices: list[int],
    pairs: list[tuple[int, int, float]],
    sample_size: int,
    score_lo: float,
    score_hi: float,
    already_scored: set[int],
) -> list[int]:
    """Stratified sample across the score range for round 1."""
    import random as _random

    available = [i for i in candidate_indices if i not in already_scored]
    if len(available) <= sample_size:
        return available

    score_range = score_hi - score_lo
    bin_width = max(0.01, score_range / 20)
    n_bins = max(1, int(score_range / bin_width))

    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i in available:
        score = pairs[i][2]
        bin_idx = max(0, min(int((score - score_lo) / bin_width), n_bins - 1))
        bins[bin_idx].append(i)

    total_available = len(available)
    result: list[int] = []
    remaining_quota = sample_size

    for b in bins:
        if not b or remaining_quota <= 0:
            continue
        quota = max(1, round(sample_size * len(b) / total_available))
        quota = min(quota, remaining_quota, len(b))
        result.extend(_random.sample(b, quota))
        remaining_quota -= quota

    if len(result) < sample_size:
        remaining = [i for i in available if i not in set(result)]
        extra = min(sample_size - len(result), len(remaining))
        if extra > 0:
            result.extend(_random.sample(remaining, extra))

    return result[:sample_size]


def _focused_sample(
    candidate_indices: list[int],
    pairs: list[tuple[int, int, float]],
    sample_size: int,
    threshold: float,
    band_width: float,
    already_scored: set[int],
) -> list[int]:
    """Focused sample near the learned threshold for rounds 2+."""
    import random as _random

    lo = threshold - band_width
    hi = threshold + band_width

    available = [
        i for i in candidate_indices
        if i not in already_scored and lo <= pairs[i][2] <= hi
    ]

    if len(available) <= sample_size:
        return available

    return _random.sample(available, sample_size)


def _iterative_calibrate(
    candidate_indices: list[int],
    pairs: list[tuple[int, int, float]],
    row_lookup: dict[int, dict],
    cols: list[str],
    provider: str,
    api_key: str,
    model: str,
    batch_size: int,
    budget: "BudgetTracker | None" = None,
    max_workers: int = 5,
    sample_size: int = 100,
    max_rounds: int = 5,
    convergence_delta: float = 0.01,
    candidate_lo: float = 0.75,
    candidate_hi: float = 0.95,
) -> tuple[float, dict[int, bool]]:
    """Iterative LLM calibration: sample, score, learn threshold, repeat.

    Returns (learned_threshold, {pair_index: is_match} for scored pairs).
    """
    all_labels: dict[int, tuple[float, bool]] = {}
    all_llm_results: dict[int, bool] = {}
    already_scored: set[int] = set()
    prev_threshold = (candidate_lo + candidate_hi) / 2

    for round_num in range(1, max_rounds + 1):
        if budget and budget.budget_exhausted:
            logger.info("LLM calibration: budget exhausted before round %d.", round_num)
            break

        if round_num == 1:
            sample = _stratified_sample(
                candidate_indices, pairs, sample_size,
                score_lo=candidate_lo, score_hi=candidate_hi,
                already_scored=already_scored,
            )
        else:
            sample = _focused_sample(
                candidate_indices, pairs, sample_size,
                threshold=prev_threshold, band_width=0.03,
                already_scored=already_scored,
            )

        if not sample:
            logger.info("LLM calibration round %d: no unscored pairs to sample. Stopping.", round_num)
            break

        round_results = _batch_score(
            sample, pairs, row_lookup, cols,
            provider, api_key, model, batch_size,
            budget=budget, max_workers=max_workers,
        )

        if not round_results:
            logger.warning(
                "LLM calibration round %d: no results returned (budget exhausted or API failure). Stopping.",
                round_num,
            )
            break

        for idx, is_match in round_results.items():
            all_labels[idx] = (pairs[idx][2], is_match)
            all_llm_results[idx] = is_match
            already_scored.add(idx)

        if not all_labels:
            break
        threshold = _compute_threshold(all_labels)

        n_match = sum(1 for m in round_results.values() if m)
        n_no = len(round_results) - n_match
        sample_scores = [pairs[i][2] for i in sample]
        score_lo_round = min(sample_scores) if sample_scores else 0
        score_hi_round = max(sample_scores) if sample_scores else 0

        if round_num == 1:
            logger.info(
                "LLM calibration round %d: %d pairs (%.3f-%.3f), %d match, %d non-match -> threshold %.3f",
                round_num, len(round_results), score_lo_round, score_hi_round,
                n_match, n_no, threshold,
            )
        else:
            delta = abs(threshold - prev_threshold)
            logger.info(
                "LLM calibration round %d: %d pairs (%.3f-%.3f), %d match, %d non-match -> threshold %.3f (delta %.3f)",
                round_num, len(round_results), score_lo_round, score_hi_round,
                n_match, n_no, threshold, delta,
            )

            if abs(threshold - prev_threshold) < convergence_delta:
                logger.info(
                    "LLM calibration converged after %d rounds (%d pairs). Threshold: %.3f",
                    round_num, len(all_llm_results), threshold,
                )
                prev_threshold = threshold
                break

        prev_threshold = threshold

    return prev_threshold, all_llm_results


def _call_openai(
    prompt: str, api_key: str, model: str, max_tokens: int = 100,
) -> tuple[str, int, int]:
    """Call OpenAI chat completions API. Returns (text, input_tokens, output_tokens)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    text = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _call_anthropic(
    prompt: str, api_key: str, model: str, max_tokens: int = 100,
) -> tuple[str, int, int]:
    """Call Anthropic messages API. Returns (text, input_tokens, output_tokens)."""
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    text = result["content"][0]["text"].strip()
    usage = result.get("usage", {})
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
