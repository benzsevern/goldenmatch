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
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

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
    display_columns: list[str] | None = None,
) -> list[tuple[int, int, float]]:
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

    Returns:
        Updated pairs list. LLM-approved pairs get score=1.0,
        LLM-rejected pairs get score=0.0, others unchanged.
    """
    if not pairs:
        return pairs

    # Auto-detect provider
    if not provider or not api_key:
        provider, api_key = _detect_provider()
        if not provider:
            logger.warning("No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
            return pairs

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
        # Nothing to score — just promote auto-accepts
        result = list(pairs)
        for i in auto_accept:
            a, b, _ = result[i]
            result[i] = (a, b, 1.0)
        return result

    # Score candidates with LLM
    t0 = time.perf_counter()
    llm_results = _batch_score(
        candidates, pairs, row_lookup, cols,
        provider, api_key, model, batch_size,
    )
    elapsed = time.perf_counter() - t0

    n_match = sum(1 for v in llm_results.values() if v)
    logger.info(
        "LLM scored %d pairs in %.1fs: %d matches, %d non-matches",
        len(candidates), elapsed, n_match, len(candidates) - n_match,
    )

    # Build result
    result = list(pairs)
    for i in auto_accept:
        a, b, _ = result[i]
        result[i] = (a, b, 1.0)
    for i in candidates:
        a, b, s = result[i]
        if llm_results.get(i, False):
            result[i] = (a, b, 1.0)
        else:
            result[i] = (a, b, 0.0)
    # Below pairs keep original scores

    return result


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
) -> dict[int, bool]:
    """Score candidate pairs in batches. Returns {pair_index: is_match}."""
    results: dict[int, bool] = {}

    for bi in range(0, len(candidate_indices), batch_size):
        batch_idx = candidate_indices[bi:bi + batch_size]

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

        # Call LLM
        for attempt in range(3):
            try:
                if provider == "openai":
                    answer = _call_openai(prompt, api_key, model, max_tokens=len(batch_idx) * 10)
                else:
                    answer = _call_anthropic(prompt, api_key, model, max_tokens=len(batch_idx) * 10)

                # Parse responses
                batch_results = []
                for line in answer.split("\n"):
                    line = line.strip().upper()
                    if "YES" in line:
                        batch_results.append(True)
                    elif "NO" in line:
                        batch_results.append(False)

                # Pad if parsing missed some
                while len(batch_results) < len(batch_idx):
                    batch_results.append(False)

                for k, idx in enumerate(batch_idx):
                    if k < len(batch_results):
                        results[idx] = batch_results[k]

                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = (attempt + 1) * 5
                    logger.warning("Rate limited, waiting %ds...", wait)
                    time.sleep(wait)
                    continue
                logger.error("LLM API error: %s", e)
                # Mark batch as non-match on failure
                for idx in batch_idx:
                    results[idx] = False
                break

        if (bi // batch_size) % 10 == 0 and bi > 0:
            logger.info("  LLM progress: %d/%d pairs", bi + len(batch_idx), len(candidate_indices))

    return results


def _call_openai(prompt: str, api_key: str, model: str, max_tokens: int = 100) -> str:
    """Call OpenAI chat completions API."""
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
    return result["choices"][0]["message"]["content"].strip()


def _call_anthropic(prompt: str, api_key: str, model: str, max_tokens: int = 100) -> str:
    """Call Anthropic messages API."""
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
    return result["content"][0]["text"].strip()
