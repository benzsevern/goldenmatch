"""LLM-based feature extraction for low-confidence records.

When heuristic extraction produces low-confidence results, routes
those records to an LLM for structured feature extraction. Operates
at O(N) on the small subset of uncertain records, not O(N^2) on pairs.

Uses the existing BudgetTracker for cost controls.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

import polars as pl

logger = logging.getLogger(__name__)


def llm_extract_features(
    df: pl.DataFrame,
    row_ids: list[int],
    text_column: str,
    domain: str = "product",
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    batch_size: int = 20,
    budget_tracker=None,
) -> dict[int, dict[str, str | None]]:
    """Extract structured features from text using an LLM.

    Args:
        df: DataFrame with __row_id__ and text_column.
        row_ids: Row IDs of low-confidence records to process.
        text_column: Column containing the text to extract from.
        domain: Domain type (product, person, bibliographic, company).
        provider: LLM provider (openai or anthropic).
        api_key: API key. Auto-detected from env if not provided.
        model: Model name. Defaults to gpt-4o-mini.
        batch_size: Records per LLM call.
        budget_tracker: Optional BudgetTracker for cost controls.

    Returns:
        Dict mapping row_id -> extracted features dict.
    """
    if not row_ids:
        return {}

    # Auto-detect provider
    if not provider or not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            provider = "openai"
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                provider = "anthropic"

    if not api_key:
        logger.warning("No API key for LLM extraction. Skipping %d records.", len(row_ids))
        return {}

    if not model:
        model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5-20251001"

    # Build row lookup
    row_lookup = {}
    for row in df.filter(pl.col("__row_id__").is_in(row_ids)).select(["__row_id__", text_column]).to_dicts():
        row_lookup[row["__row_id__"]] = str(row.get(text_column, "") or "")

    # Build prompts per domain
    if domain == "product":
        system_prompt = (
            "Extract structured product features from each line. "
            "Return JSON array with one object per line. Each object has keys: "
            "brand, model, color, specs (string of key specs). "
            "If a field can't be determined, use null. Be precise with model numbers."
        )
    elif domain == "bibliographic":
        system_prompt = (
            "Extract structured bibliographic features from each line. "
            "Return JSON array with one object per line. Each object has keys: "
            "title_normalized, year, venue, first_author_last_name. "
            "If a field can't be determined, use null."
        )
    else:
        system_prompt = (
            "Extract structured entity features from each line. "
            "Return JSON array with one object per line. Each object has keys: "
            "name_normalized, identifier, type. "
            "If a field can't be determined, use null."
        )

    results: dict[int, dict[str, str | None]] = {}

    # Process in batches
    id_list = list(row_ids)
    for bi in range(0, len(id_list), batch_size):
        batch_ids = id_list[bi:bi + batch_size]

        # Check budget
        if budget_tracker:
            estimated_tokens = len(batch_ids) * 100
            if not budget_tracker.can_send(estimated_tokens):
                logger.info("LLM extract budget exhausted after %d/%d records", bi, len(id_list))
                break

        # Build prompt
        lines = []
        for k, rid in enumerate(batch_ids):
            text = row_lookup.get(rid, "")
            lines.append(f"{k+1}. {text[:200]}")

        prompt = system_prompt + "\n\n" + "\n".join(lines)

        # Call LLM
        try:
            if provider == "openai":
                answer, in_tok, out_tok = _call_openai(prompt, api_key, model)
            else:
                answer, in_tok, out_tok = _call_anthropic(prompt, api_key, model)

            if budget_tracker:
                budget_tracker.record_usage(in_tok, out_tok, model)

            # Parse JSON response
            parsed = _parse_json_response(answer, len(batch_ids))
            for k, rid in enumerate(batch_ids):
                if k < len(parsed):
                    results[rid] = parsed[k]

        except Exception as e:
            logger.error("LLM extraction error: %s", e)
            continue

    logger.info("LLM extracted features for %d/%d records", len(results), len(row_ids))
    return results


def _parse_json_response(answer: str, expected_count: int) -> list[dict]:
    """Parse LLM JSON response, handling common formatting issues."""
    # Try direct JSON parse
    answer = answer.strip()
    if answer.startswith("```"):
        # Strip markdown code fences
        lines = answer.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        answer = "\n".join(lines).strip()

    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        pass

    # Try line-by-line JSON
    results = []
    for line in answer.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if results:
        return results

    # Fallback: return empty dicts
    return [{} for _ in range(expected_count)]


def apply_llm_extractions(
    df: pl.DataFrame,
    extractions: dict[int, dict[str, str | None]],
    domain: str = "product",
) -> pl.DataFrame:
    """Merge LLM extractions back into the DataFrame.

    Overwrites the heuristic-extracted __brand__, __model__, etc. columns
    for rows that were LLM-processed.
    """
    if not extractions:
        return df

    if domain == "product":
        col_map = {
            "brand": "__brand__",
            "model": "__model__",
            "color": "__color__",
        }
    elif domain == "bibliographic":
        col_map = {
            "title_normalized": "__title_key__",
        }
    else:
        col_map = {}

    # Build update masks
    for ext_key, df_col in col_map.items():
        if df_col not in df.columns:
            continue

        current = df[df_col].to_list()
        row_ids = df["__row_id__"].to_list()

        for i, rid in enumerate(row_ids):
            if rid in extractions:
                val = extractions[rid].get(ext_key)
                if val is not None:
                    current[i] = str(val)

        df = df.with_columns(pl.Series(df_col, current, dtype=pl.Utf8))

    # Update confidence for LLM-validated records
    if "__extract_confidence__" in df.columns:
        confs = df["__extract_confidence__"].to_list()
        row_ids = df["__row_id__"].to_list()
        for i, rid in enumerate(row_ids):
            if rid in extractions:
                confs[i] = 0.95  # LLM-validated = high confidence
        df = df.with_columns(pl.Series("__extract_confidence__", confs, dtype=pl.Float64))

    return df


def _call_openai(prompt: str, api_key: str, model: str) -> tuple[str, int, int]:
    """Call OpenAI API. Returns (text, input_tokens, output_tokens)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2000,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    text = result["choices"][0]["message"]["content"].strip()
    usage = result.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _call_anthropic(prompt: str, api_key: str, model: str) -> tuple[str, int, int]:
    """Call Anthropic API. Returns (text, input_tokens, output_tokens)."""
    body = json.dumps({
        "model": model, "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={"x-api-key": api_key, "content-type": "application/json", "anthropic-version": "2023-06-01"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    text = result["content"][0]["text"].strip()
    usage = result.get("usage", {})
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
