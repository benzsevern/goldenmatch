"""LLM-powered pair labeling for GoldenMatch boost mode."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Context auto-detection ─────────────────────────────────────────────────

_CONTEXT_MAP = {
    frozenset({"name", "email", "phone"}): "contact list",
    frozenset({"name", "email"}): "contact list",
    frozenset({"name", "phone"}): "contact list",
    frozenset({"title", "manufacturer", "price"}): "product catalog",
    frozenset({"title", "manufacturer"}): "product catalog",
    frozenset({"name", "description", "price"}): "product catalog",
    frozenset({"title", "authors", "venue"}): "publication database",
    frozenset({"title", "authors", "year"}): "publication database",
    frozenset({"address", "city", "state"}): "address database",
}


def detect_context(column_types: dict[str, str]) -> str:
    """Detect dataset context from column types for LLM prompt."""
    type_set = frozenset(column_types.values())
    col_names = frozenset(c.lower() for c in column_types.keys())

    # Try column types first
    for pattern, context in _CONTEXT_MAP.items():
        if pattern <= col_names:
            return context

    # Heuristic from column type values
    types = set(column_types.values())
    if "name" in types and ("email" in types or "phone" in types):
        return "contact list"
    if "description" in types:
        return "product catalog"

    return "dataset"


# ── Prompt building ────────────────────────────────────────────────────────

def build_prompt(
    record_a: dict, record_b: dict, columns: list[str], context: str,
) -> str:
    """Build the LLM prompt for a single pair."""
    lines_a = []
    lines_b = []
    for col in columns:
        val_a = record_a.get(col, "")
        val_b = record_b.get(col, "")
        if val_a is not None:
            lines_a.append(f"  {col}: {val_a}")
        if val_b is not None:
            lines_b.append(f"  {col}: {val_b}")

    return (
        f"You are deduplicating a {context}. Determine if these two records "
        f"refer to the same real-world entity.\n\n"
        f"Record A:\n" + "\n".join(lines_a) + "\n\n"
        f"Record B:\n" + "\n".join(lines_b) + "\n\n"
        f"Same entity? Reply with only: yes or no"
    )


# ── Response parsing ──────────────────────────────────────────────────────

def parse_response(text: str) -> bool | None:
    """Parse LLM response to bool. Returns None if ambiguous."""
    cleaned = text.strip().lower().rstrip(".")
    if cleaned.startswith("y"):
        return True
    if cleaned.startswith("n"):
        return False
    return None


# ── Provider detection ────────────────────────────────────────────────────

def detect_provider() -> tuple[str, str] | None:
    """Detect available LLM provider and API key.

    Returns (provider, api_key) or None if no key found.
    Checks env vars first, then settings file.
    """
    # Env vars first
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return ("anthropic", anthropic_key)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return ("openai", openai_key)

    # Settings file
    try:
        from goldenmatch.config.settings import load_global_settings
        settings = load_global_settings()
        raw = settings.to_dict().get("defaults", {})
        # Check for llm keys in settings (future extension)
    except Exception:
        pass

    return None


def get_default_model(provider: str) -> str:
    """Get default model for a provider."""
    if provider == "anthropic":
        return "claude-haiku-4-5-20251001"
    return "gpt-4o-mini"


# ── Cost estimation ───────────────────────────────────────────────────────

def estimate_cost(n_pairs: int, provider: str) -> float:
    """Estimate cost in USD for labeling n_pairs."""
    cost_per_pair = 0.001  # ~$0.001 for Haiku/4o-mini
    return n_pairs * cost_per_pair


# ── Single pair labeling ──────────────────────────────────────────────────

def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "LLM boost requires the anthropic package. "
            "Install with: pip install goldenmatch[llm]"
        )
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    """Call OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError(
            "LLM boost requires the openai package. "
            "Install with: pip install goldenmatch[llm]"
        )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_llm_with_retry(
    prompt: str, provider: str, api_key: str, model: str, max_retries: int = 3,
) -> str:
    """Call LLM with exponential backoff retry."""
    call_fn = _call_anthropic if provider == "anthropic" else _call_openai

    for attempt in range(max_retries):
        try:
            return call_fn(prompt, api_key, model)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("LLM API error (attempt %d/%d): %s. Retrying in %ds.", attempt + 1, max_retries, e, wait)
            time.sleep(wait)
    return ""  # unreachable


def label_pair(
    record_a: dict, record_b: dict, columns: list[str],
    context: str, provider: str, api_key: str, model: str,
) -> bool | None:
    """Ask the LLM if two records are the same entity."""
    prompt = build_prompt(record_a, record_b, columns, context)
    response = _call_llm_with_retry(prompt, provider, api_key, model)
    result = parse_response(response)

    if result is None:
        # Retry with stricter prompt
        strict_prompt = prompt + "\n\nYou must reply with exactly one word: yes or no"
        response = _call_llm_with_retry(strict_prompt, provider, api_key, model)
        result = parse_response(response)

    return result


# ── Batch labeling ────────────────────────────────────────────────────────

_PARTIAL_LABELS_FILE = ".goldenmatch_labels_partial.json"


def label_pairs(
    pairs: list[tuple[dict, dict]],
    columns: list[str],
    context: str,
    provider: str,
    api_key: str,
    model: str,
    progress_callback=None,
) -> list[bool]:
    """Label multiple pairs. Saves progress for crash recovery."""
    partial_path = Path(_PARTIAL_LABELS_FILE)
    labels: list[bool | None] = []
    start_idx = 0

    # Resume from partial if exists
    if partial_path.exists():
        try:
            saved = json.loads(partial_path.read_text())
            labels = saved.get("labels", [])
            start_idx = len(labels)
            logger.info("Resuming labeling from pair %d/%d", start_idx, len(pairs))
        except Exception:
            pass

    for i in range(start_idx, len(pairs)):
        record_a, record_b = pairs[i]
        result = label_pair(record_a, record_b, columns, context, provider, api_key, model)

        if result is None:
            logger.warning("Ambiguous LLM response for pair %d, skipping.", i)
            labels.append(False)  # conservative default
        else:
            labels.append(result)

        # Save progress every 10 pairs
        if (i + 1) % 10 == 0:
            partial_path.write_text(json.dumps({"labels": labels}))

        if progress_callback:
            progress_callback(i + 1, len(pairs))

    # Cleanup partial file
    if partial_path.exists():
        partial_path.unlink()

    return labels
