"""Auto-configuration engine for GoldenMatch zero-config mode."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    GoldenMatchConfig,
    GoldenRulesConfig,
    MatchkeyConfig,
    MatchkeyField,
    OutputConfig,
)
from goldenmatch.core.profiler import _guess_type

logger = logging.getLogger(__name__)

# ── Column name heuristics ─────────────────────────────────────────────────

_NAME_PATTERNS = re.compile(
    r"(^name$|first.?name|last.?name|full.?name|fname|lname|surname|given.?name)",
    re.IGNORECASE,
)
_EMAIL_PATTERNS = re.compile(r"(email|e.?mail|email.?addr)", re.IGNORECASE)
_PHONE_PATTERNS = re.compile(r"(phone|tel|mobile|fax|cell)", re.IGNORECASE)
_ZIP_PATTERNS = re.compile(r"(zip|postal|postcode|zip.?code)", re.IGNORECASE)
_PRICE_PATTERNS = re.compile(r"(price|cost|amount|revenue|salary|fee|charge|total|balance)", re.IGNORECASE)
_ADDRESS_PATTERNS = re.compile(r"(address|street|addr|line.?1|line.?2)", re.IGNORECASE)
_GEO_PATTERNS = re.compile(r"((?<![a-z])city|^state$|state.?cd|^country$|province|region|(?<![a-z])county)", re.IGNORECASE)
_DATE_PATTERNS = re.compile(r"(date|_dt$|_date$|registr|created|updated|birth.?d|dob)", re.IGNORECASE)
_ID_PATTERNS = re.compile(
    r"^(?i:id|key|code|sku)$|_(?i:id|key)$|(?<=[a-zA-Z])(?:ID|Id)$"
)


@dataclass
class ColumnProfile:
    """Profile of a single column for auto-configuration."""

    name: str
    dtype: str
    col_type: str  # email, name, phone, zip, address, geo, identifier, description, numeric, date, string
    confidence: float  # 0.0 to 1.0
    sample_values: list[str] = field(default_factory=list)
    null_rate: float = 0.0  # fraction of nulls (0-1)
    cardinality_ratio: float = 0.0  # unique values / total rows (0-1)
    avg_len: float = 0.0  # average string length


def _classify_by_name(col_name: str) -> str | None:
    """Phase 1: classify column by name pattern matching.

    Order matters: ID and price before phone/zip to prevent data profiling
    from overriding name-based classification (e.g., 7-digit IDs as phones,
    5-digit prices as zips).
    """
    if _DATE_PATTERNS.search(col_name):
        return "date"
    if _EMAIL_PATTERNS.search(col_name):
        return "email"
    if _ID_PATTERNS.search(col_name):
        return "identifier"
    if _PRICE_PATTERNS.search(col_name):
        return "numeric"
    if _ZIP_PATTERNS.search(col_name):
        return "zip"
    if _GEO_PATTERNS.search(col_name):
        return "geo"
    if _ADDRESS_PATTERNS.search(col_name):
        return "address"
    if _PHONE_PATTERNS.search(col_name):
        return "phone"
    if _NAME_PATTERNS.search(col_name):
        return "name"
    return None


def _classify_by_data(values: list[str]) -> tuple[str, float]:
    """Phase 2: classify column by data profiling. Returns (type, confidence)."""
    if not values:
        return "string", 0.0

    data_type = _guess_type(values)

    # Map profiler types to our types
    type_map = {
        "email": "email",
        "phone": "phone",
        "zip": "zip",
        "state": "geo",
        "numeric": "numeric",
        "name": "name",
        "address": "address",
        "date": "date",
        "text": "string",
    }

    col_type = type_map.get(data_type, "string")

    # Check for description (long freetext)
    if col_type == "string":
        avg_len = sum(len(v) for v in values) / len(values) if values else 0
        if avg_len > 50:
            col_type = "description"

    # Confidence based on how strongly data matches the type
    confidence = 0.7 if col_type != "string" else 0.3
    return col_type, confidence


def profile_columns(
    df: pl.DataFrame, sample_size: int = 1000, max_columns: int = 40,
    llm_provider: str | None = None,
) -> list[ColumnProfile]:
    """Classify columns by type using name heuristics + data profiling.

    Samples randomly to avoid bias from header-adjacent rows.
    Wide datasets (>max_columns) are trimmed: columns matching known patterns
    (name, email, phone, zip, address) are prioritized, then remaining columns
    fill up to the cap.
    """
    # Sample randomly
    if df.height > sample_size:
        sample = df.sample(sample_size, seed=42)
    else:
        sample = df

    # For wide datasets, prioritize columns likely useful for matching
    columns = [c for c in df.columns if not c.startswith("__")]
    if len(columns) > max_columns:
        # Phase 1: keep columns matching known patterns
        priority = []
        rest = []
        for col_name in columns:
            if _classify_by_name(col_name) is not None:
                priority.append(col_name)
            else:
                rest.append(col_name)
        # Fill remaining slots from unmatched columns
        remaining_slots = max(0, max_columns - len(priority))
        columns = priority + rest[:remaining_slots]
        logger.info(
            "Wide dataset (%d columns), auto-configure limited to %d columns "
            "(%d pattern-matched, %d additional)",
            len(df.columns), len(columns), len(priority), remaining_slots,
        )

    profiles = []
    for col_name in columns:
        # Skip internal columns
        if col_name.startswith("__"):
            continue

        dtype = str(df[col_name].dtype)

        # Get non-null string values for profiling
        col_series = sample[col_name]
        total_rows = col_series.len()
        null_count = col_series.null_count()
        null_rate = null_count / total_rows if total_rows > 0 else 0.0

        values = [
            str(v) for v in col_series.drop_nulls().to_list()
            if v is not None and str(v).strip()
        ]

        cardinality_ratio = len(set(values)) / total_rows if total_rows > 0 else 0.0
        avg_len = sum(len(v) for v in values) / len(values) if values else 0.0

        # Phase 1: name heuristics
        name_type = _classify_by_name(col_name)

        # Phase 2: data profiling
        data_type, data_confidence = _classify_by_data(values)

        # Combine: name heuristics are authoritative for structural types
        # (date, geo) because data profiling frequently misclassifies them
        # (e.g., ISO dates look like phone numbers, city names look like person names).
        # For other types, Phase 2 (data) wins when it contradicts Phase 1 (name).
        _name_authoritative = {"date", "geo", "identifier", "numeric"}
        if name_type and name_type in _name_authoritative:
            # Name pattern is authoritative for date/geo — trust it
            col_type = name_type
            confidence = 0.9
        elif name_type and data_type != "string":
            # Both have opinions — Phase 2 wins if types differ
            if name_type == data_type:
                col_type = name_type
                confidence = min(data_confidence + 0.2, 1.0)
            else:
                col_type = data_type
                confidence = data_confidence
        elif name_type:
            col_type = name_type
            confidence = 0.6
        else:
            col_type = data_type
            confidence = data_confidence

        profiles.append(ColumnProfile(
            name=col_name,
            dtype=dtype,
            col_type=col_type,
            confidence=confidence,
            sample_values=values[:5],
            null_rate=null_rate,
            cardinality_ratio=cardinality_ratio,
            avg_len=avg_len,
        ))

    # LLM correction pass for ambiguous columns
    if llm_provider and profiles:
        profiles = _llm_classify_columns(profiles, llm_provider)

    return profiles


def _llm_classify_columns(
    profiles: list[ColumnProfile], provider: str,
) -> list[ColumnProfile]:
    """Use LLM to correct ambiguous column classifications and rank match fields.

    Only sends columns with low confidence or generic types (string, numeric).
    High-confidence classifications (date, geo, email, identifier) are trusted.
    """
    import json as _json
    import urllib.error

    # Filter to ambiguous profiles
    high_confidence_types = {"date", "geo", "email", "identifier"}
    ambiguous = [
        p for p in profiles
        if p.confidence < 0.8 or p.col_type in ("string", "numeric")
        if p.col_type not in high_confidence_types
    ]

    if not ambiguous:
        return profiles

    # Build prompt
    col_lines = []
    for p in ambiguous:
        samples = ", ".join(p.sample_values[:5]) if p.sample_values else "no samples"
        col_lines.append(f'  "{p.name}": [{samples}]')

    all_col_names = [p.name for p in profiles if p.col_type not in high_confidence_types]

    prompt = (
        "You are classifying database columns for entity matching/deduplication.\n\n"
        "For each column below, provide:\n"
        '1. "type": one of: identifier, name, description, numeric, date, geo, '
        "email, phone, zip, address, price, string\n"
        '2. "match_rank": rank the top 5 columns most useful for entity matching '
        "(1=most useful). Only rank columns that would help identify duplicate records.\n\n"
        "Columns with sample values:\n"
        + "\n".join(col_lines)
        + "\n\nAll columns available for ranking: " + ", ".join(all_col_names)
        + '\n\nRespond in JSON: {"classifications": {"col_name": "type", ...}, '
        '"match_ranking": ["col1", "col2", "col3", "col4", "col5"]}'
    )

    try:
        raw = _call_llm_for_blocking(prompt, provider)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError) as e:
        logger.warning("LLM column classification failed: %s. Using heuristics only.", e)
        return profiles

    # Parse response
    try:
        # Extract JSON from response (may be wrapped in markdown)
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = _json.loads(text)
    except (ValueError, IndexError) as e:
        logger.warning(
            "LLM column classification returned unparseable response (error: %s). "
            "Raw response (first 200 chars): %.200s", e, raw,
        )
        return profiles

    # Normalize type aliases
    _type_aliases = {
        "id": "identifier", "ids": "identifier",
        "desc": "description", "text": "string",
        "location": "geo", "city": "geo", "state": "geo",
        "postal": "zip", "postcode": "zip",
        "cost": "numeric", "price": "numeric", "amount": "numeric",
        "tel": "phone", "telephone": "phone",
    }
    valid_types = {
        "identifier", "name", "description", "numeric", "date", "geo",
        "email", "phone", "zip", "address", "string",
    }

    # Apply type corrections
    classifications = data.get("classifications", {})
    profile_by_name = {p.name: p for p in profiles}
    for col_name, llm_type in classifications.items():
        if col_name not in profile_by_name:
            continue
        if not isinstance(llm_type, str):
            continue
        p = profile_by_name[col_name]
        # Only correct ambiguous columns
        if p.col_type in high_confidence_types:
            continue
        normalized = _type_aliases.get(llm_type.lower(), llm_type.lower())
        if normalized in valid_types:
            logger.info("LLM reclassified '%s': %s -> %s", col_name, p.col_type, normalized)
            p.col_type = normalized
            p.confidence = 0.85

    # Apply match ranking (stored as metadata for build_matchkeys to use)
    match_ranking = data.get("match_ranking", [])
    if match_ranking:
        # Store ranking as a special attribute on profiles
        for rank, col_name in enumerate(match_ranking[:5]):
            if col_name in profile_by_name:
                # Use a high utility boost so LLM-ranked fields sort first
                p = profile_by_name[col_name]
                p.cardinality_ratio = max(p.cardinality_ratio, 0.9 - rank * 0.1)
                p.avg_len = max(p.avg_len, 40 - rank * 5)
        logger.info("LLM match ranking: %s", match_ranking[:5])

    return profiles


# ── Scorer and matchkey generation ─────────────────────────────────────────

_SCORER_MAP = {
    "email": ("exact", 1.0, ["lowercase", "strip"]),
    "phone": ("exact", 0.8, ["digits_only"]),
    "zip": ("exact", 0.5, ["strip"]),
    "name": ("ensemble", 1.0, ["lowercase", "strip"]),
    "address": ("token_sort", 0.8, ["lowercase", "strip"]),
    "identifier": ("exact", 1.0, ["strip"]),
    "geo": ("exact", 0.3, ["lowercase", "strip"]),
    "string": ("token_sort", 0.5, ["lowercase", "strip"]),
}

# Domain-extracted column scorer mapping.
# These columns are added by extract_features() and start with __.
_DOMAIN_SCORER_MAP = {
    # Electronics
    "__brand__": ("exact", 0.8, ["lowercase", "strip"]),
    "__model__": ("exact", 1.0, ["strip"]),
    "__model_norm__": ("exact", 1.0, []),
    "__color__": ("exact", 0.2, ["lowercase"]),
    "__specs__": ("token_sort", 0.3, ["strip"]),
    # Software
    "__sw_name__": ("token_sort", 1.0, ["lowercase", "strip"]),
    "__sw_version__": ("exact", 0.5, ["strip"]),
    "__sw_edition__": ("exact", 0.3, ["lowercase"]),
    "__sw_platform__": ("exact", 0.3, ["lowercase"]),
    "__sw_part_num__": ("exact", 1.0, ["strip"]),
    # Bibliographic
    "__title_key__": ("exact", 0.8, ["lowercase"]),
}


def _adaptive_threshold(fields: list[MatchkeyField]) -> float:
    """Compute threshold based on field types in the matchkey."""
    exact_scorers = {"exact"}
    fuzzy_scorers = {"jaro_winkler", "levenshtein", "token_sort", "ensemble", "soundex_match"}
    embedding_scorers = {"embedding", "record_embedding"}

    scorers = {f.scorer for f in fields if f.scorer}

    if scorers <= exact_scorers:
        return 0.95
    if scorers & embedding_scorers:
        return 0.70
    if len(fields) == 1:
        return 0.85
    return 0.80


def build_matchkeys(
    profiles: list[ColumnProfile], df: pl.DataFrame | None = None,
) -> list[MatchkeyConfig]:
    """Generate matchkeys from column profiles."""
    # Separate exact and fuzzy columns
    exact_fields = []
    fuzzy_fields = []
    description_columns = []

    for p in profiles:
        if p.col_type in ("numeric", "date", "identifier"):
            continue  # skip non-matchable columns

        if p.col_type == "description":
            fuzzy_fields.append(MatchkeyField(
                field=p.name,
                scorer="token_sort",
                weight=1.5,  # higher weight ensures survival past max_fuzzy_fields truncation
                transforms=["lowercase", "strip"],
            ))
            description_columns.append(p)
            continue

        scorer_info = _SCORER_MAP.get(p.col_type)
        if not scorer_info:
            continue

        scorer, weight, transforms = scorer_info

        if scorer == "exact" and p.cardinality_ratio > 0 and p.cardinality_ratio < 0.01:
            logger.warning(
                "Skipping exact matchkey for '%s' (cardinality_ratio=%.4f; "
                "too few distinct values — would match nearly everything)",
                p.name, p.cardinality_ratio,
            )
            continue

        # Skip exact matchkeys for large datasets — exact matchkeys do a full
        # self-join which is O(N^2) without blocking. For auto-configure, use
        # exact columns only in blocking (handled by build_blocking).
        if scorer == "exact" and df is not None and df.height > 10000:
            logger.info(
                "Skipping exact matchkey for '%s' (dataset has %d rows; "
                "exact self-joins are O(N^2) — use blocking instead)",
                p.name, df.height,
            )
            continue

        mf = MatchkeyField(
            field=p.name,
            scorer=scorer,
            weight=weight,
            transforms=transforms,
        )

        if scorer == "exact":
            exact_fields.append(mf)
        else:
            fuzzy_fields.append(mf)

    # Warn if all exact-eligible columns were excluded by guards
    _exact_eligible = [
        p for p in profiles
        if p.col_type not in ("numeric", "date", "identifier", "description")
        and _SCORER_MAP.get(p.col_type, (None,))[0] == "exact"
    ]
    if _exact_eligible and not exact_fields:
        logger.warning(
            "All %d exact-eligible columns were excluded by cardinality/size guards. "
            "Falling back to fuzzy-only matchkeys. Consider providing explicit config.",
            len(_exact_eligible),
        )

    matchkeys = []

    # Exact matchkey from exact fields
    if exact_fields:
        for f in exact_fields:
            matchkeys.append(MatchkeyConfig(
                name=f"exact_{f.field}",
                type="exact",
                fields=[MatchkeyField(
                    field=f.field,
                    transforms=f.transforms,
                )],
            ))

    # Weighted matchkey from fuzzy fields
    all_weighted = list(fuzzy_fields)

    # Add description columns as record_embedding
    if description_columns:
        all_weighted.append(MatchkeyField(
            scorer="record_embedding",
            columns=[p.name for p in description_columns],
            weight=1.0,
            model=None,  # auto-selected later
        ))

    # Limit fuzzy fields to prevent OOM on wide datasets
    # Rank by match utility: cardinality * completeness * string length
    max_fuzzy_fields = 5
    if len(all_weighted) > max_fuzzy_fields:
        profile_lookup = {p.name: p for p in profiles}

        def _field_utility(f: MatchkeyField) -> float:
            if not f.field or f.field not in profile_lookup:
                return f.weight or 0.0
            p = profile_lookup[f.field]
            return p.cardinality_ratio * (1 - p.null_rate) * min(p.avg_len / 20, 1.0)

        all_weighted.sort(key=_field_utility, reverse=True)
        dropped = [f.field for f in all_weighted[max_fuzzy_fields:] if f.field]
        all_weighted = all_weighted[:max_fuzzy_fields]
        logger.info(
            "Truncated fuzzy fields from %d to %d. Dropped: %s",
            len(all_weighted) + len(dropped), max_fuzzy_fields, dropped,
        )

    if all_weighted:
        threshold = _adaptive_threshold(all_weighted)
        matchkeys.append(MatchkeyConfig(
            name="fuzzy_match",
            type="weighted",
            threshold=threshold,
            fields=all_weighted,
        ))

    # Fallback: if nothing was generated, use all string columns with token_sort
    if not matchkeys:
        string_cols = [p for p in profiles if p.dtype.startswith("String") or p.dtype.startswith("Utf8")]
        if string_cols:
            fields = [
                MatchkeyField(
                    field=p.name,
                    scorer="token_sort",
                    weight=1.0,
                    transforms=["lowercase", "strip"],
                )
                for p in string_cols[:3]  # limit to first 3
            ]
            matchkeys.append(MatchkeyConfig(
                name="fallback_fuzzy",
                type="weighted",
                threshold=0.80,
                fields=fields,
            ))

    return matchkeys


# ── Compound blocking helpers ─────────────────────────────────────────────


def _build_compound_blocking(
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    max_safe_block: int,
    max_null_rate: float,
) -> BlockingConfig | None:
    """Try to build compound blocking keys when single columns are all oversized.

    Uses greedy refinement: pick the best single column, then find the second
    column that reduces max block size the most. Generates multi-pass compound
    keys for recall.

    Returns None if no compound pair brings blocks below max_safe_block.
    """
    def _null_rate(col_name: str) -> float:
        return df[col_name].null_count() / df.height if df.height > 0 else 0.0

    # Build unified candidate pool (excludes numeric, date, identifier)
    candidates = [
        p for p in profiles
        if p.col_type not in ("numeric", "date", "identifier")
        and _null_rate(p.name) <= max_null_rate
        and _check_source_overlap(df, p.name) > 0.0
    ]
    if len(candidates) < 2:
        return None

    # Sort by cardinality descending — best single column first
    candidates.sort(key=lambda p: df[p.name].n_unique(), reverse=True)
    best = candidates[0]

    # Test compound pairs: best + each other candidate (up to 5)
    pair_results: list[tuple[ColumnProfile, int]] = []
    for other in candidates[1:6]:
        try:
            max_block = df.group_by([best.name, other.name]).len().get_column("len").max()
            pair_results.append((other, max_block))
            logger.debug(
                "Compound pair [%s, %s]: max_block=%d",
                best.name, other.name, max_block,
            )
        except Exception:
            continue

    if not pair_results:
        return None

    # Sort by max block ascending — smallest (safest) first
    pair_results.sort(key=lambda x: x[1])
    winner, winner_block = pair_results[0]

    if winner_block > max_safe_block:
        logger.info(
            "Best compound pair [%s, %s] still produces blocks of %d (> %d). "
            "No compound key is safe enough.",
            best.name, winner.name, winner_block, max_safe_block,
        )
        return None

    logger.info(
        "Compound blocking: [%s, %s] -> max_block=%d",
        best.name, winner.name, winner_block,
    )

    # Build multi-pass config for recall
    passes = [
        # Pass 1: winning compound pair
        BlockingKeyConfig(fields=[best.name, winner.name], transforms=["lowercase", "strip"]),
    ]

    # Pass 2: runner-up compound pair (if different and safe)
    if len(pair_results) > 1:
        runner_up, runner_up_block = pair_results[1]
        if runner_up_block <= max_safe_block and runner_up.name != winner.name:
            passes.append(
                BlockingKeyConfig(fields=[best.name, runner_up.name], transforms=["lowercase", "strip"]),
            )

    # Pass 3: recall-focused single-column soundex (relies on skip_oversized)
    passes.append(
        BlockingKeyConfig(fields=[best.name], transforms=["lowercase", "soundex"]),
    )

    return BlockingConfig(
        keys=[passes[0]],
        strategy="multi_pass",
        passes=passes,
        max_block_size=max_safe_block,
        skip_oversized=True,
    )


def _call_llm_for_blocking(prompt: str, provider: str) -> str:
    """Call LLM API for blocking key suggestion. Returns raw response text.

    Uses stdlib urllib (same pattern as llm_scorer.py) — no external deps.
    """
    import json as _json
    import os
    import urllib.request

    _MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}
    model = os.environ.get("GOLDENMATCH_LLM_MODEL", _MODELS.get(provider, ""))

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        body = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        body = _json.dumps({
            "model": model,
            "max_tokens": 500,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return data["content"][0]["text"]

    raise ValueError(f"Unknown provider: {provider}")


def _llm_suggest_blocking_keys(
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    provider: str,
    max_safe_block: int,
) -> BlockingConfig | None:
    """Ask LLM to suggest compound blocking keys, then validate.

    Returns a validated BlockingConfig or None if suggestions are invalid.
    """
    # Build prompt with cardinality stats (all non-numeric columns, including date)
    col_stats = []
    for p in profiles:
        if p.col_type == "numeric":
            continue
        n_unique = df[p.name].n_unique()
        max_block = df.group_by(p.name).len().get_column("len").max()
        col_stats.append(
            f"  {p.name}: type={p.col_type}, {n_unique:,} unique / {df.height:,} rows, "
            f"max_block={max_block:,}"
        )

    prompt = (
        "You are a data deduplication expert. Given these column profiles with cardinality stats:\n"
        + "\n".join(col_stats)
        + f"\n\nDataset: {df.height:,} rows. Max safe block size: {max_safe_block:,}.\n"
        "Suggest 2-3 multi-pass compound blocking key combinations.\n"
        "Each pass: 2 columns that together keep max block under the safe limit.\n"
        "Prioritize recall — different passes should cover different match scenarios "
        "(e.g., same model different location vs same model different year).\n\n"
        'Return JSON: {"passes": [{"fields": ["col_a", "col_b"], "reason": "..."}, ...]}'
    )

    try:
        raw = _call_llm_for_blocking(prompt, provider)
    except Exception as e:
        logger.warning("LLM blocking key suggestion failed: %s", e)
        return None

    # Parse JSON
    import json as _json
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = _json.loads(text)
    except (ValueError, KeyError) as e:
        logger.warning("LLM returned invalid JSON for blocking keys: %s", e)
        return None

    suggested_passes = data.get("passes", [])
    if not suggested_passes:
        logger.warning("LLM returned empty passes list")
        return None

    # Validate each suggestion
    valid_columns = set(df.columns)
    validated_passes: list[BlockingKeyConfig] = []

    for suggestion in suggested_passes:
        fields = suggestion.get("fields", [])
        reason = suggestion.get("reason", "")

        if not all(f in valid_columns for f in fields):
            bad = [f for f in fields if f not in valid_columns]
            logger.info("LLM suggestion rejected — unknown columns: %s", bad)
            continue

        try:
            max_block = df.group_by(fields).len().get_column("len").max()
        except Exception:
            logger.info("LLM suggestion rejected — group_by failed for %s", fields)
            continue

        if max_block > max_safe_block:
            logger.info(
                "LLM suggestion [%s] rejected — max_block=%d > %d. Reason: %s",
                fields, max_block, max_safe_block, reason,
            )
            continue

        logger.info(
            "LLM suggestion accepted: [%s] -> max_block=%d. Reason: %s",
            fields, max_block, reason,
        )
        validated_passes.append(
            BlockingKeyConfig(fields=fields, transforms=["lowercase", "strip"])
        )

    if not validated_passes:
        logger.info("All LLM blocking key suggestions were rejected")
        return None

    return BlockingConfig(
        keys=[validated_passes[0]],
        strategy="multi_pass",
        passes=validated_passes,
        max_block_size=max_safe_block,
        skip_oversized=True,
    )


# ── Cross-source overlap ──────────────────────────────────────────────────


def _check_source_overlap(df: pl.DataFrame, col: str) -> float:
    """Compute value overlap ratio for a column across sources.

    Returns |intersection| / |union| of unique values per source.
    Returns 1.0 if no __source__ column or only one source (no check needed).
    """
    if "__source__" not in df.columns:
        return 1.0

    sources = df["__source__"].unique().to_list()
    if len(sources) < 2:
        return 1.0

    value_sets = []
    for src in sources:
        vals = set(
            df.filter(pl.col("__source__") == src)[col]
            .drop_nulls()
            .cast(pl.Utf8)
            .to_list()
        )
        value_sets.append(vals)

    intersection = value_sets[0]
    union = value_sets[0]
    for vs in value_sets[1:]:
        intersection = intersection & vs
        union = union | vs

    if not union:
        return 1.0

    return len(intersection) / len(union)


# ── Blocking generation ────────────────────────────────────────────────────

def build_blocking(
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    llm_provider: str | None = None,
) -> BlockingConfig:
    """Generate blocking config from column profiles."""
    # Filter out high-null columns (>20% null) — they create oversized null blocks
    # that cause O(N^2) comparison explosions
    max_null_rate = 0.20

    def _null_rate(col_name: str) -> float:
        return df[col_name].null_count() / df.height if df.height > 0 else 0.0

    exact_cols = [
        p for p in profiles
        if p.col_type in ("email", "phone", "zip", "identifier")
        and _null_rate(p.name) <= max_null_rate
        and p.cardinality_ratio < 0.95
        and _check_source_overlap(df, p.name) > 0.0
    ]
    # Log skipped columns
    for p in profiles:
        if (p.col_type in ("email", "phone", "zip", "identifier")
                and _null_rate(p.name) <= max_null_rate
                and p.cardinality_ratio < 0.95
                and _check_source_overlap(df, p.name) == 0.0):
            sources = df["__source__"].unique().to_list() if "__source__" in df.columns else []
            logger.warning(
                "Blocking key '%s' has 0%% overlap between sources %s -- skipping",
                p.name, ", ".join(str(s) for s in sources),
            )
    name_cols = [
        p for p in profiles
        if p.col_type == "name"
        and _check_source_overlap(df, p.name) > 0.0
    ]
    text_cols = [p for p in profiles if p.col_type in ("description", "string", "address")]

    def _max_block_size(col_name: str) -> int:
        """Largest group size when blocking on this column."""
        return df.group_by(col_name).len().get_column("len").max()

    max_safe_block = 1000  # blocks larger than this cause OOM on ensemble scorers

    # Best case: block on highest-cardinality exact column (with low null rate + safe block size)
    if exact_cols:
        # Pre-filter: only evaluate top 5 by cardinality to avoid expensive group_by on all columns
        exact_cols_sorted = sorted(exact_cols, key=lambda p: df[p.name].n_unique(), reverse=True)
        candidates = exact_cols_sorted[:5]
        # Filter out columns that create oversized blocks
        safe_exact = [p for p in candidates if _max_block_size(p.name) <= max_safe_block]
        if safe_exact:
            best = max(safe_exact, key=lambda p: df[p.name].n_unique())
            transforms = ["lowercase", "strip"] if best.col_type == "email" else ["strip"]
            return BlockingConfig(
                keys=[BlockingKeyConfig(fields=[best.name], transforms=transforms)],
            )
        # All exact columns create oversized blocks — fall through
        logger.warning(
            "Exact blocking columns all produce oversized blocks (>%d), "
            "falling through to name-based blocking",
            max_safe_block,
        )

    # ── Check if name-based fallback would also be oversized ──
    _all_single_oversized = True
    for p in name_cols:
        try:
            if _max_block_size(p.name) <= max_safe_block:
                _all_single_oversized = False
                break
        except Exception:
            continue

    if _all_single_oversized and (name_cols or text_cols):
        # All single columns produce oversized blocks — try compound blocking
        if llm_provider:
            llm_config = _llm_suggest_blocking_keys(profiles, df, llm_provider, max_safe_block)
            if llm_config is not None:
                logger.info("Using LLM-suggested compound blocking keys")
                return llm_config
            logger.info("LLM suggestions invalid or unavailable — trying greedy compound")

        compound_config = _build_compound_blocking(profiles, df, max_safe_block, max_null_rate)
        if compound_config is not None:
            return compound_config

        logger.info("Compound blocking failed — falling through to single-column fallbacks")

    # Name columns: use multi-pass with soundex + substring
    # Prefer columns matched by name pattern (person names) over data-profiled names
    if name_cols:
        pattern_names = [p for p in name_cols if _classify_by_name(p.name) == "name"]
        best_name = (pattern_names[0] if pattern_names else name_cols[0]).name
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "soundex"])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "substring:0:5"]),
                BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "soundex"]),
                BlockingKeyConfig(fields=[best_name], transforms=["lowercase", "token_sort", "substring:0:8"]),
            ],
            max_block_size=max_safe_block,
            skip_oversized=True,
        )

    # Last resort: canopy on best text column
    if text_cols:
        from goldenmatch.config.schemas import CanopyConfig
        best_text = text_cols[0].name
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[best_text], transforms=["lowercase", "substring:0:5"])],
            strategy="canopy",
            canopy=CanopyConfig(fields=[best_text], loose_threshold=0.3, tight_threshold=0.7),
            skip_oversized=True,
        )

    # Absolute fallback
    first_string = next(
        (p for p in profiles if p.col_type != "numeric"),
        profiles[0] if profiles else None,
    )
    if first_string:
        return BlockingConfig(
            keys=[BlockingKeyConfig(fields=[first_string.name], transforms=["lowercase", "substring:0:5"])],
            skip_oversized=True,
        )

    return BlockingConfig(keys=[BlockingKeyConfig(fields=[profiles[0].name])], skip_oversized=True)


# ── Model selection ────────────────────────────────────────────────────────

def select_model(row_count: int, has_embedding_columns: bool, threshold: int = 50000) -> str | None:
    """Select embedding model. Returns None if no embedding columns needed."""
    if not has_embedding_columns:
        return None
    if row_count < threshold:
        return "gte-base-en-v1.5"
    return "all-MiniLM-L6-v2"


# ── Main entry point ──────────────────────────────────────────────────────

def auto_configure_df(df: pl.DataFrame, llm_provider: str | None = None, domain_config=None) -> GoldenMatchConfig:
    """Auto-generate a GoldenMatchConfig from a DataFrame.

    Profiles columns by name heuristics and data sampling, then builds
    matchkeys, blocking, and golden rules automatically.

    Args:
        df: Polars DataFrame to auto-configure for.

    Returns:
        A fully populated GoldenMatchConfig ready for pipeline execution.
    """
    total_rows = df.height

    logger.info("Auto-configuring %d rows, %d columns", total_rows, len(df.columns))

    # Profile columns
    profiles = profile_columns(df, llm_provider=llm_provider)

    logger.info(
        "Detected column types: %s",
        {p.name: p.col_type for p in profiles},
    )

    # ── Domain detection + conditional extraction ──
    extracted_columns = []

    if domain_config is not None:
        # Manual override: skip auto-detection
        logger.info("Domain config provided manually, skipping auto-detection")
    else:
        from goldenmatch.core.domain import detect_domain, extract_features

        user_cols = [c for c in df.columns if not c.startswith("__")]
        domain_profile = detect_domain(user_cols)

        if domain_profile.confidence > 0.7:
            original_cols = set(df.columns)
            # extract_features requires __row_id__ column
            if "__row_id__" not in df.columns:
                df = df.with_row_index("__row_id__")
            df, _low_conf_ids = extract_features(df, domain_profile)
            extracted_columns = [c for c in df.columns if c.startswith("__") and c not in original_cols]
            logger.info(
                "Domain '%s' detected (confidence=%.2f), extracted %d feature columns",
                domain_profile.name, domain_profile.confidence, len(extracted_columns),
            )
        else:
            logger.info(
                "Domain '%s' (confidence=%.2f) below threshold, skipping extraction",
                domain_profile.name, domain_profile.confidence,
            )

    # Build matchkeys
    matchkeys = build_matchkeys(profiles, df=df)

    # ── Add domain-extracted fields to matchkeys ──
    if extracted_columns:
        domain_exact = []
        domain_fuzzy = []
        for col in extracted_columns:
            if col not in _DOMAIN_SCORER_MAP:
                continue
            scorer, weight, transforms = _DOMAIN_SCORER_MAP[col]
            null_rate = df[col].null_count() / df.height if df.height > 0 else 0
            cardinality_ratio = df[col].n_unique() / df.height if df.height > 0 else 0
            if null_rate > 0.5:
                continue
            if scorer == "exact" and cardinality_ratio < 0.01:
                continue
            mf = MatchkeyField(field=col, scorer=scorer, weight=weight, transforms=transforms)
            if scorer == "exact":
                domain_exact.append(mf)
            else:
                domain_fuzzy.append(mf)

        # Add domain exact matchkeys
        for f in domain_exact:
            matchkeys.append(MatchkeyConfig(
                name=f"domain_exact_{f.field.strip('_')}",
                type="exact",
                fields=[MatchkeyField(field=f.field, transforms=f.transforms)],
            ))

        # Add domain fuzzy fields to existing weighted matchkey (or create one)
        if domain_fuzzy:
            weighted = [mk for mk in matchkeys if mk.type == "weighted"]
            if weighted:
                weighted[0].fields.extend(domain_fuzzy)
            else:
                matchkeys.append(MatchkeyConfig(
                    name="domain_fuzzy",
                    type="weighted",
                    threshold=0.80,
                    fields=domain_fuzzy,
                ))

    # Check if embeddings are needed
    has_embeddings = any(
        f.scorer in ("embedding", "record_embedding")
        for mk in matchkeys
        for f in mk.fields
    )

    # Select model and apply to embedding fields
    model = select_model(total_rows, has_embeddings)
    if model:
        for mk in matchkeys:
            for f in mk.fields:
                if f.scorer in ("embedding", "record_embedding") and not f.model:
                    f.model = model

    # ── Add domain columns to blocking candidate profiles ──
    if extracted_columns:
        for col in extracted_columns:
            if col not in _DOMAIN_SCORER_MAP:
                continue
            scorer, _weight, _transforms = _DOMAIN_SCORER_MAP[col]
            if scorer != "exact":
                continue
            null_rate = df[col].null_count() / df.height if df.height > 0 else 0
            cardinality_ratio = df[col].n_unique() / df.height if df.height > 0 else 0
            if null_rate > 0.5:
                continue
            profiles.append(ColumnProfile(
                name=col, dtype="Utf8", col_type="email",
                confidence=0.9, null_rate=null_rate,
                cardinality_ratio=cardinality_ratio, avg_len=0,
            ))

    # Build blocking (required for weighted/probabilistic matchkeys)
    has_fuzzy = any(mk.type in ("weighted", "probabilistic") for mk in matchkeys)
    blocking = build_blocking(profiles, df, llm_provider=llm_provider) if has_fuzzy else None

    # ── Data-driven strategy selection ──

    # 1. Learned blocking for large datasets
    if blocking is not None and total_rows >= 5000:
        blocking.strategy = "learned"
        blocking.learned_sample_size = 5000
        blocking.learned_min_recall = 0.95
        blocking.skip_oversized = True
        logger.info("Upgraded to learned blocking (dataset has %d rows)", total_rows)

    # 2. Reranking for multi-field matchkeys
    for mk in matchkeys:
        if mk.type == "weighted" and len(mk.fields) >= 3:
            mk.rerank = True
            logger.info("Enabled reranking for matchkey '%s' (%d fields)", mk.name, len(mk.fields))

    # 3. Adaptive threshold from data quality
    for mk in matchkeys:
        if mk.type == "weighted" and mk.threshold is not None:
            fuzzy_field_names = {f.field for f in mk.fields if f.field}
            fuzzy_profiles = [p for p in profiles if p.name in fuzzy_field_names]
            if fuzzy_profiles:
                avg_null = sum(p.null_rate for p in fuzzy_profiles) / len(fuzzy_profiles)
                avg_len = sum(p.avg_len for p in fuzzy_profiles) / len(fuzzy_profiles)
                original = mk.threshold
                if avg_null > 0.15:
                    mk.threshold = max(mk.threshold - 0.05, 0.50)
                if avg_len < 5:
                    mk.threshold = min(mk.threshold + 0.05, 0.95)
                if mk.threshold != original:
                    logger.info(
                        "Adjusted threshold for '%s': %.2f -> %.2f (avg_null=%.2f, avg_len=%.1f)",
                        mk.name, original, mk.threshold, avg_null, avg_len,
                    )

    # Build config
    config = GoldenMatchConfig(
        matchkeys=matchkeys,
        blocking=blocking,
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
        output=OutputConfig(),
    )

    return config


def auto_configure(files: list[tuple[str, str]]) -> GoldenMatchConfig:
    """Auto-generate a GoldenMatchConfig from input files.

    Args:
        files: List of (path, source_name) tuples.

    Returns:
        A fully populated GoldenMatchConfig ready for pipeline execution.
    """
    # Load and combine files
    dfs = []
    for path, source_name in files:
        p = Path(path)
        if p.suffix.lower() in (".xlsx", ".xls"):
            df = pl.read_excel(p, engine="openpyxl")
        elif p.suffix.lower() == ".parquet":
            df = pl.read_parquet(p)
        else:
            df = pl.read_csv(p, encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
        dfs.append(df)

    combined = pl.concat(dfs, how="diagonal") if len(dfs) > 1 else dfs[0]
    return auto_configure_df(combined)


def build_probabilistic_matchkeys(profiles: list[ColumnProfile]) -> list[MatchkeyConfig]:
    """Generate Fellegi-Sunter probabilistic matchkeys from column profiles.

    Produces a single probabilistic matchkey using all matchable columns
    with appropriate comparison levels and partial thresholds.
    """
    fields = []
    for p in profiles:
        if p.col_type in ("numeric", "date", "identifier", "description"):
            continue

        scorer_info = _SCORER_MAP.get(p.col_type)
        if not scorer_info:
            continue

        scorer, _weight, transforms = scorer_info

        # Determine comparison levels based on scorer type
        if scorer == "exact":
            levels = 2
            partial_threshold = 0.9
        else:
            levels = 3
            partial_threshold = 0.8

        fields.append(MatchkeyField(
            field=p.name,
            scorer=scorer,
            transforms=transforms,
            levels=levels,
            partial_threshold=partial_threshold,
        ))

    if not fields:
        return []

    return [MatchkeyConfig(
        name="probabilistic_auto",
        type="probabilistic",
        fields=fields,
    )]
