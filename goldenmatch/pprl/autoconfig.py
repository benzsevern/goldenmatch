"""Auto-configuration for PPRL -- analyze data and recommend optimal parameters.

Non-LLM default: profiles columns, measures cardinality/length, recommends
fields, bloom filter size, hash count, n-gram size, and threshold.

LLM opt-in: sends data profile to LLM for more nuanced recommendations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from goldenmatch.pprl.protocol import PPRLConfig

logger = logging.getLogger(__name__)

# Column name patterns for person-data field detection
_PERSON_FIELD_PATTERNS = {
    "first_name": ["first_name", "fname", "given_name", "first"],
    "last_name": ["last_name", "lname", "surname", "family_name", "last"],
    "middle_name": ["middle_name", "mname", "middle"],
    "dob": ["dob", "date_of_birth", "birth_date", "birthdate"],
    "birth_year": ["birth_year", "byear", "year_of_birth"],
    "zip": ["zip", "zip_code", "postal_code", "postcode"],
    "gender": ["gender", "gender_code", "sex"],
    "ssn": ["ssn", "social_security", "soc_sec_id"],
    "phone": ["phone", "phone_number", "full_phone_number", "telephone"],
    "email": ["email", "email_address"],
    "address": ["address", "street_address", "res_street_address", "address_1"],
    "city": ["city", "city_desc", "res_city_desc"],
    "state": ["state", "state_cd", "state_code"],
}


@dataclass
class FieldProfile:
    """Profile of a single field for PPRL configuration."""
    column: str
    field_type: str  # first_name, last_name, zip, etc.
    avg_length: float
    cardinality: int
    null_pct: float
    usefulness_score: float = 0.0  # 0-1, higher = more useful for linkage


@dataclass
class PPRLAutoConfigResult:
    """Result of auto-configuration analysis."""
    recommended_fields: list[str]
    recommended_config: PPRLConfig
    field_profiles: list[FieldProfile]
    explanation: str


def profile_for_pprl(df: pl.DataFrame) -> list[FieldProfile]:
    """Profile DataFrame columns for PPRL suitability."""
    profiles = []

    for col in df.columns:
        if col.startswith("__"):
            continue
        if df[col].dtype != pl.Utf8:
            continue

        col_lower = col.lower()

        # Detect field type
        field_type = "unknown"
        for ftype, patterns in _PERSON_FIELD_PATTERNS.items():
            if any(col_lower == p or col_lower.replace(" ", "_") == p for p in patterns):
                field_type = ftype
                break

        # Compute stats
        non_null = df[col].drop_nulls().cast(pl.Utf8)
        if non_null.len() == 0:
            continue

        avg_len = non_null.str.len_chars().mean()
        cardinality = non_null.n_unique()
        null_pct = df[col].null_count() / df.height

        # Usefulness score: high cardinality + short length + low null rate = good
        # Normalized to 0-1
        card_score = min(cardinality / df.height, 1.0)  # higher is better
        len_score = max(0, 1.0 - avg_len / 50)  # shorter is better for BF
        null_score = 1.0 - null_pct

        usefulness = 0.5 * card_score + 0.3 * len_score + 0.2 * null_score

        # Boost known person fields
        if field_type in ("first_name", "last_name"):
            usefulness = min(usefulness + 0.3, 1.0)
        elif field_type in ("zip", "birth_year", "dob", "gender"):
            usefulness = min(usefulness + 0.2, 1.0)
        elif field_type in ("ssn", "phone", "email"):
            usefulness = min(usefulness + 0.25, 1.0)

        profiles.append(FieldProfile(
            column=col,
            field_type=field_type,
            avg_length=avg_len,
            cardinality=cardinality,
            null_pct=null_pct,
            usefulness_score=usefulness,
        ))

    return profiles


def auto_configure_pprl(
    df: pl.DataFrame,
    security_level: str = "high",
    max_fields: int = 6,
) -> PPRLAutoConfigResult:
    """Automatically determine optimal PPRL configuration from data.

    Profiles columns, ranks by usefulness, selects top fields, and
    recommends bloom filter parameters based on field characteristics.
    """
    profiles = profile_for_pprl(df)
    if not profiles:
        raise ValueError("No string columns found for PPRL")

    # Sort by usefulness
    profiles.sort(key=lambda p: p.usefulness_score, reverse=True)

    # Select top fields
    selected = []
    for p in profiles:
        if len(selected) >= max_fields:
            break
        if p.usefulness_score < 0.2:
            break
        selected.append(p)

    if not selected:
        # Fallback: just use the first few string columns
        selected = profiles[:3]

    field_names = [p.column for p in selected]

    # Determine BF parameters from field characteristics
    avg_field_len = np.mean([p.avg_length for p in selected])

    # Short fields (names, codes) -> bigrams, smaller filter
    # Long fields (addresses, descriptions) -> trigrams, larger filter
    if avg_field_len < 10:
        ngram_size = 2
        filter_size = 512
        hash_count = 20
    elif avg_field_len < 25:
        ngram_size = 2
        filter_size = 1024
        hash_count = 30
    else:
        ngram_size = 3
        filter_size = 2048
        hash_count = 40

    # Override from security level
    _SECURITY = {
        "standard": (max(ngram_size, 2), max(hash_count, 20), max(filter_size, 512)),
        "high": (max(ngram_size, 2), max(hash_count, 30), max(filter_size, 1024)),
        "paranoid": (max(ngram_size, 3), max(hash_count, 40), max(filter_size, 2048)),
    }
    if security_level in _SECURITY:
        ngram_size, hash_count, filter_size = _SECURITY[security_level]

    # Threshold: sample-based estimation using Otsu's method
    threshold = _estimate_threshold(df, field_names, ngram_size, hash_count, filter_size)

    config = PPRLConfig(
        fields=field_names,
        threshold=threshold,
        security_level=security_level,
        ngram_size=ngram_size,
        hash_functions=hash_count,
        bloom_filter_size=filter_size,
    )

    # Build explanation
    field_summary = ", ".join(f"{p.column} ({p.field_type}, score={p.usefulness_score:.2f})" for p in selected)
    explanation = (
        f"Selected {len(field_names)} fields: {field_summary}. "
        f"BF params: {ngram_size}-gram, {hash_count} hashes, {filter_size}-bit filter. "
        f"Threshold: {threshold:.2f}. "
        f"Avg field length: {avg_field_len:.1f} chars."
    )

    return PPRLAutoConfigResult(
        recommended_fields=field_names,
        recommended_config=config,
        field_profiles=selected,
        explanation=explanation,
    )


def _estimate_threshold(
    df: pl.DataFrame,
    fields: list[str],
    ngram_size: int,
    hash_count: int,
    filter_size: int,
    sample_size: int = 200,
) -> float:
    """Estimate optimal threshold by sampling pairs and finding score distribution valley."""
    from goldenmatch.utils.transforms import apply_transforms

    transform = f"bloom_filter:{ngram_size}:{hash_count}:{filter_size}"

    # Sample records
    sample = df.sample(n=min(sample_size, df.height), seed=42)
    rows = sample.to_dicts()

    # Compute bloom filters for sample
    filters = []
    for row in rows:
        text = " ".join(str(row.get(f, "") or "") for f in fields)
        bf = apply_transforms(text, [transform])
        filters.append(bf)

    # Compute pairwise scores for a subset
    from goldenmatch.core.scorer import _hex_to_bits
    n = len(filters)
    scores = []
    for i in range(min(n, 100)):
        if filters[i] is None:
            continue
        bits_i = np.unpackbits(_hex_to_bits(filters[i])).astype(np.float32)
        pop_i = bits_i.sum()
        for j in range(i + 1, min(n, 100)):
            if filters[j] is None:
                continue
            bits_j = np.unpackbits(_hex_to_bits(filters[j])).astype(np.float32)
            intersection = np.dot(bits_i, bits_j)
            total = pop_i + bits_j.sum()
            dice = float(2.0 * intersection / total) if total > 0 else 0.0
            scores.append(dice)

    if not scores:
        return 0.85  # safe default

    # Find valley in score histogram (bimodal: non-matches vs matches)
    scores_arr = np.array(scores)
    # Use percentile-based approach: threshold at 90th percentile
    # (most pairs are non-matches, true matches cluster at the top)
    p90 = float(np.percentile(scores_arr, 90))
    p95 = float(np.percentile(scores_arr, 95))

    # Pick threshold between p90 and p95
    threshold = round((p90 + p95) / 2, 2)

    # Clamp to reasonable range
    threshold = max(0.70, min(0.95, threshold))

    return threshold


def auto_configure_pprl_llm(
    df: pl.DataFrame,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    security_level: str = "high",
) -> PPRLAutoConfigResult:
    """LLM-assisted PPRL auto-configuration.

    Sends a data profile to the LLM for more nuanced field selection
    and parameter recommendations.
    """
    import json
    import os

    # Get non-LLM baseline first
    baseline = auto_configure_pprl(df, security_level)

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No API key for LLM auto-config, using non-LLM baseline")
        return baseline

    # Build profile summary for LLM
    profile_text = "Dataset profile for PPRL configuration:\n"
    profile_text += f"Records: {df.height}\n"
    profile_text += f"Columns:\n"
    for p in baseline.field_profiles:
        profile_text += f"  - {p.column}: type={p.field_type}, avg_len={p.avg_length:.1f}, cardinality={p.cardinality}, nulls={p.null_pct:.0%}\n"

    # Also include columns NOT selected
    all_profiles = profile_for_pprl(df)
    unselected = [p for p in all_profiles if p.column not in baseline.recommended_fields]
    if unselected:
        profile_text += "Other available columns:\n"
        for p in unselected[:10]:
            profile_text += f"  - {p.column}: type={p.field_type}, avg_len={p.avg_length:.1f}, cardinality={p.cardinality}\n"

    profile_text += f"\nBaseline recommendation: fields={baseline.recommended_fields}, threshold={baseline.recommended_config.threshold}\n"
    profile_text += f"\nSecurity level: {security_level}\n"
    profile_text += "\nReturn JSON: {\"fields\": [...], \"threshold\": 0.XX, \"reasoning\": \"...\"}\n"
    profile_text += "Select fields that maximize linkage quality for person matching. Prefer short, high-cardinality fields."

    try:
        from goldenmatch.core.llm_scorer import _call_openai
        response, _, _ = _call_openai(api_key, model, profile_text)

        # Parse LLM response
        text = response.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            rec = json.loads(text[start:end])
            llm_fields = rec.get("fields", baseline.recommended_fields)
            llm_threshold = rec.get("threshold", baseline.recommended_config.threshold)
            reasoning = rec.get("reasoning", "")

            # Validate fields exist
            valid_cols = set(df.columns)
            llm_fields = [f for f in llm_fields if f in valid_cols]

            if llm_fields:
                config = PPRLConfig(
                    fields=llm_fields,
                    threshold=float(llm_threshold),
                    security_level=security_level,
                    ngram_size=baseline.recommended_config.ngram_size,
                    hash_functions=baseline.recommended_config.hash_functions,
                    bloom_filter_size=baseline.recommended_config.bloom_filter_size,
                )
                return PPRLAutoConfigResult(
                    recommended_fields=llm_fields,
                    recommended_config=config,
                    field_profiles=baseline.field_profiles,
                    explanation=f"LLM recommendation: {reasoning}",
                )
    except Exception as e:
        logger.warning("LLM auto-config failed (%s), using baseline", e)

    return baseline
