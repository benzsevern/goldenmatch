"""Schema-free matching — auto-map columns between different schemas.

Handles cases like:
  CRM has "full_name", billing has "first_name" + "last_name"
  File A has "email", File B has "contact_email"
  File A has "phone", File B has "telephone"

Uses column name similarity + value overlap analysis to find mappings.
"""

from __future__ import annotations

import logging
from collections import Counter

import polars as pl
from rapidfuzz.fuzz import ratio, partial_ratio

logger = logging.getLogger(__name__)

# Common synonyms for column name matching
SYNONYMS = {
    "name": ["full_name", "fullname", "customer_name", "person_name", "display_name", "contact_name"],
    "first_name": ["fname", "firstname", "given_name", "forename"],
    "last_name": ["lname", "lastname", "surname", "family_name"],
    "email": ["email_address", "contact_email", "e_mail", "emailaddress", "mail"],
    "phone": ["telephone", "phone_number", "tel", "mobile", "cell", "contact_phone", "phonenumber"],
    "address": ["street_address", "addr", "street", "address_line_1", "address1", "mailing_address"],
    "city": ["town", "municipality"],
    "state": ["province", "region", "st"],
    "zip": ["zipcode", "zip_code", "postal_code", "postcode", "postal"],
    "country": ["nation", "country_code"],
    "id": ["identifier", "record_id", "customer_id", "patient_id", "account_id", "uid"],
    "company": ["organization", "org", "business", "employer", "firm", "company_name"],
    "dob": ["date_of_birth", "birth_date", "birthdate", "birthday"],
    "gender": ["sex"],
    "title": ["product_name", "item_name", "description", "product_title"],
}

# Build reverse lookup
_SYNONYM_MAP: dict[str, str] = {}
for canonical, aliases in SYNONYMS.items():
    for alias in aliases:
        _SYNONYM_MAP[alias.lower()] = canonical
    _SYNONYM_MAP[canonical.lower()] = canonical


def auto_map_columns(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    min_score: float = 0.5,
) -> list[dict]:
    """Auto-detect column mappings between two DataFrames with different schemas.

    Returns list of mappings:
        [{"col_a": "full_name", "col_b": "first_name", "score": 0.85, "method": "synonym"},
         {"col_a": "email", "col_b": "contact_email", "score": 0.92, "method": "name_sim"}, ...]
    """
    cols_a = [c for c in df_a.columns if not c.startswith("__")]
    cols_b = [c for c in df_b.columns if not c.startswith("__")]

    # Score every (col_a, col_b) pair
    scores: list[tuple[str, str, float, str]] = []

    for ca in cols_a:
        for cb in cols_b:
            score, method = _score_column_pair(ca, cb, df_a, df_b)
            if score >= min_score:
                scores.append((ca, cb, score, method))

    # Greedy best-match assignment (each column used at most once)
    scores.sort(key=lambda x: -x[2])
    used_a: set[str] = set()
    used_b: set[str] = set()
    mappings = []

    for ca, cb, score, method in scores:
        if ca not in used_a and cb not in used_b:
            mappings.append({
                "col_a": ca,
                "col_b": cb,
                "score": round(score, 3),
                "method": method,
            })
            used_a.add(ca)
            used_b.add(cb)

    # Check for composite columns (e.g., full_name → first_name + last_name)
    unmapped_a = [c for c in cols_a if c not in used_a]
    unmapped_b = [c for c in cols_b if c not in used_b]
    composites = _detect_composites(unmapped_a, unmapped_b, df_a, df_b)
    mappings.extend(composites)

    logger.info("Auto-mapped %d column pairs between schemas", len(mappings))
    return mappings


def _score_column_pair(
    col_a: str, col_b: str,
    df_a: pl.DataFrame, df_b: pl.DataFrame,
) -> tuple[float, str]:
    """Score how likely two columns represent the same field."""
    best_score = 0.0
    best_method = "none"

    # 1. Exact name match
    if col_a.lower().strip() == col_b.lower().strip():
        return 1.0, "exact_name"

    # 2. Synonym match
    canonical_a = _SYNONYM_MAP.get(col_a.lower().replace(" ", "_"), "")
    canonical_b = _SYNONYM_MAP.get(col_b.lower().replace(" ", "_"), "")
    if canonical_a and canonical_a == canonical_b:
        return 0.95, "synonym"

    # 3. Fuzzy name similarity
    name_sim = ratio(col_a.lower(), col_b.lower()) / 100.0
    if name_sim > best_score:
        best_score = name_sim
        best_method = "name_sim"

    # 4. Partial name match (one name contains the other)
    partial = partial_ratio(col_a.lower(), col_b.lower()) / 100.0
    if partial > best_score:
        best_score = partial * 0.9  # slight discount
        best_method = "partial_name"

    # 5. Value overlap (sample-based)
    value_sim = _value_overlap(col_a, col_b, df_a, df_b)
    if value_sim > best_score:
        best_score = value_sim
        best_method = "value_overlap"

    # 6. Type similarity (both numeric, both string, etc.)
    type_bonus = _type_similarity(col_a, col_b, df_a, df_b)
    best_score = min(1.0, best_score + type_bonus * 0.1)

    return best_score, best_method


def _value_overlap(
    col_a: str, col_b: str,
    df_a: pl.DataFrame, df_b: pl.DataFrame,
    sample_size: int = 200,
) -> float:
    """Compute Jaccard-like overlap between value sets."""
    try:
        vals_a = set(
            str(v).lower().strip()
            for v in df_a[col_a].head(sample_size).to_list()
            if v is not None
        )
        vals_b = set(
            str(v).lower().strip()
            for v in df_b[col_b].head(sample_size).to_list()
            if v is not None
        )
        if not vals_a or not vals_b:
            return 0.0
        intersection = len(vals_a & vals_b)
        union = len(vals_a | vals_b)
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _type_similarity(
    col_a: str, col_b: str,
    df_a: pl.DataFrame, df_b: pl.DataFrame,
) -> float:
    """Check if two columns have compatible data types."""
    try:
        dtype_a = str(df_a[col_a].dtype)
        dtype_b = str(df_b[col_b].dtype)

        numeric_types = {"Int8", "Int16", "Int32", "Int64", "Float32", "Float64", "UInt8", "UInt16", "UInt32", "UInt64"}
        a_numeric = any(t in dtype_a for t in numeric_types)
        b_numeric = any(t in dtype_b for t in numeric_types)

        if a_numeric and b_numeric:
            return 1.0
        if not a_numeric and not b_numeric:
            return 0.5  # both string-ish
        return 0.0  # type mismatch
    except Exception:
        return 0.0


def _detect_composites(
    unmapped_a: list[str], unmapped_b: list[str],
    df_a: pl.DataFrame, df_b: pl.DataFrame,
) -> list[dict]:
    """Detect composite column mappings (e.g., full_name → first_name + last_name)."""
    composites = []

    # Check if any unmapped col in A could be a concatenation of B columns
    for ca in unmapped_a:
        canonical = _SYNONYM_MAP.get(ca.lower().replace(" ", "_"), ca.lower())

        if canonical == "name":
            # Look for first_name + last_name in B
            fn_candidates = [c for c in unmapped_b if _SYNONYM_MAP.get(c.lower().replace(" ", "_"), "") == "first_name"]
            ln_candidates = [c for c in unmapped_b if _SYNONYM_MAP.get(c.lower().replace(" ", "_"), "") == "last_name"]

            if fn_candidates and ln_candidates:
                composites.append({
                    "col_a": ca,
                    "col_b": f"{fn_candidates[0]} + {ln_candidates[0]}",
                    "score": 0.90,
                    "method": "composite",
                    "composite_cols": [fn_candidates[0], ln_candidates[0]],
                })

    # Check reverse (B has composite, A has parts)
    for cb in unmapped_b:
        canonical = _SYNONYM_MAP.get(cb.lower().replace(" ", "_"), cb.lower())

        if canonical == "name":
            fn_candidates = [c for c in unmapped_a if _SYNONYM_MAP.get(c.lower().replace(" ", "_"), "") == "first_name"]
            ln_candidates = [c for c in unmapped_a if _SYNONYM_MAP.get(c.lower().replace(" ", "_"), "") == "last_name"]

            if fn_candidates and ln_candidates:
                composites.append({
                    "col_a": f"{fn_candidates[0]} + {ln_candidates[0]}",
                    "col_b": cb,
                    "score": 0.90,
                    "method": "composite",
                    "composite_cols": [fn_candidates[0], ln_candidates[0]],
                })

    return composites


def apply_column_mapping(
    df: pl.DataFrame,
    mappings: list[dict],
    target_columns: list[str],
    side: str = "b",
) -> pl.DataFrame:
    """Rename columns in a DataFrame according to discovered mappings.

    Args:
        df: DataFrame to rename.
        mappings: Output from auto_map_columns.
        target_columns: The column names to map TO.
        side: "a" or "b" — which side of the mapping this df is.
    """
    rename_map = {}
    for m in mappings:
        if "composite_cols" in m:
            continue  # handle composites separately

        src = m["col_b"] if side == "b" else m["col_a"]
        dst = m["col_a"] if side == "b" else m["col_b"]

        if src in df.columns:
            rename_map[src] = dst

    if rename_map:
        df = df.rename(rename_map)

    # Handle composite columns (merge first_name + last_name → name)
    for m in mappings:
        if "composite_cols" not in m:
            continue

        parts = m["composite_cols"]
        target = m["col_a"] if side == "b" else m["col_b"]

        if all(p in df.columns for p in parts):
            df = df.with_columns(
                (pl.col(parts[0]).cast(pl.Utf8).fill_null("") + pl.lit(" ") +
                 pl.col(parts[1]).cast(pl.Utf8).fill_null(""))
                .str.strip_chars()
                .alias(target)
            )

    return df
