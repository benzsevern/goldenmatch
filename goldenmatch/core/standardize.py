"""Data standardization module for GoldenMatch.

Applies user-configurable cleaning rules to data columns before matching.
Standardization modifies the actual data (unlike matchkey transforms which
build derived columns). Runs after ingest, before matchkey computation.
"""

from __future__ import annotations

import re
import logging

import polars as pl

logger = logging.getLogger(__name__)

# ── Address abbreviations (USPS standard) ────────────────────────────────────

ADDRESS_ABBREVIATIONS = {
    "street": "St",
    "avenue": "Ave",
    "boulevard": "Blvd",
    "drive": "Dr",
    "lane": "Ln",
    "road": "Rd",
    "court": "Ct",
    "place": "Pl",
    "circle": "Cir",
    "terrace": "Ter",
    "highway": "Hwy",
    "parkway": "Pkwy",
    "expressway": "Expy",
    "freeway": "Fwy",
    "trail": "Trl",
    "way": "Way",
    "north": "N",
    "south": "S",
    "east": "E",
    "west": "W",
    "northeast": "NE",
    "northwest": "NW",
    "southeast": "SE",
    "southwest": "SW",
    "apartment": "Apt",
    "suite": "Ste",
    "building": "Bldg",
    "floor": "Fl",
    "room": "Rm",
    "unit": "Unit",
    "department": "Dept",
    "post office box": "PO Box",
    "p.o. box": "PO Box",
    "po box": "PO Box",
}

# ── Individual standardizers ─────────────────────────────────────────────────


def std_email(value: str | None) -> str | None:
    """Standardize email: lowercase, strip, validate basic structure."""
    if value is None:
        return None
    value = str(value).strip().lower()
    if not value or "@" not in value or "." not in value.split("@")[-1]:
        return None  # invalid email -> null
    return value


def std_name_proper(value: str | None) -> str | None:
    """Standardize name to proper case (Title Case)."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    # Handle hyphenated names: Mary-Jane -> Mary-Jane
    parts = value.split("-")
    return "-".join(p.strip().title() for p in parts)


def std_name_upper(value: str | None) -> str | None:
    """Standardize name to UPPER CASE."""
    if value is None:
        return None
    value = str(value).strip()
    return value.upper() if value else None


def std_name_lower(value: str | None) -> str | None:
    """Standardize name to lower case."""
    if value is None:
        return None
    value = str(value).strip()
    return value.lower() if value else None


def std_phone(value: str | None) -> str | None:
    """Standardize phone: digits only, strip country code if 11 digits starting with 1."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    # Strip US country code
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    # Must be at least 7 digits to be valid
    if len(digits) < 7:
        return None
    return digits


def std_zip5(value: str | None) -> str | None:
    """Standardize ZIP code to first 5 digits, zero-padded."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value).split("-")[0].split(" ")[0])
    if not digits:
        return None
    return digits[:5].zfill(5)


def std_address(value: str | None) -> str | None:
    """Standardize address: proper case, USPS abbreviations, normalize whitespace."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    # Normalize whitespace
    value = re.sub(r"\s+", " ", value)
    # Apply abbreviations (case-insensitive word replacement)
    words = value.split(" ")
    result = []
    i = 0
    while i < len(words):
        # Check two-word phrases first (e.g. "post office")
        if i + 1 < len(words):
            two_word = f"{words[i]} {words[i+1]}".lower()
            if two_word in ADDRESS_ABBREVIATIONS:
                result.append(ADDRESS_ABBREVIATIONS[two_word])
                i += 2
                continue
        word_lower = words[i].lower().rstrip(".,")
        if word_lower in ADDRESS_ABBREVIATIONS:
            result.append(ADDRESS_ABBREVIATIONS[word_lower])
        else:
            result.append(words[i].title())
        i += 1
    return " ".join(result)


def std_state(value: str | None) -> str | None:
    """Standardize state to uppercase 2-letter abbreviation."""
    if value is None:
        return None
    value = str(value).strip().upper()
    return value if value else None


def std_strip(value: str | None) -> str | None:
    """Strip whitespace and normalize to None if empty."""
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def std_trim_whitespace(value: str | None) -> str | None:
    """Strip and collapse internal whitespace."""
    if value is None:
        return None
    value = re.sub(r"\s+", " ", str(value).strip())
    return value if value else None


# ── Registry ─────────────────────────────────────────────────────────────────

STANDARDIZERS = {
    "email": std_email,
    "name_proper": std_name_proper,
    "name_upper": std_name_upper,
    "name_lower": std_name_lower,
    "phone": std_phone,
    "zip5": std_zip5,
    "address": std_address,
    "state": std_state,
    "strip": std_strip,
    "trim_whitespace": std_trim_whitespace,
}


def get_standardizer(name: str):
    """Get a standardizer function by name.

    Raises ValueError if name is not recognized.
    """
    if name not in STANDARDIZERS:
        raise ValueError(
            f"Unknown standardizer: {name!r}. "
            f"Available: {sorted(STANDARDIZERS.keys())}"
        )
    return STANDARDIZERS[name]


# ── Apply to DataFrame ──────────────────────────────────────────────────────


def apply_standardization(
    lf: pl.LazyFrame,
    rules: dict[str, list[str]],
) -> pl.LazyFrame:
    """Apply standardization rules to a LazyFrame.

    Args:
        lf: Input LazyFrame.
        rules: Mapping of column_name -> list of standardizer names.
            e.g. {"email": ["email"], "last_name": ["strip", "name_upper"]}
            Standardizers are applied in order (chained).

    Returns:
        LazyFrame with standardized columns.
    """
    available_cols = set(lf.collect_schema().names())
    exprs = []

    for column, std_names in rules.items():
        if column not in available_cols:
            logger.warning(
                f"Standardization: column {column!r} not found in data, skipping."
            )
            continue

        # Build a chained function from the standardizer list
        funcs = [get_standardizer(name) for name in std_names]

        def chained_fn(val, _funcs=funcs):
            for fn in _funcs:
                val = fn(val)
            return val

        expr = (
            pl.col(column)
            .cast(pl.Utf8)
            .map_elements(chained_fn, return_dtype=pl.Utf8)
            .alias(column)
        )
        exprs.append(expr)

    if exprs:
        lf = lf.with_columns(exprs)

    return lf
