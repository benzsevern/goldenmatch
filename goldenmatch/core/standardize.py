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


# ── Native Polars expressions for simple standardizers ──────────────────────

# Standardizers that can be expressed as native Polars (no Python UDF needed).
# Each returns a function that takes a pl.Expr and returns a pl.Expr.
# Must replicate exact behavior of the Python functions above, including
# empty-string-to-null handling.


def _null_if_empty(expr: pl.Expr) -> pl.Expr:
    """Replace empty strings with null."""
    return pl.when(expr.str.len_chars() == 0).then(None).otherwise(expr)


def _native_strip(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_strip: strip whitespace, empty -> null."""
    e = expr.str.strip_chars()
    return _null_if_empty(e)


def _native_name_upper(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_name_upper: strip + uppercase, empty -> null."""
    e = expr.str.strip_chars().str.to_uppercase()
    return _null_if_empty(e)


def _native_name_lower(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_name_lower: strip + lowercase, empty -> null."""
    e = expr.str.strip_chars().str.to_lowercase()
    return _null_if_empty(e)


def _native_state(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_state: strip + uppercase, empty -> null."""
    e = expr.str.strip_chars().str.to_uppercase()
    return _null_if_empty(e)


def _native_trim_whitespace(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_trim_whitespace: strip + collapse whitespace, empty -> null."""
    e = expr.str.strip_chars().str.replace_all(r"\s+", " ")
    return _null_if_empty(e)


def _native_phone(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_phone: digits only, strip US country code, null if < 7 digits."""
    digits = expr.str.replace_all(r"\D", "")
    # Strip leading "1" if 11 digits
    stripped = (
        pl.when(digits.str.len_chars() == 11)
        .then(digits.str.slice(1))
        .otherwise(digits)
    )
    # Null if < 7 digits or empty
    return (
        pl.when(stripped.str.len_chars() < 7)
        .then(None)
        .otherwise(stripped)
    )


def _native_zip5(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_zip5: first 5 digits, zero-padded."""
    # Split on dash/space, take first part, extract digits, pad to 5
    digits = expr.str.replace_all(r"[^0-9]", "")
    sliced = digits.str.slice(0, 5)
    padded = sliced.str.pad_start(5, "0")
    return (
        pl.when(digits.str.len_chars() == 0)
        .then(None)
        .otherwise(padded)
    )


def _native_email(expr: pl.Expr) -> pl.Expr:
    """Native equivalent of std_email: lowercase, strip, null if invalid."""
    cleaned = expr.str.strip_chars().str.to_lowercase()
    # Valid if contains @ and has a dot after @
    has_at = cleaned.str.contains("@")
    # Check dot after @ using regex
    has_dot_after_at = cleaned.str.contains(r"@[^@]+\.")
    return (
        pl.when(has_at & has_dot_after_at & (cleaned.str.len_chars() > 0))
        .then(cleaned)
        .otherwise(None)
    )


# Map of standardizer names to native expression builders.
# Each takes a pl.Expr and returns a pl.Expr.
_NATIVE_STANDARDIZERS: dict[str, object] = {
    "strip": _native_strip,
    "name_upper": _native_name_upper,
    "name_lower": _native_name_lower,
    "state": _native_state,
    "trim_whitespace": _native_trim_whitespace,
    "phone": _native_phone,
    "zip5": _native_zip5,
    "email": _native_email,
}


def _try_build_native_chain(column: str, std_names: list[str]) -> pl.Expr | None:
    """Try to build a fully native Polars expression chain for the given standardizers.

    Returns None if any standardizer in the chain requires map_elements.
    """
    # Only use native path if ALL standardizers in the chain are natively expressible
    for name in std_names:
        if name not in _NATIVE_STANDARDIZERS:
            return None

    expr = pl.col(column).cast(pl.Utf8)
    for name in std_names:
        native_fn = _NATIVE_STANDARDIZERS[name]
        expr = native_fn(expr)

    return expr.alias(column)


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

        # Try fully native chain first (fastest)
        native_expr = _try_build_native_chain(column, std_names)
        if native_expr is not None:
            exprs.append(native_expr)
            continue

        # If only one standardizer has no native, use native for the ones that do
        # and map_elements only for the non-native ones
        non_native = [n for n in std_names if n not in _NATIVE_STANDARDIZERS]
        if len(non_native) < len(std_names):
            # Split into native prefix, non-native middle, native suffix
            # For simplicity: apply native ones as expressions, non-native as map_elements
            native_names = [n for n in std_names if n in _NATIVE_STANDARDIZERS]
            non_native_names = [n for n in std_names if n not in _NATIVE_STANDARDIZERS]

            # Apply native first
            expr = pl.col(column).cast(pl.Utf8)
            for name in native_names:
                expr = _NATIVE_STANDARDIZERS[name](expr)

            # Then apply non-native via map_elements
            funcs = [get_standardizer(name) for name in non_native_names]

            def chained_fn(val, _funcs=funcs):
                for fn in _funcs:
                    val = fn(val)
                return val

            expr = expr.map_elements(chained_fn, return_dtype=pl.Utf8).alias(column)
            exprs.append(expr)
        else:
            # All non-native, pure map_elements
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
