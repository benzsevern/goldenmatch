"""Matchkey builder for GoldenMatch."""

from __future__ import annotations

import re

import polars as pl

from goldenmatch.config.schemas import MatchkeyConfig
from goldenmatch.utils.transforms import apply_transforms


def _try_native_chain(column: str, transforms: list[str]) -> pl.Expr | None:
    """Try to build a fully native Polars expression chain for transforms.

    Returns a Polars expression if ALL transforms are natively expressible,
    or None if any requires a Python UDF.
    """
    expr = pl.col(column).cast(pl.Utf8)
    for t in transforms:
        result = _try_native_transform(expr, t)
        if result is None:
            return None
        expr = result
    return expr


def _try_native_transform(expr: pl.Expr, transform: str) -> pl.Expr | None:
    """Try to apply a transform using native Polars expressions.

    Returns the transformed expression, or None if the transform
    requires a Python UDF (map_elements).
    """
    if transform == "lowercase":
        return expr.str.to_lowercase()
    elif transform == "uppercase":
        return expr.str.to_uppercase()
    elif transform == "strip":
        return expr.str.strip_chars()
    elif transform.startswith("substring:"):
        parts = transform.split(":")
        start = int(parts[1])
        length = int(parts[2]) - start
        return expr.str.slice(start, length)
    elif transform == "normalize_whitespace":
        return expr.str.replace_all(r"\s+", " ").str.strip_chars()
    elif transform == "strip_all":
        return expr.str.replace_all(r"\s+", "")
    elif transform == "digits_only":
        return expr.str.replace_all(r"[^0-9]", "")
    elif transform == "alpha_only":
        return expr.str.replace_all(r"[^a-zA-Z]", "")
    else:
        # soundex, metaphone, etc. need Python UDF
        return None


def _build_field_expr_native(field_name: str, transforms: list[str]) -> pl.Expr | None:
    """Try to build a fully native Polars expression for a field's transforms.

    Returns None if any transform requires map_elements.
    """
    expr = pl.col(field_name).cast(pl.Utf8)
    for t in transforms:
        result = _try_native_transform(expr, t)
        if result is None:
            return None
        expr = result
    return expr


def build_matchkey_expr(mk: MatchkeyConfig) -> pl.Expr:
    """Build a Polars expression for a matchkey.

    For exact matchkeys: transforms each field using native Polars expressions
    when possible, falling back to map_elements + apply_transforms for complex
    transforms (soundex, metaphone). Concatenates with "||" separator.
    Returns expr aliased as ``__mk_{mk.name}__``.

    For weighted matchkeys: returns pl.lit(None) placeholder (fuzzy scoring handled
    in scorer).

    Args:
        mk: The matchkey configuration.

    Returns:
        A Polars expression producing the matchkey column.
    """
    alias = f"__mk_{mk.name}__"

    if mk.type == "weighted":
        return pl.lit(None).alias(alias)

    # Exact matchkey: transform each field, then concatenate with "||"
    field_exprs = []
    for f in mk.fields:
        if f.transforms:
            # Try native Polars expressions first
            native_expr = _build_field_expr_native(f.field, f.transforms)
            if native_expr is not None:
                expr = native_expr
            else:
                # Fall back to map_elements for complex transforms
                expr = pl.col(f.field).map_elements(
                    lambda val, transforms=f.transforms: apply_transforms(val, transforms),
                    return_dtype=pl.Utf8,
                )
        else:
            expr = pl.col(f.field).cast(pl.Utf8)
        field_exprs.append(expr)

    if len(field_exprs) == 1:
        return field_exprs[0].alias(alias)

    return pl.concat_str(field_exprs, separator="||").alias(alias)


def compute_matchkeys(
    lf: pl.LazyFrame, matchkeys: list[MatchkeyConfig]
) -> pl.LazyFrame:
    """Add matchkey columns for all exact matchkeys.

    Args:
        lf: Input LazyFrame.
        matchkeys: List of matchkey configurations.

    Returns:
        LazyFrame with additional matchkey columns for each exact matchkey.
    """
    exprs = []
    for mk in matchkeys:
        if mk.type == "exact":
            exprs.append(build_matchkey_expr(mk))
    if exprs:
        lf = lf.with_columns(exprs)
    return lf
