"""Matchkey builder for GoldenMatch."""

from __future__ import annotations

import polars as pl

from goldenmatch.config.schemas import MatchkeyConfig
from goldenmatch.utils.transforms import apply_transforms


def build_matchkey_expr(mk: MatchkeyConfig) -> pl.Expr:
    """Build a Polars expression for a matchkey.

    For exact matchkeys: transforms each field using map_elements + apply_transforms,
    then concat_str with "||" separator. Returns expr aliased as ``__mk_{mk.name}__``.

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
