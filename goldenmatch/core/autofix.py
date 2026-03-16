"""Auto-fix module for common data issues in GoldenMatch."""

from __future__ import annotations

import re

import polars as pl


# Strings that should be treated as null
_NULL_STRINGS = frozenset({
    "NULL", "N/A", "NA", "n/a", "None", "none", "", "-", ".", "nan", "NaN",
})

# Control characters to strip (everything below 0x20 except \t and \n)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def auto_fix_dataframe(
    df: pl.DataFrame,
    profile: dict | None = None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Automatically detect and fix common data issues.

    Returns (fixed_df, list_of_fixes_applied). Each fix is a dict with keys:
    fix, column, rows_affected, detail.

    Fixes applied in order:
    1. Strip BOM characters
    2. Drop fully empty rows
    3. Drop fully null columns
    4. Trim whitespace
    5. Normalize null representations
    6. Collapse repeated whitespace
    7. Remove non-printable characters
    """
    fixes: list[dict] = []
    string_cols = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String)]

    # ── 1. Strip BOM characters ──────────────────────────────────────────
    bom_affected = 0
    if string_cols:
        for col in string_cols:
            series = df[col]
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                has_bom = non_null.str.contains("\ufeff")
                count = int(has_bom.sum())
                if count > 0:
                    bom_affected += count
                    df = df.with_columns(
                        pl.col(col).str.replace_all("\ufeff", "").alias(col)
                    )
    fixes.append({
        "fix": "strip_bom",
        "column": None,
        "rows_affected": bom_affected,
        "detail": f"Removed BOM characters from {bom_affected} cell(s)",
    })

    # ── 2. Drop fully empty rows ─────────────────────────────────────────
    # Recalculate string_cols in case columns were modified
    empty_conditions = []
    for col in df.columns:
        if df[col].dtype in (pl.Utf8, pl.String):
            empty_conditions.append(
                pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")
            )
        else:
            empty_conditions.append(pl.col(col).is_null())

    if empty_conditions:
        all_empty_expr = empty_conditions[0]
        for cond in empty_conditions[1:]:
            all_empty_expr = all_empty_expr & cond
        empty_mask = df.select(all_empty_expr.alias("__empty__"))["__empty__"]
        empty_count = int(empty_mask.sum())
        if empty_count > 0:
            df = df.filter(~all_empty_expr)
    else:
        empty_count = 0

    fixes.append({
        "fix": "drop_empty_rows",
        "column": None,
        "rows_affected": empty_count,
        "detail": f"Dropped {empty_count} fully empty row(s)",
    })

    # ── 3. Drop fully null columns ───────────────────────────────────────
    null_cols_dropped: list[str] = []
    if profile is not None and "columns" in profile:
        # Use profile data
        for cp in profile["columns"]:
            if cp["name"] in df.columns and cp["null_rate"] >= 1.0:
                null_cols_dropped.append(cp["name"])
    else:
        # Compute
        for col in df.columns:
            if df[col].null_count() == df.height:
                null_cols_dropped.append(col)

    if null_cols_dropped:
        df = df.drop(null_cols_dropped)

    fixes.append({
        "fix": "drop_null_columns",
        "column": ", ".join(null_cols_dropped) if null_cols_dropped else None,
        "rows_affected": len(null_cols_dropped),
        "detail": f"Dropped {len(null_cols_dropped)} fully null column(s): {null_cols_dropped}" if null_cols_dropped else "No fully null columns found",
    })

    # Refresh string cols after column drops
    string_cols = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String)]

    # ── 4. Trim whitespace ───────────────────────────────────────────────
    trim_affected = 0
    if string_cols:
        for col in string_cols:
            series = df[col]
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                stripped = non_null.str.strip_chars()
                changed = (non_null != stripped)
                count = int(changed.sum())
                if count > 0:
                    trim_affected += count
                    df = df.with_columns(
                        pl.col(col).str.strip_chars().alias(col)
                    )
    fixes.append({
        "fix": "trim_whitespace",
        "column": None,
        "rows_affected": trim_affected,
        "detail": f"Trimmed whitespace in {trim_affected} cell(s)",
    })

    # ── 5. Normalize null representations ────────────────────────────────
    null_norm_affected = 0
    if string_cols:
        for col in string_cols:
            series = df[col]
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                is_null_str = non_null.is_in(list(_NULL_STRINGS))
                count = int(is_null_str.sum())
                if count > 0:
                    null_norm_affected += count
                    df = df.with_columns(
                        pl.when(pl.col(col).is_in(list(_NULL_STRINGS)))
                        .then(None)
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
    fixes.append({
        "fix": "normalize_nulls",
        "column": None,
        "rows_affected": null_norm_affected,
        "detail": f"Normalized {null_norm_affected} null-like string(s) to actual null",
    })

    # Refresh string cols (types haven't changed)
    string_cols = [c for c in df.columns if df[c].dtype in (pl.Utf8, pl.String)]

    # ── 6. Collapse repeated whitespace ──────────────────────────────────
    collapse_affected = 0
    if string_cols:
        for col in string_cols:
            series = df[col]
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                collapsed = non_null.str.replace_all(r"\s{2,}", " ")
                changed = (non_null != collapsed)
                count = int(changed.sum())
                if count > 0:
                    collapse_affected += count
                    df = df.with_columns(
                        pl.col(col).str.replace_all(r"\s{2,}", " ").alias(col)
                    )
    fixes.append({
        "fix": "collapse_whitespace",
        "column": None,
        "rows_affected": collapse_affected,
        "detail": f"Collapsed repeated whitespace in {collapse_affected} cell(s)",
    })

    # ── 7. Remove non-printable characters ───────────────────────────────
    nonprint_affected = 0
    if string_cols:
        for col in string_cols:
            series = df[col]
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                has_ctrl = non_null.str.contains(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
                count = int(has_ctrl.sum())
                if count > 0:
                    nonprint_affected += count
                    # Use map_elements for control char removal
                    df = df.with_columns(
                        pl.col(col).map_elements(
                            lambda v, _re=_CONTROL_CHAR_RE: _re.sub("", v) if v is not None else None,
                            return_dtype=pl.Utf8,
                        ).alias(col)
                    )
    fixes.append({
        "fix": "remove_non_printable",
        "column": None,
        "rows_affected": nonprint_affected,
        "detail": f"Removed non-printable characters from {nonprint_affected} cell(s)",
    })

    return df, fixes
