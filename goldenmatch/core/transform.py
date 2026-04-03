"""GoldenFlow integration -- data transformation before matching."""
from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def _goldenflow_available() -> bool:
    """Check if goldenflow is installed."""
    try:
        import goldenflow  # noqa: F401
        return True
    except ImportError as e:
        logger.debug("goldenflow not available: %s", e)
        return False


def _do_transform(df: pl.DataFrame):
    """Call goldenflow.transform_df. Separated for testability."""
    from goldenflow import transform_df
    return transform_df(df)


def run_transform(
    df: pl.DataFrame,
    config=None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Run GoldenFlow transform if available.

    Returns (transformed_df, list_of_fixes) matching autofix format.
    Falls back gracefully if goldenflow is not installed.
    """
    if not _goldenflow_available():
        if config is not None and getattr(config, "enabled", True):
            logger.warning(
                "GoldenFlow transforms configured but goldenflow is not installed. "
                "Install with: pip install goldenmatch[transform]"
            )
        return df, []

    # Parse config
    enabled = True
    mode = "announced"

    if config is not None:
        mode = getattr(config, "mode", "announced")
        enabled = getattr(config, "enabled", True)

    if not enabled or mode == "disabled":
        return df, []

    try:
        result = _do_transform(df)
    except Exception:
        logger.warning("GoldenFlow: transform failed, skipping", exc_info=True)
        return df, []

    # Convert manifest to autofix-compatible format
    fixes = []
    for record in result.manifest.records:
        fixes.append({
            "fix": f"goldenflow:{record.transform}",
            "column": record.column,
            "rows_affected": record.affected_rows,
            "detail": (
                f"{record.transform}: {record.affected_rows} rows"
                + (f" (e.g., {record.sample_before[0]} -> {record.sample_after[0]})"
                   if record.sample_before and record.sample_after else "")
            ),
        })

    if mode == "announced" and fixes:
        fix_types = set(record.transform for record in result.manifest.records)
        logger.info(
            "GoldenFlow: %d transforms applied (%s)",
            len(fixes), ", ".join(sorted(fix_types)),
        )
    elif mode == "announced":
        logger.info("GoldenFlow: no transforms needed")

    return result.df, fixes
