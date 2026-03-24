"""GoldenCheck integration — enhanced data quality scanning before matching."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _goldencheck_available() -> bool:
    """Check if goldencheck is installed."""
    try:
        import goldencheck  # noqa: F401
        return True
    except ImportError:
        return False


def run_quality_check(
    df: pl.DataFrame,
    config=None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Run GoldenCheck scan + fix if available.

    Returns (fixed_df, list_of_fixes) matching autofix format.
    Falls back gracefully if goldencheck is not installed.
    """
    if not _goldencheck_available():
        return df, []

    # Parse config
    enabled = True
    mode = "announced"
    fix_mode = "safe"
    domain = None

    if config is not None:
        mode = getattr(config, "mode", "announced")
        fix_mode = getattr(config, "fix_mode", "safe")
        domain = getattr(config, "domain", None)
        enabled = getattr(config, "enabled", True)

    if not enabled or mode == "disabled":
        return df, []

    if fix_mode == "none":
        # Scan only, no fixes
        return _scan_only(df, mode, domain)

    return _scan_and_fix(df, mode, fix_mode, domain)


def _scan_only(
    df: pl.DataFrame,
    mode: str,
    domain: str | None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Run GoldenCheck scan without fixes. Reports findings."""
    from goldencheck.engine.scanner import scan_file
    from goldencheck.engine.confidence import apply_confidence_downgrade
    from goldencheck.models.finding import Severity

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.write_csv(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        findings, _ = scan_file(tmp_path, domain=domain)
        findings = apply_confidence_downgrade(findings, llm_boost=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    errors = sum(1 for f in findings if f.severity == Severity.ERROR)
    warnings = sum(1 for f in findings if f.severity == Severity.WARNING)

    if mode == "announced":
        logger.info(
            "GoldenCheck: %d issues found (%d errors, %d warnings)",
            len(findings), errors, warnings,
        )

    return df, []


def _scan_and_fix(
    df: pl.DataFrame,
    mode: str,
    fix_mode: str,
    domain: str | None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Run GoldenCheck scan + apply fixes."""
    from goldencheck.engine.scanner import scan_file
    from goldencheck.engine.confidence import apply_confidence_downgrade
    from goldencheck.engine.fixer import apply_fixes

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.write_csv(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        findings, _ = scan_file(tmp_path, domain=domain)
        findings = apply_confidence_downgrade(findings, llm_boost=False)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Apply fixes
    fixed_df, report = apply_fixes(df, findings, mode=fix_mode)

    # Convert to autofix-compatible format
    fixes = []
    for entry in report.entries:
        fixes.append({
            "fix": f"goldencheck:{entry.fix_type}",
            "column": entry.column,
            "rows_affected": entry.rows_affected,
            "detail": (
                f"{entry.fix_type}: {entry.rows_affected} rows"
                + (f" (e.g., {entry.sample_before[0]} → {entry.sample_after[0]})"
                   if entry.sample_before and entry.sample_after else "")
            ),
        })

    if mode == "announced" and fixes:
        fix_types = set(e.fix_type for e in report.entries)
        print(
            f"GoldenCheck: scanning data quality... "
            f"{len(findings)} issues found, {len(fixes)} auto-fixed "
            f"({', '.join(sorted(fix_types))})"
        )
    elif mode == "announced":
        print("GoldenCheck: scanning data quality... no fixes needed")

    return fixed_df, fixes
