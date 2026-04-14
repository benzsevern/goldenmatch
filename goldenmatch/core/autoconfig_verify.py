"""Auto-configuration verification: preflight (and later, postflight) checks.

See spec: docs/superpowers/specs/2026-04-14-autoconfig-verification-design.md §4.

Preflight runs right before `auto_configure_df` returns. It validates that the
generated `GoldenMatchConfig` is internally consistent with the DataFrame it
was derived from, auto-repairs issues where it can, and raises
`ConfigValidationError` on unrepairable errors.

Every check produces a `PreflightFinding`. A `PreflightReport` aggregates them
and is attached to the returned config as ``config._preflight_report`` for
downstream introspection (Postflight, diagnostics, tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl

    from goldenmatch.config.schemas import GoldenMatchConfig
    from goldenmatch.core.autoconfig import ColumnProfile


@dataclass
class PreflightFinding:
    """One check result — informational, warning, or hard error."""

    check: str
    severity: Literal["error", "warning", "info"]
    subject: str
    message: str
    repaired: bool
    repair_note: str | None


@dataclass
class PreflightReport:
    """Aggregated result of running all preflight checks."""

    findings: list[PreflightFinding] = field(default_factory=list)
    config_was_modified: bool = False

    @property
    def has_errors(self) -> bool:
        """True if any unrepaired error-severity finding exists."""
        return any(
            f.severity == "error" and not f.repaired for f in self.findings
        )


class ConfigValidationError(Exception):
    """Raised when preflight finds unrepairable configuration errors.

    The full `PreflightReport` is attached as ``self.report`` for callers that
    want to inspect findings programmatically.
    """

    def __init__(self, *, report: PreflightReport) -> None:
        self.report = report
        unrepaired = [
            f for f in report.findings
            if f.severity == "error" and not f.repaired
        ]
        msg_parts = [f"{f.check}: {f.message}" for f in unrepaired]
        super().__init__(
            f"auto-config produced {len(unrepaired)} unrepairable error(s): "
            + "; ".join(msg_parts)
        )


# ── Column collection ────────────────────────────────────────────────────


def _collect_referenced_columns(config: "GoldenMatchConfig") -> set[str]:
    """Walk the config and return every raw column name it references.

    Collects from blocking (keys + passes) and all matchkey fields.
    """
    cols: set[str] = set()
    if config.blocking is not None:
        for key in config.blocking.keys or []:
            cols.update(key.fields)
        for key in (config.blocking.passes or []):
            cols.update(key.fields)
    for mk in config.get_matchkeys():
        for f in mk.fields:
            if f.field is not None and f.field != "__record__":
                cols.add(f.field)
            if f.column is not None:
                cols.add(f.column)
            if f.columns:
                cols.update(f.columns)
    return cols


def _check_columns(
    df: "pl.DataFrame", config: "GoldenMatchConfig", report: PreflightReport
) -> None:
    """Check 1: every referenced column exists, or is pipeline-synthesized,
    or is a domain-extracted column recoverable via domain repair.
    """
    from goldenmatch.core.domain import _DOMAIN_EXTRACTED_COLS

    df_cols = set(df.columns)
    referenced = _collect_referenced_columns(config)
    domain_profile = getattr(config, "_domain_profile", None)

    for col in sorted(referenced):
        if col in df_cols:
            continue
        # Pipeline-synthesized matchkey columns — safe, created at runtime.
        if col.startswith("__mk_"):
            continue
        # Domain-extracted column: auto-repair by enabling DomainConfig if
        # a domain profile was stashed by auto_configure_df.
        if col in _DOMAIN_EXTRACTED_COLS and domain_profile is not None:
            _repair_domain(config, domain_profile, report, subject=col)
            continue
        # Unrepairable: raw column does not exist in the DataFrame.
        report.findings.append(
            PreflightFinding(
                check="missing_column",
                severity="error",
                subject=col,
                message=(
                    f"column '{col}' referenced by config but not present in "
                    f"DataFrame (columns: {sorted(df_cols)[:10]}...)"
                ),
                repaired=False,
                repair_note=None,
            )
        )


def _repair_domain(
    config: "GoldenMatchConfig",
    domain_profile: object,
    report: PreflightReport,
    *,
    subject: str,
) -> None:
    """Enable DomainConfig so the pipeline produces the extracted columns."""
    from goldenmatch.config.schemas import DomainConfig

    already_enabled = (
        config.domain is not None and config.domain.enabled is True
    )
    if not already_enabled:
        config.domain = DomainConfig(
            enabled=True, mode=getattr(domain_profile, "name", None)
        )
        report.config_was_modified = True

    report.findings.append(
        PreflightFinding(
            check="missing_column",
            severity="error",
            subject=subject,
            message=(
                f"column '{subject}' is produced by domain extraction; "
                f"enabled config.domain (mode={getattr(domain_profile, 'name', None)})"
            ),
            repaired=True,
            repair_note=(
                "config.domain enabled so the pipeline's domain-extraction "
                "step produces this column at runtime"
            ),
        )
    )


# ── Entry point ──────────────────────────────────────────────────────────


def preflight(
    df: "pl.DataFrame",
    config: "GoldenMatchConfig",
    *,
    profiles: "list[ColumnProfile] | None" = None,
    allow_remote_assets: bool = False,
) -> PreflightReport:
    """Run all preflight checks on (df, config).

    Auto-repairs what it can (annotating each finding with ``repaired=True``)
    and records unrepairable issues as error findings. Callers should inspect
    ``report.has_errors`` and raise `ConfigValidationError` if strict behavior
    is desired — `auto_configure_df` does this.
    """
    report = PreflightReport()
    _check_columns(df, config, report)
    return report
