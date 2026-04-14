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
from typing import Literal


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
