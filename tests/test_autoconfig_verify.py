"""Unit tests for autoconfig_verify preflight + postflight."""
import polars as pl
import pytest
from goldenmatch.core.autoconfig_verify import (
    PreflightReport, PreflightFinding, ConfigValidationError,
)


def test_preflight_report_dataclass_shape():
    f = PreflightFinding(check="x", severity="info", subject="y",
                         message="z", repaired=False, repair_note=None)
    r = PreflightReport(findings=[f], config_was_modified=False)
    assert not r.has_errors
    assert len(r.findings) == 1


def test_config_validation_error_carries_report():
    f = PreflightFinding(check="missing_column", severity="error", subject="bad_col",
                         message="missing", repaired=False, repair_note=None)
    r = PreflightReport(findings=[f], config_was_modified=False)
    with pytest.raises(ConfigValidationError) as exc:
        raise ConfigValidationError(report=r)
    assert exc.value.report is r
    assert r.has_errors
