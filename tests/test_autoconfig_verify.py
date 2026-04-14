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


def test_preflight_check1_missing_raw_column_errors():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"name": ["a"]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["nonexistent"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="exact", fields=[MatchkeyField(field="nonexistent")])],
    )
    report = preflight(df, cfg)
    assert report.has_errors
    err = [f for f in report.findings if f.check == "missing_column"][0]
    assert "nonexistent" in err.message


def test_preflight_check1_domain_auto_repair():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"title": ["a"], "authors": ["b"]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["__title_key__"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="exact", fields=[MatchkeyField(field="__title_key__")])],
    )
    class StubProfile:
        name = "bibliographic"
    cfg._domain_profile = StubProfile()
    report = preflight(df, cfg)
    assert not report.has_errors
    assert cfg.domain is not None and cfg.domain.enabled is True
    assert cfg.domain.mode == "bibliographic"
    repaired = [f for f in report.findings if f.repaired]
    assert any(f.check == "missing_column" for f in repaired)


def test_preflight_check2_drops_high_cardinality_exact_matchkey():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"id": list(range(100)), "name": ["alice"] * 100})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[
            MatchkeyConfig(name="mk_id", type="exact", fields=[MatchkeyField(field="id")]),
            MatchkeyConfig(name="mk_name", type="weighted", threshold=0.7,
                           fields=[MatchkeyField(field="name", scorer="exact", weight=1.0)]),
        ],
    )
    report = preflight(df, cfg)
    remaining_names = [mk.name for mk in cfg.get_matchkeys()]
    assert "mk_id" not in remaining_names
    warnings = [f for f in report.findings if f.check == "cardinality_high"]
    assert warnings and warnings[0].repaired


def test_preflight_check3_drops_low_cardinality_exact_matchkey():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"state": ["NC"] * 100, "last_name": [f"name{i}" for i in range(100)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["last_name"])]),
        matchkeys=[
            MatchkeyConfig(name="mk_state", type="exact", fields=[MatchkeyField(field="state")]),
            MatchkeyConfig(name="mk_name", type="weighted", threshold=0.7,
                           fields=[MatchkeyField(field="last_name", scorer="exact", weight=1.0)]),
        ],
    )
    report = preflight(df, cfg)
    assert "mk_state" not in [mk.name for mk in cfg.get_matchkeys()]
    warnings = [f for f in report.findings if f.check == "cardinality_low"]
    assert warnings and warnings[0].repaired


def test_preflight_no_matchkeys_after_drops_is_hard_error():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"id": list(range(100))})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["id"])]),
        matchkeys=[MatchkeyConfig(name="mk_id", type="exact", fields=[MatchkeyField(field="id")])],
    )
    report = preflight(df, cfg)
    assert report.has_errors
    assert any(f.check == "no_matchkeys_remain" for f in report.findings)


def test_preflight_check4_warns_on_mega_block():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    # 10000 rows, all same state code → one mega-block
    n = 10000
    df = pl.DataFrame({"state": ["NC"] * n, "last_name": [f"name{i}" for i in range(n)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["state"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7,
                                  fields=[MatchkeyField(field="last_name", scorer="token_sort", weight=1.0)])],
    )
    report = preflight(df, cfg)
    warnings = [f for f in report.findings if f.check == "block_size"]
    assert warnings and not warnings[0].repaired


def test_preflight_check5_demotes_embedding_by_default():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"name": ["alice", "bob"], "desc": ["x", "y"]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="desc", scorer="embedding", weight=1.0),
        ])],
    )
    report = preflight(df, cfg, allow_remote_assets=False)
    demoted = [mk for mk in cfg.get_matchkeys() for f in mk.fields if f.scorer == "ensemble"]
    assert demoted
    findings = [f for f in report.findings if f.check == "remote_asset"]
    assert findings and findings[0].repaired


def test_preflight_check5_keeps_embedding_when_allowed():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"name": ["alice", "bob"], "desc": ["x", "y"]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="desc", scorer="embedding", weight=1.0),
        ])],
    )
    preflight(df, cfg, allow_remote_assets=True)
    assert any(f.scorer == "embedding" for mk in cfg.get_matchkeys() for f in mk.fields)
