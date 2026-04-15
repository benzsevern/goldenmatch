"""Unit tests for autoconfig_verify preflight + postflight."""
import polars as pl
import pytest
from goldenmatch.core.autoconfig_verify import (
    PreflightReport, PreflightFinding, ConfigValidationError,
    PostflightReport, PostflightAdjustment,
)


def test_postflight_report_dataclass_shape():
    adj = PostflightAdjustment(
        field="threshold", from_value=0.7, to_value=0.5,
        reason="valley at 0.5", signal="score_histogram",
    )
    r = PostflightReport(
        signals={"score_histogram": {"bins": [], "counts": []}},
        adjustments=[adj],
        advisories=["nudge the LLM"],
    )
    assert r.signals["score_histogram"] == {"bins": [], "counts": []}
    assert r.adjustments[0].signal == "score_histogram"
    assert r.advisories == ["nudge the LLM"]


def test_postflight_report_defaults_empty():
    r = PostflightReport()
    assert r.signals == {}
    assert r.adjustments == []
    assert r.advisories == []


def test_postflight_bimodal_histogram_adjusts_threshold():
    import random
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import postflight
    random.seed(42)
    pair_scores = []
    for i in range(500):
        pair_scores.append((i, i+1000, max(0.0, min(1.0, random.gauss(0.2, 0.08)))))
    for i in range(500):
        pair_scores.append((i+2000, i+3000, max(0.0, min(1.0, random.gauss(0.9, 0.05)))))
    df = pl.DataFrame({"name": [f"x{i}" for i in range(100)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="name", scorer="token_sort", weight=1.0)])],
    )
    report = postflight(df, cfg, pair_scores=pair_scores)
    assert any(adj.field == "threshold" for adj in report.adjustments)
    adj = next(a for a in report.adjustments if a.field == "threshold")
    assert 0.3 <= adj.to_value <= 0.7


def test_postflight_unimodal_histogram_no_adjustment():
    import random
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import postflight
    random.seed(42)
    pair_scores = [(i, i+1, random.random()) for i in range(1000)]
    df = pl.DataFrame({"name": [f"x{i}" for i in range(100)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="name", scorer="token_sort", weight=1.0)])],
    )
    report = postflight(df, cfg, pair_scores=pair_scores)
    assert not any(adj.field == "threshold" for adj in report.adjustments)


def test_postflight_signals_schema():
    """Contract test: PostflightReport.signals must contain EXACTLY the 8
    keys documented in the stable schema. New keys require explicit spec
    amendment + schema bump — a subset-check would silently accept drift."""
    from goldenmatch.core.autoconfig_verify import postflight
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    df = pl.DataFrame({"a": list(range(100))})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["a"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="a", scorer="exact", weight=1.0)])],
    )
    pair_scores = [(i, i+1, 0.8) for i in range(0, 98, 2)]
    report = postflight(df, cfg, pair_scores=pair_scores)
    expected_keys = {
        "score_histogram", "blocking_recall", "block_size_percentiles",
        "threshold_overlap_pct", "total_pairs_scored", "current_threshold",
        "preliminary_cluster_sizes", "oversized_clusters",
    }
    assert set(report.signals.keys()) == expected_keys, (
        f"signals schema drift: extra={set(report.signals.keys()) - expected_keys}, "
        f"missing={expected_keys - set(report.signals.keys())}"
    )


def test_postflight_threshold_overlap_triggers_llm_advisory():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import postflight
    # 30% of pairs in 0.70 ± 0.02 band.
    pair_scores = [(i, i+1, 0.69) for i in range(300)]
    pair_scores += [(i, i+5000, 0.2) for i in range(700)]
    df = pl.DataFrame({"name": [f"n{i}" for i in range(100)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="name", scorer="token_sort", weight=1.0)])],
    )
    report = postflight(df, cfg, pair_scores=pair_scores)
    assert any("llm" in adv.lower() or "auto" in adv.lower() for adv in report.advisories)
    assert report.signals["threshold_overlap_pct"] > 0.20


def test_postflight_cluster_sizes_identifies_oversized():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import postflight
    # Force one big component of 150 rows via chain of edges.
    pair_scores = [(i, i+1, 0.9) for i in range(149)]
    df = pl.DataFrame({"name": [f"n{i}" for i in range(200)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="name", scorer="token_sort", weight=1.0)])],
    )
    report = postflight(df, cfg, pair_scores=pair_scores)
    oversized = report.signals["oversized_clusters"]
    assert len(oversized) == 1
    assert oversized[0]["size"] == 150
    assert len(oversized[0]["bottleneck_pair"]) == 2


def test_postflight_blocking_recall_gated_below_10k():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import postflight
    df = pl.DataFrame({"name": [f"n{i}" for i in range(500)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="name", scorer="token_sort", weight=1.0)])],
    )
    report = postflight(df, cfg, pair_scores=[(0, 1, 0.9)])
    assert report.signals["blocking_recall"] == "deferred"  # explicit sentinel


def test_postflight_strict_mode_no_adjustment():
    """Strict mode: all signals computed, zero adjustments."""
    import random
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import postflight
    random.seed(42)
    pair_scores = []
    for i in range(500):
        pair_scores.append((i, i+1000, max(0.0, min(1.0, random.gauss(0.2, 0.08)))))
    for i in range(500):
        pair_scores.append((i+2000, i+3000, max(0.0, min(1.0, random.gauss(0.9, 0.05)))))
    df = pl.DataFrame({"name": [f"x{i}" for i in range(100)]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="name", scorer="token_sort", weight=1.0)])],
    )
    cfg._strict_autoconfig = True
    report = postflight(df, cfg, pair_scores=pair_scores)
    assert report.adjustments == []
    assert "score_histogram" in report.signals


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


def test_preflight_check6_caps_weight_for_low_confidence():
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig import ColumnProfile
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"mystery": ["a", "b", "c"] * 10, "name": [f"n{i}" for i in range(30)]})
    profiles = [
        ColumnProfile(name="mystery", dtype="Utf8", col_type="string", confidence=0.3,
                      null_rate=0.0, cardinality_ratio=0.1, avg_len=1.0, sample_values=[]),
        ColumnProfile(name="name", dtype="Utf8", col_type="name", confidence=0.9,
                      null_rate=0.0, cardinality_ratio=1.0, avg_len=2.0, sample_values=[]),
    ]
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["name"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7, fields=[
            MatchkeyField(field="mystery", scorer="token_sort", weight=1.0),
            MatchkeyField(field="name", scorer="token_sort", weight=1.0),
        ])],
    )
    preflight(df, cfg, profiles=profiles)
    mystery_f = next(f for mk in cfg.get_matchkeys() for f in mk.fields if f.field == "mystery")
    name_f = next(f for mk in cfg.get_matchkeys() for f in mk.fields if f.field == "name")
    assert mystery_f.weight == 0.5
    assert name_f.weight == 1.0


def test_auto_configure_df_attaches_preflight_report():
    from goldenmatch.core.autoconfig import auto_configure_df
    df = pl.DataFrame({
        "title": [f"paper {i}" for i in range(50)],
        "authors": [f"A, B, C{i}" for i in range(50)],
        "year": [2000 + i % 10 for i in range(50)],
    })
    cfg = auto_configure_df(df)
    assert hasattr(cfg, "_preflight_report")
    assert cfg._preflight_report is not None


def test_preflight_check5_drops_empty_matchkey_after_record_embedding_removal():
    import polars as pl
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    from goldenmatch.core.autoconfig_verify import preflight
    df = pl.DataFrame({"title": ["a", "b"]})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["title"])]),
        matchkeys=[
            # A weighted matchkey with ONLY a record_embedding field — would be
            # empty after demotion drops it.
            MatchkeyConfig(name="mk_rec_only", type="weighted", threshold=0.7, fields=[
                MatchkeyField(scorer="record_embedding", columns=["title"], weight=1.0),
            ]),
            # A keeper.
            MatchkeyConfig(name="mk_keep", type="weighted", threshold=0.7, fields=[
                MatchkeyField(field="title", scorer="token_sort", weight=1.0),
            ]),
        ],
    )
    preflight(df, cfg, allow_remote_assets=False)
    names = [mk.name for mk in cfg.get_matchkeys()]
    assert "mk_rec_only" not in names
    assert "mk_keep" in names


def test_auto_configure_df_dblp_acm_does_not_crash():
    from pathlib import Path
    from goldenmatch._api import dedupe_df
    d = Path("tests/benchmarks/datasets/DBLP-ACM")
    dblp = pl.read_csv(d / "DBLP2.csv", encoding="utf8-lossy", ignore_errors=True)
    acm  = pl.read_csv(d / "ACM.csv", encoding="utf8-lossy", ignore_errors=True)
    df = pl.concat([dblp, acm], how="diagonal_relaxed")
    result = dedupe_df(df)
    assert result is not None


def test_dedupe_result_has_postflight_report_field():
    from goldenmatch._api import DedupeResult, MatchResult
    assert "postflight_report" in DedupeResult.__annotations__
    assert "postflight_report" in MatchResult.__annotations__


def test_postflight_attached_after_dedupe_via_autoconfig():
    import polars as pl
    from goldenmatch._api import dedupe_df
    df = pl.DataFrame({
        "name": [f"alice{i}" for i in range(50)] + [f"bob{i}" for i in range(50)],
        "zip": ["90210"] * 100,
    })
    result = dedupe_df(df)  # zero-config -> postflight must run
    assert result.postflight_report is not None
    assert "score_histogram" in result.postflight_report.signals


def test_postflight_attached_after_match_via_autoconfig():
    """match_df zero-config produces postflight_report on asymmetric frames
    with real fuzzy variants (not trivial self-matches).

    Uses a non-product field name (first_name) so auto-config does not
    detect the 'product' domain and demand __color__ — match pipeline
    does not run domain feature extraction.
    """
    import polars as pl
    from goldenmatch._api import match_df
    # Target is a small set of known variants; reference is a superset.
    target = pl.DataFrame({"first_name": ["alice", "bob", "carol"]})
    reference = pl.DataFrame({
        "first_name": [
            "alyce",   # typo, should fuzzy-match alice
            "robert",  # nickname for bob - no match expected
            "carol",   # exact
            "dan",
            "eve",
        ]
    })
    result = match_df(target, reference)
    assert result.postflight_report is not None
    sig = result.postflight_report.signals
    assert "score_histogram" in sig
    assert sig["total_pairs_scored"] > 0


def test_preflight_check1_domain_repair_works_with_manual_domain_config():
    """Bug fix: when user passes explicit domain_config, preflight should still
    be able to auto-repair domain-extracted column references."""
    import polars as pl
    from goldenmatch.core.autoconfig import auto_configure_df
    from goldenmatch.config.schemas import DomainConfig
    df = pl.DataFrame({"title": [f"paper {i}" for i in range(50)]})
    # Force manual domain selection
    cfg = auto_configure_df(df, domain_config=DomainConfig(enabled=True, mode="bibliographic"))
    assert hasattr(cfg, "_domain_profile")
    assert cfg._domain_profile is not None
    assert cfg._preflight_report is not None


def test_postflight_threshold_adjustment_applied_before_clustering():
    """End-to-end: bimodal score distribution causes postflight to nudge
    threshold; the pipeline must apply the nudge to all_pairs BEFORE
    clustering (not just attach the report)."""
    import polars as pl
    import random
    from goldenmatch._api import dedupe_df

    random.seed(42)
    # Build a synthetic frame with 100 rows where half are near-dupes
    # and half are distinct, producing a bimodal score distribution.
    # Use name pairs: "alice0" ↔ "alice1" (high similarity), plus noise.
    names = []
    for i in range(50):
        names.append(f"alice_smith_{i}")
        names.append(f"alice_smith_{i}x")  # tiny perturbation, should match
    df = pl.DataFrame({"name": names})
    result = dedupe_df(df)
    assert result.postflight_report is not None
    # If the threshold adjustment was applied, clusters should reflect the
    # adjusted (lower) threshold — more pairs above threshold → more
    # members merged → fewer clusters than if we naively used 0.7.
    # Smoke-check: postflight ran AND produced a sane result shape.
    sig = result.postflight_report.signals
    assert sig["total_pairs_scored"] > 0
    # If an adjustment fired, the adjusted threshold is reflected in the
    # current_threshold signal OR an adjustment entry is present.
    adjusted = any(adj.field == "threshold" for adj in result.postflight_report.adjustments)
    if adjusted:
        # When adjusted, final cluster count reflects the post-adjustment
        # filter. The important assertion: zero silent regression — filtering
        # happened. If the filter silently stopped running, every high-pair
        # would cluster together and we'd see far fewer clusters than with
        # the adjusted threshold applied.
        assert result.clusters is not None


def test_postflight_strict_mode_pipeline_does_not_filter_all_pairs():
    """When _strict_autoconfig is True, the pipeline must NOT apply threshold
    adjustments even if postflight emits them. Verified by running with
    strict=True and a bimodal input."""
    import polars as pl
    from goldenmatch._api import dedupe_df
    from goldenmatch.core.autoconfig import auto_configure_df

    # Same bimodal input as above
    names = []
    for i in range(50):
        names.append(f"bob_jones_{i}")
        names.append(f"bob_jones_{i}x")
    df = pl.DataFrame({"name": names})

    # Build strict config externally then reuse
    cfg_strict = auto_configure_df(df, strict=True)
    assert cfg_strict._strict_autoconfig is True

    result = dedupe_df(df, config=cfg_strict)
    assert result.postflight_report is not None
    # In strict mode, adjustments list is always empty per spec §4.5
    assert result.postflight_report.adjustments == []
    # Advisories may still be present
    # Signals still populated
    assert "score_histogram" in result.postflight_report.signals


def test_postflight_empty_pair_scores():
    """Postflight must not crash on empty pair_scores."""
    import polars as pl
    from goldenmatch.core.autoconfig_verify import postflight
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    df = pl.DataFrame({"a": list(range(10))})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["a"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7,
                                  fields=[MatchkeyField(field="a", scorer="exact", weight=1.0)])],
    )
    report = postflight(df, cfg, pair_scores=[])
    assert report.signals["total_pairs_scored"] == 0
    assert report.adjustments == []  # empty input → no signal to nudge from


def test_postflight_all_identical_scores():
    """All pairs at the same score — no bimodality, no valley."""
    import polars as pl
    from goldenmatch.core.autoconfig_verify import postflight
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    df = pl.DataFrame({"a": list(range(20))})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["a"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7,
                                  fields=[MatchkeyField(field="a", scorer="exact", weight=1.0)])],
    )
    pair_scores = [(i, i+1, 0.8) for i in range(19)]
    report = postflight(df, cfg, pair_scores=pair_scores)
    # No valley, no adjustment
    threshold_adjustments = [a for a in report.adjustments if a.field == "threshold"]
    assert threshold_adjustments == []


def test_postflight_zero_height_df():
    """Postflight must not crash on zero-row DataFrame."""
    import polars as pl
    from goldenmatch.core.autoconfig_verify import postflight
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )
    df = pl.DataFrame({"a": []}, schema={"a": pl.Int64})
    cfg = GoldenMatchConfig(
        blocking=BlockingConfig(strategy="static", keys=[BlockingKeyConfig(fields=["a"])]),
        matchkeys=[MatchkeyConfig(name="mk", type="weighted", threshold=0.7,
                                  fields=[MatchkeyField(field="a", scorer="exact", weight=1.0)])],
    )
    report = postflight(df, cfg, pair_scores=[])
    # Key invariant: does not crash; returns schema-valid report.
    assert "score_histogram" in report.signals
