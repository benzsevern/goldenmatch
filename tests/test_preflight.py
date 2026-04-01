"""Tests for preflight system."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.config.schemas import SafetyPolicy, GoldenMatchConfig, BlockingConfig, BlockingKeyConfig, MatchkeyConfig, MatchkeyField, OutputConfig
from goldenmatch.core.preflight import (
    preflight,
    RunPlan,
    ResourceProjection,
    SampleStats,
    Downgrade,
    _take_sample,
    _extrapolate,
    _apply_downgrades,
    PreflightError,
)


def _make_df(n: int = 500) -> pl.DataFrame:
    return pl.DataFrame({
        "name": [f"Person {i}" for i in range(n)],
        "email": [f"p{i}@test.com" for i in range(n)],
        "zip": [f"{10000 + i % 100}" for i in range(n)],
    })


class TestSampling:
    def test_small_dataset_no_sample(self):
        df = _make_df(100)
        sample = _take_sample(df)
        assert sample.height == 100

    def test_default_sample_5k(self):
        df = _make_df(50000)
        sample = _take_sample(df)
        assert sample.height == 5000

    def test_one_percent_when_larger(self):
        df = _make_df(800000)
        sample = _take_sample(df)
        assert sample.height == 8000

    def test_cap_at_10k(self):
        df = _make_df(2000000)
        sample = _take_sample(df)
        assert sample.height == 10000


class TestExtrapolation:
    def test_comparison_count_from_blocking_stats(self):
        blocking_stats = {"a": 10, "b": 20, "c": 5}
        proj = _extrapolate(
            blocking_stats=blocking_stats,
            sample_comparisons=50,
            sample_peak_memory_mb=100.0,
            sample_llm_calls=5,
            sample_llm_cost=0.01,
            sample_wall_time=2.0,
            sample_rows=500,
            total_rows=5000,
        )
        assert proj.total_comparisons == 45 + 190 + 10  # n*(n-1)/2

    def test_risk_level_safe(self):
        proj = _extrapolate(
            blocking_stats={"a": 10},
            sample_comparisons=10,
            sample_peak_memory_mb=50.0,
            sample_llm_calls=0,
            sample_llm_cost=0.0,
            sample_wall_time=1.0,
            sample_rows=100,
            total_rows=1000,
        )
        assert proj.risk_level == "safe"


class TestDowngrades:
    def test_conservative_adds_skip_oversized(self):
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="test", type="weighted", threshold=0.8,
                fields=[MatchkeyField(field="name", scorer="ensemble", weight=1.0)])],
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase"])],
                skip_oversized=False,
            ),
            output=OutputConfig(),
        )
        policy = SafetyPolicy(max_comparisons=100, mode="conservative")
        proj = ResourceProjection(
            total_comparisons=1_000_000,
            estimated_memory_mb=500,
            estimated_llm_calls=0,
            estimated_llm_cost_usd=0.0,
            estimated_wall_time_seconds=60,
            risk_level="danger",
        )
        adjusted, downgrades = _apply_downgrades(config, proj, policy)
        assert adjusted.blocking.skip_oversized is True
        assert len(downgrades) > 0

    def test_safety_none_skips_downgrades(self):
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="test", type="weighted", threshold=0.8,
                fields=[MatchkeyField(field="name", scorer="ensemble", weight=1.0)])],
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase"])],
            ),
            output=OutputConfig(),
        )
        policy = SafetyPolicy(mode="none")
        proj = ResourceProjection(
            total_comparisons=999_999_999,
            estimated_memory_mb=99999,
            estimated_llm_calls=0,
            estimated_llm_cost_usd=0.0,
            estimated_wall_time_seconds=99999,
            risk_level="danger",
        )
        adjusted, downgrades = _apply_downgrades(config, proj, policy)
        assert len(downgrades) == 0


class TestPreflightSmallDataset:
    def test_small_df_returns_plan(self):
        df = _make_df(100)
        plan = preflight(df, safety="conservative")
        assert plan is not None
        assert isinstance(plan, RunPlan)
        assert len(plan.downgrades) == 0
