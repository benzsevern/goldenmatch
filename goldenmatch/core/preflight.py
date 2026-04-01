"""Preflight system -- sample-based resource estimation and auto-downgrade engine.

Runs a quick sample of the dataset through the pipeline to estimate resource
usage at full scale, then applies configuration downgrades if projections
exceed the safety policy limits.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import polars as pl
import psutil

from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    GoldenMatchConfig,
    SafetyPolicy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class PreflightError(Exception):
    """Raised when preflight encounters a fatal, unrecoverable problem."""


@dataclass
class ResourceProjection:
    """Projected resource usage at full dataset scale."""

    total_comparisons: int
    estimated_memory_mb: float
    estimated_llm_calls: int
    estimated_llm_cost_usd: float
    estimated_wall_time_seconds: float
    risk_level: str  # "safe", "caution", "danger"


@dataclass
class Downgrade:
    """A single configuration adjustment applied by the preflight engine."""

    field: str
    old_value: object
    new_value: object
    reason: str


@dataclass
class SampleStats:
    """Statistics collected from the sample run."""

    sample_rows: int
    comparisons: int
    peak_memory_mb: float
    llm_calls: int
    llm_cost: float
    wall_time_seconds: float


@dataclass
class RunPlan:
    """Complete preflight result: projections, downgrades, and the adjusted config."""

    projection: ResourceProjection
    downgrades: list[Downgrade] = field(default_factory=list)
    config: GoldenMatchConfig | None = None
    sample_stats: SampleStats | None = None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

_DEFAULT_SAMPLE = 5_000
_MAX_SAMPLE = 10_000


def _take_sample(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    """Return a representative sample of *df*.

    Rules:
    - If the dataset has <= 5 000 rows, return the full frame.
    - Default sample size is 5 000.
    - For large datasets, use 1% of rows (but at least 5 000, at most 10 000).
    """
    n = df.height
    if n <= _DEFAULT_SAMPLE:
        return df

    sample_size = max(_DEFAULT_SAMPLE, int(n * 0.01))
    sample_size = min(sample_size, _MAX_SAMPLE)
    return df.sample(n=sample_size, seed=seed)


# ---------------------------------------------------------------------------
# Blocking stats helper
# ---------------------------------------------------------------------------


def _get_blocking_stats(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
) -> dict[str, int]:
    """Compute block sizes by grouping on blocking keys.

    Returns a dict mapping block key value -> number of rows in that block.
    """
    if config.blocking is None or not config.blocking.keys:
        # No blocking keys -- the entire dataset is one block.
        return {"__all__": df.height}

    from goldenmatch.core.blocker import _build_block_key_expr

    stats: dict[str, int] = {}
    for key_config in config.blocking.keys:
        block_key_expr = _build_block_key_expr(key_config)
        grouped = (
            df.lazy()
            .with_columns(block_key_expr)
            .filter(pl.col("__block_key__").is_not_null())
            .group_by("__block_key__")
            .agg(pl.len().alias("__count__"))
            .collect()
        )
        for row in grouped.iter_rows(named=True):
            bk = row["__block_key__"]
            cnt = row["__count__"]
            if cnt >= 2:
                stats[bk] = cnt
    return stats


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------


def _extrapolate(
    blocking_stats: dict[str, int],
    sample_comparisons: int,
    sample_peak_memory_mb: float,
    sample_llm_calls: int,
    sample_llm_cost: float,
    sample_wall_time: float,
    sample_rows: int,
    total_rows: int,
    policy: SafetyPolicy | None = None,
) -> ResourceProjection:
    """Extrapolate full-scale resource usage from sample statistics.

    Total comparisons are computed exactly from blocking stats as
    sum(n*(n-1)/2) for each block of size n.
    Memory, LLM calls/cost, and wall time are scaled linearly from the
    sample ratio (total_rows / sample_rows).
    """
    # Exact comparison count from blocking stats
    total_comparisons = sum(n * (n - 1) // 2 for n in blocking_stats.values())

    # Scale factor for linear extrapolation
    if sample_rows > 0 and sample_comparisons > 0:
        scale = total_comparisons / max(sample_comparisons, 1)
    else:
        scale = total_rows / max(sample_rows, 1)

    estimated_memory_mb = sample_peak_memory_mb * max(scale, 1.0)
    estimated_llm_calls = int(sample_llm_calls * scale)
    estimated_llm_cost = sample_llm_cost * scale
    estimated_wall_time = sample_wall_time * max(scale, 1.0)

    # Determine risk level
    if policy is None:
        policy = SafetyPolicy()

    risk_level = _assess_risk(
        total_comparisons, estimated_memory_mb, estimated_llm_calls,
        estimated_llm_cost, estimated_wall_time, policy,
    )

    return ResourceProjection(
        total_comparisons=total_comparisons,
        estimated_memory_mb=round(estimated_memory_mb, 1),
        estimated_llm_calls=estimated_llm_calls,
        estimated_llm_cost_usd=round(estimated_llm_cost, 4),
        estimated_wall_time_seconds=round(estimated_wall_time, 1),
        risk_level=risk_level,
    )


def _assess_risk(
    comparisons: int,
    memory_mb: float,
    llm_calls: int,
    llm_cost: float,
    wall_time: float,
    policy: SafetyPolicy,
) -> str:
    """Assign a risk level: 'safe', 'caution', or 'danger'."""
    danger = False
    caution = False

    if comparisons > policy.max_comparisons:
        danger = True
    elif comparisons > policy.max_comparisons * 0.7:
        caution = True

    if memory_mb > policy.max_memory_mb:
        danger = True
    elif memory_mb > policy.max_memory_mb * 0.7:
        caution = True

    if llm_cost > policy.max_llm_cost_usd:
        danger = True
    elif llm_cost > policy.max_llm_cost_usd * 0.7:
        caution = True

    if wall_time > policy.max_wall_time_seconds:
        danger = True
    elif wall_time > policy.max_wall_time_seconds * 0.7:
        caution = True

    if danger:
        return "danger"
    if caution:
        return "caution"
    return "safe"


# ---------------------------------------------------------------------------
# Safety check
# ---------------------------------------------------------------------------


def _is_safe(proj: ResourceProjection, policy: SafetyPolicy) -> bool:
    """Return True if projections are within all policy limits."""
    if proj.total_comparisons > policy.max_comparisons:
        return False
    if proj.estimated_memory_mb > policy.max_memory_mb:
        return False
    if proj.estimated_llm_cost_usd > policy.max_llm_cost_usd:
        return False
    if proj.estimated_wall_time_seconds > policy.max_wall_time_seconds:
        return False
    return True


# ---------------------------------------------------------------------------
# Auto-downgrade cascade
# ---------------------------------------------------------------------------


def _apply_downgrades(
    config: GoldenMatchConfig,
    proj: ResourceProjection,
    policy: SafetyPolicy,
) -> tuple[GoldenMatchConfig, list[Downgrade]]:
    """Apply a cascade of configuration downgrades to bring projections in line.

    Returns the adjusted config (deep copy) and list of downgrades applied.
    For mode='none', no downgrades are applied regardless of projections.
    """
    downgrades: list[Downgrade] = []

    if policy.mode == "none":
        return config, downgrades

    # Work on a deep copy so the caller's config is not mutated.
    adjusted = config.model_copy(deep=True)

    if _is_safe(proj, policy):
        return adjusted, downgrades

    # --- Conservative downgrades (always applied when mode != "none") ---

    # 1. Enable skip_oversized if not already set
    if adjusted.blocking and not adjusted.blocking.skip_oversized:
        adjusted.blocking.skip_oversized = True
        downgrades.append(Downgrade(
            field="blocking.skip_oversized",
            old_value=False,
            new_value=True,
            reason="Skip oversized blocks to reduce comparison count",
        ))

    # 2. Halve max_block_size (floor 500)
    if adjusted.blocking and adjusted.blocking.max_block_size > 500:
        old_val = adjusted.blocking.max_block_size
        new_val = max(old_val // 2, 500)
        if new_val != old_val:
            adjusted.blocking.max_block_size = new_val
            downgrades.append(Downgrade(
                field="blocking.max_block_size",
                old_value=old_val,
                new_value=new_val,
                reason="Reduce max block size to limit per-block comparisons",
            ))

    # 3. Switch to ANN strategy if currently static and comparisons are huge
    if (
        adjusted.blocking
        and adjusted.blocking.strategy == "static"
        and proj.total_comparisons > policy.max_comparisons * 2
    ):
        adjusted.blocking.strategy = "ann"
        downgrades.append(Downgrade(
            field="blocking.strategy",
            old_value="static",
            new_value="ann",
            reason="Switch to ANN blocking to reduce comparison count",
        ))

    # 4. Switch to DuckDB backend for memory relief
    if (
        proj.estimated_memory_mb > policy.max_memory_mb
        and adjusted.backend != "duckdb"
    ):
        old_backend = adjusted.backend
        adjusted.backend = "duckdb"
        downgrades.append(Downgrade(
            field="backend",
            old_value=old_backend,
            new_value="duckdb",
            reason="Switch to DuckDB backend for out-of-core processing",
        ))

    # --- Aggressive downgrades (only in "aggressive" mode) ---
    if policy.mode == "aggressive":
        # 5. Tighten LLM scoring band (raise candidate_lo)
        if adjusted.llm_scorer and adjusted.llm_scorer.enabled:
            old_lo = adjusted.llm_scorer.candidate_lo
            new_lo = min(old_lo + 0.10, adjusted.llm_scorer.candidate_hi - 0.05)
            if new_lo != old_lo:
                adjusted.llm_scorer.candidate_lo = new_lo
                downgrades.append(Downgrade(
                    field="llm_scorer.candidate_lo",
                    old_value=old_lo,
                    new_value=new_lo,
                    reason="Tighten LLM scoring band to reduce LLM calls",
                ))

        # 6. Raise matchkey thresholds by 0.05
        for mk in adjusted.get_matchkeys():
            if mk.threshold is not None and mk.threshold < 0.95:
                old_thresh = mk.threshold
                mk.threshold = min(old_thresh + 0.05, 0.99)
                downgrades.append(Downgrade(
                    field=f"matchkey.{mk.name}.threshold",
                    old_value=old_thresh,
                    new_value=mk.threshold,
                    reason="Raise match threshold to reduce false positive pairs",
                ))

    return adjusted, downgrades


# ---------------------------------------------------------------------------
# Main preflight function
# ---------------------------------------------------------------------------

_SMALL_DATASET_THRESHOLD = 10_000


def preflight(
    df: pl.DataFrame,
    config: GoldenMatchConfig | None = None,
    safety: str | SafetyPolicy = "conservative",
) -> RunPlan:
    """Run preflight analysis: estimate resources, apply auto-downgrades.

    Parameters
    ----------
    df
        The full input DataFrame.
    config
        Optional GoldenMatchConfig. If None, auto-configuration is used.
    safety
        A SafetyPolicy instance or one of "conservative", "aggressive", "none".

    Returns
    -------
    RunPlan
        Contains projections, any applied downgrades, and the adjusted config.
    """
    # Resolve safety policy
    if isinstance(safety, str):
        policy = SafetyPolicy(mode=safety)  # type: ignore[arg-type]
    else:
        policy = safety

    # Build config if not provided
    if config is None:
        try:
            from goldenmatch.core.autoconfig import auto_configure_df
            config = auto_configure_df(df)
        except Exception:
            logger.warning("Preflight: auto_configure_df failed, using minimal config")
            config = GoldenMatchConfig(output={})

    # Attach safety policy to config
    config = config.model_copy(deep=True)
    config.safety = policy

    # For small datasets, skip sample run and compute stats directly
    if df.height <= _SMALL_DATASET_THRESHOLD:
        return _preflight_small(df, config, policy)

    # For larger datasets, take a sample and run it
    return _preflight_sampled(df, config, policy)


def _preflight_small(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    policy: SafetyPolicy,
) -> RunPlan:
    """Preflight for small datasets -- blocking stats only, no sample run."""
    blocking_stats = _get_blocking_stats(df, config)
    total_comparisons = sum(n * (n - 1) // 2 for n in blocking_stats.values())

    proj = ResourceProjection(
        total_comparisons=total_comparisons,
        estimated_memory_mb=0.0,
        estimated_llm_calls=0,
        estimated_llm_cost_usd=0.0,
        estimated_wall_time_seconds=0.0,
        risk_level=_assess_risk(
            total_comparisons, 0.0, 0, 0.0, 0.0, policy,
        ),
    )

    adjusted, downgrades = _apply_downgrades(config, proj, policy)

    return RunPlan(
        projection=proj,
        downgrades=downgrades,
        config=adjusted,
        sample_stats=SampleStats(
            sample_rows=df.height,
            comparisons=total_comparisons,
            peak_memory_mb=0.0,
            llm_calls=0,
            llm_cost=0.0,
            wall_time_seconds=0.0,
        ),
    )


def _preflight_sampled(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    policy: SafetyPolicy,
) -> RunPlan:
    """Preflight for larger datasets -- sample run + extrapolation."""
    sample = _take_sample(df)

    # Measure the sample run
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)
    t0 = time.monotonic()

    sample_comparisons = 0
    sample_llm_calls = 0
    sample_llm_cost = 0.0

    try:
        # Import inside function to avoid circular imports
        from goldenmatch.core.pipeline import run_dedupe_df

        result = run_dedupe_df(sample, config, source_name="__preflight__")

        # Try to extract comparison count from the result
        scored_pairs = result.get("scored_pairs", [])
        sample_comparisons = len(scored_pairs) if scored_pairs else 0
    except Exception as exc:
        logger.warning("Preflight sample run failed: %s", exc)
        # Fall back to blocking-stats-only estimation

    t1 = time.monotonic()
    mem_after = process.memory_info().rss / (1024 * 1024)

    sample_wall_time = t1 - t0
    sample_peak_memory_mb = max(mem_after - mem_before, 0.0)

    # Compute blocking stats on full dataset for accurate comparison counts
    blocking_stats = _get_blocking_stats(df, config)

    proj = _extrapolate(
        blocking_stats=blocking_stats,
        sample_comparisons=max(sample_comparisons, 1),
        sample_peak_memory_mb=max(sample_peak_memory_mb, 1.0),
        sample_llm_calls=sample_llm_calls,
        sample_llm_cost=sample_llm_cost,
        sample_wall_time=sample_wall_time,
        sample_rows=sample.height,
        total_rows=df.height,
        policy=policy,
    )

    sample_stats = SampleStats(
        sample_rows=sample.height,
        comparisons=sample_comparisons,
        peak_memory_mb=round(sample_peak_memory_mb, 1),
        llm_calls=sample_llm_calls,
        llm_cost=sample_llm_cost,
        wall_time_seconds=round(sample_wall_time, 2),
    )

    adjusted, downgrades = _apply_downgrades(config, proj, policy)

    return RunPlan(
        projection=proj,
        downgrades=downgrades,
        config=adjusted,
        sample_stats=sample_stats,
    )
