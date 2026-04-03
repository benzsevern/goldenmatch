"""Parameter sensitivity analysis using CCMS cluster comparison.

Sweeps configuration parameters across a range, running the pipeline
for each value and comparing the resulting clusters against a baseline.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from goldenmatch.config.schemas import GoldenMatchConfig
from goldenmatch.core.compare_clusters import CompareResult, compare_clusters
from goldenmatch.core.pipeline import run_dedupe

logger = logging.getLogger(__name__)

SUPPORTED_FIELDS = {
    "threshold",
    "blocking.max_block_size",
    # matchkey.<name>.threshold handled dynamically
}


@dataclass
class SweepParam:
    """Definition of a parameter to sweep."""
    field: str
    start: float
    stop: float
    step: float


@dataclass
class SweepPoint:
    """Result of a single sweep value."""
    param_value: float
    comparison: CompareResult


@dataclass
class SensitivityResult:
    """Result of sweeping one parameter across a range."""
    param: SweepParam
    baseline_value: float
    points: list[SweepPoint] = field(default_factory=list)

    def stability_report(self) -> dict:
        """Identify the value range with the most unchanged clusters."""
        if not self.points:
            return {"best_value": self.baseline_value, "best_unchanged_pct": 1.0, "points": []}

        best = max(self.points, key=lambda p: p.comparison.unchanged)
        total = best.comparison.cc1 or 1
        return {
            "best_value": best.param_value,
            "best_unchanged_pct": round(best.comparison.unchanged / total, 4),
            "points": [
                {
                    "value": p.param_value,
                    "unchanged": p.comparison.unchanged,
                    "merged": p.comparison.merged,
                    "partitioned": p.comparison.partitioned,
                    "overlapping": p.comparison.overlapping,
                    "twi": round(p.comparison.twi, 4),
                }
                for p in self.points
            ],
        }


def _validate_field(field: str, config: GoldenMatchConfig) -> None:
    """Validate that a sweep field is supported and exists in config."""
    if field in SUPPORTED_FIELDS:
        return
    if field.startswith("matchkey.") and field.endswith(".threshold"):
        name = field.split(".")[1]
        matchkeys = config.get_matchkeys()
        if not any(mk.name == name for mk in matchkeys):
            available = [mk.name for mk in matchkeys]
            raise ValueError(
                f"Matchkey '{name}' not found in config. Available: {available}"
            )
        return
    raise ValueError(
        f"Unsupported sweep field '{field}'. "
        f"Supported: {sorted(SUPPORTED_FIELDS)} or 'matchkey.<name>.threshold'"
    )


def _get_current_value(field: str, config: GoldenMatchConfig) -> float:
    """Get the current value of a sweep field from the config."""
    if field == "threshold":
        matchkeys = config.get_matchkeys()
        fuzzy = [mk for mk in matchkeys if mk.threshold is not None]
        if fuzzy:
            return fuzzy[0].threshold
        return 0.85  # default

    if field == "blocking.max_block_size":
        if config.blocking:
            return float(config.blocking.max_block_size)
        return 5000.0

    if field.startswith("matchkey.") and field.endswith(".threshold"):
        name = field.split(".")[1]
        for mk in config.get_matchkeys():
            if mk.name == name:
                return mk.threshold if mk.threshold is not None else 0.85

    raise ValueError(f"Cannot read current value for field '{field}' -- no handler defined")


def _apply_value(field: str, value: float, config: GoldenMatchConfig) -> None:
    """Apply a sweep value to the config (mutates in-place)."""
    if field == "threshold":
        applied = False
        for mk in config.get_matchkeys():
            if mk.threshold is not None:
                mk.threshold = value
                applied = True
        if not applied:
            logger.warning("No fuzzy matchkeys found to apply threshold sweep")
        return

    if field == "blocking.max_block_size":
        if config.blocking is None:
            from goldenmatch.config.schemas import BlockingConfig
            config.blocking = BlockingConfig()
        config.blocking.max_block_size = int(value)
        return

    if field.startswith("matchkey.") and field.endswith(".threshold"):
        name = field.split(".")[1]
        for mk in config.get_matchkeys():
            if mk.name == name:
                mk.threshold = value
                return

    raise ValueError(f"Cannot apply sweep value for field '{field}' -- no handler defined")


def _generate_values(param: SweepParam) -> list[float]:
    """Generate the list of values to sweep."""
    values = []
    v = param.start
    while v <= param.stop + 1e-9:  # epsilon for float comparison
        values.append(round(v, 6))
        v += param.step
    return values


def _sample_files(file_specs: list, sample_size: int, seed: int = 42) -> list:
    """Load, sample, and write to temp files. Returns new file_specs."""
    import tempfile

    import polars as pl

    from goldenmatch.core.ingest import load_file

    frames = []
    for spec in file_specs:
        path = spec[0] if isinstance(spec, tuple) else spec
        lf = load_file(path)
        frames.append(lf.collect())

    combined = pl.concat(frames) if len(frames) > 1 else frames[0]
    if sample_size < combined.height:
        combined = combined.sample(n=sample_size, seed=seed)

    # Write sampled data to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    combined.write_parquet(tmp.name)
    tmp.close()

    source_name = file_specs[0][1] if isinstance(file_specs[0], tuple) and len(file_specs[0]) > 1 else "sampled"
    return [(tmp.name, source_name)]


def run_sensitivity(
    file_specs: list,
    config: GoldenMatchConfig,
    sweep_params: list[SweepParam],
    sample_size: int | None = None,
) -> list[SensitivityResult]:
    """Run parameter sensitivity analysis.

    Sweeps each parameter independently, comparing each run's clusters
    against a baseline run.

    Args:
        file_specs: File specs for the pipeline (same as run_dedupe).
        config: Base config to use.
        sweep_params: Parameters to sweep.
        sample_size: If set, randomly sample this many records before sweeping.

    Returns:
        One SensitivityResult per sweep parameter.
    """
    for param in sweep_params:
        _validate_field(param.field, config)

    # Sample data once, reuse for all runs
    effective_specs = file_specs
    if sample_size is not None:
        effective_specs = _sample_files(file_specs, sample_size)

    # Run baseline
    baseline_config = copy.deepcopy(config)
    logger.info("Running baseline pipeline...")
    baseline_result = run_dedupe(effective_specs, baseline_config)
    baseline_clusters = baseline_result["clusters"]

    results: list[SensitivityResult] = []

    for param in sweep_params:
        baseline_value = _get_current_value(param.field, config)
        values = _generate_values(param)
        points: list[SweepPoint] = []

        logger.info("Sweeping %s: %s", param.field, values)

        for value in values:
            sweep_config = copy.deepcopy(config)
            _apply_value(param.field, value, sweep_config)

            logger.info("  %s = %s", param.field, value)
            try:
                sweep_result = run_dedupe(effective_specs, sweep_config)
                sweep_clusters = sweep_result["clusters"]
                comparison = compare_clusters(baseline_clusters, sweep_clusters)
                points.append(SweepPoint(param_value=value, comparison=comparison))
            except Exception as exc:
                logger.error(
                    "Sweep point %s=%s failed: %s", param.field, value, exc
                )

        results.append(SensitivityResult(
            param=param,
            baseline_value=baseline_value,
            points=points,
        ))

    return results
