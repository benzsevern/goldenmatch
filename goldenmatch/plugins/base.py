"""Base protocols for GoldenMatch plugins.

Plugin authors implement these protocols and register via entry points:

    [project.entry-points."goldenmatch.plugins.scorer"]
    my_scorer = "my_package.scorers:MyScorer"
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl


@runtime_checkable
class ScorerPlugin(Protocol):
    """Plugin protocol for custom field scorers."""

    name: str

    def score_pair(self, val_a: str | None, val_b: str | None) -> float | None:
        """Score two field values. Return None if either is None."""
        ...

    def score_matrix(self, values_a: list[str], values_b: list[str]) -> np.ndarray:
        """Score all pairs NxM. Optional -- falls back to pairwise if not implemented."""
        ...


@runtime_checkable
class TransformPlugin(Protocol):
    """Plugin protocol for custom field transforms."""

    name: str

    def transform(self, value: str) -> str:
        """Transform a single value."""
        ...

    def transform_series(self, series: pl.Series) -> pl.Series:
        """Transform a Polars Series. Optional -- falls back to map_elements."""
        ...


@runtime_checkable
class ConnectorPlugin(Protocol):
    """Plugin protocol for data source/sink connectors."""

    name: str

    def read(self, config: dict) -> pl.LazyFrame:
        """Read data from external source."""
        ...

    def write(self, df: pl.DataFrame, config: dict) -> None:
        """Write data to external sink."""
        ...


@runtime_checkable
class GoldenStrategyPlugin(Protocol):
    """Plugin protocol for custom golden record merge strategies."""

    name: str

    def merge(self, values: list, sources: list[str] | None = None) -> tuple[Any, float]:
        """Merge field values. Returns (merged_value, confidence)."""
        ...
