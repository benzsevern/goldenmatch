"""MatchEngine — shared foundation for TUI and preview mode.

Wraps the existing pipeline modules into a clean API with sample
extraction, scored-pairs caching, and threshold re-clustering.
No Textual dependency — pure Python + Polars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass
class EngineStats:
    total_records: int
    total_clusters: int
    singleton_count: int
    match_rate: float
    cluster_sizes: list[int]
    avg_cluster_size: float
    max_cluster_size: int
    oversized_count: int
    hit_rate: float | None = None
    avg_score: float | None = None


@dataclass
class EngineResult:
    clusters: dict[int, dict]
    golden: pl.DataFrame | None
    unique: pl.DataFrame | None
    dupes: pl.DataFrame | None
    quarantine: pl.DataFrame | None
    matched: pl.DataFrame | None
    unmatched: pl.DataFrame | None
    scored_pairs: list[tuple[int, int, float]]
    stats: EngineStats


class MatchEngine:
    """Wraps the pipeline into a clean API for the TUI and preview mode."""

    def __init__(self, files: list[Path | str]):
        self._files = [Path(f) for f in files]
        self._data: pl.DataFrame | None = None
        self._profile: dict | None = None
        self._last_result: EngineResult | None = None
        self._load()

    def _load(self) -> None:
        from goldenmatch.core.ingest import load_file
        from goldenmatch.core.profiler import profile_dataframe

        frames = []
        for f in self._files:
            lf = load_file(f)
            lf = lf.with_columns(pl.lit(f.stem).alias("__source__"))
            frames.append(lf.collect())
        combined = pl.concat(frames)
        # Add row IDs
        combined = combined.with_row_index("__row_id__").with_columns(
            pl.col("__row_id__").cast(pl.Int64)
        )
        self._data = combined
        # Profile without internal columns
        profile_cols = [c for c in combined.columns if not c.startswith("__")]
        self._profile = profile_dataframe(combined.select(profile_cols))

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def profile(self) -> dict:
        return self._profile

    @property
    def columns(self) -> list[str]:
        return [c for c in self._data.columns if not c.startswith("__")]

    @property
    def row_count(self) -> int:
        return self._data.height

    def get_sample(self, n: int) -> pl.DataFrame:
        if n >= self._data.height:
            return self._data
        return self._data.head(n)
