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
