"""Rich preview formatter for MatchEngine results.

Provides formatted text output for stats, clusters, golden records,
and score histograms using Rich tables and console capture.
"""
from __future__ import annotations

import polars as pl
from rich.console import Console
from rich.table import Table

from goldenmatch.tui.engine import EngineStats


def format_preview_stats(stats: EngineStats) -> str:
    """Rich table showing engine statistics."""
    console = Console(record=True, width=120)

    table = Table(title="Engine Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Records", str(stats.total_records))
    table.add_row("Clusters", str(stats.total_clusters))
    table.add_row("Singletons", str(stats.singleton_count))
    table.add_row("Match Rate", f"{stats.match_rate:.2%}")
    table.add_row("Avg Cluster Size", f"{stats.avg_cluster_size:.2f}")
    table.add_row("Max Cluster Size", str(stats.max_cluster_size))
    table.add_row("Oversized Clusters", str(stats.oversized_count))

    if stats.hit_rate is not None:
        table.add_row("Hit Rate", f"{stats.hit_rate:.2%}")
    if stats.avg_score is not None:
        table.add_row("Avg Score", f"{stats.avg_score:.2f}")

    console.print(table)
    return console.export_text()


def format_preview_clusters(
    clusters: dict, data: pl.DataFrame, max_clusters: int = 10
) -> str:
    """Show top N clusters sorted by size descending."""
    console = Console(record=True, width=120)

    # Filter to multi-member clusters only
    multi = {
        cid: info for cid, info in clusters.items() if info["size"] > 1
    }

    if not multi:
        console.print("[dim]No clusters found.[/dim]")
        return console.export_text()

    # Sort by size descending, take top N
    sorted_clusters = sorted(multi.items(), key=lambda x: x[1]["size"], reverse=True)
    sorted_clusters = sorted_clusters[:max_clusters]

    # Determine display columns (first 6 non-__ columns)
    display_cols = [c for c in data.columns if not c.startswith("__")][:6]

    for cluster_id, cluster_info in sorted_clusters:
        member_ids = cluster_info["members"]
        member_df = data.filter(pl.col("__row_id__").is_in(member_ids))

        table = Table(title=f"Cluster {cluster_id} (size: {cluster_info['size']})")
        for col in display_cols:
            table.add_column(col)

        for row in member_df.iter_rows(named=True):
            table.add_row(*[str(row.get(col, "")) for col in display_cols])

        console.print(table)

    return console.export_text()


def format_preview_golden(
    golden: pl.DataFrame | None, max_records: int = 10
) -> str:
    """Show top N golden records in a Rich table."""
    console = Console(record=True, width=120)

    if golden is None or golden.height == 0:
        console.print("[dim]No golden records found.[/dim]")
        return console.export_text()

    display_df = golden.head(max_records)

    table = Table(title="Golden Records")
    # Always include cluster_id and confidence first
    table.add_column("__cluster_id__")
    table.add_column("__golden_confidence__")

    other_cols = [
        c for c in display_df.columns
        if c not in ("__cluster_id__", "__golden_confidence__")
    ]
    for col in other_cols:
        table.add_column(col)

    for row in display_df.iter_rows(named=True):
        values = [
            str(row.get("__cluster_id__", "")),
            f"{row.get('__golden_confidence__', 0.0):.2f}",
        ]
        values.extend(str(row.get(col, "")) for col in other_cols)
        table.add_row(*values)

    console.print(table)
    return console.export_text()


def format_score_histogram(scores: list[float], bins: int = 10) -> str:
    """Text-based histogram of score distribution."""
    console = Console(record=True, width=120)

    if not scores:
        console.print("[dim]No scores to display.[/dim]")
        return console.export_text()

    min_score = min(scores)
    max_score = max(scores)

    # Handle all-same-score edge case
    if min_score == max_score:
        console.print(f"All {len(scores)} scores = {min_score:.4f}")
        return console.export_text()

    # Build histogram bins
    bin_width = (max_score - min_score) / bins
    bin_counts = [0] * bins
    for s in scores:
        idx = int((s - min_score) / bin_width)
        if idx >= bins:
            idx = bins - 1
        bin_counts[idx] += 1

    max_count = max(bin_counts) if bin_counts else 1
    bar_max_width = 40

    console.print("Score Distribution")
    console.print("")
    for i in range(bins):
        lo = min_score + i * bin_width
        hi = lo + bin_width
        count = bin_counts[i]
        bar_len = int((count / max_count) * bar_max_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        console.print(f"  {lo:.4f} - {hi:.4f} | {bar} ({count})")

    return console.export_text()
