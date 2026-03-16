"""Report generators for GoldenMatch runs."""

from __future__ import annotations

from collections import Counter


def generate_dedupe_report(
    total_records: int,
    total_clusters: int,
    cluster_sizes: list[int],
    oversized_clusters: int,
    matchkeys_used: list[str],
) -> dict:
    """Generate a summary report for a dedupe run.

    Returns a dict with key metrics.
    """
    size_dist = Counter(cluster_sizes)
    return {
        "total_records": total_records,
        "total_clusters": total_clusters,
        "match_rate": total_clusters / total_records if total_records else 0.0,
        "cluster_size_distribution": size_dist,
        "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "oversized_count": oversized_clusters,
        "matchkeys_used": matchkeys_used,
    }


def generate_match_report(
    total_targets: int,
    matched: int,
    unmatched: int,
    scores: list[float],
) -> dict:
    """Generate a summary report for a match run.

    Returns a dict with key metrics.
    """
    return {
        "total_targets": total_targets,
        "matched": matched,
        "unmatched": unmatched,
        "hit_rate": matched / total_targets if total_targets else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
    }
