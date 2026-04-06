"""Golden record builder with per-field merge strategies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import polars as pl

from goldenmatch.config.schemas import GoldenFieldRule, GoldenRulesConfig


@dataclass
class FieldProvenance:
    value: Any
    source_row_id: int
    strategy: str
    confidence: float
    candidates: list[dict] = dataclass_field(default_factory=list)


@dataclass
class ClusterProvenance:
    cluster_id: int
    cluster_quality: str
    cluster_confidence: float
    fields: dict[str, FieldProvenance] = dataclass_field(default_factory=dict)


@dataclass
class GoldenRecordResult:
    df: pl.DataFrame
    provenance: list[ClusterProvenance] = dataclass_field(default_factory=list)

# Columns to skip when building golden records
_INTERNAL_PREFIXES = ("__row_id__", "__source__", "__block_key__", "__mk_")


def _is_internal(col: str) -> bool:
    return any(col.startswith(p) for p in _INTERNAL_PREFIXES) or col == "__mk_"


def merge_field(
    values: list,
    rule: GoldenFieldRule,
    sources: list[str] | None = None,
    dates: list | None = None,
    quality_weights: list[float] | None = None,
) -> tuple[object, float, int | None]:
    """Merge a list of values using the given rule's strategy.

    Returns (value, confidence, source_index) where source_index is the
    index into the values list that the winning value came from.
    """
    non_null = [(i, v) for i, v in enumerate(values) if v is not None]

    if not non_null:
        return (None, 0.0, None)

    # If all non-null values are identical, return with confidence 1.0
    unique_vals = set(v for _, v in non_null)
    if len(unique_vals) == 1:
        return (non_null[0][1], 1.0, non_null[0][0])

    strategy = rule.strategy

    if strategy == "most_complete":
        return _most_complete(non_null, quality_weights)
    elif strategy == "majority_vote":
        return _majority_vote(non_null, quality_weights)
    elif strategy == "source_priority":
        return _source_priority(values, rule, sources)
    elif strategy == "most_recent":
        return _most_recent(values, dates)
    elif strategy == "first_non_null":
        return _first_non_null(non_null, quality_weights)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _most_complete(non_null: list[tuple[int, object]], quality_weights: list[float] | None = None) -> tuple:
    str_vals = [(i, str(v), v) for i, v in non_null]
    max_len = max(len(s) for _, s, _ in str_vals)
    longest = [(i, s, v) for i, s, v in str_vals if len(s) == max_len]
    if len(longest) == 1:
        return (longest[0][2], 1.0, longest[0][0])
    # Tie-break by quality weight if available
    if quality_weights is not None:
        best = max(longest, key=lambda x: quality_weights[x[0]] if x[0] < len(quality_weights) else 1.0)
        conf = min(1.0, 0.7 * quality_weights[best[0]]) if best[0] < len(quality_weights) else 0.7
        return (best[2], conf, best[0])
    return (longest[0][2], 0.7, longest[0][0])


def _majority_vote(non_null: list[tuple[int, object]], quality_weights: list[float] | None = None) -> tuple:
    if quality_weights is not None:
        # Weighted vote: sum quality weights per value
        value_weights: dict[object, float] = {}
        value_idx: dict[object, int] = {}
        for i, v in non_null:
            w = quality_weights[i] if i < len(quality_weights) else 1.0
            value_weights[v] = value_weights.get(v, 0.0) + w
            if v not in value_idx:
                value_idx[v] = i
        winner = max(value_weights, key=value_weights.__getitem__)
        total_weight = sum(value_weights.values())
        conf = value_weights[winner] / total_weight if total_weight > 0 else 0.0
        return (winner, conf, value_idx[winner])
    counts = Counter(v for _, v in non_null)
    winner, count = counts.most_common(1)[0]
    total = len(non_null)
    # Find the index of the first occurrence of the winner
    winner_idx = next(i for i, v in non_null if v == winner)
    return (winner, count / total, winner_idx)


def _source_priority(
    values: list,
    rule: GoldenFieldRule,
    sources: list[str] | None,
) -> tuple:
    if sources is None:
        raise ValueError("source_priority strategy requires sources list")
    source_val = {}
    source_idx = {}
    for i, (src, val) in enumerate(zip(sources, values)):
        if src not in source_val:
            source_val[src] = val
            source_idx[src] = i

    for idx, src in enumerate(rule.source_priority):
        val = source_val.get(src)
        if val is not None:
            conf = max(0.1, 1.0 - idx * 0.1)
            return (val, conf, source_idx[src])

    # Fallback: no match found in priority list
    return (None, 0.0, None)


def _most_recent(values: list, dates: list | None) -> tuple:
    if dates is None:
        raise ValueError("most_recent strategy requires dates list")
    indexed_pairs = [(i, d, v) for i, (d, v) in enumerate(zip(dates, values)) if v is not None and d is not None]
    if not indexed_pairs:
        return (None, 0.0, None)
    indexed_pairs.sort(key=lambda x: x[1], reverse=True)
    top_date = indexed_pairs[0][1]
    tied = [p for p in indexed_pairs if p[1] == top_date]
    conf = 1.0 if len(tied) == 1 else 0.5
    return (indexed_pairs[0][2], conf, indexed_pairs[0][0])


def _first_non_null(non_null: list[tuple[int, object]], quality_weights: list[float] | None = None) -> tuple:
    if quality_weights is not None:
        # Pick the non-null value with the highest quality weight
        best = max(non_null, key=lambda x: quality_weights[x[0]] if x[0] < len(quality_weights) else 1.0)
        return (best[1], 0.6, best[0])
    return (non_null[0][1], 0.6, non_null[0][0])


def build_golden_record(
    cluster_df: pl.DataFrame,
    rules: GoldenRulesConfig,
    quality_scores: dict[tuple[int, str], float] | None = None,
) -> dict:
    """Build a golden record from a cluster DataFrame.

    Returns dict of {col: {"value": v, "confidence": c}, ...,
    "__golden_confidence__": mean_of_confidences}.
    """
    result = {}
    confidences = []
    row_ids = cluster_df["__row_id__"].to_list() if "__row_id__" in cluster_df.columns else None

    for col in cluster_df.columns:
        if _is_internal(col):
            continue

        values = cluster_df[col].to_list()

        # Look up field rule or build default
        if col in rules.field_rules:
            field_rule = rules.field_rules[col]
        else:
            field_rule = GoldenFieldRule(strategy=rules.default_strategy)

        # Gather optional lists
        sources = None
        dates = None
        weights = None
        if field_rule.strategy == "source_priority" and "__source__" in cluster_df.columns:
            sources = cluster_df["__source__"].to_list()
        if field_rule.strategy == "most_recent" and field_rule.date_column:
            if field_rule.date_column in cluster_df.columns:
                dates = cluster_df[field_rule.date_column].to_list()
        if quality_scores is not None and row_ids is not None:
            weights = [quality_scores.get((rid, col), 1.0) for rid in row_ids]

        val, conf, _idx = merge_field(values, field_rule, sources=sources, dates=dates, quality_weights=weights)
        result[col] = {"value": val, "confidence": conf}
        confidences.append(conf)

    if confidences:
        result["__golden_confidence__"] = sum(confidences) / len(confidences)
    else:
        result["__golden_confidence__"] = 0.0

    return result


def build_golden_record_with_provenance(
    df: pl.DataFrame,
    rules: GoldenRulesConfig,
    clusters: dict[int, dict],
    quality_scores: dict[tuple[int, str], float] | None = None,
) -> GoldenRecordResult:
    """Build golden records with field-level provenance tracking."""
    golden_rows = []
    provenance_list = []

    cluster_col = "__cluster_id__"
    if cluster_col not in df.columns:
        # Single cluster case
        cluster_ids = [1]
        cluster_dfs = {1: df}
    else:
        cluster_ids = sorted(df[cluster_col].unique().to_list())
        cluster_dfs = {cid: df.filter(pl.col(cluster_col) == cid) for cid in cluster_ids}

    for cid in cluster_ids:
        cluster_df = cluster_dfs[cid]
        cinfo = clusters.get(cid, {})
        row_ids = cluster_df["__row_id__"].to_list() if "__row_id__" in cluster_df.columns else list(range(len(cluster_df)))

        # Build golden record + provenance in a single pass (no double merge_field call)
        field_provenance = {}
        golden_row = {"__cluster_id__": cid}
        confidences = []

        for col in cluster_df.columns:
            if _is_internal(col):
                continue
            values = cluster_df[col].to_list()
            if col in rules.field_rules:
                field_rule = rules.field_rules[col]
            else:
                field_rule = GoldenFieldRule(strategy=rules.default_strategy)

            sources = None
            dates = None
            weights = None
            if field_rule.strategy == "source_priority" and "__source__" in cluster_df.columns:
                sources = cluster_df["__source__"].to_list()
            if field_rule.strategy == "most_recent" and field_rule.date_column:
                if field_rule.date_column in cluster_df.columns:
                    dates = cluster_df[field_rule.date_column].to_list()
            if quality_scores is not None and row_ids:
                weights = [quality_scores.get((rid, col), 1.0) for rid in row_ids]

            val, conf, src_idx = merge_field(values, field_rule, sources=sources, dates=dates, quality_weights=weights)
            confidences.append(conf)

            source_row_id = row_ids[src_idx] if src_idx is not None and src_idx < len(row_ids) else row_ids[0]

            candidates = []
            for rid, v in zip(row_ids, values):
                q = quality_scores.get((rid, col), 1.0) if quality_scores else 1.0
                candidates.append({"row_id": rid, "value": v, "quality": q})

            field_provenance[col] = FieldProvenance(
                value=val,
                source_row_id=source_row_id,
                strategy=field_rule.strategy,
                confidence=conf,
                candidates=candidates,
            )
            golden_row[col] = val

        golden_row["__golden_confidence__"] = sum(confidences) / len(confidences) if confidences else 0.0
        golden_rows.append(golden_row)

        provenance_list.append(ClusterProvenance(
            cluster_id=cid,
            cluster_quality=cinfo.get("cluster_quality", "strong"),
            cluster_confidence=cinfo.get("confidence", 0.0),
            fields=field_provenance,
        ))

    golden_df = pl.DataFrame(golden_rows) if golden_rows else pl.DataFrame()
    return GoldenRecordResult(df=golden_df, provenance=provenance_list)
