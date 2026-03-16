"""Golden record builder with per-field merge strategies."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime

import polars as pl

from goldenmatch.config.schemas import GoldenFieldRule, GoldenRulesConfig

# Columns to skip when building golden records
_INTERNAL_PREFIXES = ("__row_id__", "__source__", "__block_key__", "__mk_")


def _is_internal(col: str) -> bool:
    return any(col.startswith(p) for p in _INTERNAL_PREFIXES) or col == "__mk_"


def merge_field(
    values: list,
    rule: GoldenFieldRule,
    sources: list[str] | None = None,
    dates: list | None = None,
) -> tuple:
    """Merge a list of values using the given rule's strategy.

    Returns (value, confidence).
    """
    non_null = [(i, v) for i, v in enumerate(values) if v is not None]

    if not non_null:
        return (None, 0.0)

    # If all non-null values are identical, return with confidence 1.0
    unique_vals = set(v for _, v in non_null)
    if len(unique_vals) == 1:
        return (non_null[0][1], 1.0)

    strategy = rule.strategy

    if strategy == "most_complete":
        return _most_complete(non_null)
    elif strategy == "majority_vote":
        return _majority_vote(non_null)
    elif strategy == "source_priority":
        return _source_priority(values, rule, sources)
    elif strategy == "most_recent":
        return _most_recent(values, dates)
    elif strategy == "first_non_null":
        return _first_non_null(non_null)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _most_complete(non_null: list[tuple[int, object]]) -> tuple:
    str_vals = [(i, str(v), v) for i, v in non_null]
    max_len = max(len(s) for _, s, _ in str_vals)
    longest = [(i, s, v) for i, s, v in str_vals if len(s) == max_len]
    if len(longest) == 1:
        return (longest[0][2], 1.0)
    return (longest[0][2], 0.7)


def _majority_vote(non_null: list[tuple[int, object]]) -> tuple:
    counts = Counter(v for _, v in non_null)
    winner, count = counts.most_common(1)[0]
    total = len(non_null)
    return (winner, count / total)


def _source_priority(
    values: list,
    rule: GoldenFieldRule,
    sources: list[str] | None,
) -> tuple:
    if sources is None:
        raise ValueError("source_priority strategy requires sources list")
    source_val = {}
    for src, val in zip(sources, values):
        if src not in source_val:
            source_val[src] = val

    for idx, src in enumerate(rule.source_priority):
        val = source_val.get(src)
        if val is not None:
            conf = max(0.1, 1.0 - idx * 0.1)
            return (val, conf)

    # Fallback: no match found in priority list
    return (None, 0.0)


def _most_recent(values: list, dates: list | None) -> tuple:
    if dates is None:
        raise ValueError("most_recent strategy requires dates list")
    pairs = [(d, v) for d, v in zip(dates, values) if v is not None and d is not None]
    if not pairs:
        return (None, 0.0)
    pairs.sort(key=lambda x: x[0], reverse=True)
    top_date = pairs[0][0]
    tied = [p for p in pairs if p[0] == top_date]
    conf = 1.0 if len(tied) == 1 else 0.5
    return (pairs[0][1], conf)


def _first_non_null(non_null: list[tuple[int, object]]) -> tuple:
    return (non_null[0][1], 0.6)


def build_golden_record(
    cluster_df: pl.DataFrame,
    rules: GoldenRulesConfig,
) -> dict:
    """Build a golden record from a cluster DataFrame.

    Returns dict of {col: {"value": v, "confidence": c}, ...,
    "__golden_confidence__": mean_of_confidences}.
    """
    result = {}
    confidences = []

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
        if field_rule.strategy == "source_priority" and "__source__" in cluster_df.columns:
            sources = cluster_df["__source__"].to_list()
        if field_rule.strategy == "most_recent" and field_rule.date_column:
            if field_rule.date_column in cluster_df.columns:
                dates = cluster_df[field_rule.date_column].to_list()

        val, conf = merge_field(values, field_rule, sources=sources, dates=dates)
        result[col] = {"value": val, "confidence": conf}
        confidences.append(conf)

    if confidences:
        result["__golden_confidence__"] = sum(confidences) / len(confidences)
    else:
        result["__golden_confidence__"] = 0.0

    return result
