"""Scorer for GoldenMatch — field-level and pair-level scoring."""

from __future__ import annotations

from itertools import combinations

import jellyfish
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_sort_ratio

from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
from goldenmatch.utils.transforms import apply_transforms


def score_field(val_a: str | None, val_b: str | None, scorer: str) -> float | None:
    """Score two field values using the specified scorer.

    Returns None if either value is None.
    """
    if val_a is None or val_b is None:
        return None

    if scorer == "exact":
        return 1.0 if val_a == val_b else 0.0
    elif scorer == "jaro_winkler":
        return JaroWinkler.similarity(val_a, val_b)
    elif scorer == "levenshtein":
        return Levenshtein.normalized_similarity(val_a, val_b)
    elif scorer == "token_sort":
        return token_sort_ratio(val_a, val_b) / 100.0
    elif scorer == "soundex_match":
        return 1.0 if jellyfish.soundex(val_a) == jellyfish.soundex(val_b) else 0.0
    else:
        raise ValueError(f"Unknown scorer: {scorer!r}")


def score_pair(row_a: dict, row_b: dict, fields: list[MatchkeyField]) -> float:
    """Score a pair of rows across all fields using weighted aggregation.

    Fields that produce None scores are excluded from the average.
    If all fields are None, returns 0.0.
    """
    weighted_sum = 0.0
    weight_sum = 0.0

    for f in fields:
        val_a = apply_transforms(row_a.get(f.field), f.transforms)
        val_b = apply_transforms(row_b.get(f.field), f.transforms)
        field_score = score_field(val_a, val_b, f.scorer)

        if field_score is not None:
            weighted_sum += field_score * f.weight
            weight_sum += f.weight

    if weight_sum == 0.0:
        return 0.0

    return weighted_sum / weight_sum


def find_exact_matches(
    lf: pl.LazyFrame, mk: MatchkeyConfig
) -> list[tuple[int, int, float]]:
    """Find exact matches by grouping on the matchkey column.

    Groups by __mk_{name}__ column, then generates all combinations of
    __row_id__ pairs within each group that has 2+ records, each with score 1.0.
    """
    mk_col = f"__mk_{mk.name}__"
    df = lf.collect()

    results: list[tuple[int, int, float]] = []

    for _key, group_df in df.group_by(mk_col):
        key_val = _key[0]
        if key_val is None:
            continue
        if len(group_df) < 2:
            continue

        row_ids = group_df["__row_id__"].to_list()
        for id_a, id_b in combinations(row_ids, 2):
            results.append((id_a, id_b, 1.0))

    return results


def find_fuzzy_matches(
    block_df: pl.DataFrame, mk: MatchkeyConfig
) -> list[tuple[int, int, float]]:
    """Find fuzzy matches within a block DataFrame.

    Converts to dicts, compares all pairs, and keeps those >= mk.threshold.
    """
    rows = block_df.to_dicts()
    results: list[tuple[int, int, float]] = []

    for row_a, row_b in combinations(rows, 2):
        score = score_pair(row_a, row_b, mk.fields)
        if score >= mk.threshold:
            results.append((row_a["__row_id__"], row_b["__row_id__"], score))

    return results
