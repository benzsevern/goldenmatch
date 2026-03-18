"""Scorer for GoldenMatch — field-level and pair-level scoring."""

from __future__ import annotations

from itertools import combinations

import jellyfish
import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_sort_ratio
from rapidfuzz.process import cdist

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

    Uses a Polars self-join on the matchkey column to find all pairs of
    __row_id__ that share the same matchkey value, each with score 1.0.
    Null matchkey values are excluded.
    """
    mk_col = f"__mk_{mk.name}__"
    df = lf.select("__row_id__", mk_col).collect()

    # Drop nulls — they should not match
    df = df.filter(pl.col(mk_col).is_not_null())

    if df.height < 2:
        return []

    # Self-join on matchkey — produces all (left, right) combinations per group
    joined = df.join(df, on=mk_col, suffix="_right")

    # Keep only pairs where left < right (avoid duplicates and self-matches)
    joined = joined.filter(pl.col("__row_id__") < pl.col("__row_id___right"))

    if joined.height == 0:
        return []

    ids_a = joined["__row_id__"].to_list()
    ids_b = joined["__row_id___right"].to_list()
    return [(a, b, 1.0) for a, b in zip(ids_a, ids_b)]


# ---------------------------------------------------------------------------
# Vectorized helpers for find_fuzzy_matches
# ---------------------------------------------------------------------------

def _get_transformed_values(block_df: pl.DataFrame, field: MatchkeyField) -> list:
    """Get transformed values for a field as a list, using Polars expressions when possible."""
    from goldenmatch.core.matchkey import _try_native_chain
    col = field.field

    # Try native Polars transforms (fast path)
    native_expr = _try_native_chain(col, field.transforms)
    if native_expr is not None:
        result_df = block_df.select(native_expr.alias("__tmp__"))
        return result_df["__tmp__"].to_list()

    # Fallback: Python per-row
    values = block_df[col].to_list()
    return [apply_transforms(v, field.transforms) if v is not None else None for v in values]


def _exact_score_matrix(values: list) -> np.ndarray:
    """NxN exact match matrix using hash-based grouping."""
    n = len(values)
    scores = np.zeros((n, n))
    # Group indices by value (O(n) hash map)
    groups: dict[str, list[int]] = {}
    for i, v in enumerate(values):
        if v is not None:
            groups.setdefault(v, []).append(i)
    # For each group, set all pairs to 1.0
    for indices in groups.values():
        if len(indices) > 1:
            idx = np.array(indices)
            scores[np.ix_(idx, idx)] = 1.0
    return scores


def _fuzzy_score_matrix(
    values: list, scorer_name: str, model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """NxN fuzzy score matrix using rapidfuzz cdist or embedding cosine similarity."""
    if scorer_name == "embedding":
        from goldenmatch.core.embedder import get_embedder

        embedder = get_embedder(model_name)
        embeddings = embedder.embed_column(values, cache_key=f"_block_{id(values)}")
        sim = embedder.cosine_similarity_matrix(embeddings)
        return np.asarray(sim, dtype=np.float64)

    # Replace None with empty string for cdist (we handle nulls separately)
    clean = [v if v is not None else "" for v in values]

    if scorer_name == "jaro_winkler":
        matrix = cdist(clean, clean, scorer=JaroWinkler.similarity)
    elif scorer_name == "levenshtein":
        matrix = cdist(clean, clean, scorer=Levenshtein.normalized_similarity)
    elif scorer_name == "token_sort":
        matrix = cdist(clean, clean, scorer=token_sort_ratio) / 100.0
    else:
        raise ValueError(f"Unknown fuzzy scorer: {scorer_name!r}")

    return np.asarray(matrix, dtype=np.float64)


def _record_embedding_score_matrix(
    block_df: pl.DataFrame, columns: list[str], model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """NxN score matrix from record-level embeddings.

    Concatenates columns into a single text string per record,
    embeds the full string, and computes cosine similarity.
    """
    from goldenmatch.core.embedder import get_embedder

    concat_values = []
    for row in block_df.iter_rows(named=True):
        parts = []
        for col in columns:
            val = row.get(col)
            if val is not None:
                parts.append(f"{col}: {val}")
        concat_values.append(" | ".join(parts) if parts else "")

    row_ids = block_df["__row_id__"].to_list()
    cache_key = f"_rec_emb_{hash(tuple(columns))}_{hash(tuple(row_ids))}"

    embedder = get_embedder(model_name)
    embeddings = embedder.embed_column(concat_values, cache_key=cache_key)
    sim = embedder.cosine_similarity_matrix(embeddings)
    return np.asarray(sim, dtype=np.float64)


def _soundex_score_matrix(values: list) -> np.ndarray:
    """NxN soundex match matrix."""
    codes = [jellyfish.soundex(v) if v is not None else None for v in values]
    return _exact_score_matrix(codes)


def _build_null_mask(values: list) -> np.ndarray:
    """NxN boolean mask — True where either value is null."""
    null_arr = np.array([v is None for v in values])
    return null_arr[:, None] | null_arr[None, :]


def find_fuzzy_matches(
    block_df: pl.DataFrame,
    mk: MatchkeyConfig,
    exclude_pairs: set[tuple[int, int]] | None = None,
    pre_scored_pairs: list[tuple[int, int, float]] | None = None,
) -> list[tuple[int, int, float]]:
    """Find fuzzy matches within a block DataFrame.

    Uses vectorized rapidfuzz cdist for batch scoring, with early termination
    when exact fields make it mathematically impossible to reach the threshold.

    Args:
        block_df: Block DataFrame with __row_id__ and field columns.
        mk: Matchkey configuration with fields, weights, and threshold.
        exclude_pairs: Optional set of (min_id, max_id) pairs to skip.
        pre_scored_pairs: Optional pre-computed (id_a, id_b, score) pairs
            from ANN blocking. When set, skip NxN scoring.

    Returns:
        List of (row_id_a, row_id_b, score) tuples above threshold.
    """
    # Fast path: pre-scored pairs from ANN (skip NxN scoring)
    if pre_scored_pairs is not None:
        results = []
        for a, b, score in pre_scored_pairs:
            if score >= mk.threshold:
                pair_key = (min(a, b), max(a, b))
                if exclude_pairs and pair_key in exclude_pairs:
                    continue
                results.append((pair_key[0], pair_key[1], score))
        return results

    n = block_df.height
    if n < 2:
        return []

    row_ids = block_df["__row_id__"].to_list()

    # Separate exact (cheap), record_embedding, and fuzzy (expensive) fields
    exact_fields = [f for f in mk.fields if f.scorer == "exact" or f.scorer == "soundex_match"]
    record_emb_fields = [f for f in mk.fields if f.scorer == "record_embedding"]
    fuzzy_fields = [f for f in mk.fields if f.scorer not in ("exact", "soundex_match", "record_embedding")]

    total_weight = sum(f.weight for f in mk.fields)
    if total_weight == 0.0:
        return []

    # Phase 1: Score cheap fields (exact + soundex) and build null masks
    cheap_numerator = np.zeros((n, n))
    cheap_denominator = np.zeros((n, n))

    for f in exact_fields:
        values = _get_transformed_values(block_df, f)
        null_mask = _build_null_mask(values)
        valid = ~null_mask

        if f.scorer == "exact":
            scores = _exact_score_matrix(values)
        else:  # soundex_match
            scores = _soundex_score_matrix(values)

        cheap_numerator += scores * f.weight * valid
        cheap_denominator += f.weight * valid

    # Phase 2: Early termination check
    # For each pair, the maximum possible score is:
    #   (cheap_contribution + fuzzy_max_weight) / (cheap_denom + fuzzy_max_weight)
    # where fuzzy_max_weight assumes all fuzzy fields score 1.0
    fuzzy_total_weight = sum(f.weight for f in fuzzy_fields) + sum(f.weight for f in record_emb_fields)

    # If no fuzzy or record_embedding fields, just use cheap scores
    if not fuzzy_fields and not record_emb_fields:
        with np.errstate(divide="ignore", invalid="ignore"):
            combined = np.where(cheap_denominator > 0, cheap_numerator / cheap_denominator, 0.0)
    else:
        # Check which pairs can possibly reach threshold even if all fuzzy fields score 1.0
        max_possible_numerator = cheap_numerator + fuzzy_total_weight
        max_possible_denominator = cheap_denominator + fuzzy_total_weight

        with np.errstate(divide="ignore", invalid="ignore"):
            max_possible = np.where(
                max_possible_denominator > 0,
                max_possible_numerator / max_possible_denominator,
                0.0,
            )

        # Pairs that can't possibly reach threshold — mark them
        impossible = max_possible < mk.threshold

        # Phase 3: Score fuzzy fields
        fuzzy_numerator = np.zeros((n, n))
        fuzzy_denominator = np.zeros((n, n))

        for f in fuzzy_fields:
            values = _get_transformed_values(block_df, f)
            null_mask = _build_null_mask(values)
            valid = ~null_mask

            scores = _fuzzy_score_matrix(values, f.scorer, model_name=f.model or "all-MiniLM-L6-v2")

            fuzzy_numerator += scores * f.weight * valid
            fuzzy_denominator += f.weight * valid

        for f in record_emb_fields:
            scores = _record_embedding_score_matrix(
                block_df, f.columns, model_name=f.model or "all-MiniLM-L6-v2"
            )
            fuzzy_numerator += scores * f.weight
            fuzzy_denominator += f.weight

        # Combine cheap + fuzzy
        total_numerator = cheap_numerator + fuzzy_numerator
        total_denominator = cheap_denominator + fuzzy_denominator

        with np.errstate(divide="ignore", invalid="ignore"):
            combined = np.where(total_denominator > 0, total_numerator / total_denominator, 0.0)

        # Zero out impossible pairs (early termination)
        combined[impossible] = 0.0

    # Extract upper triangle pairs above threshold using numpy
    # Zero out lower triangle and diagonal
    upper = np.triu(combined, k=1)
    rows_idx, cols_idx = np.where(upper >= mk.threshold)

    if len(rows_idx) == 0:
        return []

    row_id_arr = np.array(row_ids)
    ids_a = row_id_arr[rows_idx]
    ids_b = row_id_arr[cols_idx]
    scores = upper[rows_idx, cols_idx]

    if exclude_pairs is not None and len(exclude_pairs) > 0:
        results = []
        for a, b, s in zip(ids_a, ids_b, scores):
            pair_key = (min(int(a), int(b)), max(int(a), int(b)))
            if pair_key not in exclude_pairs:
                results.append((int(a), int(b), float(s)))
        return results

    return [(int(a), int(b), float(s)) for a, b, s in zip(ids_a, ids_b, scores)]
