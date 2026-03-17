"""Block analyzer for GoldenMatch — auto-suggests optimal blocking keys."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from itertools import combinations

import polars as pl

from goldenmatch.utils.transforms import apply_transforms

logger = logging.getLogger(__name__)


# ── Column type detection ───────────────────────────────────────────────────


def detect_column_type(column_name: str) -> str:
    """Heuristic name-based type detection for a column.

    Returns one of: "name", "zip", "email", "phone", "state", "generic".
    """
    lower = column_name.lower()

    if re.search(r"(name|fname|lname)", lower):
        return "name"
    if re.search(r"(zip|postal)", lower):
        return "zip"
    if re.search(r"(email|mail)", lower):
        return "email"
    if re.search(r"(phone|tel|mobile)", lower):
        return "phone"
    if re.search(r"(state)", lower):
        return "state"
    return "generic"


# ── Candidate generation ────────────────────────────────────────────────────


def _single_column_candidates(column: str) -> list[dict]:
    """Generate single-column blocking key candidates based on detected type."""
    col_type = detect_column_type(column)
    candidates = []

    if col_type == "name":
        for length in (3, 4, 5):
            candidates.append({
                "key_fields": [column],
                "transforms": ["lowercase", f"substring:0:{length}"],
                "description": f"{column}[:{length}]",
            })
        candidates.append({
            "key_fields": [column],
            "transforms": ["lowercase", "soundex"],
            "description": f"soundex({column})",
        })
    elif col_type == "zip":
        for length in (3, 5):
            candidates.append({
                "key_fields": [column],
                "transforms": [f"substring:0:{length}"],
                "description": f"{column}[:{length}]",
            })
        candidates.append({
            "key_fields": [column],
            "transforms": [],
            "description": column,
        })
    elif col_type == "state":
        candidates.append({
            "key_fields": [column],
            "transforms": [],
            "description": column,
        })
    elif col_type == "email":
        candidates.append({
            "key_fields": [column],
            "transforms": ["lowercase", "substring:0:5"],
            "description": f"{column}[:5]",
        })
    elif col_type == "phone":
        for length in (3, 6):
            candidates.append({
                "key_fields": [column],
                "transforms": [f"substring:0:{length}"],
                "description": f"{column}[:{length}]",
            })
    else:  # generic
        for length in (3, 4, 5):
            candidates.append({
                "key_fields": [column],
                "transforms": [f"substring:0:{length}"],
                "description": f"{column}[:{length}]",
            })

    return candidates


def generate_candidates(matchkey_columns: list[str]) -> list[dict]:
    """Generate blocking key candidates from matchkey columns.

    Produces single-column candidates based on column type heuristics,
    plus compound candidates combining pairs of single-column candidates.
    """
    # Single-column candidates
    single_candidates: dict[str, list[dict]] = {}
    all_candidates: list[dict] = []

    for col in matchkey_columns:
        col_candidates = _single_column_candidates(col)
        single_candidates[col] = col_candidates
        all_candidates.extend(col_candidates)

    # Compound candidates: combine pairs of columns (max 2)
    if len(matchkey_columns) >= 2:
        for col_a, col_b in combinations(matchkey_columns, 2):
            for cand_a in single_candidates[col_a]:
                for cand_b in single_candidates[col_b]:
                    all_candidates.append({
                        "key_fields": [col_a, col_b],
                        "transforms": [cand_a["transforms"], cand_b["transforms"]],
                        "description": f"{cand_a['description']} + {cand_b['description']}",
                    })

    return all_candidates


# ── Scoring ──────────────────────────────────────────────────────────────────


def _apply_candidate_transforms(df: pl.DataFrame, candidate: dict) -> pl.DataFrame:
    """Apply a candidate's transforms and add __block_key__ column."""
    key_fields = candidate["key_fields"]
    transforms = candidate["transforms"]

    if len(key_fields) == 1:
        col = key_fields[0]
        tfms = transforms  # flat list of transforms
        if tfms:
            expr = pl.col(col).cast(pl.Utf8).map_elements(
                lambda val, t=tfms: apply_transforms(val, t),
                return_dtype=pl.Utf8,
            )
        else:
            expr = pl.col(col).cast(pl.Utf8)
        return df.with_columns(expr.alias("__block_key__"))
    else:
        # Compound: transforms is a list of lists
        field_exprs = []
        for i, col in enumerate(key_fields):
            tfms = transforms[i] if i < len(transforms) else []
            if tfms:
                expr = pl.col(col).cast(pl.Utf8).map_elements(
                    lambda val, t=tfms: apply_transforms(val, t),
                    return_dtype=pl.Utf8,
                )
            else:
                expr = pl.col(col).cast(pl.Utf8)
            field_exprs.append(expr)
        concat_expr = pl.concat_str(field_exprs, separator="||")
        return df.with_columns(concat_expr.alias("__block_key__"))


def score_candidate(
    df: pl.DataFrame,
    candidate: dict,
    target_block_size: int = 5000,
) -> dict:
    """Score a blocking key candidate on the given data.

    Returns a dict with group_count, max_group_size, mean_group_size,
    std_group_size, total_comparisons, and score.
    """
    # Check columns exist
    for col in candidate["key_fields"]:
        if col not in df.columns:
            return {
                "group_count": 0,
                "max_group_size": 0,
                "mean_group_size": 0.0,
                "std_group_size": 0.0,
                "total_comparisons": 0,
                "score": 0.0,
            }

    df_with_key = _apply_candidate_transforms(df, candidate)

    # Drop nulls in block key
    df_valid = df_with_key.filter(pl.col("__block_key__").is_not_null())

    if len(df_valid) == 0:
        return {
            "group_count": 0,
            "max_group_size": 0,
            "mean_group_size": 0.0,
            "std_group_size": 0.0,
            "total_comparisons": 0,
            "score": 0.0,
        }

    # Group by block key using Polars expressions
    stats = (
        df_valid
        .group_by("__block_key__")
        .agg(pl.len().alias("block_size"))
    )

    group_count = len(stats)
    total_records = len(df_valid)

    if group_count == 0:
        return {
            "group_count": 0,
            "max_group_size": 0,
            "mean_group_size": 0.0,
            "std_group_size": 0.0,
            "total_comparisons": 0,
            "score": 0.0,
        }

    max_group_size = stats["block_size"].max()
    mean_group_size = stats["block_size"].mean()
    std_group_size = stats["block_size"].std() if group_count > 1 else 0.0
    if std_group_size is None:
        std_group_size = 0.0

    # total_comparisons = sum(n*(n-1)/2) using Polars
    total_comparisons = int(
        stats.select(
            (pl.col("block_size") * (pl.col("block_size") - 1) / 2).sum()
        ).item()
    )

    # Score formula
    if mean_group_size == 0:
        score = 0.0
    else:
        score = (
            (group_count / total_records)
            * (1 / (1 + max_group_size / target_block_size))
            * (1 / (1 + std_group_size / mean_group_size))
        )

    return {
        "group_count": group_count,
        "max_group_size": int(max_group_size),
        "mean_group_size": float(mean_group_size),
        "std_group_size": float(std_group_size),
        "total_comparisons": total_comparisons,
        "score": float(score),
    }


# ── Coverage check ───────────────────────────────────────────────────────────


def check_coverage(candidate: dict, matchkey_columns: list[str]) -> bool:
    """Check if all key_fields in the candidate are in matchkey_columns."""
    return all(f in matchkey_columns for f in candidate["key_fields"])


# ── Recall estimation ────────────────────────────────────────────────────────


def estimate_recall(
    df: pl.DataFrame,
    candidate: dict,
    matchkey_columns: list[str],
    sample_size: int = 1000,
) -> float:
    """Estimate recall for a blocking candidate using pair sampling.

    Takes a random sample, finds fuzzy-similar pairs via JaroWinkler on the
    highest-cardinality matchkey column, then checks what fraction would land
    in the same block under this candidate.
    """
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.process import cdist
    import numpy as np

    n = len(df)
    if n < 2:
        return 0.0

    actual_sample = min(sample_size, n)
    sample_df = df.sample(actual_sample, seed=42)

    # Pick highest-cardinality matchkey column
    valid_cols = [c for c in matchkey_columns if c in sample_df.columns]
    if not valid_cols:
        return 0.0

    best_col = max(valid_cols, key=lambda c: sample_df[c].n_unique())

    # Prepare string values for cdist
    values = (
        sample_df[best_col]
        .cast(pl.Utf8)
        .fill_null("")
        .to_list()
    )
    values = [str(v).lower().strip() for v in values]

    # Compute pairwise JaroWinkler scores
    scores = cdist(values, values, scorer=JaroWinkler.similarity, workers=1)

    # Find pairs above threshold (upper triangle only)
    threshold = 0.7
    pairs_above = set()
    for i in range(actual_sample):
        for j in range(i + 1, actual_sample):
            if scores[i][j] >= threshold:
                pairs_above.add((i, j))

    if not pairs_above:
        return 1.0  # No pairs to miss

    # Apply candidate transforms to sample and get block keys
    sample_with_key = _apply_candidate_transforms(sample_df, candidate)
    block_keys = sample_with_key["__block_key__"].to_list()

    # Check how many pairs share the same block key
    pairs_in_same_block = sum(
        1 for i, j in pairs_above
        if block_keys[i] is not None and block_keys[i] == block_keys[j]
    )

    return pairs_in_same_block / len(pairs_above)


# ── BlockingSuggestion ───────────────────────────────────────────────────────


@dataclass
class BlockingSuggestion:
    """A ranked blocking strategy suggestion."""

    keys: list[dict]
    group_count: int
    max_group_size: int
    mean_group_size: float
    total_comparisons: int
    estimated_recall: float
    score: float
    description: str


# ── Main analyzer ────────────────────────────────────────────────────────────


def analyze_blocking(
    df: pl.DataFrame,
    matchkey_columns: list[str],
    sample_size: int = 1000,
    target_block_size: int = 5000,
) -> list[BlockingSuggestion]:
    """Analyze data and return ranked blocking strategy suggestions.

    Pipeline:
    1. Generate candidates from matchkey_columns
    2. Score each candidate
    3. Check coverage (demote non-covering ones)
    4. Estimate recall for top candidates (top 10 by score)
    5. Sort by score * recall_bonus
    6. Return ranked list
    """
    candidates = generate_candidates(matchkey_columns)

    # Score each candidate
    scored = []
    for cand in candidates:
        metrics = score_candidate(df, cand, target_block_size=target_block_size)
        if metrics["group_count"] == 0:
            continue
        scored.append((cand, metrics))

    if not scored:
        return []

    # Sort by score descending to pick top candidates for recall estimation
    scored.sort(key=lambda x: x[1]["score"], reverse=True)

    # Estimate recall for top 10
    top_n = min(10, len(scored))
    for i in range(top_n):
        cand, metrics = scored[i]
        try:
            recall = estimate_recall(df, cand, matchkey_columns, sample_size=sample_size)
        except Exception:
            logger.warning(f"Recall estimation failed for {cand['description']}", exc_info=True)
            recall = 0.0
        metrics["estimated_recall"] = recall

    # For the rest, set recall to 0.0
    for i in range(top_n, len(scored)):
        scored[i][1]["estimated_recall"] = 0.0

    # Build suggestions with coverage-based ranking
    suggestions = []
    for cand, metrics in scored:
        covers = check_coverage(cand, matchkey_columns)
        recall_bonus = 1.0 if covers else 0.5
        adjusted_score = metrics["score"] * recall_bonus

        suggestions.append(BlockingSuggestion(
            keys=[cand],
            group_count=metrics["group_count"],
            max_group_size=metrics["max_group_size"],
            mean_group_size=metrics["mean_group_size"],
            total_comparisons=metrics["total_comparisons"],
            estimated_recall=metrics.get("estimated_recall", 0.0),
            score=adjusted_score,
            description=cand["description"],
        ))

    # Sort by final score descending
    suggestions.sort(key=lambda s: s.score, reverse=True)

    return suggestions
