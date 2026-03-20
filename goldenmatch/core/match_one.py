"""Single-record matching -- match one record against an existing dataset.

This is the streaming primitive: embed, query ANN for top-K candidates,
score each candidate pair, return matches above threshold. No full pipeline
re-run needed.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from goldenmatch.config.schemas import MatchkeyConfig
from goldenmatch.core.scorer import score_pair
from goldenmatch.utils.transforms import apply_transforms

logger = logging.getLogger(__name__)


def match_one(
    record: dict,
    df: pl.DataFrame,
    mk: MatchkeyConfig,
    ann_blocker=None,
    embedder=None,
    ann_column: str | None = None,
    top_k: int = 20,
) -> list[tuple[int, float]]:
    """Match a single record against a dataset.

    Uses ANN index for candidate retrieval when available, then scores
    each candidate pair using the matchkey's fields/scorers/weights.
    Falls back to brute-force scoring when no ANN index is provided
    (suitable for small datasets only).

    Args:
        record: Dict of field->value for the new record.
        df: Existing dataset with __row_id__ column.
        mk: Matchkey config with fields, scorers, weights, threshold.
        ann_blocker: Optional ANNBlocker with a built FAISS index.
        embedder: Optional Embedder for computing the new record's embedding.
        ann_column: Column to embed for ANN candidate retrieval.
        top_k: Number of ANN candidates to retrieve.

    Returns:
        List of (row_id, score) tuples for matches above mk.threshold.
    """
    if mk.threshold is None:
        return []

    # ANN candidate retrieval
    if ann_blocker is not None and embedder is not None and ann_column is not None:
        return _match_one_ann(record, df, mk, ann_blocker, embedder, ann_column, top_k)

    # Brute-force fallback
    return _match_one_brute(record, df, mk)


def _match_one_ann(
    record: dict,
    df: pl.DataFrame,
    mk: MatchkeyConfig,
    ann_blocker,
    embedder,
    ann_column: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """ANN-accelerated single-record matching."""
    # Build embedding text from the record
    text = str(record.get(ann_column, "") or "")
    embedding = embedder.embed_column([text], cache_key=f"_match_one_{hash(text)}")

    # Query top-K candidates
    candidates = ann_blocker.query_one(embedding[0])

    if not candidates:
        return []

    # Map FAISS index positions to row_ids
    row_ids = df["__row_id__"].to_list()
    rows = df.to_dicts()

    results = []
    for faiss_idx, ann_score in candidates:
        if faiss_idx >= len(rows):
            continue
        candidate_row = rows[faiss_idx]
        candidate_id = row_ids[faiss_idx]

        # Score using matchkey fields
        pair_score = score_pair(record, candidate_row, mk.fields)

        if pair_score >= mk.threshold:
            results.append((int(candidate_id), pair_score))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        "match_one: %d ANN candidates, %d above threshold %.2f",
        len(candidates), len(results), mk.threshold,
    )
    return results


def _match_one_brute(
    record: dict,
    df: pl.DataFrame,
    mk: MatchkeyConfig,
) -> list[tuple[int, float]]:
    """Brute-force single-record matching (no ANN index)."""
    if df.height > 10000:
        logger.warning(
            "Brute-force match_one on %d records. Consider using ANN blocking.",
            df.height,
        )

    row_ids = df["__row_id__"].to_list()
    rows = df.to_dicts()

    results = []
    for i, candidate_row in enumerate(rows):
        pair_score = score_pair(record, candidate_row, mk.fields)
        if pair_score >= mk.threshold:
            results.append((int(row_ids[i]), pair_score))

    results.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        "match_one (brute): %d records scanned, %d above threshold %.2f",
        df.height, len(results), mk.threshold,
    )
    return results
