"""Apply pair-level corrections during scoring."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from goldenmatch.core.memory.store import MemoryStore

log = logging.getLogger("goldenmatch.memory")


@dataclass
class CorrectionStats:
    """Statistics from applying corrections."""
    applied: int = 0
    stale: int = 0
    total_pairs: int = 0
    stale_pairs: list[tuple[int, int]] = field(default_factory=list)


def build_row_lookup(df: pl.DataFrame, fields: list[str]) -> dict[int, tuple]:
    """Build row ID to field values lookup once for all pairs."""
    available = [f for f in fields if f in df.columns]
    if "__row_id__" not in df.columns:
        log.warning("DataFrame missing __row_id__ column, corrections cannot be applied")
        return {}
    if not available:
        log.warning("No matchkey fields found in DataFrame: %s", fields)
        return {}
    rows = df.select(["__row_id__"] + available).to_dicts()
    return {r["__row_id__"]: tuple(r[f] for f in available) for r in rows}


def compute_field_hash(row_a_vals: tuple, row_b_vals: tuple) -> str:
    """Hash matched field values for staleness detection."""
    combined = "|".join(str(v) for v in row_a_vals + row_b_vals)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def compute_record_hash(df: pl.DataFrame, row_id: int) -> str:
    """Hash ALL fields (sorted by name) for entity identity check."""
    filtered = df.filter(pl.col("__row_id__") == row_id)
    if filtered.is_empty():
        log.warning("Row ID %d not found in DataFrame, returning empty hash", row_id)
        return ""
    row = filtered.select(sorted(df.columns)).row(0)
    return hashlib.sha256("|".join(str(v) for v in row).encode()).hexdigest()[:16]


def apply_corrections(
    scored_pairs: list[tuple[int, int, float]],
    store: MemoryStore,
    df: pl.DataFrame,
    matchkey_fields: list[str],
    dataset: str | None = None,
) -> tuple[list[tuple[int, int, float]], CorrectionStats]:
    """Apply pair-level corrections to scored pairs.

    Returns adjusted pairs and correction stats.
    """
    stats = CorrectionStats(total_pairs=len(scored_pairs))

    pair_keys = [(a, b) for a, b, _ in scored_pairs]
    corrections = store.get_pair_corrections_bulk(pair_keys, dataset=dataset)

    if not corrections:
        return scored_pairs, stats

    field_lookup = build_row_lookup(df, matchkey_fields)

    # Pre-compute record hashes for correction-involved rows
    record_hashes: dict[int, str] = {}
    for (id_a, id_b) in corrections:
        if id_a not in record_hashes:
            record_hashes[id_a] = compute_record_hash(df, id_a)
        if id_b not in record_hashes:
            record_hashes[id_b] = compute_record_hash(df, id_b)

    adjusted = []
    for id_a, id_b, score in scored_pairs:
        correction = corrections.get((id_a, id_b))
        if correction is None:
            adjusted.append((id_a, id_b, score))
            continue

        # Check if row IDs are in lookup
        if id_a not in field_lookup or id_b not in field_lookup:
            log.warning("Row ID(s) not in lookup for correction (%d, %d), marking stale", id_a, id_b)
            adjusted.append((id_a, id_b, score))
            stats.stale += 1
            stats.stale_pairs.append((id_a, id_b))
            continue

        current_field_hash = compute_field_hash(
            field_lookup[id_a], field_lookup[id_b],
        )
        current_record_hash = (
            f"{record_hashes.get(id_a, '')}:{record_hashes.get(id_b, '')}"
        )

        # Empty hashes = collected without DataFrame access, skip staleness check
        hashes_empty = (not correction.field_hash and not correction.record_hash)
        hashes_match = (
            current_field_hash == correction.field_hash
            and current_record_hash == correction.record_hash
        )

        if hashes_empty or hashes_match:
            new_score = 1.0 if correction.decision == "approve" else 0.0
            adjusted.append((id_a, id_b, new_score))
            stats.applied += 1
        else:
            adjusted.append((id_a, id_b, score))
            stats.stale += 1
            stats.stale_pairs.append((id_a, id_b))

    log.info("Corrections: %d applied, %d stale, %d total pairs",
             stats.applied, stats.stale, stats.total_pairs)
    return adjusted, stats
