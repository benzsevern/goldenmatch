"""Reconciliation engine — merges new records into existing clusters."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import polars as pl

from goldenmatch.config.schemas import GoldenRulesConfig
from goldenmatch.db.clusters import (
    add_to_cluster,
    create_cluster,
    get_cluster_members,
    get_cluster_size,
    merge_clusters,
)
from goldenmatch.db.connector import DatabaseConnector, _quote_ident

logger = logging.getLogger(__name__)


@dataclass
class ReconcileResult:
    """Result of reconciling a new record against existing clusters."""

    action: str  # "merged", "new", "conflict"
    cluster_id: int
    golden_record: dict
    merged_clusters: list[int] = field(default_factory=list)
    previous_version_id: int | None = None


def reconcile_match(
    new_record: dict,
    new_record_id: int,
    matched_cluster_ids: list[int],
    match_scores: dict[int, float],
    connector: DatabaseConnector,
    source_table: str,
    golden_rules: GoldenRulesConfig | None = None,
    max_cluster_size: int = 100,
    merge_mode: str = "recompute",
    run_id: str = "",
    id_column: str = "id",
) -> ReconcileResult:
    """Reconcile a new record against matched clusters.

    Args:
        new_record: The new record as a dict.
        new_record_id: ID of the new record.
        matched_cluster_ids: Cluster IDs that the new record matched.
        match_scores: {cluster_id: best_score} for ranking.
        connector: Database connection.
        source_table: Source table name.
        golden_rules: Golden record merge strategy.
        max_cluster_size: Safety cap for cluster merging.
        merge_mode: "recompute" or "incremental".
        run_id: Current sync run ID.
        id_column: Name of the ID column.

    Returns:
        ReconcileResult with action taken and updated golden record.
    """
    # Deduplicate cluster IDs
    unique_clusters = list(set(matched_cluster_ids))

    if not unique_clusters:
        # No match — create new single-record cluster
        return _handle_new_entity(
            new_record, new_record_id, connector, source_table, run_id,
        )

    if len(unique_clusters) == 1:
        # Single cluster match — add to existing
        return _handle_single_match(
            new_record, new_record_id, unique_clusters[0],
            connector, source_table, golden_rules, merge_mode, run_id, id_column,
        )

    # Multiple cluster matches — merge or conflict
    return _handle_multi_match(
        new_record, new_record_id, unique_clusters, match_scores,
        connector, source_table, golden_rules, max_cluster_size,
        merge_mode, run_id, id_column,
    )


def _handle_new_entity(
    new_record, new_record_id, connector, source_table, run_id,
) -> ReconcileResult:
    """Create a new cluster for an unmatched record."""
    cluster_id = create_cluster(connector, [new_record_id], source_table, run_id)

    # Golden record is just the record itself
    golden = {k: v for k, v in new_record.items() if not k.startswith("__")}

    _write_golden_version(connector, cluster_id, source_table, [new_record_id], golden, run_id)

    logger.debug("New entity: record %d → cluster %d", new_record_id, cluster_id)
    return ReconcileResult(action="new", cluster_id=cluster_id, golden_record=golden)


def _handle_single_match(
    new_record, new_record_id, cluster_id,
    connector, source_table, golden_rules, merge_mode, run_id, id_column,
) -> ReconcileResult:
    """Add new record to existing cluster and update golden record."""
    # Add to cluster
    add_to_cluster(connector, cluster_id, [new_record_id], source_table, run_id)

    # Get all members
    members = get_cluster_members(connector, cluster_id)

    # Compute golden record
    golden = _compute_golden(
        members, connector, source_table, golden_rules, merge_mode,
        new_record, id_column,
    )

    # Get previous version ID
    prev_id = _get_current_version_id(connector, cluster_id)

    # Write new version
    _write_golden_version(connector, cluster_id, source_table, members, golden, run_id)

    logger.debug("Merged: record %d → cluster %d (%d members)", new_record_id, cluster_id, len(members))
    return ReconcileResult(
        action="merged", cluster_id=cluster_id,
        golden_record=golden, previous_version_id=prev_id,
    )


def _handle_multi_match(
    new_record, new_record_id, cluster_ids, match_scores,
    connector, source_table, golden_rules, max_cluster_size,
    merge_mode, run_id, id_column,
) -> ReconcileResult:
    """Handle new record matching multiple clusters."""
    # Calculate merged size
    total_size = 1  # the new record
    for cid in cluster_ids:
        total_size += get_cluster_size(connector, cid)

    if total_size <= max_cluster_size:
        # Safe to merge all clusters
        survivor = merge_clusters(connector, cluster_ids, source_table, run_id)
        add_to_cluster(connector, survivor, [new_record_id], source_table, run_id)

        members = get_cluster_members(connector, survivor)
        golden = _compute_golden(
            members, connector, source_table, golden_rules, merge_mode,
            new_record, id_column,
        )

        # Mark old golden records as not current
        for cid in cluster_ids:
            _mark_not_current(connector, cid)

        _write_golden_version(connector, survivor, source_table, members, golden, run_id)

        logger.info(
            "Multi-merge: record %d bridged clusters %s → %d (%d members)",
            new_record_id, cluster_ids, survivor, len(members),
        )
        return ReconcileResult(
            action="merged", cluster_id=survivor,
            golden_record=golden, merged_clusters=cluster_ids,
        )
    else:
        # Would exceed size cap — assign to best match, log conflict
        best_cluster = max(cluster_ids, key=lambda cid: match_scores.get(cid, 0.0))
        add_to_cluster(connector, best_cluster, [new_record_id], source_table, run_id)

        members = get_cluster_members(connector, best_cluster)
        golden = _compute_golden(
            members, connector, source_table, golden_rules, merge_mode,
            new_record, id_column,
        )

        prev_id = _get_current_version_id(connector, best_cluster)
        _write_golden_version(connector, best_cluster, source_table, members, golden, run_id)

        logger.warning(
            "Conflict: record %d matches clusters %s but merge would exceed max size %d. "
            "Assigned to cluster %d.",
            new_record_id, cluster_ids, max_cluster_size, best_cluster,
        )
        return ReconcileResult(
            action="conflict", cluster_id=best_cluster,
            golden_record=golden, merged_clusters=cluster_ids,
            previous_version_id=prev_id,
        )


# ── Golden record computation ─────────────────────────────────────────────

def _compute_golden(
    member_ids, connector, source_table, golden_rules, merge_mode,
    new_record=None, id_column="id",
) -> dict:
    """Compute golden record from cluster members."""
    if merge_mode == "incremental" and new_record is not None:
        return _incremental_merge(connector, member_ids, source_table, new_record, golden_rules)

    # Recompute: fetch all members and run build_golden_record
    return _recompute_golden(connector, member_ids, source_table, golden_rules, id_column)


def _recompute_golden(connector, member_ids, source_table, golden_rules, id_column) -> dict:
    """Full recompute of golden record from all cluster members."""
    if not member_ids:
        return {}

    id_list = ", ".join(str(int(i)) for i in member_ids)
    df = connector.read_query(
        f"SELECT * FROM {_quote_ident(source_table)} WHERE {_quote_ident(id_column)} IN ({id_list})"
    )

    if df.height == 0:
        return {}

    try:
        from goldenmatch.core.golden import build_golden_record
        golden = build_golden_record(df, golden_rules)
        return golden if golden else {}
    except Exception as e:
        logger.warning("Golden record computation failed: %s", e)
        # Fallback: use first record
        return {k: v for k, v in df.to_dicts()[0].items() if not k.startswith("__")}


def _incremental_merge(connector, member_ids, source_table, new_record, golden_rules) -> dict:
    """Incrementally merge new record into existing golden record."""
    # Get current golden record
    current = _get_current_golden(connector, source_table)
    if not current:
        return {k: v for k, v in new_record.items() if not k.startswith("__")}

    # Simple incremental: for each field, apply golden rule
    merged = dict(current)
    for key, new_val in new_record.items():
        if key.startswith("__"):
            continue
        old_val = merged.get(key)

        if old_val is None and new_val is not None:
            merged[key] = new_val
        elif new_val is not None:
            # Default: keep most complete (longest)
            if len(str(new_val)) > len(str(old_val or "")):
                merged[key] = new_val

    return merged


# ── Golden record versioning helpers ──────────────────────────────────────

def _get_current_version_id(connector, cluster_id) -> int | None:
    """Get the ID of the current golden record version."""
    df = connector.read_query(
        f"SELECT id FROM gm_golden_records "
        f"WHERE cluster_id = {cluster_id} AND is_current = TRUE "
        f"LIMIT 1"
    )
    return int(df["id"][0]) if df.height > 0 else None


def _get_current_golden(connector, source_table) -> dict | None:
    """Get current golden record data for the most recent cluster."""
    df = connector.read_query(
        f"SELECT record_data FROM gm_golden_records "
        f"WHERE source_table = '{source_table}' AND is_current = TRUE "
        f"ORDER BY merged_at DESC LIMIT 1"
    )
    if df.height == 0:
        return None
    try:
        data = df["record_data"][0]
        return json.loads(data) if isinstance(data, str) else data
    except Exception:
        return None


def _mark_not_current(connector, cluster_id) -> None:
    """Mark existing golden records for a cluster as not current."""
    connector.execute(
        "UPDATE gm_golden_records SET is_current = FALSE "
        f"WHERE cluster_id = {cluster_id} AND is_current = TRUE"
    )


def _write_golden_version(
    connector, cluster_id, source_table, member_ids, golden_data, run_id,
) -> None:
    """Insert a new golden record version."""
    # Get next version number
    df = connector.read_query(
        f"SELECT COALESCE(MAX(version), 0) + 1 AS next_ver "
        f"FROM gm_golden_records WHERE cluster_id = {cluster_id}"
    )
    next_version = int(df["next_ver"][0]) if df.height > 0 else 1

    # Mark old as not current
    _mark_not_current(connector, cluster_id)

    # Insert new version
    cursor = connector.conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO gm_golden_records "
            "(cluster_id, source_table, source_ids, record_data, is_current, version, run_id) "
            "VALUES (%s, %s, %s, %s, TRUE, %s, %s)",
            (cluster_id, source_table, member_ids, json.dumps(golden_data),
             next_version, run_id or None),
        )
        connector.conn.commit()
    except Exception:
        connector.conn.rollback()
        raise
    finally:
        cursor.close()

    logger.debug("Golden record v%d for cluster %d (%d members)", next_version, cluster_id, len(member_ids))
