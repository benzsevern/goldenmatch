"""Persistent cluster management for database integration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from goldenmatch.db.connector import DatabaseConnector

logger = logging.getLogger(__name__)


def next_cluster_id(connector: DatabaseConnector) -> int:
    """Get next available cluster ID."""
    df = connector.read_query("SELECT COALESCE(MAX(cluster_id), 0) + 1 AS next_id FROM gm_clusters")
    return int(df["next_id"][0])


def create_cluster(
    connector: DatabaseConnector,
    record_ids: list[int],
    source_table: str,
    run_id: str,
) -> int:
    """Create a new cluster with the given records. Returns cluster_id."""
    cluster_id = next_cluster_id(connector)

    cursor = connector.conn.cursor()
    try:
        for rid in record_ids:
            cursor.execute(
                "INSERT INTO gm_clusters (cluster_id, record_id, source_table, run_id) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (cluster_id, record_id, source_table) DO NOTHING",
                (cluster_id, rid, source_table, run_id),
            )
        connector.conn.commit()
    except Exception:
        connector.conn.rollback()
        raise
    finally:
        cursor.close()

    logger.debug("Created cluster %d with %d records", cluster_id, len(record_ids))
    return cluster_id


def add_to_cluster(
    connector: DatabaseConnector,
    cluster_id: int,
    record_ids: list[int],
    source_table: str,
    run_id: str,
) -> None:
    """Add records to an existing cluster."""
    cursor = connector.conn.cursor()
    try:
        for rid in record_ids:
            cursor.execute(
                "INSERT INTO gm_clusters (cluster_id, record_id, source_table, run_id) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (cluster_id, record_id, source_table) DO NOTHING",
                (cluster_id, rid, source_table, run_id),
            )
        connector.conn.commit()
    except Exception:
        connector.conn.rollback()
        raise
    finally:
        cursor.close()

    logger.debug("Added %d records to cluster %d", len(record_ids), cluster_id)


def merge_clusters(
    connector: DatabaseConnector,
    cluster_ids: list[int],
    source_table: str,
    run_id: str,
) -> int:
    """Merge multiple clusters into one. Returns the surviving cluster_id.

    Uses the lowest cluster_id as the survivor. All members from other
    clusters are reassigned to the survivor.
    """
    if len(cluster_ids) < 2:
        return cluster_ids[0] if cluster_ids else 0

    survivor = min(cluster_ids)
    others = [cid for cid in cluster_ids if cid != survivor]

    cursor = connector.conn.cursor()
    try:
        for old_cid in others:
            # Reassign members to survivor
            cursor.execute(
                "UPDATE gm_clusters SET cluster_id = %s "
                "WHERE cluster_id = %s AND source_table = %s",
                (survivor, old_cid, source_table),
            )
        connector.conn.commit()
    except Exception:
        connector.conn.rollback()
        raise
    finally:
        cursor.close()

    logger.info("Merged clusters %s → %d", cluster_ids, survivor)
    return survivor


def get_cluster_members(connector: DatabaseConnector, cluster_id: int) -> list[int]:
    """Get all record IDs in a cluster."""
    df = connector.read_query(
        f"SELECT record_id FROM gm_clusters WHERE cluster_id = {cluster_id} ORDER BY record_id"
    )
    return df["record_id"].to_list() if df.height > 0 else []


def get_cluster_for_record(
    connector: DatabaseConnector, record_id: int, source_table: str,
) -> int | None:
    """Find which cluster a record belongs to, if any."""
    df = connector.read_query(
        f"SELECT cluster_id FROM gm_clusters "
        f"WHERE record_id = {record_id} AND source_table = '{source_table}' "
        f"LIMIT 1"
    )
    return int(df["cluster_id"][0]) if df.height > 0 else None


def get_cluster_size(connector: DatabaseConnector, cluster_id: int) -> int:
    """Get the number of records in a cluster."""
    df = connector.read_query(
        f"SELECT COUNT(*) as cnt FROM gm_clusters WHERE cluster_id = {cluster_id}"
    )
    return int(df["cnt"][0]) if df.height > 0 else 0


def get_clusters_for_records(
    connector: DatabaseConnector, record_ids: list[int], source_table: str,
) -> dict[int, int]:
    """Find clusters for multiple records. Returns {record_id: cluster_id}."""
    if not record_ids:
        return {}

    id_list = ", ".join(str(int(i)) for i in record_ids)
    df = connector.read_query(
        f"SELECT record_id, cluster_id FROM gm_clusters "
        f"WHERE record_id IN ({id_list}) AND source_table = '{source_table}'"
    )

    return {int(row["record_id"]): int(row["cluster_id"]) for row in df.to_dicts()}
