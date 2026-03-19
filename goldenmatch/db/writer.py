"""Result write-back to database."""

from __future__ import annotations

import json
import logging

import polars as pl

from goldenmatch.db.connector import DatabaseConnector, _quote_ident

logger = logging.getLogger(__name__)


def write_golden_records(
    connector: DatabaseConnector,
    clusters: dict[int, dict],
    golden_df: pl.DataFrame | None,
    source_table: str,
    mode: str = "separate",
) -> int:
    """Write golden records to database.

    Args:
        connector: Database connection
        clusters: Cluster dict from build_clusters
        golden_df: Golden record DataFrame
        source_table: Name of source table
        mode: "separate" (gm_golden_records) or "in_place" (update source)

    Returns:
        Number of records written
    """
    if golden_df is None or golden_df.height == 0:
        return 0

    if mode == "separate":
        return _write_separate(connector, clusters, golden_df, source_table)
    elif mode == "in_place":
        return _write_in_place(connector, clusters, source_table)
    else:
        raise ValueError(f"Unknown output mode: {mode}")


def _write_separate(
    connector: DatabaseConnector,
    clusters: dict[int, dict],
    golden_df: pl.DataFrame,
    source_table: str,
) -> int:
    """Write golden records to gm_golden_records table."""
    rows_written = 0

    cursor = connector.conn.cursor()
    try:
        for cluster_id, cluster_info in clusters.items():
            if cluster_info.get("size", 0) < 2:
                continue

            members = cluster_info.get("members", [])

            # Get golden record data for this cluster
            cluster_golden = golden_df.filter(
                pl.col("__cluster_id__") == cluster_id
            ) if "__cluster_id__" in golden_df.columns else None

            if cluster_golden is not None and cluster_golden.height > 0:
                record_data = {}
                for col in cluster_golden.columns:
                    if not col.startswith("__"):
                        val = cluster_golden[col][0]
                        if val is not None:
                            record_data[col] = str(val)

                cursor.execute(
                    """
                    INSERT INTO gm_golden_records (cluster_id, source_table, source_ids, record_data)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (cluster_id, source_table, members, json.dumps(record_data)),
                )
                rows_written += 1

        connector.conn.commit()
    except Exception:
        connector.conn.rollback()
        raise
    finally:
        cursor.close()

    logger.info("Wrote %d golden records to gm_golden_records", rows_written)
    return rows_written


def _write_in_place(
    connector: DatabaseConnector,
    clusters: dict[int, dict],
    source_table: str,
) -> int:
    """Add cluster columns to source table and update matched rows."""
    table = _quote_ident(source_table)

    # Add columns if they don't exist
    for col, col_type in [
        ("__cluster_id__", "BIGINT"),
        ("__is_golden__", "BOOLEAN DEFAULT FALSE"),
    ]:
        try:
            connector.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                f"{_quote_ident(col)} {col_type}"
            )
        except Exception as e:
            logger.debug("Column %s may already exist: %s", col, e)

    # Update cluster assignments
    cursor = connector.conn.cursor()
    rows_updated = 0
    try:
        for cluster_id, cluster_info in clusters.items():
            members = cluster_info.get("members", [])
            if len(members) < 2:
                continue

            # Update all members with cluster ID
            placeholders = ",".join(["%s"] * len(members))
            cursor.execute(
                f"UPDATE {table} SET {_quote_ident('__cluster_id__')} = %s "
                f"WHERE id IN ({placeholders})",
                [cluster_id] + members,
            )
            rows_updated += cursor.rowcount

            # Mark first member as golden (or use golden record logic)
            cursor.execute(
                f"UPDATE {table} SET {_quote_ident('__is_golden__')} = TRUE "
                f"WHERE id = %s",
                (members[0],),
            )

        connector.conn.commit()
    except Exception:
        connector.conn.rollback()
        raise
    finally:
        cursor.close()

    logger.info("Updated %d rows in %s with cluster assignments", rows_updated, source_table)
    return rows_updated
