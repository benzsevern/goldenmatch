"""GoldenMatch metadata table management (gm_* tables)."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime

from goldenmatch.db.connector import DatabaseConnector

logger = logging.getLogger(__name__)


# ── Table DDL ─────────────────────────────────────────────────────────────

_STATE_DDL = """
CREATE TABLE IF NOT EXISTS gm_state (
    id SERIAL PRIMARY KEY,
    source_table TEXT NOT NULL,
    last_processed_at TIMESTAMP,
    last_row_id BIGINT,
    last_incremental_value TEXT,
    config_hash TEXT,
    record_count BIGINT DEFAULT 0
);
"""

_EMBEDDINGS_DDL = """
CREATE TABLE IF NOT EXISTS gm_embeddings (
    record_id BIGINT NOT NULL,
    source_table TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (record_id, source_table, model_name)
);
"""

_MATCH_LOG_DDL = """
CREATE TABLE IF NOT EXISTS gm_match_log (
    id SERIAL PRIMARY KEY,
    record_id_a BIGINT NOT NULL,
    record_id_b BIGINT NOT NULL,
    score DOUBLE PRECISION,
    action TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

_GOLDEN_RECORDS_DDL = """
CREATE TABLE IF NOT EXISTS gm_golden_records (
    id SERIAL PRIMARY KEY,
    cluster_id BIGINT NOT NULL,
    source_table TEXT NOT NULL,
    source_ids BIGINT[] NOT NULL,
    record_data JSONB NOT NULL,
    merged_at TIMESTAMP DEFAULT NOW(),
    is_current BOOLEAN DEFAULT TRUE,
    version INT DEFAULT 1,
    run_id TEXT
);
"""

_CLUSTERS_DDL = """
CREATE TABLE IF NOT EXISTS gm_clusters (
    cluster_id BIGINT NOT NULL,
    record_id BIGINT NOT NULL,
    source_table TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT NOW(),
    run_id TEXT,
    PRIMARY KEY (cluster_id, record_id, source_table)
);
"""


def ensure_metadata_tables(connector: DatabaseConnector) -> None:
    """Create gm_* tables if they don't exist."""
    for ddl in [_STATE_DDL, _EMBEDDINGS_DDL, _MATCH_LOG_DDL, _GOLDEN_RECORDS_DDL, _CLUSTERS_DDL]:
        connector.execute(ddl)
    logger.info("GoldenMatch metadata tables ready")


# ── State management ──────────────────────────────────────────────────────

def config_hash(config_dict: dict) -> str:
    """Hash a config dict for change detection."""
    import json
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def get_state(connector: DatabaseConnector, source_table: str) -> dict | None:
    """Get last processing state for a table."""
    df = connector.read_query(
        f"SELECT * FROM gm_state WHERE source_table = '{source_table}' "
        f"ORDER BY last_processed_at DESC LIMIT 1"
    )
    if df.height == 0:
        return None
    return df.to_dicts()[0]


def update_state(
    connector: DatabaseConnector,
    source_table: str,
    last_row_id: int | None = None,
    last_incremental_value: str | None = None,
    cfg_hash: str | None = None,
    record_count: int = 0,
) -> None:
    """Insert new state record."""
    connector.execute(
        """
        INSERT INTO gm_state (source_table, last_processed_at, last_row_id,
                              last_incremental_value, config_hash, record_count)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (source_table, datetime.utcnow(), last_row_id,
         last_incremental_value, cfg_hash, record_count),
    )


# ── Match logging ─────────────────────────────────────────────────────────

def new_run_id() -> str:
    return str(uuid.uuid4())


def log_match(
    connector: DatabaseConnector,
    record_id_a: int,
    record_id_b: int,
    score: float,
    action: str,
    run_id: str,
) -> None:
    """Log a match decision."""
    connector.execute(
        """
        INSERT INTO gm_match_log (record_id_a, record_id_b, score, action, run_id)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (record_id_a, record_id_b, score, action, run_id),
    )


def log_matches_batch(
    connector: DatabaseConnector,
    matches: list[tuple[int, int, float, str]],
    run_id: str,
) -> None:
    """Batch log match decisions."""
    if not matches:
        return
    cursor = connector.conn.cursor()
    try:
        from psycopg2.extras import execute_values
        execute_values(
            cursor,
            "INSERT INTO gm_match_log (record_id_a, record_id_b, score, action, run_id) "
            "VALUES %s",
            [(a, b, s, action, run_id) for a, b, s, action in matches],
        )
        connector.conn.commit()
    except Exception:
        connector.conn.rollback()
        raise
    finally:
        cursor.close()
