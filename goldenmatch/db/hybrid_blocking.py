"""Hybrid blocking — SQL + ANN in parallel, union results."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from goldenmatch.config.schemas import BlockingConfig
from goldenmatch.db.ann_index import PersistentANNIndex
from goldenmatch.db.blocking import build_blocking_query
from goldenmatch.db.connector import DatabaseConnector, _quote_ident

logger = logging.getLogger(__name__)


def find_candidates(
    new_record: dict,
    connector: DatabaseConnector,
    ann_index: PersistentANNIndex | None,
    blocking_config: BlockingConfig,
    source_table: str,
    columns: list[str],
    id_column: str = "id",
    model_name: str = "all-MiniLM-L6-v2",
    ann_top_k: int = 20,
) -> pl.DataFrame:
    """Find candidate matches using SQL + ANN blocking, return union.

    SQL blocking handles exact/phonetic fields. ANN blocking handles
    semantic fields. Results are unioned and deduplicated before
    fetching full records from DB.

    Falls back to SQL-only if ANN index is not available.
    """
    candidate_ids: set[int] = set()

    # ── SQL Blocking ──────────────────────────────────────────────
    record_id = new_record.get(id_column)
    sql_query = build_blocking_query(
        source_table, new_record, blocking_config,
        exclude_id=record_id, id_column=id_column,
    )

    if sql_query:
        try:
            sql_candidates = connector.read_query(sql_query)
            if sql_candidates.height > 0 and id_column in sql_candidates.columns:
                sql_ids = set(sql_candidates[id_column].to_list())
                candidate_ids.update(sql_ids)
                logger.debug("SQL blocking found %d candidates", len(sql_ids))
        except Exception as e:
            logger.warning("SQL blocking failed: %s", e)

    # ── ANN Blocking ──────────────────────────────────────────────
    if ann_index is not None and ann_index.is_available:
        try:
            embedding = _embed_record(new_record, columns, model_name)
            if embedding is not None:
                neighbors = ann_index.query(embedding, top_k=ann_top_k)
                ann_ids = {db_id for _, db_id, _ in neighbors}
                # Exclude self
                if record_id is not None:
                    ann_ids.discard(record_id)
                candidate_ids.update(ann_ids)
                logger.debug("ANN blocking found %d candidates", len(ann_ids))
        except Exception as e:
            logger.warning("ANN blocking failed: %s", e)

    # ── Fetch full records ────────────────────────────────────────
    if not candidate_ids:
        return pl.DataFrame()

    # Single query to fetch all candidates by ID
    id_list = ", ".join(str(int(i)) for i in candidate_ids)
    fetch_query = (
        f"SELECT * FROM {_quote_ident(source_table)} "
        f"WHERE {_quote_ident(id_column)} IN ({id_list})"
    )

    try:
        return connector.read_query(fetch_query)
    except Exception as e:
        logger.warning("Failed to fetch candidates: %s", e)
        return pl.DataFrame()


def find_candidates_batch(
    new_records: list[dict],
    connector: DatabaseConnector,
    ann_index: PersistentANNIndex | None,
    blocking_config: BlockingConfig,
    source_table: str,
    columns: list[str],
    id_column: str = "id",
    model_name: str = "all-MiniLM-L6-v2",
    ann_top_k: int = 20,
) -> dict[int, pl.DataFrame]:
    """Find candidates for a batch of new records.

    Returns {new_record_id: candidates_df}.
    Optimizes by batching ANN queries and DB fetches.
    """
    all_candidate_ids: dict[int, set[int]] = {}

    # ── SQL Blocking (per record) ─────────────────────────────────
    for record in new_records:
        record_id = record.get(id_column)
        if record_id is None:
            continue

        all_candidate_ids[record_id] = set()

        sql_query = build_blocking_query(
            source_table, record, blocking_config,
            exclude_id=record_id, id_column=id_column,
        )
        if sql_query:
            try:
                sql_candidates = connector.read_query(sql_query)
                if sql_candidates.height > 0 and id_column in sql_candidates.columns:
                    all_candidate_ids[record_id].update(
                        sql_candidates[id_column].to_list()
                    )
            except Exception:
                pass

    # ── ANN Blocking (batched) ────────────────────────────────────
    if ann_index is not None and ann_index.is_available:
        embeddings_list = []
        record_ids_order = []

        for record in new_records:
            record_id = record.get(id_column)
            if record_id is None:
                continue
            emb = _embed_record(record, columns, model_name)
            if emb is not None:
                embeddings_list.append(emb[0])
                record_ids_order.append(record_id)

        if embeddings_list:
            batch_embeddings = np.array(embeddings_list, dtype=np.float32)
            neighbors = ann_index.query(batch_embeddings, top_k=ann_top_k)

            for query_idx, db_id, score in neighbors:
                if query_idx < len(record_ids_order):
                    rid = record_ids_order[query_idx]
                    if rid in all_candidate_ids and db_id != rid:
                        all_candidate_ids[rid].add(db_id)

    # ── Fetch all candidate records in one query ──────────────────
    all_ids = set()
    for ids in all_candidate_ids.values():
        all_ids.update(ids)

    if not all_ids:
        return {rid: pl.DataFrame() for rid in all_candidate_ids}

    id_list = ", ".join(str(int(i)) for i in all_ids)
    fetch_query = (
        f"SELECT * FROM {_quote_ident(source_table)} "
        f"WHERE {_quote_ident(id_column)} IN ({id_list})"
    )

    try:
        all_candidates_df = connector.read_query(fetch_query)
    except Exception:
        return {rid: pl.DataFrame() for rid in all_candidate_ids}

    # Split by record
    result = {}
    for rid, cand_ids in all_candidate_ids.items():
        if not cand_ids:
            result[rid] = pl.DataFrame()
        else:
            mask = all_candidates_df[id_column].is_in(list(cand_ids))
            result[rid] = all_candidates_df.filter(mask)

    return result


def _embed_record(
    record: dict, columns: list[str], model_name: str,
) -> np.ndarray | None:
    """Embed a single record for ANN query. Returns (1, dim) array."""
    try:
        from goldenmatch.core.embedder import get_embedder

        parts = []
        for col in columns:
            val = record.get(col)
            if val is not None:
                parts.append(f"{col}: {val}")
        text = " | ".join(parts) if parts else ""

        if not text.strip():
            return None

        embedder = get_embedder(model_name)
        embedding = embedder.embed_column([text], cache_key=f"_hybrid_{hash(text)}")
        return embedding
    except ImportError:
        return None
    except Exception as e:
        logger.debug("Failed to embed record: %s", e)
        return None
