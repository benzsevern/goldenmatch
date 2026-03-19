"""Incremental sync orchestrator — matches new records against existing DB."""

from __future__ import annotations

import logging
from datetime import datetime

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig
from goldenmatch.db.connector import DatabaseConnector
from goldenmatch.db.metadata import (
    config_hash,
    ensure_metadata_tables,
    get_state,
    log_matches_batch,
    new_run_id,
    update_state,
)
from goldenmatch.db.blocking import build_blocking_query
from goldenmatch.db.writer import write_golden_records

logger = logging.getLogger(__name__)


def run_sync(
    connector: DatabaseConnector,
    source_table: str,
    config: GoldenMatchConfig,
    output_mode: str = "separate",
    full_rescan: bool = False,
    dry_run: bool = False,
    chunk_size: int = 10000,
    incremental_column: str | None = None,
) -> dict:
    """Run incremental entity resolution against database.

    Args:
        connector: Active database connection
        source_table: Table to match against
        config: GoldenMatch matching configuration
        output_mode: "separate" or "in_place"
        full_rescan: Force reprocess all records
        dry_run: Match but don't write results
        chunk_size: Records per chunk for reading
        incremental_column: Column for incremental detection

    Returns:
        Summary dict with match counts, actions taken
    """
    from goldenmatch.core.autofix import auto_fix_dataframe
    from goldenmatch.core.blocker import build_blocks
    from goldenmatch.core.cluster import build_clusters
    from goldenmatch.core.golden import build_golden_record
    from goldenmatch.core.matchkey import compute_matchkeys
    from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches

    # Ensure metadata tables exist
    ensure_metadata_tables(connector)

    run_id = new_run_id()
    cfg_hash = config_hash(config.model_dump())
    matchkeys = config.get_matchkeys()

    # Check last state
    state = get_state(connector, source_table)
    total_rows = connector.get_row_count(source_table)

    if state and not full_rescan:
        if state.get("config_hash") != cfg_hash:
            logger.info("Config changed since last run. Forcing full rescan.")
            full_rescan = True

    # Determine which records to process
    if full_rescan or state is None:
        logger.info("Full scan: reading %d records from %s", total_rows, source_table)
        new_records = _read_all(connector, source_table, chunk_size)
        existing_records = pl.DataFrame()  # no existing for full scan
    else:
        new_records, existing_records = _read_incremental(
            connector, source_table, state, incremental_column, chunk_size,
        )
        logger.info(
            "Incremental: %d new records, %d existing in %s",
            new_records.height, existing_records.height if existing_records.height else "N/A", source_table,
        )

    if new_records.height == 0:
        logger.info("No new records to process.")
        return {"new_records": 0, "matches": 0, "actions": []}

    # Add internal columns
    new_records = new_records.with_columns(pl.lit("new").alias("__source__"))
    new_records = new_records.with_row_index("__row_id__").with_columns(
        pl.col("__row_id__").cast(pl.Int64)
    )

    # Auto-fix
    new_records, _ = auto_fix_dataframe(new_records)

    # For full scan: run dedupe pipeline on all records
    if full_rescan or state is None:
        return _full_scan_pipeline(
            connector, new_records, source_table, config, matchkeys,
            output_mode, dry_run, run_id, cfg_hash, total_rows,
        )

    # For incremental: match new records against existing via DB-side blocking
    return _incremental_pipeline(
        connector, new_records, source_table, config, matchkeys,
        output_mode, dry_run, run_id, cfg_hash, total_rows,
    )


def _read_all(connector: DatabaseConnector, table: str, chunk_size: int) -> pl.DataFrame:
    """Read entire table into DataFrame."""
    chunks = []
    for chunk in connector.read_table(table, chunk_size):
        chunks.append(chunk)
    return pl.concat(chunks) if chunks else pl.DataFrame()


def _read_incremental(
    connector: DatabaseConnector,
    table: str,
    state: dict,
    incremental_column: str | None,
    chunk_size: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read only new records since last state."""
    from goldenmatch.db.connector import _quote_ident

    if incremental_column and state.get("last_incremental_value"):
        last_val = state["last_incremental_value"]
        query = (
            f"SELECT * FROM {_quote_ident(table)} "
            f"WHERE {_quote_ident(incremental_column)} > '{last_val}' "
            f"ORDER BY {_quote_ident(incremental_column)}"
        )
        new_records = connector.read_query(query)
    elif state.get("last_row_id"):
        last_id = state["last_row_id"]
        query = f"SELECT * FROM {_quote_ident(table)} WHERE id > {last_id} ORDER BY id"
        new_records = connector.read_query(query)
    else:
        new_records = _read_all(connector, table, chunk_size)

    return new_records, pl.DataFrame()  # existing loaded on-demand via blocking


def _full_scan_pipeline(
    connector, df, source_table, config, matchkeys,
    output_mode, dry_run, run_id, cfg_hash, total_rows,
):
    """Run full dedupe on all records."""
    from goldenmatch.core.blocker import build_blocks
    from goldenmatch.core.cluster import build_clusters
    from goldenmatch.core.golden import build_golden_record
    from goldenmatch.core.matchkey import compute_matchkeys
    from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches

    # Compute matchkeys
    lf = df.lazy()
    lf = compute_matchkeys(lf, matchkeys)
    df = lf.collect()

    # Score pairs
    all_pairs = []
    matched_pairs = set()

    for mk in matchkeys:
        if mk.type == "exact":
            pairs = find_exact_matches(df.lazy(), mk)
            all_pairs.extend(pairs)
            for a, b, s in pairs:
                matched_pairs.add((min(a, b), max(a, b)))

    if config.blocking:
        for mk in matchkeys:
            if mk.type == "weighted":
                blocks = build_blocks(df.lazy(), config.blocking)
                for block in blocks:
                    bdf = block.df.collect()
                    pairs = find_fuzzy_matches(
                        bdf, mk,
                        exclude_pairs=matched_pairs,
                        pre_scored_pairs=block.pre_scored_pairs,
                    )
                    all_pairs.extend(pairs)
                    for a, b, s in pairs:
                        matched_pairs.add((min(a, b), max(a, b)))

    # Cluster
    all_ids = df["__row_id__"].to_list()
    clusters = build_clusters(all_pairs, all_ids, max_cluster_size=100)

    # Golden records
    golden_rules = config.golden_rules
    golden_records = []
    for cluster_id, cluster_info in clusters.items():
        if cluster_info["size"] > 1 and not cluster_info.get("oversized"):
            cluster_df = df.filter(pl.col("__row_id__").is_in(cluster_info["members"]))
            golden = build_golden_record(cluster_df, golden_rules)
            if golden:
                golden["__cluster_id__"] = cluster_id
                golden_records.append(golden)

    golden_df = pl.DataFrame(golden_records) if golden_records else None

    # Log matches
    match_actions = []
    for a, b, s in all_pairs:
        match_actions.append((int(a), int(b), float(s), "merged"))

    if not dry_run:
        log_matches_batch(connector, match_actions, run_id)
        write_golden_records(connector, clusters, golden_df, source_table, output_mode)
        update_state(connector, source_table, cfg_hash=cfg_hash, record_count=total_rows)

    multi_clusters = {k: v for k, v in clusters.items() if v["size"] > 1}
    logger.info(
        "Sync complete: %d records, %d pairs, %d clusters",
        df.height, len(all_pairs), len(multi_clusters),
    )

    return {
        "new_records": df.height,
        "matches": len(all_pairs),
        "clusters": len(multi_clusters),
        "golden_records": len(golden_records),
        "actions": match_actions,
        "run_id": run_id,
    }


def _incremental_pipeline(
    connector, new_df, source_table, config, matchkeys,
    output_mode, dry_run, run_id, cfg_hash, total_rows,
    embed_chunk_size=100000,
):
    """Match new records against existing database using hybrid blocking."""
    from goldenmatch.core.scorer import score_pair
    from goldenmatch.db.ann_index import PersistentANNIndex
    from goldenmatch.db.hybrid_blocking import find_candidates

    all_pairs = []
    match_actions = []
    new_embeddings_ids = []
    new_embeddings_vecs = []

    id_col = "id" if "id" in new_df.columns else new_df.columns[0]
    matchable_cols = [c for c in new_df.columns if not c.startswith("__")]

    # Load or build ANN index
    ann_index = None
    try:
        ann_index = PersistentANNIndex(
            connector=connector, source_table=source_table,
        )
        ann_index.load_or_build()
    except Exception as e:
        logger.debug("ANN index not available: %s", e)

    for row in new_df.iter_rows(named=True):
        record_id = row.get(id_col)

        if not config.blocking:
            match_actions.append((record_id, 0, 0.0, "new"))
            continue

        # Hybrid blocking: SQL + ANN union
        candidates = find_candidates(
            new_record=row,
            connector=connector,
            ann_index=ann_index,
            blocking_config=config.blocking,
            source_table=source_table,
            columns=matchable_cols,
            id_column=id_col,
        )

        if candidates.height == 0:
            match_actions.append((record_id, 0, 0.0, "new"))
            continue

        # Score against each candidate
        best_score = 0.0
        best_match_id = None

        for mk in matchkeys:
            if mk.type != "weighted":
                continue

            for candidate in candidates.iter_rows(named=True):
                cand_id = candidate.get(id_col)
                score = score_pair(row, candidate, mk.fields)

                if score >= (mk.threshold or 0.0) and score > best_score:
                    best_score = score
                    best_match_id = cand_id

        if best_match_id is not None:
            all_pairs.append((record_id, best_match_id, best_score))
            match_actions.append((record_id, best_match_id, best_score, "merged"))
        else:
            match_actions.append((record_id, 0, 0.0, "new"))

    # Add new record embeddings to index
    if ann_index is not None and new_embeddings_ids:
        import numpy as np
        ann_index.add(new_embeddings_ids, np.array(new_embeddings_vecs))

    # Progressive: embed next chunk of existing records
    if ann_index is not None:
        _embed_next_chunk(connector, ann_index, source_table, matchable_cols, embed_chunk_size)

    # Save index
    if ann_index is not None:
        try:
            ann_index.save()
        except Exception as e:
            logger.debug("Failed to save ANN index: %s", e)

    if not dry_run:
        log_matches_batch(connector, match_actions, run_id)
        update_state(
            connector, source_table,
            cfg_hash=cfg_hash, record_count=total_rows,
        )

    merged = sum(1 for _, _, _, a in match_actions if a == "merged")
    new_entities = sum(1 for _, _, _, a in match_actions if a == "new")

    logger.info(
        "Incremental sync: %d new records — %d merged, %d new entities",
        new_df.height, merged, new_entities,
    )

    return {
        "new_records": new_df.height,
        "matches": len(all_pairs),
        "merged": merged,
        "new_entities": new_entities,
        "actions": match_actions,
        "run_id": run_id,
    }


def _embed_next_chunk(
    connector: DatabaseConnector,
    ann_index: PersistentANNIndex,
    source_table: str,
    columns: list[str],
    chunk_size: int = 100000,
) -> int:
    """Embed next chunk of existing records for progressive ANN coverage."""
    try:
        from goldenmatch.core.embedder import get_embedder
        from goldenmatch.db.connector import _quote_ident

        # Find records not yet embedded
        already_embedded = ann_index.record_count
        query = (
            f"SELECT id FROM {_quote_ident(source_table)} "
            f"WHERE id NOT IN ("
            f"  SELECT record_id FROM gm_embeddings "
            f"  WHERE source_table = '{source_table}'"
            f") ORDER BY id LIMIT {chunk_size}"
        )

        df = connector.read_query(query)
        if df.height == 0:
            logger.info("All records already embedded.")
            return 0

        record_ids = df["id"].to_list()

        # Fetch full records for embedding
        id_list = ", ".join(str(int(i)) for i in record_ids)
        records_df = connector.read_query(
            f"SELECT * FROM {_quote_ident(source_table)} WHERE id IN ({id_list})"
        )

        # Build text for embedding
        texts = []
        for row in records_df.iter_rows(named=True):
            parts = [f"{c}: {row.get(c, '')}" for c in columns if row.get(c) is not None]
            texts.append(" | ".join(parts) if parts else "")

        embedder = get_embedder(ann_index.model_name)
        embeddings = embedder.embed_column(texts, cache_key=f"_progressive_{source_table}")

        ann_index.add(record_ids, embeddings)
        logger.info("Progressive embedding: added %d records (%d total)", len(record_ids), ann_index.record_count)
        return len(record_ids)

    except ImportError:
        logger.debug("sentence-transformers not available for progressive embedding")
        return 0
    except Exception as e:
        logger.warning("Progressive embedding failed: %s", e)
        return 0
