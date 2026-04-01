"""Blocker for GoldenMatch — groups records into blocks for comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, CanopyConfig
from goldenmatch.utils.transforms import apply_transforms

logger = logging.getLogger(__name__)


@dataclass
class BlockResult:
    """Result of blocking: a block key and its associated LazyFrame."""

    block_key: str
    df: pl.LazyFrame
    strategy: str = "static"
    depth: int = 0
    parent_key: str | None = None
    pre_scored_pairs: list[tuple[int, int, float]] | None = None


def _build_block_key_expr(key_config: BlockingKeyConfig) -> pl.Expr:
    """Build a block key expression from a BlockingKeyConfig.

    Transforms each field and concatenates with || separator.
    """
    field_exprs = []
    for field_name in key_config.fields:
        if key_config.transforms:
            expr = pl.col(field_name).map_elements(
                lambda val, transforms=key_config.transforms: apply_transforms(val, transforms),
                return_dtype=pl.Utf8,
            )
        else:
            expr = pl.col(field_name).cast(pl.Utf8)
        field_exprs.append(expr)

    if len(field_exprs) == 1:
        return field_exprs[0].alias("__block_key__")
    else:
        return pl.concat_str(field_exprs, separator="||").alias("__block_key__")


def _build_static_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build static blocks — original blocking logic.

    Groups records by each blocking key, skipping blocks with < 2 records
    and handling oversized blocks per config.skip_oversized.
    """
    results: list[BlockResult] = []

    for key_config in config.keys:
        block_key_expr = _build_block_key_expr(key_config)

        # Add block key column and collect
        df_with_key = lf.with_columns(block_key_expr).collect()

        # Group by block key
        groups = df_with_key.group_by("__block_key__")

        for key, group_df in groups:
            key_str = key[0]  # group_by returns tuple of key values
            if key_str is None:
                continue

            size = len(group_df)

            if size < 2:
                continue

            if size > config.max_block_size:
                if config.skip_oversized and config.ann_column:
                    # ANN fallback: embed oversized block's records and sub-block
                    ann_sub = _ann_sub_block(
                        group_df, config.ann_column, config.ann_top_k,
                        config.ann_model, config.max_block_size, key_str,
                    )
                    if ann_sub:
                        results.extend(ann_sub)
                    continue
                elif config.skip_oversized:
                    logger.warning(
                        f"Block {key_str!r} has {size} records "
                        f"(exceeds max_block_size={config.max_block_size}). Skipping."
                    )
                    continue
                else:
                    logger.warning(
                        f"Block {key_str!r} has {size} records "
                        f"(exceeds max_block_size={config.max_block_size}). Processing anyway."
                    )

            results.append(BlockResult(
                block_key=key_str,
                df=group_df.lazy(),
            ))

    return results


def _ann_sub_block(
    block_df: pl.DataFrame,
    ann_column: str,
    ann_top_k: int,
    ann_model: str,
    max_block_size: int,
    parent_key: str,
) -> list[BlockResult]:
    """ANN fallback for oversized blocks.

    Embeds only the unique text values in the block, maps embeddings back
    to all records, then uses FAISS to find neighbors and create sub-blocks.
    """
    from goldenmatch.core.ann_blocker import ANNBlocker
    from goldenmatch.core.cluster import UnionFind
    from goldenmatch.core.embedder import get_embedder

    size = len(block_df)

    # Cap: only ANN sub-block moderately oversized blocks (up to 10x max_block_size)
    # Truly massive blocks (60K+) would still be too expensive to embed
    if size > max_block_size * 10:
        logger.info(
            "ANN fallback: block %r has %d records (>%dx max). Too large, skipping.",
            parent_key, size, 10,
        )
        return []

    if ann_column not in block_df.columns:
        logger.warning(
            "ANN fallback: column %r not in block %r. Skipping %d records.",
            ann_column, parent_key, size,
        )
        return []

    # Deduplicate texts — embed only unique values
    all_texts = block_df[ann_column].to_list()
    unique_texts = list(set(t for t in all_texts if t is not None and str(t).strip()))

    if len(unique_texts) < 2:
        logger.info("ANN fallback: block %r has <2 unique texts. Skipping.", parent_key)
        return []

    logger.info(
        "ANN fallback: block %r has %d records, %d unique texts. Embedding...",
        parent_key, size, len(unique_texts),
    )

    embedder = get_embedder(ann_model)
    unique_embeddings = embedder.embed_column(
        unique_texts, cache_key=f"ann_sub_{parent_key}",
    )

    # Map unique embeddings back to all records
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    record_indices = []  # index into unique_embeddings for each record
    valid_records = []   # indices into block_df that have valid text
    for i, t in enumerate(all_texts):
        if t is not None and str(t).strip() and t in text_to_idx:
            record_indices.append(text_to_idx[t])
            valid_records.append(i)

    if len(valid_records) < 2:
        return []

    import numpy as np
    record_embeddings = unique_embeddings[np.array(record_indices)]

    # Build FAISS index and query
    blocker = ANNBlocker(top_k=min(ann_top_k, len(valid_records) - 1))
    blocker.build_index(record_embeddings)
    pairs = blocker.query(record_embeddings)

    # Group into sub-blocks via Union-Find
    row_ids = block_df["__row_id__"].to_list()
    uf = UnionFind()
    for a, b in pairs:
        real_a = valid_records[a]
        real_b = valid_records[b]
        uf.add(real_a)
        uf.add(real_b)
        uf.union(real_a, real_b)

    clusters = uf.get_clusters()
    results: list[BlockResult] = []
    n_oversized = 0
    for members in clusters:
        if len(members) < 2:
            continue
        member_list = sorted(members)
        if len(member_list) > max_block_size:
            n_oversized += 1
            continue  # still too big even after ANN sub-blocking
        sub_df = block_df[member_list]
        results.append(BlockResult(
            block_key=f"{parent_key}_ann_{min(member_list)}",
            df=sub_df.lazy(),
            strategy="ann",
        ))

    logger.info(
        "ANN fallback: block %r -> %d sub-blocks (%d still oversized)",
        parent_key, len(results), n_oversized,
    )
    return results


def _sub_block(
    block_df: pl.DataFrame,
    sub_block_keys: list[BlockingKeyConfig],
    max_block_size: int,
    depth: int,
    parent_key: str,
) -> list[BlockResult]:
    """Recursively sub-block an oversized block using sub_block_keys.

    Args:
        block_df: The oversized block DataFrame.
        sub_block_keys: Remaining sub-block keys to try.
        max_block_size: Maximum block size threshold.
        depth: Current recursion depth (1-indexed).
        parent_key: The parent block key value.

    Returns:
        List of BlockResult with adaptive metadata.
    """
    if depth > 3 or not sub_block_keys:
        # Max depth reached or no more keys — return as-is with warning
        logger.warning(
            f"Sub-block of {parent_key!r} has {len(block_df)} records at depth {depth}. "
            f"No further sub-blocking possible. Processing anyway."
        )
        return [BlockResult(
            block_key=parent_key,
            df=block_df.lazy(),
            strategy="adaptive",
            depth=depth,
            parent_key=parent_key,
        )]

    current_key_config = sub_block_keys[0]
    remaining_keys = sub_block_keys[1:]

    block_key_expr = _build_block_key_expr(current_key_config)
    df_with_key = block_df.with_columns(block_key_expr)

    groups = df_with_key.group_by("__block_key__")
    results: list[BlockResult] = []

    for key, group_df in groups:
        key_str = key[0]
        if key_str is None:
            continue

        size = len(group_df)

        if size < 2:
            continue

        if size > max_block_size and remaining_keys and depth < 3:
            # Recurse with next sub_block_key
            sub_results = _sub_block(
                group_df,
                remaining_keys,
                max_block_size,
                depth + 1,
                parent_key,
            )
            results.extend(sub_results)
        else:
            if size > max_block_size:
                logger.warning(
                    f"Sub-block {key_str!r} of {parent_key!r} has {size} records at depth {depth}. "
                    f"Processing anyway."
                )
            results.append(BlockResult(
                block_key=key_str,
                df=group_df.lazy(),
                strategy="adaptive",
                depth=depth,
                parent_key=parent_key,
            ))

    return results


def _auto_split_block(
    block_df: pl.DataFrame,
    max_block_size: int,
    parent_key: str,
) -> list[BlockResult]:
    """Auto-split an oversized block using the highest-cardinality column.

    When no sub_block_keys are configured, this provides a zero-config fallback
    that splits by the column with the most unique values.
    """
    # Find non-internal columns
    candidates = [c for c in block_df.columns if not c.startswith("__")]
    if not candidates:
        logger.warning(
            "Auto-split of %r: no non-internal columns available. Processing as-is.",
            parent_key,
        )
        return [BlockResult(block_key=parent_key, df=block_df.lazy(), strategy="adaptive", depth=1, parent_key=parent_key)]

    # Pick column whose cardinality best splits blocks near max_block_size.
    # Ideal: each group has ~max_block_size records.
    # Score = number of groups with >= 2 records (useful groups).
    n = len(block_df)
    best_col = candidates[0]
    best_useful_groups = 0
    best_nunique = 0

    for col in candidates:
        nunique = block_df[col].n_unique()
        # Estimate: if we split by this column, avg group size = n / nunique
        avg_group = n / nunique if nunique > 0 else n
        # Count groups that will have >= 2 records (useful for matching)
        useful_groups = block_df.group_by(pl.col(col).cast(pl.Utf8)).agg(
            pl.len().alias("cnt")
        ).filter(pl.col("cnt") >= 2).height

        if useful_groups > best_useful_groups or (
            useful_groups == best_useful_groups and avg_group <= max_block_size and nunique > best_nunique
        ):
            best_useful_groups = useful_groups
            best_nunique = nunique
            best_col = col

    split_expr = pl.col(best_col).cast(pl.Utf8).alias("__auto_split__")

    df_with_key = block_df.with_columns(split_expr)
    groups = df_with_key.group_by("__auto_split__")

    results: list[BlockResult] = []
    for key, group_df in groups:
        key_str = key[0]
        if key_str is None:
            continue
        if len(group_df) < 2:
            continue
        if len(group_df) > max_block_size:
            logger.warning(
                "Auto-split sub-block %r of %r has %d records (still oversized). Processing anyway.",
                key_str, parent_key, len(group_df),
            )
        results.append(BlockResult(
            block_key=f"{parent_key}||{key_str}",
            df=group_df.drop("__auto_split__").lazy(),
            strategy="adaptive",
            depth=1,
            parent_key=parent_key,
        ))

    logger.info(
        "Auto-split %r (%d records) into %d sub-blocks using column %r (cardinality=%d)",
        parent_key, len(block_df), len(results), best_col, best_nunique,
    )
    return results if results else [BlockResult(block_key=parent_key, df=block_df.lazy(), strategy="adaptive", depth=1, parent_key=parent_key)]


def _build_sorted_neighborhood_blocks(
    lf: pl.LazyFrame, config: BlockingConfig,
) -> list[BlockResult]:
    """Build sorted neighborhood blocks with a sliding window.

    For each SortKeyField in config.sort_key, transform the column and
    concatenate into a sort key. Collect, sort, then slide a window through.
    """
    if not config.sort_key:
        raise ValueError("sorted_neighborhood strategy requires sort_key configuration.")

    # Build sort key expression
    sort_field_exprs = []
    for skf in config.sort_key:
        if skf.transforms:
            expr = pl.col(skf.column).map_elements(
                lambda val, transforms=skf.transforms: apply_transforms(val, transforms),
                return_dtype=pl.Utf8,
            )
        else:
            expr = pl.col(skf.column).cast(pl.Utf8)
        sort_field_exprs.append(expr)

    if len(sort_field_exprs) == 1:
        sort_key_expr = sort_field_exprs[0].alias("__sort_key__")
    else:
        sort_key_expr = pl.concat_str(sort_field_exprs, separator="||").alias("__sort_key__")

    # Collect and sort
    df = lf.with_columns(sort_key_expr).collect().sort("__sort_key__")
    n = len(df)
    window_size = config.window_size

    results: list[BlockResult] = []

    if n <= window_size:
        # Dataset smaller than window — single block
        if n >= 2:
            results.append(BlockResult(
                block_key="sorted_window_0",
                df=df.lazy(),
                strategy="sorted_neighborhood",
            ))
        return results

    # Slide window through sorted data
    for i in range(n - window_size + 1):
        window_df = df.slice(i, window_size)
        results.append(BlockResult(
            block_key=f"sorted_window_{i}",
            df=window_df.lazy(),
            strategy="sorted_neighborhood",
        ))

    return results


def _build_multi_pass_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Run multiple blocking passes and union candidate blocks.

    Each pass uses a different BlockingKeyConfig. Blocks with duplicate keys
    across passes are deduplicated so each unique block key appears once.
    """
    all_blocks: list[BlockResult] = []
    seen_keys: set[str] = set()

    for pass_config in config.passes or []:
        temp_config = BlockingConfig(
            keys=[pass_config],
            max_block_size=config.max_block_size,
            skip_oversized=config.skip_oversized,
            ann_column=config.ann_column,
            ann_top_k=config.ann_top_k,
            ann_model=config.ann_model,
        )
        blocks = _build_static_blocks(lf, temp_config)
        for block in blocks:
            if block.block_key not in seen_keys:
                block.strategy = "multi_pass"
                all_blocks.append(block)
                seen_keys.add(block.block_key)

    return all_blocks


def _build_ann_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build blocks using ANN (approximate nearest neighbor) on embeddings.

    Embeds the configured column, queries top-K neighbors with FAISS,
    then groups connected pairs into micro-blocks via Union-Find.
    """
    from goldenmatch.core.ann_blocker import ANNBlocker
    from goldenmatch.core.cluster import UnionFind
    from goldenmatch.core.embedder import get_embedder

    if not config.ann_column:
        raise ValueError("ANN blocking requires 'ann_column' to be set.")

    df = lf.collect()
    values = df[config.ann_column].to_list()

    embedder = get_embedder(config.ann_model)
    embeddings = embedder.embed_column(values, cache_key=f"ann_{config.ann_column}")

    blocker = ANNBlocker(top_k=config.ann_top_k)
    blocker.build_index(embeddings)
    pairs = blocker.query(embeddings)

    # Group nearby records into micro-blocks using Union-Find
    row_ids = df["__row_id__"].to_list()
    uf = UnionFind()
    for a, b in pairs:
        uf.add(a)
        uf.add(b)
        uf.union(a, b)

    clusters = uf.get_clusters()
    results: list[BlockResult] = []
    for members in clusters:
        if len(members) < 2:
            continue
        member_list = sorted(members)
        block_df = df.filter(pl.col("__row_id__").is_in([row_ids[m] for m in member_list]))
        results.append(BlockResult(
            block_key=f"ann_{min(member_list)}",
            df=block_df.lazy(),
            strategy="ann",
        ))

    return results


def _build_ann_pair_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build direct-pair ANN blocks without Union-Find.

    Returns a single BlockResult with pre_scored_pairs set.
    FAISS similarity scores are propagated directly.
    """
    from goldenmatch.core.ann_blocker import ANNBlocker
    from goldenmatch.core.embedder import get_embedder

    if not config.ann_column:
        raise ValueError("ann_pairs blocking requires 'ann_column' to be set.")

    df = lf.collect()
    values = df[config.ann_column].to_list()

    embedder = get_embedder(config.ann_model)
    embeddings = embedder.embed_column(values, cache_key=f"ann_{config.ann_column}")

    blocker = ANNBlocker(top_k=config.ann_top_k)
    blocker.build_index(embeddings)
    scored_pairs = blocker.query_with_scores(embeddings)

    # Map positional indices to __row_id__ values
    row_ids = df["__row_id__"].to_list()
    mapped_pairs = [
        (int(row_ids[a]), int(row_ids[b]), score)
        for a, b, score in scored_pairs
    ]

    return [BlockResult(
        block_key="ann_pairs",
        df=df.lazy(),
        strategy="ann_pairs",
        pre_scored_pairs=mapped_pairs,
    )]


def _build_learned_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build blocks using learned predicates.

    Two-pass approach:
    1. If cached rules exist, load and apply them
    2. Otherwise, run a fast sample with static blocking to generate training pairs,
       then learn predicates from those pairs
    """
    from goldenmatch.core.learned_blocking import (
        apply_learned_blocks,
        learn_blocking_rules,
        load_learned_rules,
        save_learned_rules,
    )

    # Try loading cached rules
    if config.learned_cache_path:
        cached = load_learned_rules(config.learned_cache_path)
        if cached:
            logger.info("Using cached learned blocking rules from %s", config.learned_cache_path)
            return apply_learned_blocks(lf, cached, config.max_block_size)

    # Pass 1: fast static blocking on first key to generate training pairs
    df = lf.collect()
    sample_size = min(config.learned_sample_size, df.height)
    if sample_size < df.height:
        sample_df = df.sample(sample_size, seed=42)
    else:
        sample_df = df

    # Use static blocking with the configured keys for the sample run
    sample_config = config.model_copy(update={"strategy": "static"})
    sample_blocks = _build_static_blocks(sample_df.lazy(), sample_config)

    # Score sample blocks to get training pairs
    from goldenmatch.core.scorer import find_fuzzy_matches
    from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField

    # Build a simple weighted matchkey for scoring
    cols = [c for c in df.columns if not c.startswith("__")]
    if not cols:
        return _build_static_blocks(lf, sample_config)

    # Use first few columns for a quick score
    score_fields = [
        MatchkeyField(field=c, scorer="token_sort", weight=1.0, transforms=["lowercase"])
        for c in cols[:3]
    ]
    score_mk = MatchkeyConfig(name="_learned_score", type="weighted", threshold=0.5, fields=score_fields)

    scored_pairs = []
    for block in sample_blocks:
        block_df = block.df.collect() if hasattr(block.df, 'collect') else block.df
        pairs = find_fuzzy_matches(block_df, score_mk)
        scored_pairs.extend(pairs)

    if not scored_pairs:
        logger.warning("No scored pairs from sample run. Falling back to static blocking.")
        return _build_static_blocks(lf, sample_config)

    # Pass 2: learn rules from scored pairs
    rules = learn_blocking_rules(
        sample_df,
        scored_pairs,
        columns=cols,
        min_recall=config.learned_min_recall,
        min_reduction=config.learned_min_reduction,
        predicate_depth=config.learned_predicate_depth,
    )

    # Cache rules
    if config.learned_cache_path and rules:
        save_learned_rules(rules, config.learned_cache_path)
        logger.info("Saved learned blocking rules to %s", config.learned_cache_path)

    # Apply to full dataset
    return apply_learned_blocks(lf, rules, config.max_block_size)


def _build_canopy_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build blocks using TF-IDF canopy clustering.

    Forms overlapping canopies based on cosine similarity of TF-IDF vectors.
    Records can appear in multiple canopies.
    """
    from goldenmatch.core.canopy import build_canopies

    if not config.canopy:
        raise ValueError("Canopy blocking requires 'canopy' config to be set.")

    df = lf.collect()
    canopy_cfg = config.canopy

    # Concatenate canopy fields into a single text value per record
    text_values = []
    for row in df.iter_rows(named=True):
        parts = [str(row.get(f, "") or "") for f in canopy_cfg.fields]
        text_values.append(" ".join(parts))

    canopies = build_canopies(
        text_values,
        loose_threshold=canopy_cfg.loose_threshold,
        tight_threshold=canopy_cfg.tight_threshold,
        max_canopy_size=canopy_cfg.max_canopy_size,
    )

    row_ids = df["__row_id__"].to_list()
    results: list[BlockResult] = []
    for i, members in enumerate(canopies):
        if len(members) < 2:
            continue
        block_df = df.filter(pl.col("__row_id__").is_in([row_ids[m] for m in members]))
        results.append(BlockResult(
            block_key=f"canopy_{i}",
            df=block_df.lazy(),
            strategy="canopy",
        ))

    return results


def select_best_blocking_key(
    lf: pl.LazyFrame,
    keys: list[BlockingKeyConfig],
    max_block_size: int = 5000,
) -> BlockingKeyConfig:
    """Evaluate blocking keys and select the one with smallest max block size.

    Computes group-size histogram for each candidate key, then picks the key
    that minimizes max_group_size while maintaining >= 50% coverage.
    """
    if len(keys) <= 1:
        return keys[0]

    df = lf.collect()
    total = len(df)

    best_key = keys[0]
    best_max_size = float("inf")

    for key_config in keys:
        block_key_expr = _build_block_key_expr(key_config)
        df_with_key = df.with_columns(block_key_expr)

        # Count non-null block keys (coverage)
        non_null = df_with_key.filter(pl.col("__block_key__").is_not_null()).height
        coverage = non_null / total if total > 0 else 0.0

        if coverage < 0.5:
            logger.debug(
                "Auto-select: skipping key %s (coverage %.1f%% < 50%%)",
                key_config.fields, coverage * 100,
            )
            continue

        # Compute group sizes
        groups = df_with_key.filter(pl.col("__block_key__").is_not_null()).group_by("__block_key__").agg(
            pl.len().alias("size")
        )
        max_size = groups["size"].max()
        group_count = groups.height

        logger.debug(
            "Auto-select: key %s -> groups=%d, max_size=%d, coverage=%.1f%%",
            key_config.fields, group_count, max_size, coverage * 100,
        )

        if max_size < best_max_size or (max_size == best_max_size and group_count > 0):
            best_max_size = max_size
            best_key = key_config

    logger.info(
        "Auto-select: chose key %s (max_block_size=%d)",
        best_key.fields, best_max_size,
    )
    return best_key


def build_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build blocks from a LazyFrame based on blocking configuration.

    Routes by config.strategy:
    - "static": original blocking behavior
    - "adaptive": primary blocks + recursive sub-blocking for oversized blocks
    - "sorted_neighborhood": sliding window over sorted data
    - "ann": ANN blocking with FAISS on embeddings
    - "canopy": TF-IDF canopy clustering

    Args:
        lf: Input LazyFrame.
        config: Blocking configuration with keys, max_block_size, skip_oversized.

    Returns:
        List of BlockResult, one per valid block.
    """
    # Auto-select: pick best key based on histogram analysis
    if config.auto_select and config.keys and len(config.keys) > 1:
        best_key = select_best_blocking_key(lf, config.keys, config.max_block_size)
        config = config.model_copy(update={"keys": [best_key], "auto_select": False})

    if config.strategy == "learned":
        return _build_learned_blocks(lf, config)

    if config.strategy == "canopy":
        return _build_canopy_blocks(lf, config)

    if config.strategy == "ann_pairs":
        return _build_ann_pair_blocks(lf, config)

    if config.strategy == "ann":
        return _build_ann_blocks(lf, config)

    if config.strategy == "sorted_neighborhood":
        return _build_sorted_neighborhood_blocks(lf, config)

    if config.strategy == "multi_pass":
        return _build_multi_pass_blocks(lf, config)

    if config.strategy == "static":
        return _build_static_blocks(lf, config)

    # strategy == "adaptive"
    primary_blocks = _build_static_blocks(lf, config)
    sub_block_keys = config.sub_block_keys or []

    results: list[BlockResult] = []
    for block in primary_blocks:
        block_df = block.df.collect()
        size = len(block_df)

        if size > config.max_block_size and sub_block_keys:
            sub_results = _sub_block(
                block_df,
                sub_block_keys,
                config.max_block_size,
                depth=1,
                parent_key=block.block_key,
            )
            results.extend(sub_results)
        elif size > config.max_block_size and not config.skip_oversized:
            # Auto-split: no sub_block_keys configured, split by highest-cardinality column
            auto_results = _auto_split_block(block_df, config.max_block_size, block.block_key)
            results.extend(auto_results)
        else:
            results.append(block)

    return results
