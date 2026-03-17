"""Blocker for GoldenMatch — groups records into blocks for comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
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
                if config.skip_oversized:
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
        )
        blocks = _build_static_blocks(lf, temp_config)
        for block in blocks:
            if block.block_key not in seen_keys:
                block.strategy = "multi_pass"
                all_blocks.append(block)
                seen_keys.add(block.block_key)

    return all_blocks


def build_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build blocks from a LazyFrame based on blocking configuration.

    Routes by config.strategy:
    - "static": original blocking behavior
    - "adaptive": primary blocks + recursive sub-blocking for oversized blocks
    - "sorted_neighborhood": sliding window over sorted data

    Args:
        lf: Input LazyFrame.
        config: Blocking configuration with keys, max_block_size, skip_oversized.

    Returns:
        List of BlockResult, one per valid block.
    """
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
        else:
            results.append(block)

    return results
