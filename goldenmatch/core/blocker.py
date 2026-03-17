"""Blocker for GoldenMatch — groups records into blocks for comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from goldenmatch.config.schemas import BlockingConfig
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


def build_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build blocks from a LazyFrame based on blocking configuration.

    For each blocking key config, build a block key expression (transform +
    concat_str fields), collect the LazyFrame, group by the block key, skip
    blocks with < 2 records, and warn/skip if block exceeds max_block_size.

    Args:
        lf: Input LazyFrame.
        config: Blocking configuration with keys, max_block_size, skip_oversized.

    Returns:
        List of BlockResult, one per valid block.
    """
    results: list[BlockResult] = []

    for key_config in config.keys:
        # Build expression for each field in this blocking key
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

        # Concatenate fields with || separator
        if len(field_exprs) == 1:
            block_key_expr = field_exprs[0].alias("__block_key__")
        else:
            block_key_expr = pl.concat_str(field_exprs, separator="||").alias("__block_key__")

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
