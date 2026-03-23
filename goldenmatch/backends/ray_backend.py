"""Ray distributed backend for large-scale entity resolution.

Replaces ThreadPoolExecutor block scoring with Ray distributed tasks.
Each block is scored as an independent Ray task, enabling parallelism
across all CPU cores (local) or a Ray cluster (distributed).

Usage:
    pip install goldenmatch[ray]
    goldenmatch dedupe huge.parquet --backend ray
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_ray = None


def _ensure_ray():
    """Import and initialize Ray lazily."""
    global _ray
    if _ray is not None:
        return _ray
    try:
        import ray
        _ray = ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
            logger.info(
                "Ray initialized: %d CPUs, %s",
                ray.cluster_resources().get("CPU", 0),
                "local" if ray.util.client.ray.is_connected() is False else "cluster",
            )
        return ray
    except ImportError:
        raise ImportError(
            "Ray backend requires ray. Install with: pip install goldenmatch[ray]"
        )


def score_blocks_ray(
    blocks: list,
    mk: "MatchkeyConfig",
    matched_pairs: set[tuple[int, int]],
    across_files_only: bool = False,
    source_lookup: dict[int, str] | None = None,
    target_ids: set[int] | None = None,
) -> list[tuple[int, int, float]]:
    """Score all blocks using Ray distributed tasks.

    Drop-in replacement for score_blocks_parallel. Each block is submitted
    as an independent Ray task. Ray handles scheduling across all available
    CPU cores (local mode) or cluster nodes.

    Args:
        blocks: List of BlockResult objects.
        mk: Matchkey configuration.
        matched_pairs: Set of already-matched (min_id, max_id) pairs.
        across_files_only: Filter to cross-source pairs only.
        source_lookup: Row ID to source name mapping.
        target_ids: For match mode — filter to target/ref cross pairs.

    Returns:
        All fuzzy pairs found across blocks.
    """
    ray = _ensure_ray()

    if not blocks:
        return []

    # For very small block counts, use the regular scorer (no Ray overhead)
    if len(blocks) <= 4:
        from goldenmatch.core.scorer import score_blocks_parallel
        return score_blocks_parallel(
            blocks, mk, matched_pairs,
            across_files_only=across_files_only,
            source_lookup=source_lookup,
            target_ids=target_ids,
        )

    from goldenmatch.core.scorer import _score_one_block

    # Freeze exclude pairs for immutable sharing
    frozen_exclude = frozenset(matched_pairs)

    # Put shared data in Ray object store (zero-copy for large objects)
    mk_ref = ray.put(mk)
    exclude_ref = ray.put(frozen_exclude)
    source_ref = ray.put(source_lookup) if source_lookup else None

    @ray.remote
    def _score_block_remote(block, mk_config, exclude, across_only, src_lookup):
        """Ray remote task: score one block."""
        return _score_one_block(
            block, mk_config, exclude,
            across_files_only=across_only,
            source_lookup=src_lookup,
        )

    # Submit all blocks as Ray tasks
    futures = []
    for block in blocks:
        # Collect the lazy DataFrame before sending to Ray
        if hasattr(block, 'df') and hasattr(block.df, 'collect'):
            import polars as pl
            collected_block = type(block)(
                block_key=block.block_key,
                df=block.df.collect().lazy(),
                strategy=block.strategy,
                depth=getattr(block, 'depth', 0),
                parent_key=getattr(block, 'parent_key', None),
                pre_scored_pairs=getattr(block, 'pre_scored_pairs', None),
            )
        else:
            collected_block = block

        future = _score_block_remote.remote(
            collected_block, mk_ref, exclude_ref,
            across_files_only, source_ref,
        )
        futures.append(future)

    logger.info("Submitted %d blocks to Ray (%d CPUs available)",
                len(futures), int(ray.cluster_resources().get("CPU", 0)))

    # Collect results
    all_pairs = []
    results = ray.get(futures)
    for pairs in results:
        if target_ids is not None:
            pairs = [
                (a, b, s) for a, b, s in pairs
                if (a in target_ids) != (b in target_ids)
            ]
        all_pairs.extend(pairs)
        for a, b, s in pairs:
            matched_pairs.add((min(a, b), max(a, b)))

    return all_pairs


def shutdown_ray():
    """Shut down the Ray runtime if initialized."""
    global _ray
    if _ray is not None and _ray.is_initialized():
        _ray.shutdown()
        logger.info("Ray shut down")
    _ray = None
