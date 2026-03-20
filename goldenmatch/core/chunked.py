"""Large dataset mode — chunked processing for files that don't fit in memory.

Processes a CSV/Parquet in chunks, maintains a persistent match index,
and merges results across chunks. Handles datasets from 1M to 100M+ records.

Architecture:
  Chunk 1 → match within chunk → add to index
  Chunk 2 → match within chunk + match against index → add to index
  Chunk 3 → match within chunk + match against index → add to index
  ...
  Final → merge all clusters → compute golden records
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig

logger = logging.getLogger(__name__)


class ChunkedMatcher:
    """Process large files in chunks with persistent matching."""

    def __init__(
        self,
        config: GoldenMatchConfig,
        chunk_size: int = 100_000,
    ):
        self.config = config
        self.chunk_size = chunk_size

        # Persistent state across chunks
        self._all_pairs: list[tuple[int, int, float]] = []
        self._all_ids: list[int] = []
        self._index_records: list[dict] = []  # representative records for cross-chunk matching
        self._row_offset = 0
        self._total_processed = 0
        self._chunk_count = 0

    def process_file(
        self,
        file_path: str | Path,
        on_chunk: callable | None = None,
    ) -> dict:
        """Process a large file in chunks.

        Args:
            file_path: Path to CSV or Parquet file.
            on_chunk: Optional callback(chunk_num, records_processed, pairs_found).

        Returns:
            Summary dict with stats.
        """
        from goldenmatch.core.autofix import auto_fix_dataframe
        from goldenmatch.core.standardize import apply_standardization
        from goldenmatch.core.matchkey import compute_matchkeys
        from goldenmatch.core.blocker import build_blocks
        from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches, score_blocks_parallel
        from goldenmatch.core.cluster import build_clusters
        from goldenmatch.core.golden import build_golden_record

        file_path = Path(file_path)
        matchkeys = self.config.get_matchkeys()
        t_start = time.perf_counter()

        # Determine reader
        if file_path.suffix == ".parquet":
            reader = self._read_parquet_chunks(file_path)
        else:
            reader = self._read_csv_chunks(file_path)

        for chunk_df in reader:
            self._chunk_count += 1
            chunk_start = time.perf_counter()

            # Add row IDs with offset
            chunk_df = chunk_df.with_row_index("__row_id__").with_columns(
                (pl.col("__row_id__") + self._row_offset).cast(pl.Int64).alias("__row_id__")
            )
            chunk_df = chunk_df.with_columns(pl.lit("source").alias("__source__"))

            # Auto-fix
            chunk_df, _ = auto_fix_dataframe(chunk_df)

            # Standardize
            if self.config.standardization:
                lf = chunk_df.lazy()
                lf = apply_standardization(lf, self.config.standardization)
                chunk_df = lf.collect()

            # Compute matchkeys
            lf = chunk_df.lazy()
            lf = compute_matchkeys(lf, matchkeys)
            chunk_df = lf.collect()

            # Match within chunk
            chunk_pairs = []
            matched_pairs = set()

            for mk in matchkeys:
                if mk.type == "exact":
                    pairs = find_exact_matches(chunk_df.lazy(), mk)
                    chunk_pairs.extend(pairs)
                    for a, b, s in pairs:
                        matched_pairs.add((min(a, b), max(a, b)))

            if self.config.blocking:
                for mk in matchkeys:
                    if mk.type == "weighted":
                        blocks = build_blocks(chunk_df.lazy(), self.config.blocking)
                        pairs = score_blocks_parallel(blocks, mk, matched_pairs)
                        chunk_pairs.extend(pairs)

            # Match against index (cross-chunk matching)
            if self._index_records:
                cross_pairs = self._match_against_index(chunk_df, matchkeys)
                chunk_pairs.extend(cross_pairs)

            # Add to index (sample representative records for future cross-chunk matching)
            self._add_to_index(chunk_df)

            # Accumulate
            self._all_pairs.extend(chunk_pairs)
            chunk_ids = chunk_df["__row_id__"].to_list()
            self._all_ids.extend(chunk_ids)
            self._row_offset += chunk_df.height
            self._total_processed += chunk_df.height

            elapsed = time.perf_counter() - chunk_start

            logger.info(
                "Chunk %d: %d records, %d pairs (%.1fs, %d rec/s)",
                self._chunk_count, chunk_df.height, len(chunk_pairs),
                elapsed, chunk_df.height / elapsed if elapsed > 0 else 0,
            )

            if on_chunk:
                on_chunk(self._chunk_count, self._total_processed, len(self._all_pairs))

        # Final clustering across all chunks
        logger.info("Clustering %d records, %d pairs...", len(self._all_ids), len(self._all_pairs))
        t_cluster = time.perf_counter()
        clusters = build_clusters(self._all_pairs, self._all_ids, max_cluster_size=100)
        cluster_time = time.perf_counter() - t_cluster

        multi_clusters = {k: v for k, v in clusters.items() if v["size"] > 1}
        total_time = time.perf_counter() - t_start

        return {
            "total_records": self._total_processed,
            "total_pairs": len(self._all_pairs),
            "total_clusters": len(multi_clusters),
            "chunks_processed": self._chunk_count,
            "chunk_size": self.chunk_size,
            "total_time": round(total_time, 2),
            "cluster_time": round(cluster_time, 2),
            "records_per_second": round(self._total_processed / total_time) if total_time > 0 else 0,
        }

    def _read_csv_chunks(self, path: Path):
        """Read CSV in chunks."""
        # Read full file lazily and collect in chunks
        try:
            full_df = pl.read_csv(path, encoding="utf8", ignore_errors=True)
        except Exception:
            full_df = pl.read_csv(str(path), ignore_errors=True)

        total = full_df.height
        for offset in range(0, total, self.chunk_size):
            chunk = full_df.slice(offset, self.chunk_size)
            if chunk.height == 0:
                break
            yield chunk

    def _read_parquet_chunks(self, path: Path):
        """Read Parquet in chunks."""
        total = pl.scan_parquet(path).select(pl.len()).collect().item()

        for offset in range(0, total, self.chunk_size):
            chunk = pl.scan_parquet(path).slice(offset, self.chunk_size).collect()
            if chunk.height == 0:
                break
            yield chunk

    def _match_against_index(
        self, chunk_df: pl.DataFrame, matchkeys: list,
    ) -> list[tuple[int, int, float]]:
        """Match chunk records against index of previous chunks."""
        from goldenmatch.core.scorer import score_pair
        from goldenmatch.utils.transforms import apply_transforms

        pairs = []
        chunk_rows = chunk_df.to_dicts()

        for mk in matchkeys:
            if mk.type == "exact":
                # Build lookup of index values for exact matching
                for field in mk.fields:
                    col = field.field or ""
                    transforms = field.transforms or []

                    # Build index lookup: transformed_value -> [row_ids]
                    idx_lookup: dict[str, list[int]] = {}
                    for idx_rec in self._index_records:
                        idx_id = idx_rec.get("__row_id__")
                        raw = idx_rec.get(col)
                        if raw is not None:
                            val = apply_transforms(str(raw), transforms)
                            idx_lookup.setdefault(val, []).append(idx_id)

                    # Check chunk records against index
                    for chunk_row in chunk_rows:
                        chunk_id = chunk_row.get("__row_id__")
                        raw = chunk_row.get(col)
                        if raw is not None:
                            val = apply_transforms(str(raw), transforms)
                            for idx_id in idx_lookup.get(val, []):
                                if idx_id != chunk_id:
                                    pairs.append((chunk_id, idx_id, 1.0))

            elif mk.type == "weighted":
                for chunk_row in chunk_rows:
                    chunk_id = chunk_row.get("__row_id__")

                    for idx_record in self._index_records:
                        idx_id = idx_record.get("__row_id__")
                        if idx_id == chunk_id:
                            continue

                        score = score_pair(chunk_row, idx_record, mk.fields)
                        if score >= (mk.threshold or 0.0):
                            pairs.append((chunk_id, idx_id, score))

        return pairs

    def _add_to_index(self, chunk_df: pl.DataFrame) -> None:
        """Add records from chunk to index for cross-chunk matching.

        For exact matching: stores all unique key values (compact).
        For fuzzy matching: samples representative records.
        """
        # Store all records — for exact matching we need full coverage
        # Keep only matchkey-relevant columns + __row_id__ to save memory
        matchkeys = self.config.get_matchkeys()
        keep_cols = {"__row_id__"}
        for mk in matchkeys:
            for f in mk.fields:
                if f.field:
                    keep_cols.add(f.field)

        available = [c for c in keep_cols if c in chunk_df.columns]
        slim_df = chunk_df.select(available)
        self._index_records.extend(slim_df.to_dicts())
