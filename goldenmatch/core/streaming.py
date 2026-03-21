"""Streaming / CDC mode -- continuous record matching.

Watches a data source for new records and incrementally matches them
against the existing dataset. Supports immediate (per-record) and
micro-batch (every N seconds) processing modes.

Usage in config:
    streaming:
      enabled: true
      mode: micro_batch
      batch_interval_sec: 30
      source:
        connector: snowflake
        poll_interval_sec: 60
        query: "SELECT * FROM customers WHERE updated_at > :last_seen"
      write_back:
        connector: snowflake
        table: customer_golden
        mode: upsert
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig, MatchkeyConfig
from goldenmatch.core.cluster import add_to_cluster
from goldenmatch.core.match_one import match_one
from goldenmatch.core.scorer import score_pair

logger = logging.getLogger(__name__)


@dataclass
class StreamStats:
    """Running statistics for the stream processor."""

    records_processed: int = 0
    records_matched: int = 0
    records_new_entity: int = 0
    total_match_time_ms: float = 0.0
    llm_cost_usd: float = 0.0
    started_at: str = ""
    last_record_at: str | None = None
    errors: int = 0

    @property
    def records_per_sec(self) -> float:
        if self.total_match_time_ms == 0:
            return 0.0
        return self.records_processed / (self.total_match_time_ms / 1000)

    @property
    def match_rate(self) -> float:
        if self.records_processed == 0:
            return 0.0
        return self.records_matched / self.records_processed

    def summary(self) -> dict:
        return {
            "records_processed": self.records_processed,
            "records_matched": self.records_matched,
            "records_new_entity": self.records_new_entity,
            "match_rate": round(self.match_rate, 4),
            "records_per_sec": round(self.records_per_sec, 2),
            "llm_cost_usd": round(self.llm_cost_usd, 4),
            "errors": self.errors,
            "started_at": self.started_at,
            "last_record_at": self.last_record_at,
        }


class StreamProcessor:
    """Processes new records against an existing dataset.

    Supports two modes:
    - immediate: each record is matched as it arrives
    - micro_batch: records are collected and matched in batches
    """

    def __init__(
        self,
        df: pl.DataFrame,
        config: GoldenMatchConfig,
        clusters: dict[int, dict] | None = None,
    ) -> None:
        self._df = df
        self._config = config
        self._clusters = clusters or {}
        self._stats = StreamStats(started_at=datetime.now().isoformat())
        self._matchkeys = config.get_matchkeys()
        self._stop_event = Event()

    @property
    def stats(self) -> StreamStats:
        return self._stats

    @property
    def clusters(self) -> dict[int, dict]:
        return self._clusters

    @property
    def data(self) -> pl.DataFrame:
        return self._df

    def process_record(self, record: dict) -> list[tuple[int, float]]:
        """Match a single record against the dataset and update clusters.

        Args:
            record: Dict of field->value for the new record.

        Returns:
            List of (matched_row_id, score) tuples.
        """
        t0 = time.perf_counter()

        # Assign a new row_id
        max_id = self._df["__row_id__"].max() if self._df.height > 0 else 0
        new_id = (max_id or 0) + 1
        record["__row_id__"] = new_id

        all_matches = []
        for mk in self._matchkeys:
            if mk.type in ("weighted", "probabilistic") and mk.threshold is not None:
                matches = match_one(record, self._df, mk)
                all_matches.extend(matches)

        # Deduplicate by row_id, keeping highest score
        best: dict[int, float] = {}
        for rid, score in all_matches:
            if rid not in best or score > best[rid]:
                best[rid] = score
        matches = [(rid, score) for rid, score in sorted(best.items(), key=lambda x: -x[1])]

        # Update clusters
        self._clusters = add_to_cluster(new_id, matches, self._clusters)

        # Add record to dataset
        new_row = pl.DataFrame([record])
        # Ensure column alignment
        for col in self._df.columns:
            if col not in new_row.columns:
                new_row = new_row.with_columns(pl.lit(None).alias(col))
        new_row = new_row.select(self._df.columns)
        self._df = pl.concat([self._df, new_row])

        # Update stats
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._stats.records_processed += 1
        self._stats.total_match_time_ms += elapsed_ms
        self._stats.last_record_at = datetime.now().isoformat()

        if matches:
            self._stats.records_matched += 1
        else:
            self._stats.records_new_entity += 1

        return matches

    def process_batch(self, records: list[dict]) -> list[list[tuple[int, float]]]:
        """Process a batch of records. Returns matches per record."""
        results = []
        for record in records:
            try:
                matches = self.process_record(record)
                results.append(matches)
            except Exception as e:
                logger.error("Error processing record: %s", e)
                self._stats.errors += 1
                results.append([])
        return results

    def stop(self) -> None:
        """Signal the processor to stop."""
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()


def run_stream(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    source_fn,
    poll_interval: int = 30,
    batch_mode: bool = True,
    on_match=None,
    on_batch_complete=None,
    clusters: dict[int, dict] | None = None,
) -> StreamProcessor:
    """Run the streaming processor in a polling loop.

    Args:
        df: Initial dataset.
        config: GoldenMatch config.
        source_fn: Callable that returns new records as list[dict].
                   Called every poll_interval seconds.
        poll_interval: Seconds between polls.
        batch_mode: If True, process records in micro-batches.
        on_match: Callback(record, matches) called when a record matches.
        on_batch_complete: Callback(stats) called after each batch.
        clusters: Initial clusters (optional).

    Returns:
        StreamProcessor instance (call .stop() to halt).
    """
    processor = StreamProcessor(df, config, clusters)

    # Handle SIGTERM/SIGINT
    def _signal_handler(signum, frame):
        logger.info("Received signal %d, stopping stream...", signum)
        processor.stop()

    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("Stream processor started (poll=%ds, batch=%s)", poll_interval, batch_mode)

    while not processor.stopped:
        try:
            new_records = source_fn()
            if not new_records:
                time.sleep(poll_interval)
                continue

            logger.info("Stream: received %d new records", len(new_records))

            if batch_mode:
                results = processor.process_batch(new_records)
                for record, matches in zip(new_records, results):
                    if on_match and matches:
                        on_match(record, matches)
            else:
                for record in new_records:
                    matches = processor.process_record(record)
                    if on_match and matches:
                        on_match(record, matches)

            if on_batch_complete:
                on_batch_complete(processor.stats)

        except Exception as e:
            logger.error("Stream error: %s", e)
            processor._stats.errors += 1

        time.sleep(poll_interval)

    logger.info("Stream processor stopped. %s", processor.stats.summary())
    return processor
