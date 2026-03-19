"""Live stream mode — poll database for changes and match continuously.

Usage:
    goldenmatch watch --table customers --connection-string $DB --config config.yaml

Polls the database every N seconds for new records, runs incremental
matching, and logs results. Runs until Ctrl+C.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from goldenmatch.config.schemas import GoldenMatchConfig
from goldenmatch.db.connector import DatabaseConnector
from goldenmatch.db.metadata import ensure_metadata_tables, get_state
from goldenmatch.db.sync import run_sync

logger = logging.getLogger(__name__)


def watch(
    connector: DatabaseConnector,
    source_table: str,
    config: GoldenMatchConfig,
    poll_interval: int = 30,
    output_mode: str = "separate",
    incremental_column: str | None = None,
    on_sync: callable | None = None,
) -> None:
    """Poll database for changes and match continuously.

    Args:
        connector: Active database connection.
        source_table: Table to watch.
        config: Matching configuration.
        poll_interval: Seconds between polls.
        output_mode: "separate" or "in_place".
        incremental_column: Column for change detection.
        on_sync: Optional callback(results_dict) after each sync.
    """
    ensure_metadata_tables(connector)

    total_syncs = 0
    total_merged = 0
    total_new = 0
    start_time = datetime.now()

    _print_header(source_table, poll_interval, connector)

    # First run — may be full scan if no prior state
    state = get_state(connector, source_table)
    is_first = state is None
    if is_first:
        _log_event("First run — performing full table scan...")
    else:
        _log_event(f"Resuming from last sync at {state.get('last_processed_at', 'unknown')}")

    try:
        while True:
            sync_start = time.perf_counter()

            try:
                results = run_sync(
                    connector=connector,
                    source_table=source_table,
                    config=config,
                    output_mode=output_mode,
                    full_rescan=is_first,
                    incremental_column=incremental_column,
                )
                is_first = False

                elapsed = time.perf_counter() - sync_start
                new_records = results.get("new_records", 0)
                merged = results.get("merged", results.get("matches", 0))
                new_entities = results.get("new_entities", 0)

                total_syncs += 1
                total_merged += merged
                total_new += new_entities

                if new_records > 0:
                    _log_event(
                        f"Sync #{total_syncs}: {new_records} records processed — "
                        f"{merged} merged, {new_entities} new [{elapsed:.1f}s]"
                    )
                else:
                    _log_event(f"Sync #{total_syncs}: no new records [{elapsed:.1f}s]", dim=True)

                if on_sync:
                    on_sync(results)

            except Exception as e:
                _log_event(f"Sync error: {e}", error=True)

            # Wait for next poll
            _log_waiting(poll_interval)
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        elapsed_total = (datetime.now() - start_time).total_seconds()
        _print_summary(total_syncs, total_merged, total_new, elapsed_total)


def _print_header(table: str, interval: int, connector: DatabaseConnector) -> None:
    """Print watch mode header."""
    try:
        row_count = connector.get_row_count(table)
    except Exception:
        row_count = 0

    print()
    print(f"  \033[33m⚡ GoldenMatch Watch Mode\033[0m")
    print(f"  {'─' * 40}")
    print(f"  Table:    \033[1m{table}\033[0m ({row_count:,} rows)")
    print(f"  Interval: every {interval}s")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'─' * 40}")
    print(f"  Press Ctrl+C to stop.")
    print()


def _log_event(msg: str, dim: bool = False, error: bool = False) -> None:
    """Log a timestamped event."""
    ts = datetime.now().strftime("%H:%M:%S")
    if error:
        print(f"  \033[31m[{ts}]\033[0m {msg}")
    elif dim:
        print(f"  \033[90m[{ts}] {msg}\033[0m")
    else:
        print(f"  \033[33m[{ts}]\033[0m {msg}")


def _log_waiting(interval: int) -> None:
    """Log waiting message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  \033[90m[{ts}] Waiting {interval}s...\033[0m", end="\r")


def _print_summary(total_syncs: int, total_merged: int, total_new: int, elapsed: float) -> None:
    """Print summary on shutdown."""
    print(f"\n")
    print(f"  \033[33m⚡ Watch Mode Summary\033[0m")
    print(f"  {'─' * 40}")
    print(f"  Syncs:    {total_syncs}")
    print(f"  Merged:   {total_merged}")
    print(f"  New:      {total_new}")
    print(f"  Duration: {elapsed:.0f}s")
    print(f"  {'─' * 40}")
    print()
