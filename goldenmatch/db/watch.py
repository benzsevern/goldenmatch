"""Live stream mode — poll database for changes and match continuously.

Usage:
    goldenmatch watch --table customers --connection-string $DB --config config.yaml

    # Daemon mode with health endpoint:
    goldenmatch watch --table customers --connection-string $DB --daemon --health-port 9090

Polls the database every N seconds for new records, runs incremental
matching, and logs results. Runs until Ctrl+C or SIGTERM.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

from goldenmatch.config.schemas import GoldenMatchConfig
from goldenmatch.db.connector import DatabaseConnector
from goldenmatch.db.metadata import ensure_metadata_tables, get_state
from goldenmatch.db.sync import run_sync

logger = logging.getLogger(__name__)

# Daemon state
_daemon_stats = {
    "status": "starting",
    "syncs": 0,
    "merged": 0,
    "new_entities": 0,
    "last_sync_at": None,
    "last_sync_duration": None,
    "errors": 0,
    "started_at": None,
    "pid": os.getpid(),
}


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


# ── Daemon mode ──────────────────────────────────────────────────────────


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal health check endpoint for daemon mode."""

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_daemon_stats, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass  # suppress access logs


def _write_pid(pid_path: Path) -> None:
    """Write PID file for daemon management."""
    pid_path.write_text(str(os.getpid()))
    logger.info("PID file written: %s", pid_path)


def _remove_pid(pid_path: Path) -> None:
    """Remove PID file on shutdown."""
    if pid_path.exists():
        pid_path.unlink()


def watch_daemon(
    connector: DatabaseConnector,
    source_table: str,
    config: GoldenMatchConfig,
    poll_interval: int = 30,
    output_mode: str = "separate",
    incremental_column: str | None = None,
    health_port: int = 9090,
    pid_file: str | None = None,
) -> None:
    """Run watch mode as a daemon with health endpoint and PID file.

    Like watch() but adds:
    - HTTP health endpoint at /health (for monitoring)
    - PID file (for process management)
    - Graceful SIGTERM handling
    - Structured status reporting
    """
    global _daemon_stats
    _daemon_stats["started_at"] = datetime.now().isoformat()
    _daemon_stats["status"] = "initializing"

    # PID file
    pid_path = Path(pid_file) if pid_file else Path(f".goldenmatch_watch_{source_table}.pid")
    _write_pid(pid_path)

    # Health server in background thread
    health_server = HTTPServer(("0.0.0.0", health_port), _HealthHandler)
    health_thread = Thread(target=health_server.serve_forever, daemon=True)
    health_thread.start()
    logger.info("Health endpoint running at http://0.0.0.0:%d/health", health_port)

    # Graceful shutdown
    _running = True

    def _handle_signal(signum, frame):
        nonlocal _running
        _running = False
        logger.info("Received signal %d, shutting down...", signum)

    signal.signal(signal.SIGTERM, _handle_signal)

    ensure_metadata_tables(connector)
    state = get_state(connector, source_table)
    is_first = state is None

    _daemon_stats["status"] = "running"
    _print_header(source_table, poll_interval, connector)
    print(f"  Health:   http://0.0.0.0:{health_port}/health")
    print(f"  PID:      {os.getpid()} ({pid_path})")
    print()

    try:
        while _running:
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

                merged = results.get("merged", results.get("matches", 0))
                new_entities = results.get("new_entities", 0)

                _daemon_stats["syncs"] += 1
                _daemon_stats["merged"] += merged
                _daemon_stats["new_entities"] += new_entities
                _daemon_stats["last_sync_at"] = datetime.now().isoformat()
                _daemon_stats["last_sync_duration"] = round(elapsed, 2)

                new_records = results.get("new_records", 0)
                if new_records > 0:
                    _log_event(
                        f"Sync #{_daemon_stats['syncs']}: {new_records} records - "
                        f"{merged} merged, {new_entities} new [{elapsed:.1f}s]"
                    )

            except Exception as e:
                _daemon_stats["errors"] += 1
                _daemon_stats["status"] = "error"
                _log_event(f"Sync error: {e}", error=True)
                _daemon_stats["status"] = "running"

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        pass
    finally:
        _daemon_stats["status"] = "stopped"
        health_server.shutdown()
        _remove_pid(pid_path)
        elapsed_total = (datetime.now() - datetime.fromisoformat(_daemon_stats["started_at"])).total_seconds()
        _print_summary(
            _daemon_stats["syncs"], _daemon_stats["merged"],
            _daemon_stats["new_entities"], elapsed_total,
        )
