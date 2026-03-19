"""Scheduled runs -- cron-like scheduling with email digest.

Usage:
  goldenmatch schedule --cron "0 6 * * *" --file customers.csv --config config.yaml
  goldenmatch schedule --every 1h --file data.csv
  goldenmatch schedule --list
  goldenmatch schedule --cancel job-id

Runs as a foreground process. For background execution,
use system cron, Task Scheduler, or a process manager.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class ScheduledJob:
    """A scheduled deduplication job."""

    def __init__(
        self,
        job_id: str,
        file_paths: list[str],
        config_path: str | None = None,
        interval_seconds: int = 3600,
        output_dir: str = ".",
        on_complete: Callable | None = None,
    ):
        self.job_id = job_id
        self.file_paths = file_paths
        self.config_path = config_path
        self.interval_seconds = interval_seconds
        self.output_dir = output_dir
        self.on_complete = on_complete

        self.run_count = 0
        self.last_run: datetime | None = None
        self.last_result: dict | None = None
        self._running = False

    def run_once(self) -> dict:
        """Execute one deduplication run."""
        from goldenmatch.tui.engine import MatchEngine

        engine = MatchEngine(self.file_paths)

        if self.config_path:
            from goldenmatch.config.loader import load_config
            config = load_config(self.config_path)
        else:
            from goldenmatch.core.autoconfig import auto_configure
            parsed = [(f, Path(f).stem) for f in self.file_paths]
            config = auto_configure(parsed)

        result = engine.run_full(config)

        summary = {
            "job_id": self.job_id,
            "run_number": self.run_count + 1,
            "timestamp": datetime.now().isoformat(),
            "records": result.stats.total_records,
            "clusters": result.stats.total_clusters,
            "match_rate": round(result.stats.match_rate, 2),
            "pairs": len(result.scored_pairs),
        }

        self.run_count += 1
        self.last_run = datetime.now()
        self.last_result = summary

        return summary

    def start(self) -> None:
        """Start the scheduled job loop."""
        self._running = True

        _print_schedule_header(self)

        try:
            while self._running:
                run_start = time.perf_counter()

                try:
                    result = self.run_once()
                    elapsed = time.perf_counter() - run_start

                    _print_run_result(result, elapsed)

                    if self.on_complete:
                        self.on_complete(result)

                except Exception as e:
                    _print_run_error(str(e))

                # Wait for next run
                next_run = datetime.now().timestamp() + self.interval_seconds
                _print_waiting(self.interval_seconds, next_run)

                while time.perf_counter() - run_start < self.interval_seconds and self._running:
                    time.sleep(1)

        except KeyboardInterrupt:
            self._running = False
            _print_schedule_summary(self)

    def stop(self) -> None:
        self._running = False


def parse_interval(spec: str) -> int:
    """Parse interval spec like '1h', '30m', '6h', '1d'."""
    spec = spec.strip().lower()

    if spec.endswith("s"):
        return int(spec[:-1])
    elif spec.endswith("m"):
        return int(spec[:-1]) * 60
    elif spec.endswith("h"):
        return int(spec[:-1]) * 3600
    elif spec.endswith("d"):
        return int(spec[:-1]) * 86400
    else:
        try:
            return int(spec)
        except ValueError:
            raise ValueError(f"Invalid interval: {spec}. Use format: 30m, 1h, 6h, 1d")


def parse_cron(cron_spec: str) -> int:
    """Parse a simple cron spec and return interval in seconds.

    Supports simplified cron: just calculates interval from the spec.
    For full cron support, use system cron.
    """
    parts = cron_spec.strip().split()
    if len(parts) != 5:
        raise ValueError("Cron spec must have 5 fields: minute hour day month weekday")

    minute, hour, day, month, weekday = parts

    if hour != "*" and minute != "*":
        return 86400  # daily
    elif minute != "*":
        return 3600  # hourly
    else:
        return 3600  # default hourly


# ── Display helpers ───────────────────────────────────────────────────────

def _print_schedule_header(job: ScheduledJob) -> None:
    hrs = job.interval_seconds / 3600
    interval_str = f"{hrs:.0f}h" if hrs >= 1 else f"{job.interval_seconds / 60:.0f}m"

    print()
    print(f"  \033[33mGoldenMatch Scheduled Job\033[0m")
    print(f"  {'=' * 40}")
    print(f"  Job ID:   \033[1m{job.job_id}\033[0m")
    print(f"  Files:    {', '.join(Path(f).name for f in job.file_paths)}")
    print(f"  Interval: every {interval_str}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'=' * 40}")
    print(f"  Press Ctrl+C to stop.")
    print()


def _print_run_result(result: dict, elapsed: float) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(
        f"  \033[33m[{ts}]\033[0m Run #{result['run_number']}: "
        f"{result['records']} records, {result['clusters']} clusters, "
        f"{result['match_rate']}% match rate [{elapsed:.1f}s]"
    )


def _print_run_error(error: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  \033[31m[{ts}]\033[0m Error: {error}")


def _print_waiting(interval: int, next_ts: float) -> None:
    next_str = datetime.fromtimestamp(next_ts).strftime("%H:%M:%S")
    print(f"  \033[90mNext run at {next_str}\033[0m", end="\r")


def _print_schedule_summary(job: ScheduledJob) -> None:
    print(f"\n")
    print(f"  \033[33mSchedule Summary\033[0m")
    print(f"  {'=' * 40}")
    print(f"  Total runs: {job.run_count}")
    if job.last_result:
        print(f"  Last result: {job.last_result['clusters']} clusters, {job.last_result['match_rate']}%")
    print(f"  {'=' * 40}")
    print()
