"""CLI command for scheduled runs."""

from __future__ import annotations

import uuid
from typing import Optional

import typer


def schedule_cmd(
    files: list[str] = typer.Argument(..., help="Data files to process"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML file"),
    every: Optional[str] = typer.Option(None, "--every", help="Run interval (e.g. 1h, 30m, 6h, 1d)"),
    cron: Optional[str] = typer.Option(None, "--cron", help="Cron schedule (e.g. '0 6 * * *')"),
    output_dir: str = typer.Option(".", "--output-dir", help="Output directory"),
) -> None:
    """Run deduplication on a schedule."""
    from goldenmatch.core.scheduler import ScheduledJob, parse_interval, parse_cron

    if every:
        interval = parse_interval(every)
    elif cron:
        interval = parse_cron(cron)
    else:
        typer.echo("Error: specify --every or --cron", err=True)
        raise typer.Exit(code=1)

    job_id = f"gm-{uuid.uuid4().hex[:8]}"

    job = ScheduledJob(
        job_id=job_id,
        file_paths=files,
        config_path=config,
        interval_seconds=interval,
        output_dir=output_dir,
    )

    job.start()
