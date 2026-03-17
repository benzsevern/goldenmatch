"""CLI match command for GoldenMatch."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.config.loader import load_config
from goldenmatch.cli.dedupe import _parse_file_source, _resolve_column_maps

console = Console()
err_console = Console(stderr=True)


def match_cmd(
    target: str = typer.Argument(
        ..., help="Target file as path or path:source_name"
    ),
    against: list[str] = typer.Option(
        ..., "--against", "-a", help="Reference files as path or path:source_name"
    ),
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file"
    ),
    preview: bool = typer.Option(False, "--preview", help="Preview results without writing files"),
    preview_size: int = typer.Option(10000, "--preview-size", help="Number of records for preview sample"),
    preview_random: bool = typer.Option(False, "--preview-random", help="Random sample instead of first N"),
    output_matched: bool = typer.Option(False, "--output-matched", help="Output matched records"),
    output_unmatched: bool = typer.Option(False, "--output-unmatched", help="Output unmatched records"),
    output_scores: bool = typer.Option(False, "--output-scores", help="Output score details"),
    output_all: bool = typer.Option(False, "--output-all", help="Output all result types"),
    output_report: bool = typer.Option(False, "--output-report", help="Generate summary report"),
    match_mode: str = typer.Option("best", "--match-mode", help="Match mode: best or all"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Output directory"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (csv, parquet)"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name for output files"),
    auto_fix: bool = typer.Option(False, "--auto-fix", help="Run auto-fix before matching"),
    auto_block: bool = typer.Option(False, "--auto-block", help="Auto-suggest blocking keys"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Run list-matching: match a target file against reference files."""
    # Parse target and reference files
    target_parsed = _parse_file_source(target)
    refs_parsed = [_parse_file_source(f) for f in against]

    # Load config
    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        if not quiet:
            console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)

    # ── Preview mode ──
    if preview:
        from goldenmatch.tui.engine import MatchEngine
        from goldenmatch.core.preview import (
            format_preview_stats,
            format_preview_clusters,
            format_preview_golden,
            format_score_histogram,
        )

        # Load all files into engine for a dedupe-style preview
        all_file_paths = [target_parsed[0]] + [fp for fp, _name in refs_parsed]
        engine = MatchEngine(all_file_paths)

        err_console.print(
            "[cyan]Preview shows cross-file matching results.[/cyan]",
        )

        if preview_size < engine.row_count:
            err_console.print(
                f"[yellow]Previewing {preview_size} of {engine.row_count} records.[/yellow]",
            )

        result = engine.run_sample(cfg, sample_size=preview_size)

        err_console.print(format_preview_stats(result.stats))
        err_console.print(
            format_preview_clusters(result.clusters, engine.data, max_clusters=10),
        )
        err_console.print(format_preview_golden(result.golden, max_records=10))
        err_console.print(
            format_score_histogram([s for _, _, s in result.scored_pairs]),
        )

        run_full = typer.confirm("Run full job now?", default=False)
        if not run_full:
            raise typer.Exit(code=0)
        # Fall through to normal match

    # Apply CLI overrides
    if output_dir:
        cfg.output.directory = output_dir
    if format:
        cfg.output.format = format
    if run_name:
        cfg.output.run_name = run_name

    if output_all:
        output_matched = True
        output_unmatched = True
        output_scores = True
        output_report = True

    # Enable auto-fix from CLI flag
    if auto_fix:
        from goldenmatch.config.schemas import ValidationConfig
        if cfg.validation is None:
            cfg.validation = ValidationConfig(auto_fix=True)
        else:
            cfg.validation.auto_fix = True

    # Enable auto-block from CLI flag
    if auto_block:
        from goldenmatch.config.schemas import BlockingConfig
        if cfg.blocking is None:
            cfg.blocking = BlockingConfig(keys=[], auto_suggest=True)
        else:
            cfg.blocking.auto_suggest = True

    # Resolve column maps
    all_parsed = [target_parsed] + refs_parsed
    all_resolved = _resolve_column_maps(all_parsed, cfg)
    target_file = all_resolved[0]
    reference_files = all_resolved[1:]

    # Run match
    try:
        from goldenmatch.core.pipeline import run_match

        results = run_match(
            target_file=target_file,
            reference_files=reference_files,
            config=cfg,
            output_matched=output_matched,
            output_unmatched=output_unmatched,
            output_scores=output_scores,
            output_report=output_report,
            match_mode=match_mode,
        )
    except Exception as exc:
        if not quiet:
            console.print(f"[red]Runtime error:[/red] {exc}")
        raise typer.Exit(code=3)

    # Print report
    if not quiet and results.get("report"):
        report = results["report"]
        table = Table(title="Match Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key, val in report.items():
            table.add_row(str(key), str(val))
        console.print(table)
    elif not quiet:
        matched = results.get("matched")
        count = len(matched) if matched is not None else 0
        console.print(f"Match complete. {count} matches found.")
