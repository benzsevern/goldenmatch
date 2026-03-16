"""CLI match command for GoldenMatch."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.config.loader import load_config
from goldenmatch.cli.dedupe import _parse_file_source, _resolve_column_maps

console = Console()


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
    output_matched: bool = typer.Option(False, "--output-matched", help="Output matched records"),
    output_unmatched: bool = typer.Option(False, "--output-unmatched", help="Output unmatched records"),
    output_scores: bool = typer.Option(False, "--output-scores", help="Output score details"),
    output_all: bool = typer.Option(False, "--output-all", help="Output all result types"),
    output_report: bool = typer.Option(False, "--output-report", help="Generate summary report"),
    match_mode: str = typer.Option("best", "--match-mode", help="Match mode: best or all"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Output directory"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (csv, parquet)"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name for output files"),
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
