"""CLI dedupe command for GoldenMatch."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.config.loader import load_config

console = Console()
err_console = Console(stderr=True)


def _parse_file_source(raw: str) -> tuple[str, str]:
    """Parse 'file_path:source_name' handling Windows drive letters.

    Only split on the last colon if the first part is longer than 1 char
    (to avoid treating C: in C:\\path as a separator).
    """
    # Find last colon
    idx = raw.rfind(":")
    if idx <= 0:
        # No colon or colon at position 0 -> treat whole thing as path
        return (raw, Path(raw).stem)
    # Check if first part is a single char (Windows drive letter like C:)
    left = raw[:idx]
    if len(left) == 1 and left.isalpha():
        # This is a drive letter, not a separator
        return (raw, Path(raw).stem)
    return (left, raw[idx + 1:])


def _resolve_column_maps(parsed_files, cfg):
    """Match CLI files against config input.files to pick up column_map settings.

    Returns list of (path, source_name, column_map_or_None) tuples.
    """
    config_files = {}
    if cfg.input and hasattr(cfg.input, "files") and cfg.input.files:
        for fc in cfg.input.files:
            config_files[Path(fc.path).name] = fc
    elif cfg.input and hasattr(cfg.input, "file_a") and cfg.input.file_a:
        config_files[Path(cfg.input.file_a.path).name] = cfg.input.file_a
        if cfg.input.file_b:
            config_files[Path(cfg.input.file_b.path).name] = cfg.input.file_b

    result = []
    for file_path, source_name in parsed_files:
        fname = Path(file_path).name
        col_map = None
        if fname in config_files:
            fc = config_files[fname]
            col_map = fc.column_map
            if fc.source_name and source_name == Path(file_path).stem:
                source_name = fc.source_name
        result.append((file_path, source_name, col_map))
    return result


def dedupe_cmd(
    files: list[str] = typer.Argument(
        ..., help="Input files as path or path:source_name"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file (optional — auto-detects if omitted)"
    ),
    no_tui: bool = typer.Option(False, "--no-tui", help="Skip TUI, run with auto-config directly"),
    model: Optional[str] = typer.Option(None, "--model", help="Override embedding model selection"),
    preview: bool = typer.Option(False, "--preview", help="Preview results without writing files"),
    preview_size: int = typer.Option(10000, "--preview-size", help="Number of records for preview sample"),
    preview_random: bool = typer.Option(False, "--preview-random", help="Random sample instead of first N"),
    output_golden: bool = typer.Option(False, "--output-golden", help="Output golden records"),
    output_clusters: bool = typer.Option(False, "--output-clusters", help="Output cluster info"),
    output_dupes: bool = typer.Option(False, "--output-dupes", help="Output duplicate records"),
    output_unique: bool = typer.Option(False, "--output-unique", help="Output unique records"),
    output_all: bool = typer.Option(False, "--output-all", help="Output all result types"),
    output_report: bool = typer.Option(False, "--output-report", help="Generate summary report"),
    across_files_only: bool = typer.Option(False, "--across-files-only", help="Only match across different sources"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Output directory"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (csv, parquet)"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name for output files"),
    auto_fix: bool = typer.Option(False, "--auto-fix", help="Run auto-fix before matching"),
    auto_block: bool = typer.Option(False, "--auto-block", help="Auto-suggest blocking keys"),
    llm_boost: bool = typer.Option(False, "--llm-boost", help="Boost accuracy with LLM-labeled training data"),
    llm_retrain: bool = typer.Option(False, "--llm-retrain", help="Force re-labeling (ignore saved model)"),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider", help="LLM provider: auto, anthropic, or openai"),
    llm_max_labels: int = typer.Option(500, "--llm-max-labels", help="Max pairs to label with LLM"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """Run deduplication on one or more input files."""
    # Parse file:source pairs
    parsed_files = [_parse_file_source(f) for f in files]

    # Load config — from file, project settings, or auto-detect
    if config:
        try:
            cfg = load_config(config)
        except (FileNotFoundError, ValueError) as exc:
            if not quiet:
                console.print(f"[red]Config error:[/red] {exc}")
            raise typer.Exit(code=1)
    else:
        # Try project settings first
        from goldenmatch.config.settings import load_project_settings
        project = load_project_settings()
        if project and "matchkeys" in project:
            try:
                from goldenmatch.config.schemas import GoldenMatchConfig
                cfg = GoldenMatchConfig(**project)
                if not quiet:
                    console.print("[green]Loaded project settings from .goldenmatch.yaml[/green]")
            except Exception:
                project = None

        if not project or "matchkeys" not in (project or {}):
            # Auto-configure from input files
            try:
                from goldenmatch.core.autoconfig import auto_configure
                if not quiet:
                    console.print("[yellow]No config file — auto-detecting column types...[/yellow]")
                cfg = auto_configure(parsed_files)
                if not quiet:
                    from goldenmatch.core.autoconfig import profile_columns
                    console.print("[green]Auto-config complete. Launching TUI for review...[/green]")
            except Exception as exc:
                if not quiet:
                    console.print(f"[red]Auto-config error:[/red] {exc}")
                raise typer.Exit(code=1)

            # Override model if specified
            if model:
                for mk in cfg.get_matchkeys():
                    for f in mk.fields:
                        if f.scorer in ("embedding", "record_embedding"):
                            f.model = model

            # Launch TUI for review (unless --no-tui)
            if not no_tui and not preview:
                from goldenmatch.tui.app import GoldenMatchApp
                file_paths = [fp for fp, _name in parsed_files]
                tui_app = GoldenMatchApp(files=file_paths)
                tui_app.current_config = cfg
                tui_app.run()
                raise typer.Exit(code=0)

    # ── Preview mode ──
    if preview:
        from goldenmatch.tui.engine import MatchEngine
        from goldenmatch.core.preview import (
            format_preview_stats,
            format_preview_clusters,
            format_preview_golden,
            format_score_histogram,
        )

        file_paths = [fp for fp, _name in parsed_files]
        engine = MatchEngine(file_paths)

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
        # Fall through to normal dedupe

    # Apply CLI overrides
    if output_dir:
        cfg.output.directory = output_dir
    if format:
        cfg.output.format = format
    if run_name:
        cfg.output.run_name = run_name

    if output_all:
        output_golden = True
        output_clusters = True
        output_dupes = True
        output_unique = True
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

    # Enable LLM boost from CLI flag
    if llm_boost or llm_retrain:
        cfg.llm_boost = True

    # Resolve column maps from config input.files section
    file_specs = _resolve_column_maps(parsed_files, cfg)

    # Run dedupe
    try:
        from goldenmatch.core.pipeline import run_dedupe

        results = run_dedupe(
            files=file_specs,
            config=cfg,
            output_golden=output_golden,
            output_clusters=output_clusters,
            output_dupes=output_dupes,
            output_unique=output_unique,
            output_report=output_report,
            across_files_only=across_files_only,
            llm_retrain=llm_retrain,
            llm_provider=llm_provider,
            llm_max_labels=llm_max_labels,
        )
    except Exception as exc:
        if not quiet:
            console.print(f"[red]Runtime error:[/red] {exc}")
        raise typer.Exit(code=3)

    # Print report
    if not quiet and results.get("report"):
        report = results["report"]
        table = Table(title="Dedupe Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for key, val in report.items():
            if key == "cluster_size_distribution":
                val = dict(val)
            table.add_row(str(key), str(val))
        console.print(table)
    elif not quiet:
        clusters = results.get("clusters", {})
        console.print(f"Dedupe complete. {len(clusters)} clusters found.")
