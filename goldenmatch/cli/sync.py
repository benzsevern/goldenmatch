"""CLI command for database sync."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def sync_cmd(
    source_type: str = typer.Option("postgres", "--source-type", help="Database type"),
    connection_string: Optional[str] = typer.Option(None, "--connection-string", help="Database URL"),
    table: str = typer.Option(..., "--table", help="Source table name"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML file"),
    output_mode: str = typer.Option("separate", "--output-mode", help="separate or in_place"),
    full_rescan: bool = typer.Option(False, "--full-rescan", help="Force full rescan"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Match without writing results"),
    incremental_column: Optional[str] = typer.Option(None, "--incremental-column", help="Column for incremental detection"),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Records per chunk"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Sync database table — match new records against existing."""
    import logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif not quiet:
        logging.basicConfig(level=logging.INFO)

    # Load config
    if config:
        from goldenmatch.config.loader import load_config
        cfg = load_config(config)
    else:
        # Try project settings
        from goldenmatch.config.settings import load_project_settings
        project = load_project_settings()
        if project and "matchkeys" in project:
            from goldenmatch.config.schemas import GoldenMatchConfig
            cfg = GoldenMatchConfig(**project)
        else:
            # Auto-configure from table sample
            if not quiet:
                console.print("[yellow]No config — will auto-detect after connecting...[/yellow]")
            cfg = None

    # Create connector
    source_config = {"type": source_type, "connection": connection_string}
    from goldenmatch.db.connector import create_connector
    connector = create_connector(source_config)

    try:
        connector.connect()

        if not quiet:
            row_count = connector.get_row_count(table)
            console.print(f"Connected. Table [cyan]{table}[/cyan] has [green]{row_count:,}[/green] rows.")

        # Auto-configure if no config
        if cfg is None:
            from goldenmatch.core.autoconfig import auto_configure
            # Read sample for profiling
            sample = next(connector.read_table(table, chunk_size=1000))
            import tempfile
            from pathlib import Path
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
                sample.write_csv(f.name)
                cfg = auto_configure([(f.name, table)])
            if not quiet:
                console.print("[green]Auto-configured matching rules.[/green]")

        # Run sync
        from goldenmatch.db.sync import run_sync

        if not quiet:
            console.print(f"{'[yellow]DRY RUN — ' if dry_run else ''}Syncing...")

        results = run_sync(
            connector=connector,
            source_table=table,
            config=cfg,
            output_mode=output_mode,
            full_rescan=full_rescan,
            dry_run=dry_run,
            chunk_size=chunk_size,
            incremental_column=incremental_column,
        )

        # Print results
        if not quiet:
            result_table = Table(title="Sync Results")
            result_table.add_column("Metric", style="cyan")
            result_table.add_column("Value", style="green")

            result_table.add_row("Records processed", str(results.get("new_records", 0)))
            result_table.add_row("Matches found", str(results.get("matches", 0)))
            result_table.add_row("Clusters", str(results.get("clusters", results.get("merged", 0))))
            result_table.add_row("New entities", str(results.get("new_entities", 0)))
            result_table.add_row("Golden records", str(results.get("golden_records", 0)))
            result_table.add_row("Run ID", results.get("run_id", "N/A"))

            if dry_run:
                result_table.title = "Sync Results (DRY RUN)"

            console.print(result_table)

    except Exception as exc:
        if not quiet:
            console.print(f"[red]Sync error:[/red] {exc}")
        raise typer.Exit(code=1)
    finally:
        connector.close()
