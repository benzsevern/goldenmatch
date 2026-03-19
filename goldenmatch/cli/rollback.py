"""CLI commands for rollback and run history."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def rollback_cmd(
    run_id: str = typer.Argument(..., help="Run ID to rollback"),
    output_dir: str = typer.Option(".", "--output-dir", help="Directory containing run log"),
) -> None:
    """Rollback a previous merge run by deleting its output files."""
    from goldenmatch.core.rollback import rollback_run

    result = rollback_run(run_id, output_dir)

    if "error" in result:
        console.print(f"[red]Error:[/] {result['error']}")
        if "available_runs" in result:
            console.print(f"Available runs: {', '.join(result['available_runs'][:5])}")
        raise typer.Exit(code=1)

    console.print(f"[#2ecc71]Rolled back run {run_id}[/]")
    if result["deleted"]:
        for f in result["deleted"]:
            console.print(f"  Deleted: {f}")
    if result["not_found"]:
        for f in result["not_found"]:
            console.print(f"  [dim]Not found: {f}[/dim]")


def runs_cmd(
    output_dir: str = typer.Option(".", "--output-dir", help="Directory containing run log"),
) -> None:
    """List previous runs (for rollback)."""
    from goldenmatch.core.rollback import list_runs

    runs = list_runs(output_dir)

    if not runs:
        console.print("[dim]No runs found. Run a dedupe first.[/dim]")
        return

    table = Table(title="Run History", border_style="#d4a017", header_style="bold #d4a017")
    table.add_column("Run ID", style="bold")
    table.add_column("Timestamp")
    table.add_column("Files")
    table.add_column("Status")

    for run in reversed(runs[-10:]):
        status = "[red]rolled back[/]" if run.get("rolled_back") else "[#2ecc71]active[/]"
        ts = run.get("timestamp", "")[:19]
        files = str(len(run.get("output_files", [])))
        table.add_row(run["run_id"][:12], ts, files, status)

    console.print(table)
