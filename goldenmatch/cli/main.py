"""GoldenMatch CLI application."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax

from goldenmatch.cli.dedupe import dedupe_cmd
from goldenmatch.cli.match import match_cmd
from goldenmatch.prefs.store import PresetStore

app = typer.Typer(
    name="goldenmatch",
    help="GoldenMatch: deduplication and list-matching toolkit.",
    no_args_is_help=True,
)

app.command("dedupe", help="Run deduplication on one or more files.")(dedupe_cmd)
app.command("match", help="Match a target file against reference files.")(match_cmd)

# ── Config subcommands ─────────────────────────────────────────────────────

config_app = typer.Typer(
    name="config",
    help="Manage saved config presets.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")

console = Console()


def _get_store() -> PresetStore:
    return PresetStore()


@config_app.command("save")
def config_save(
    name: str = typer.Argument(..., help="Preset name"),
    config_path: str = typer.Argument(..., help="Path to config YAML file"),
) -> None:
    """Save a config file as a named preset."""
    store = _get_store()
    try:
        dest = store.save(name, config_path)
        console.print(f"[green]Preset '{name}' saved to {dest}[/green]")
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


@config_app.command("load")
def config_load(
    name: str = typer.Argument(..., help="Preset name to load"),
    dest: str = typer.Option("goldenmatch.yaml", "--dest", "-d", help="Destination path"),
) -> None:
    """Load a named preset to a local file."""
    store = _get_store()
    try:
        out = store.load(name, dest)
        console.print(f"[green]Preset '{name}' loaded to {out}[/green]")
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


@config_app.command("list")
def config_list() -> None:
    """List all saved presets."""
    store = _get_store()
    presets = store.list_presets()
    if not presets:
        console.print("[dim]No presets found.[/dim]")
        return
    for name in presets:
        console.print(f"  - {name}")


@config_app.command("delete")
def config_delete(
    name: str = typer.Argument(..., help="Preset name to delete"),
) -> None:
    """Delete a named preset."""
    store = _get_store()
    try:
        store.delete(name)
        console.print(f"[green]Preset '{name}' deleted.[/green]")
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


@config_app.command("show")
def config_show(
    name: str = typer.Argument(..., help="Preset name to display"),
) -> None:
    """Show the contents of a named preset."""
    store = _get_store()
    try:
        content = store.show(name)
        syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
