"""GoldenMatch CLI application."""

from __future__ import annotations

import platform
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from goldenmatch import __version__
from goldenmatch.cli.dedupe import dedupe_cmd
from goldenmatch.cli.match import match_cmd
from goldenmatch.prefs.store import PresetStore

LOGO = r"""[bold bright_yellow]
   ██████╗  ██████╗ ██╗     ██████╗ ███████╗███╗   ██╗
  ██╔════╝ ██╔═══██╗██║     ██╔══██╗██╔════╝████╗  ██║
  ██║  ███╗██║   ██║██║     ██║  ██║█████╗  ██╔██╗ ██║
  ██║   ██║██║   ██║██║     ██║  ██║██╔══╝  ██║╚██╗██║
  ╚██████╔╝╚██████╔╝███████╗██████╔╝███████╗██║ ╚████║
   ╚═════╝  ╚═════╝ ╚══════╝╚═════╝ ╚══════╝╚═╝  ╚═══╝
  ███╗   ███╗ █████╗ ████████╗ ██████╗██╗  ██╗
  ████╗ ████║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
  ██╔████╔██║███████║   ██║   ██║     ███████║
  ██║╚██╔╝██║██╔══██║   ██║   ██║     ██╔══██║
  ██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╗██║  ██║
  ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝[/bold bright_yellow]"""


def _print_banner() -> None:
    console = Console(stderr=True)
    console.print(LOGO)
    console.print()

    info = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    info.add_column(style="bold cyan", no_wrap=True)
    info.add_column(style="white")

    info.add_row("version", __version__)
    info.add_row("python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    info.add_row("platform", platform.system() + " " + platform.machine())

    store = PresetStore()
    presets = store.list_presets()
    info.add_row("presets", str(len(presets)) + " saved")

    import importlib
    for pkg in ["polars", "rapidfuzz", "typer", "pydantic"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "installed")
            info.add_row(pkg, ver)
        except ImportError:
            info.add_row(pkg, "[red]missing[/red]")

    console.print(info)
    console.print()
    console.print("[dim]Usage: goldenmatch <command> [options][/dim]")
    console.print("[dim]Commands: dedupe, match, config, init[/dim]")
    console.print("[dim]Run goldenmatch <command> --help for details[/dim]")
    console.print()


def _callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _print_banner()


app = typer.Typer(
    name="goldenmatch",
    help="GoldenMatch: deduplication and list-matching toolkit.",
    invoke_without_command=True,
    callback=_callback,
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


# ── Init command (wizard) ──────────────────────────────────────────────────


@app.command("init")
def init_cmd(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for generated config"),
) -> None:
    """Launch the interactive config wizard."""
    from goldenmatch.config.wizard import run_wizard

    run_wizard(output_path=output)


# ── Profile command ───────────────────────────────────────────────────────


@app.command("interactive")
def interactive_cmd(
    files: list[str] = typer.Argument(..., help="File(s) to load"),
) -> None:
    """Launch the interactive TUI for building configs with live feedback."""
    from goldenmatch.tui.app import GoldenMatchApp

    tui_app = GoldenMatchApp(files=files)
    tui_app.run()


@app.command("profile")
def profile_cmd(
    files: list[str] = typer.Argument(..., help="File(s) to profile"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed per-column info"),
    suggest_fixes: bool = typer.Option(False, "--suggest-fixes", help="Show what auto-fix would do (dry run)"),
) -> None:
    """Scan input files and generate a data quality report."""
    from pathlib import Path

    from goldenmatch.core.ingest import load_file
    from goldenmatch.core.profiler import format_profile_report, profile_dataframe

    for file_path in files:
        p = Path(file_path)
        console.print(f"\n[bold cyan]Profiling:[/bold cyan] {p.name}")
        try:
            lf = load_file(p)
            df = lf.collect()
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            continue

        profile = profile_dataframe(df)
        report = format_profile_report(profile, df=df if verbose else None)
        console.print(report)

        if suggest_fixes:
            from goldenmatch.core.autofix import auto_fix_dataframe

            _, fixes = auto_fix_dataframe(df, profile=profile)
            console.print("\n[bold yellow]Suggested Fixes (dry run):[/bold yellow]")
            any_fix = False
            for fix in fixes:
                if fix["rows_affected"] > 0:
                    any_fix = True
                    console.print(f"  [green]{fix['fix']}[/green]: {fix['detail']}")
            if not any_fix:
                console.print("  [dim]No fixes needed.[/dim]")
