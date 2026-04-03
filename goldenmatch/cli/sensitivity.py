"""CLI sensitivity command for GoldenMatch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.cli.dedupe import _parse_file_source, _resolve_column_maps
from goldenmatch.config.loader import load_config
from goldenmatch.core.sensitivity import SweepParam, run_sensitivity

console = Console()
err_console = Console(stderr=True)


def _parse_sweep(sweep_str: str) -> SweepParam:
    """Parse a sweep string like 'threshold:0.70:0.95:0.05'."""
    parts = sweep_str.split(":")
    if len(parts) != 4:
        raise typer.BadParameter(
            f"Sweep format must be 'field:start:stop:step', got '{sweep_str}'"
        )
    field = parts[0]
    try:
        start = float(parts[1])
        stop = float(parts[2])
        step = float(parts[3])
    except ValueError:
        raise typer.BadParameter(
            f"Sweep start/stop/step must be numbers, got '{sweep_str}'"
        )
    if step <= 0:
        raise typer.BadParameter("Sweep step must be positive")
    if start > stop:
        raise typer.BadParameter("Sweep start must be <= stop")
    return SweepParam(field=field, start=start, stop=stop, step=step)


def sensitivity_cmd(
    files: list[str] = typer.Argument(..., help="Input files (path or path:source_name)"),
    config: Path = typer.Option(..., "--config", "-c", help="Config YAML path"),
    sweep: list[str] = typer.Option(..., "--sweep", "-s", help="Sweep spec: field:start:stop:step (repeatable)"),
    sample: Optional[int] = typer.Option(None, "--sample", help="Random sample size for speed"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
) -> None:
    """Analyze parameter sensitivity using CCMS cluster comparison.

    Sweeps each parameter independently, comparing clusters against a baseline.

    Example:
        goldenmatch sensitivity data.csv -c config.yaml --sweep threshold:0.70:0.95:0.05
    """
    cfg = load_config(str(config))

    parsed = [_parse_file_source(f) for f in files]
    file_specs = _resolve_column_maps(parsed, cfg)

    sweep_params = [_parse_sweep(s) for s in sweep]

    console.print("[bold]Running sensitivity analysis...[/bold]")
    console.print(f"  Sweeping {len(sweep_params)} parameter(s)")
    if sample:
        console.print(f"  Sample size: {sample}")
    console.print()

    try:
        results = run_sensitivity(
            file_specs=file_specs,
            config=cfg,
            sweep_params=sweep_params,
            sample_size=sample,
        )
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    # Display results per parameter
    for r in results:
        table = Table(
            title=f"Sensitivity: {r.param.field} (baseline={r.baseline_value})",
            show_header=True,
        )
        table.add_column("Value", style="bold", justify="right")
        table.add_column("UC", justify="right")
        table.add_column("MC", justify="right")
        table.add_column("PC", justify="right")
        table.add_column("OC", justify="right")
        table.add_column("CC2", justify="right")
        table.add_column("TWI", justify="right")
        table.add_column("UC%", justify="right", style="cyan")

        # Find best UC% for highlighting
        best_uc = max(
            (p.comparison.unchanged for p in r.points), default=0
        )

        for pt in r.points:
            c = pt.comparison
            uc_pct = c.unchanged / (c.cc1 or 1)
            is_best = c.unchanged == best_uc
            style = "bold green" if is_best else ""
            table.add_row(
                f"{pt.param_value:.4f}",
                str(c.unchanged),
                str(c.merged),
                str(c.partitioned),
                str(c.overlapping),
                str(c.cc2),
                f"{c.twi:.4f}",
                f"{uc_pct:.1%}",
                style=style,
            )

        console.print(table)

        report = r.stability_report()
        console.print(
            f"  [bold]Stability plateau:[/bold] {r.param.field} = {report['best_value']} "
            f"({report['best_unchanged_pct']:.1%} unchanged)\n"
        )

    # Output JSON
    if output:
        out = []
        for r in results:
            report = r.stability_report()
            report["field"] = r.param.field
            report["baseline_value"] = r.baseline_value
            out.append(report)
        output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        console.print(f"[green]Results saved to {output}[/green]")
