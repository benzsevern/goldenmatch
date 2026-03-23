"""CLI evaluate command for GoldenMatch."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.cli.dedupe import _parse_file_source, _resolve_column_maps
from goldenmatch.config.loader import load_config
from goldenmatch.core.evaluate import evaluate_clusters, load_ground_truth_csv

console = Console()
err_console = Console(stderr=True)


def evaluate_cmd(
    files: list[str] = typer.Argument(..., help="Input files (path or path:source_name)"),
    config: Path = typer.Option(..., "--config", "-c", help="Config YAML path"),
    ground_truth: Path = typer.Option(..., "--ground-truth", "--gt", help="Ground truth CSV path"),
    col_a: str = typer.Option("id_a", "--col-a", help="Ground truth column A"),
    col_b: str = typer.Option("id_b", "--col-b", help="Ground truth column B"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Override match threshold"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
    min_f1: Optional[float] = typer.Option(None, "--min-f1", help="Minimum F1 score (exit code 1 if below). For CI/CD quality gates."),
    min_precision: Optional[float] = typer.Option(None, "--min-precision", help="Minimum precision (exit code 1 if below)"),
    min_recall: Optional[float] = typer.Option(None, "--min-recall", help="Minimum recall (exit code 1 if below)"),
) -> None:
    """Evaluate matching quality against ground truth pairs.

    Use --min-f1, --min-precision, --min-recall as CI/CD quality gates:
    goldenmatch evaluate data.csv -c config.yaml --gt gt.csv --min-f1 0.90
    """
    from goldenmatch.core.pipeline import run_dedupe

    if not ground_truth.exists():
        err_console.print(f"[red]Ground truth file not found: {ground_truth}[/red]")
        raise typer.Exit(1)

    cfg = load_config(str(config))

    # Override threshold if specified
    if threshold is not None:
        for mk in cfg.get_matchkeys():
            if mk.threshold is not None:
                mk.threshold = threshold

    parsed = [_parse_file_source(f) for f in files]
    file_specs = _resolve_column_maps(parsed, cfg)

    gt_pairs = load_ground_truth_csv(str(ground_truth), col_a, col_b)

    console.print(f"[bold]Evaluating with {len(gt_pairs)} ground truth pairs...[/bold]\n")

    result = run_dedupe(file_specs, cfg)
    clusters = result["clusters"]

    eval_result = evaluate_clusters(clusters, gt_pairs)

    # Display results
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    summary = eval_result.summary()
    table.add_row("Precision", f"{summary['precision']:.1%}")
    table.add_row("Recall", f"{summary['recall']:.1%}")
    table.add_row("F1 Score", f"{summary['f1']:.1%}")
    table.add_row("True Positives", str(summary["tp"]))
    table.add_row("False Positives", str(summary["fp"]))
    table.add_row("False Negatives", str(summary["fn"]))
    table.add_row("Predicted Pairs", str(summary["predicted_pairs"]))
    table.add_row("Ground Truth Pairs", str(summary["ground_truth_pairs"]))

    console.print(table)

    if output:
        import json
        output.write_text(json.dumps(summary, indent=2))
        console.print(f"\n[green]Results saved to {output}[/green]")

    # CI/CD quality gates
    failed = False
    if min_f1 is not None and summary["f1"] < min_f1:
        err_console.print(f"[red]FAIL: F1 {summary['f1']:.1%} < minimum {min_f1:.1%}[/red]")
        failed = True
    if min_precision is not None and summary["precision"] < min_precision:
        err_console.print(f"[red]FAIL: Precision {summary['precision']:.1%} < minimum {min_precision:.1%}[/red]")
        failed = True
    if min_recall is not None and summary["recall"] < min_recall:
        err_console.print(f"[red]FAIL: Recall {summary['recall']:.1%} < minimum {min_recall:.1%}[/red]")
        failed = True
    if failed:
        raise typer.Exit(1)
