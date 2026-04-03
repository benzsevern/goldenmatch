"""CLI compare-clusters command for GoldenMatch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.core.compare_clusters import compare_clusters

console = Console()
err_console = Console(stderr=True)


def _load_clusters(path: Path) -> dict[int, dict]:
    """Load clusters from a JSON file.

    Handles pair_scores keys stored as "a,b" strings by converting
    back to (int, int) tuples.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Accept either {"clusters": {...}} or bare cluster dict
    if "clusters" in raw and isinstance(raw["clusters"], dict):
        raw = raw["clusters"]

    clusters: dict[int, dict] = {}
    for cid_str, info in raw.items():
        cid = int(cid_str)
        # Reconstruct pair_scores with tuple keys
        pair_scores = {}
        for k, v in info.get("pair_scores", {}).items():
            if isinstance(k, str) and "," in k:
                parts = k.split(",")
                pair_scores[(int(parts[0]), int(parts[1]))] = v
            else:
                pair_scores[k] = v
        info["pair_scores"] = pair_scores
        clusters[cid] = info
    return clusters


def compare_clusters_cmd(
    file_a: str = typer.Argument(..., help="First cluster JSON file (baseline / ER1)"),
    file_b: str = typer.Argument(..., help="Second cluster JSON file (comparison / ER2)"),
    details: bool = typer.Option(False, "--details", "-d", help="Show per-cluster transformation details"),
    case_type: Optional[str] = typer.Option(
        None, "--case-type",
        help="Filter details by case: unchanged, merged, partitioned, overlapping",
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON"),
) -> None:
    """Compare two ER clustering outcomes using the CCMS framework.

    Classifies each cluster from file A into: unchanged, merged,
    partitioned, or overlapping relative to file B.
    """
    path_a = Path(file_a)
    path_b = Path(file_b)

    for p in (path_a, path_b):
        if not p.exists():
            err_console.print(f"[red]File not found: {p}[/red]")
            raise typer.Exit(1)

    try:
        clusters_a = _load_clusters(path_a)
        clusters_b = _load_clusters(path_b)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        err_console.print(f"[red]Error reading cluster files:[/red] {exc}")
        raise typer.Exit(1)

    try:
        result = compare_clusters(clusters_a, clusters_b)
    except ValueError as exc:
        err_console.print(f"[red]Comparison error:[/red] {exc}")
        raise typer.Exit(1)

    # Summary table
    table = Table(title="CCMS Cluster Comparison", show_header=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_column("% of ER1", justify="right", style="dim")

    s = result.summary()
    table.add_row("Unchanged (UC)", str(s["unchanged"]), f"{s['unchanged_pct']:.1%}")
    table.add_row("Merged (MC)", str(s["merged"]), f"{s['merged_pct']:.1%}")
    table.add_row("Partitioned (PC)", str(s["partitioned"]), f"{s['partitioned_pct']:.1%}")
    table.add_row("Overlapping (OC)", str(s["overlapping"]), f"{s['overlapping_pct']:.1%}")
    table.add_row("", "", "")
    table.add_row("Total References", str(s["rc"]), "")
    table.add_row("ER1 Clusters", str(s["cc1"]), "")
    table.add_row("ER2 Clusters", str(s["cc2"]), "")
    table.add_row("ER1 Singletons", str(s["sc1"]), "")
    table.add_row("ER2 Singletons", str(s["sc2"]), "")
    table.add_row("TWI", f"{s['twi']:.4f}", "")

    console.print(table)

    # Details
    if details:
        detail_table = Table(title="Cluster Details", show_header=True)
        detail_table.add_column("ER1 Cluster", style="bold", justify="right")
        detail_table.add_column("Case", style="cyan")
        detail_table.add_column("Members", style="dim")
        detail_table.add_column("ER2 Mapping")

        for case in result.cases:
            if case_type and case.case != case_type:
                continue
            er2_str = ", ".join(
                f"{cid}: {members}" for cid, members in case.er2_clusters.items()
            )
            detail_table.add_row(
                str(case.cluster_id),
                case.case,
                str(case.members),
                er2_str,
            )

        console.print(detail_table)

    # Output JSON
    if output:
        out = s.copy()
        out["cases"] = [
            {
                "cluster_id": c.cluster_id,
                "case": c.case,
                "members": c.members,
                "er2_clusters": {str(k): v for k, v in c.er2_clusters.items()},
            }
            for c in result.cases
        ]
        output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        console.print(f"\n[green]Results saved to {output}[/green]")
