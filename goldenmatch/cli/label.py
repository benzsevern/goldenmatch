"""CLI label command -- build ground truth by labeling pairs interactively."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.cli.dedupe import _parse_file_source, _resolve_column_maps
from goldenmatch.config.loader import load_config

console = Console()


def label_cmd(
    files: list[str] = typer.Argument(..., help="Input files (path or path:source_name)"),
    config: Path = typer.Option(..., "--config", "-c", help="Config YAML path"),
    output: Path = typer.Option("ground_truth.csv", "--output", "-o", help="Output ground truth CSV"),
    n: int = typer.Option(50, "--n", "-n", help="Number of pairs to label"),
    strategy: str = typer.Option("borderline", "--strategy", help="Pair selection: borderline, random, or hardest"),
    append: bool = typer.Option(False, "--append", "-a", help="Append to existing ground truth file"),
) -> None:
    """Build ground truth by labeling record pairs interactively.

    Shows pairs one at a time. Type y (match), n (no match), or s (skip).
    Saves labeled pairs to a CSV for use with 'goldenmatch evaluate'.
    """
    import polars as pl
    from goldenmatch.core.pipeline import run_dedupe

    cfg = load_config(str(config))
    parsed = [_parse_file_source(f) for f in files]
    file_specs = _resolve_column_maps(parsed, cfg)

    console.print("[bold]Running pipeline to generate candidate pairs...[/bold]")
    result = run_dedupe(file_specs, cfg)
    clusters = result["clusters"]

    # Extract all scored pairs from clusters
    all_pairs = []
    for cid, cinfo in clusters.items():
        for (a, b), score in cinfo.get("pair_scores", {}).items():
            all_pairs.append((a, b, score))

    if not all_pairs:
        console.print("[yellow]No pairs found. Check your config.[/yellow]")
        raise typer.Exit(1)

    # Select pairs based on strategy
    if strategy == "borderline":
        # Sort by distance from 0.85 (most ambiguous first)
        all_pairs.sort(key=lambda p: abs(p[2] - 0.85))
    elif strategy == "hardest":
        # Lowest scores first (hardest to decide)
        all_pairs.sort(key=lambda p: p[2])
    else:
        # Random
        import random
        random.shuffle(all_pairs)

    pairs_to_label = all_pairs[:n * 2]  # extra buffer for skips

    # Build row lookup
    df = result.get("_df")
    if df is None:
        # Reconstruct from files
        from goldenmatch.core.ingest import load_file
        from goldenmatch.core.autofix import auto_fix_dataframe
        frames = []
        for spec in file_specs:
            path = spec[0] if isinstance(spec, tuple) else spec
            lf = load_file(path)
            frames.append(lf.collect())
        combined = pl.concat(frames, how="diagonal")
        combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
        combined, _ = auto_fix_dataframe(combined)
    else:
        combined = df

    row_lookup = {r["__row_id__"]: r for r in combined.to_dicts()}
    display_cols = [c for c in combined.columns if not c.startswith("__")][:6]

    # Load existing labels if appending
    existing = set()
    if append and output.exists():
        existing_df = pl.read_csv(output)
        for r in existing_df.to_dicts():
            existing.add((int(r["id_a"]), int(r["id_b"])))
        console.print(f"[dim]Loaded {len(existing)} existing labels from {output}[/dim]")

    # Interactive labeling loop
    labels = []
    labeled = 0
    skipped = 0

    console.print(f"\n[bold]Label {n} pairs. Type: y=match, n=no match, s=skip, q=quit[/bold]\n")

    for a, b, score in pairs_to_label:
        if labeled >= n:
            break
        if (a, b) in existing or (b, a) in existing:
            continue

        row_a = row_lookup.get(a, {})
        row_b = row_lookup.get(b, {})

        # Display pair
        table = Table(title=f"Pair {labeled + 1}/{n} (score: {score:.3f})", show_header=True)
        table.add_column("Field", style="bold")
        table.add_column("Record A", style="cyan")
        table.add_column("Record B", style="green")
        for col in display_cols:
            val_a = str(row_a.get(col, ""))[:60]
            val_b = str(row_b.get(col, ""))[:60]
            style_a = style_b = ""
            if val_a.lower() == val_b.lower() and val_a:
                style_a = style_b = "bold"
            table.add_row(col, f"[{style_a}]{val_a}[/{style_a}]", f"[{style_b}]{val_b}[/{style_b}]")
        console.print(table)

        # Get input
        while True:
            response = console.input("[y/n/s/q] > ").strip().lower()
            if response in ("y", "n", "s", "q"):
                break
            console.print("[dim]Type y, n, s, or q[/dim]")

        if response == "q":
            break
        elif response == "s":
            skipped += 1
            continue
        else:
            labels.append({
                "id_a": a,
                "id_b": b,
                "label": 1 if response == "y" else 0,
                "score": round(score, 4),
            })
            labeled += 1
            console.print()

    # Save results
    if labels:
        labels_df = pl.DataFrame(labels)
        if append and output.exists():
            existing_df = pl.read_csv(output)
            labels_df = pl.concat([existing_df, labels_df], how="diagonal")
        labels_df.write_csv(output)

        match_count = sum(1 for l in labels if l["label"] == 1)
        console.print(f"\n[green]Saved {len(labels)} labels to {output}[/green]")
        console.print(f"  Matches: {match_count}, Non-matches: {len(labels) - match_count}, Skipped: {skipped}")
        console.print(f"\n[dim]Use with: goldenmatch evaluate {' '.join(files)} -c {config} --gt {output}[/dim]")
    else:
        console.print("\n[yellow]No labels saved.[/yellow]")
