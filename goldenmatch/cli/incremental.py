"""CLI incremental command for GoldenMatch."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from goldenmatch.config.loader import load_config

console = Console()
err_console = Console(stderr=True)


def incremental_cmd(
    base_file: str = typer.Argument(..., help="Base dataset file path"),
    new_records: Path = typer.Option(..., "--new-records", "-n", help="New records CSV to match"),
    config: Path = typer.Option(..., "--config", "-c", help="Config YAML path"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Override threshold"),
) -> None:
    """Match new records against an existing base dataset incrementally."""
    import polars as pl
    from goldenmatch.core.ingest import load_file
    from goldenmatch.core.autofix import auto_fix_dataframe
    from goldenmatch.core.standardize import apply_standardization
    from goldenmatch.core.matchkey import compute_matchkeys
    from goldenmatch.core.match_one import match_one
    from goldenmatch.core.scorer import find_exact_matches

    if not new_records.exists():
        err_console.print(f"[red]New records file not found: {new_records}[/red]")
        raise typer.Exit(1)

    cfg = load_config(str(config))
    matchkeys = cfg.get_matchkeys()

    if threshold is not None:
        for mk in matchkeys:
            if mk.threshold is not None:
                mk.threshold = threshold

    # Load base dataset
    console.print("[bold]Loading base dataset...[/bold]")
    base_lf = load_file(base_file)
    base_df = base_lf.collect()
    base_df = base_df.with_row_index("__row_id__").with_columns(
        pl.col("__row_id__").cast(pl.Int64),
        pl.lit("base").alias("__source__"),
    )
    base_df, _ = auto_fix_dataframe(base_df)

    # Load new records
    console.print("[bold]Loading new records...[/bold]")
    new_lf = load_file(str(new_records))
    new_df = new_lf.collect()
    base_max_id = base_df["__row_id__"].max() + 1 if base_df.height > 0 else 0
    new_df = new_df.with_row_index("__row_id__").with_columns(
        (pl.col("__row_id__").cast(pl.Int64) + base_max_id).alias("__row_id__"),
        pl.lit("new").alias("__source__"),
    )
    new_df, _ = auto_fix_dataframe(new_df)

    # Standardize and compute matchkeys on combined data
    combined = pl.concat([base_df, new_df], how="diagonal")
    lf = combined.lazy()
    if cfg.standardization:
        lf = apply_standardization(lf, cfg.standardization)
    for mk in matchkeys:
        lf = compute_matchkeys(lf, [mk])
    combined = lf.collect()

    # Match each new record against the base
    console.print(f"[bold]Matching {new_df.height} new records against {base_df.height} base records...[/bold]")
    t0 = time.perf_counter()

    all_matches = []
    new_ids = set(range(base_max_id, base_max_id + new_df.height))

    # Handle exact matchkeys via Polars join (match_one doesn't support exact)
    exact_mks = [mk for mk in matchkeys if mk.type == "exact"]
    fuzzy_mks = [mk for mk in matchkeys if mk.type != "exact"]

    for mk in exact_mks:
        mk_col = f"__mk_{mk.name}__"
        if mk_col not in combined.columns:
            continue
        pairs = find_exact_matches(combined.lazy(), mk)
        for a, b, score in pairs:
            # Keep only cross-source pairs (one new, one base)
            if (a in new_ids) != (b in new_ids):
                new_id = a if a in new_ids else b
                base_id = b if a in new_ids else a
                all_matches.append((new_id, base_id, score))

    # Handle fuzzy matchkeys via match_one
    if fuzzy_mks:
        row_index = {}
        for row in combined.to_dicts():
            row_index[row["__row_id__"]] = row

        for new_id in sorted(new_ids):
            row = row_index.get(new_id)
            if not row:
                continue
            for mk in fuzzy_mks:
                matches = match_one(row, combined, mk)
                for rid, score in matches:
                    if rid not in new_ids:
                        all_matches.append((new_id, rid, score))

    # Deduplicate: keep best score per (new_id, base_id) pair
    best = {}
    for new_id, base_id, score in all_matches:
        key = (new_id, base_id)
        if key not in best or score > best[key]:
            best[key] = score
    all_matches = [(n, b, s) for (n, b), s in best.items()]

    matched_new_ids = {n for n, _, _ in all_matches}
    matched_count = len(matched_new_ids)
    new_entity_count = len(new_ids) - matched_count

    elapsed = time.perf_counter() - t0

    # Build results
    table = Table(title="Incremental Match Results")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("New records processed", str(new_df.height))
    table.add_row("Matched to base", str(matched_count))
    table.add_row("New entities", str(new_entity_count))
    table.add_row("Total match pairs", str(len(all_matches)))
    table.add_row("Time", f"{elapsed:.2f}s")
    console.print(table)

    if output and all_matches:
        rows = []
        for new_id, base_id, score in all_matches:
            rows.append({
                "new_row_id": new_id,
                "base_row_id": base_id,
                "score": round(score, 4),
            })
        result_df = pl.DataFrame(rows)
        result_df.write_csv(output)
        console.print(f"\n[green]Results saved to {output}[/green]")
    elif output:
        console.print("\n[yellow]No matches found - no output written[/yellow]")
