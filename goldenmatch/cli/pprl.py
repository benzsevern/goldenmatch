"""CLI PPRL commands for GoldenMatch."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)

pprl_app = typer.Typer(
    name="pprl",
    help="Privacy-preserving record linkage commands.",
    no_args_is_help=True,
)


@pprl_app.command("link")
def pprl_link(
    file_a: Path = typer.Option(..., "--file-a", "-a", help="Party A data file (CSV)"),
    file_b: Path = typer.Option(..., "--file-b", "-b", help="Party B data file (CSV)"),
    fields: str = typer.Option(..., "--fields", "-f", help="Comma-separated field names to match on"),
    threshold: float = typer.Option(0.85, "--threshold", "-t", help="Match threshold"),
    security: str = typer.Option("high", "--security", "-s", help="Security level: standard, high, paranoid"),
    protocol: str = typer.Option("trusted_third_party", "--protocol", "-p", help="Protocol: trusted_third_party or smc"),
    scorer: str = typer.Option("dice", "--scorer", help="Similarity scorer: dice or jaccard"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV path for cluster assignments"),
) -> None:
    """Link records across two parties without sharing raw data."""
    import polars as pl
    from goldenmatch.pprl.protocol import PPRLConfig, run_pprl

    if not file_a.exists():
        err_console.print(f"[red]File not found: {file_a}[/red]")
        raise typer.Exit(1)
    if not file_b.exists():
        err_console.print(f"[red]File not found: {file_b}[/red]")
        raise typer.Exit(1)

    field_list = [f.strip() for f in fields.split(",")]

    config = PPRLConfig(
        fields=field_list,
        threshold=threshold,
        security_level=security,
        protocol=protocol,
        scorer=scorer,
    )

    # Map security level to bloom filter params
    _LEVELS = {
        "standard": (2, 20, 512),
        "high": (2, 30, 1024),
        "paranoid": (3, 40, 2048),
    }
    if security in _LEVELS:
        config.ngram_size, config.hash_functions, config.bloom_filter_size = _LEVELS[security]

    console.print(f"[bold]Loading data...[/bold]")
    df_a = pl.read_csv(file_a)
    df_b = pl.read_csv(file_b)

    console.print(f"  Party A: {df_a.height} records")
    console.print(f"  Party B: {df_b.height} records")
    console.print(f"  Fields: {', '.join(field_list)}")
    console.print(f"  Protocol: {protocol}")
    console.print(f"  Security: {security}")
    console.print()

    console.print(f"[bold]Running PPRL linkage...[/bold]")
    result = run_pprl(
        df_a, df_b, config,
        party_a_id="party_a", party_b_id="party_b",
    )

    # Display results
    table = Table(title="PPRL Linkage Results")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("Clusters found", str(len(result.clusters)))
    table.add_row("Match pairs", str(result.match_count))
    table.add_row("Total comparisons", f"{result.total_comparisons:,}")
    table.add_row("Protocol", protocol)
    table.add_row("Security level", security)
    console.print(table)

    if result.clusters:
        console.print(f"\n[bold]Cluster details:[/bold]")
        for cid, members in sorted(result.clusters.items())[:10]:
            member_strs = [f"{pid}:{rid}" for pid, rid in members]
            console.print(f"  Cluster {cid}: {', '.join(member_strs)}")
        if len(result.clusters) > 10:
            console.print(f"  ... and {len(result.clusters) - 10} more")

    if output and result.clusters:
        rows = []
        for cid, members in result.clusters.items():
            for party_id, record_id in members:
                rows.append({
                    "cluster_id": cid,
                    "party": party_id,
                    "record_id": record_id,
                })
        pl.DataFrame(rows).write_csv(output)
        console.print(f"\n[green]Results saved to {output}[/green]")


@pprl_app.command("auto-config")
def pprl_auto_config(
    file: Path = typer.Argument(..., help="Data file to analyze (CSV)"),
    security: str = typer.Option("high", "--security", "-s", help="Security level: standard, high, paranoid"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM for enhanced recommendations"),
) -> None:
    """Analyze data and recommend optimal PPRL configuration."""
    import polars as pl
    from goldenmatch.pprl.autoconfig import auto_configure_pprl, auto_configure_pprl_llm

    if not file.exists():
        err_console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Analyzing {file.name}...[/bold]")
    df = pl.read_csv(file, ignore_errors=True, encoding="utf8-lossy")
    console.print(f"  {df.height} records, {len(df.columns)} columns\n")

    if use_llm:
        result = auto_configure_pprl_llm(df, security_level=security)
    else:
        result = auto_configure_pprl(df, security_level=security)

    # Display field profiles
    table = Table(title="Field Analysis")
    table.add_column("Field", style="bold")
    table.add_column("Type")
    table.add_column("Avg Len", justify="right")
    table.add_column("Cardinality", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Selected", justify="center")

    for p in result.field_profiles:
        selected = "Y" if p.column in result.recommended_fields else ""
        table.add_row(
            p.column, p.field_type,
            f"{p.avg_length:.1f}", str(p.cardinality),
            f"{p.usefulness_score:.2f}", selected,
        )
    console.print(table)

    # Display recommended config
    config = result.recommended_config
    console.print(f"\n[bold]Recommended Configuration:[/bold]")
    console.print(f"  Fields: {', '.join(result.recommended_fields)}")
    console.print(f"  Threshold: {config.threshold}")
    console.print(f"  Security: {config.security_level}")
    console.print(f"  Bloom filter: {config.ngram_size}-gram, {config.hash_functions} hashes, {config.bloom_filter_size}-bit")

    # Print as YAML config
    console.print(f"\n[bold]YAML config:[/bold]")
    console.print(f"  pprl:")
    console.print(f"    fields: [{', '.join(result.recommended_fields)}]")
    console.print(f"    threshold: {config.threshold}")
    console.print(f"    security_level: {config.security_level}")
    console.print(f"    bloom_filter_size: {config.bloom_filter_size}")
    console.print(f"    hash_functions: {config.hash_functions}")
    console.print(f"    ngram_size: {config.ngram_size}")
