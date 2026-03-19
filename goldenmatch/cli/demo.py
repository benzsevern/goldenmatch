"""Built-in demo — zero-friction showcase of GoldenMatch."""

from __future__ import annotations

import csv
import tempfile
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


DEMO_DATA = [
    ["id", "name", "email", "phone", "zip", "specialty"],
    [1, "John Smith", "john.smith@gmail.com", "555-0101", "10001", "Cardiology"],
    [2, "Jon Smith", "jon.smith@gmail.com", "555-0101", "10001", "Cardiology"],
    [3, "Jane Doe", "jane.doe@yahoo.com", "555-0202", "90210", "Radiology"],
    [4, "Janet Doe", "janet.doe@yahoo.com", "555-0203", "90210", "Radiology"],
    [5, "Bob Johnson", "bob.j@outlook.com", "555-0303", "30301", "Neurology"],
    [6, "Robert Johnson", "robert.johnson@outlook.com", "555-0303", "30301", "Neurology"],
    [7, "Alice Brown", "alice.b@gmail.com", "555-0404", "60601", "Pediatrics"],
    [8, "Alicia Brown", "alicia.brown@gmail.com", "555-0405", "60601", "Pediatrics"],
    [9, "Mike Wilson", "mike.w@test.com", "555-0505", "20001", "Oncology"],
    [10, "Michael Wilson", "michael.wilson@test.com", "555-0505", "20001", "Oncology"],
    [11, "Sarah Lee", "sarah.lee@hospital.org", "555-0606", "40201", "Dermatology"],
    [12, "Emily Chen", "emily.chen@clinic.com", "555-0707", "50301", "Surgery"],
    [13, "Chris Taylor", "chris.t@medical.net", "555-0808", "70401", "Cardiology"],
    [14, "Christopher Taylor", "christopher.taylor@medical.net", "555-0808", "70401", "Cardiology"],
    [15, "Lisa Anderson", "lisa.a@health.com", "555-0909", "80501", "Neurology"],
    [16, "Elisabeth Anderson", "liz.anderson@health.com", "555-0910", "80501", "Neurology"],
]


def demo_cmd(
    tui: bool = typer.Option(False, "--tui", help="Launch interactive TUI with demo data"),
    report: bool = typer.Option(False, "--report", help="Generate HTML report"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Generate before/after dashboard"),
    graph: bool = typer.Option(False, "--graph", help="Generate cluster graph"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress animated output"),
) -> None:
    """Run a built-in demo with sample data — no files needed."""
    from goldenmatch.tui.engine import MatchEngine
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
        BlockingConfig, BlockingKeyConfig, GoldenRulesConfig, OutputConfig,
    )

    # Write demo data to temp file
    tmp = Path(tempfile.mkdtemp()) / "demo_patients.csv"
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        for row in DEMO_DATA:
            writer.writerow(row)

    if tui:
        from goldenmatch.tui.app import GoldenMatchApp
        app = GoldenMatchApp(files=[str(tmp)])
        app.run()
        return

    # CLI demo with animated output
    if not quiet:
        console.print()
        console.print(Panel(
            "[bold #d4a017]GoldenMatch Demo[/]\n\n"
            "16 patient records with intentional duplicates.\n"
            "Watch GoldenMatch find them automatically.",
            border_style="#d4a017",
            width=55,
        ))
        console.print()

    # Show the data
    if not quiet:
        data_table = Table(title="Demo Data (16 records)", border_style="#d4a017", header_style="bold #d4a017")
        for col in DEMO_DATA[0]:
            data_table.add_column(col)
        for row in DEMO_DATA[1:6]:
            data_table.add_row(*[str(v) for v in row])
        data_table.add_row("...", "...", "...", "...", "...", "...")
        console.print(data_table)
        console.print()

    # Build config
    config = GoldenMatchConfig(
        matchkeys=[
            MatchkeyConfig(
                name="phone_exact",
                type="exact",
                fields=[MatchkeyField(field="phone", transforms=["strip"])],
            ),
            MatchkeyConfig(
                name="name_zip",
                type="weighted",
                threshold=0.80,
                fields=[
                    MatchkeyField(field="name", scorer="jaro_winkler", weight=0.5, transforms=["lowercase", "strip"]),
                    MatchkeyField(field="zip", scorer="exact", weight=0.2),
                    MatchkeyField(field="specialty", scorer="exact", weight=0.3),
                ],
            ),
        ],
        blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"])],
        ),
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
        output=OutputConfig(),
    )

    # Run matching
    if not quiet:
        console.print("[#d4a017]Running pipeline...[/]")

    engine = MatchEngine([str(tmp)])
    t0 = time.perf_counter()
    result = engine.run_full(config)
    elapsed = time.perf_counter() - t0

    if not quiet:
        console.print(f"[#2ecc71]Done in {elapsed:.2f}s[/]\n")

    # Show results
    stats = result.stats
    multi_clusters = {k: v for k, v in result.clusters.items() if v["size"] > 1}

    result_table = Table(title="Results", border_style="#d4a017", header_style="bold #d4a017")
    result_table.add_column("Metric", style="#d4a017")
    result_table.add_column("Value", style="#2ecc71")
    result_table.add_row("Records", str(stats.total_records))
    result_table.add_row("Duplicate Clusters", str(len(multi_clusters)))
    result_table.add_row("Match Rate", f"{stats.match_rate:.1f}%")
    result_table.add_row("Singletons", str(stats.singleton_count))
    result_table.add_row("Time", f"{elapsed:.2f}s")
    console.print(result_table)
    console.print()

    # Show clusters
    rows = engine.data.to_dicts()
    id_to_row = {r["__row_id__"]: r for r in rows}

    for cid, info in sorted(multi_clusters.items(), key=lambda x: -x[1]["size"]):
        members = info["members"]
        cluster_table = Table(
            title=f"Cluster #{cid} ({info['size']} records)",
            border_style="#d4a017",
            header_style="bold #d4a017",
        )
        cluster_table.add_column("name")
        cluster_table.add_column("email")
        cluster_table.add_column("phone")
        cluster_table.add_column("zip")

        for mid in members:
            r = id_to_row.get(mid, {})
            cluster_table.add_row(
                str(r.get("name", "")),
                str(r.get("email", "")),
                str(r.get("phone", "")),
                str(r.get("zip", "")),
            )
        console.print(cluster_table)

    # Golden records
    if result.golden is not None and result.golden.height > 0:
        console.print()
        golden_table = Table(title="Golden Records (merged)", border_style="#2ecc71", header_style="bold #2ecc71")
        cols = [c for c in result.golden.columns if not c.startswith("__")]
        for col in cols[:5]:
            golden_table.add_column(col)
        for row in result.golden.to_dicts():
            golden_table.add_row(*[str(row.get(c, ""))[:25] for c in cols[:5]])
        console.print(golden_table)

    # Optional: generate report
    if report:
        from goldenmatch.core.report import generate_report
        report_path = generate_report(
            engine.data, result.clusters, result.scored_pairs,
            golden_df=result.golden,
            matchkey_fields=config.matchkeys[1].fields,
            output_path="goldenmatch_demo_report.html",
            title="GoldenMatch Demo Report",
        )
        console.print(f"\n[#d4a017]Report saved:[/] {report_path}")

    if dashboard:
        from goldenmatch.core.dashboard import generate_dashboard
        dash_path = generate_dashboard(
            engine.data, result.clusters, result.scored_pairs,
            golden_df=result.golden,
            output_path="goldenmatch_demo_dashboard.html",
            title="GoldenMatch Demo",
        )
        console.print(f"[#d4a017]Dashboard saved:[/] {dash_path}")

    if graph:
        from goldenmatch.core.graph import generate_cluster_graph
        graph_path = generate_cluster_graph(
            engine.data, result.clusters, result.scored_pairs,
            output_path="goldenmatch_demo_graph.html",
        )
        console.print(f"[#d4a017]Graph saved:[/] {graph_path}")

    if not quiet:
        console.print()
        console.print(Panel(
            "[bold #d4a017]Try it yourself:[/]\n\n"
            "  goldenmatch dedupe your_file.csv\n"
            "  goldenmatch setup               [dim]# configure GPU/API[/dim]\n"
            "  goldenmatch demo --tui           [dim]# interactive TUI[/dim]\n"
            "  goldenmatch demo --report        [dim]# HTML report[/dim]\n"
            "  goldenmatch demo --graph         [dim]# cluster graph[/dim]",
            border_style="#d4a017",
            width=55,
        ))
