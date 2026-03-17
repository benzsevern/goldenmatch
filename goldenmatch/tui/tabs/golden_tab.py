"""Golden tab — golden record preview with confidence scores."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static

from goldenmatch.tui.engine import EngineResult


class GoldenTab(Static):
    """Golden record preview with per-field confidence and color coding."""

    DEFAULT_CSS = """
    GoldenTab {
        height: 1fr;
    }
    #golden-table {
        height: 1fr;
        border: solid $primary;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result: EngineResult | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(
                "[dim]Run matching to see golden records here.[/dim]",
                id="golden-placeholder",
            )
            yield DataTable(id="golden-table")

    def on_mount(self) -> None:
        table = self.query_one("#golden-table", DataTable)
        table.display = False

    def update_results(self, result: EngineResult) -> None:
        """Populate table with golden records from the engine result."""
        self._result = result
        placeholder = self.query_one("#golden-placeholder", Static)
        table = self.query_one("#golden-table", DataTable)

        golden_df = result.golden
        if golden_df is None or golden_df.height == 0:
            placeholder.update("[dim]Run matching to see golden records here.[/dim]")
            placeholder.display = True
            table.display = False
            return

        # Hide placeholder, show table
        placeholder.display = False
        table.clear(columns=True)
        table.display = True

        # Determine data columns (exclude internal columns)
        data_cols = [
            c for c in golden_df.columns
            if c not in ("__cluster_id__", "__golden_confidence__")
        ]

        # Add columns: Cluster ID, Confidence, then data columns
        table.add_column("Cluster ID")
        table.add_column("Confidence")
        for col in data_cols:
            table.add_column(col)

        # Add rows
        for row in golden_df.iter_rows(named=True):
            cluster_id = str(row.get("__cluster_id__", ""))
            confidence = row.get("__golden_confidence__", 0.0)
            if confidence is None:
                confidence = 0.0

            # Format confidence with color coding
            conf_str = f"{confidence:.2f}"
            if confidence > 0.9:
                conf_str = f"[green]{conf_str}[/green]"
            elif confidence >= 0.7:
                conf_str = f"[yellow]{conf_str}[/yellow]"
            else:
                conf_str = f"[red]{conf_str}[/red]"

            values = [str(row.get(c, "")) for c in data_cols]
            table.add_row(cluster_id, conf_str, *values)
