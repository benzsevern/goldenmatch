"""Data tab — file browser and profiler display."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static


class DataTab(Static):
    """File browser and profiler display."""

    def compose(self):
        yield VerticalScroll(
            Static("[dim]Load files to see profile data[/dim]", id="data-overview"),
            DataTable(id="profile-table"),
        )

    def show_profile(self, engine) -> None:
        """Populate the data tab with profiler results from the engine."""
        overview = self.query_one("#data-overview", Static)
        cols_preview = ", ".join(engine.columns[:10])
        extra = f" (+{len(engine.columns) - 10} more)" if len(engine.columns) > 10 else ""
        overview.update(
            f"[bold]Loaded {engine.row_count:,} records[/bold] "
            f"across {len(engine._files)} file(s)\n"
            f"Columns: {cols_preview}{extra}"
        )

        table = self.query_one("#profile-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Column", "Type", "Detected", "Null%", "Unique%", "Sample")

        for col_profile in engine.profile.get("columns", []):
            null_pct = f"{col_profile.get('null_rate', 0):.0%}"
            unique_pct = f"{col_profile.get('unique_rate', 0):.0%}"
            samples = col_profile.get("sample_values", [])
            sample_str = ", ".join(str(s) for s in samples[:3])
            table.add_row(
                col_profile.get("name", ""),
                str(col_profile.get("dtype", "")),
                col_profile.get("suspected_type", "text"),
                null_pct,
                unique_pct,
                sample_str,
            )
