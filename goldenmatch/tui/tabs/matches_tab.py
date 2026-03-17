"""Matches tab -- cluster/match viewer with color-coded scores."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

import polars as pl

from goldenmatch.tui.engine import EngineResult


class MatchesTab(Static):
    """Match preview with cluster list and detail drill-down."""

    DEFAULT_CSS = """
    MatchesTab {
        height: 1fr;
    }
    #cluster-table {
        height: 40%;
        border: solid $primary;
    }
    #detail-table {
        height: 55%;
        border: solid $accent;
    }
    .no-results {
        padding: 2;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result: EngineResult | None = None
        self._data: pl.DataFrame | None = None
        self._clusters: dict[int, dict] | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "[dim]Run matching from the Config tab to see results here.[/dim]",
                id="no-results-msg",
                classes="no-results",
            )
            yield DataTable(id="cluster-table")
            yield DataTable(id="detail-table")

    def on_mount(self) -> None:
        cluster_table = self.query_one("#cluster-table", DataTable)
        cluster_table.add_columns("Cluster ID", "Size", "Top Score")
        cluster_table.display = False
        cluster_table.cursor_type = "row"

        detail_table = self.query_one("#detail-table", DataTable)
        detail_table.display = False

    def update_results(self, result: EngineResult, data: pl.DataFrame) -> None:
        """Populate cluster list and detail tables from engine results."""
        self._result = result
        self._data = data
        self._clusters = result.clusters

        # Hide placeholder
        no_msg = self.query_one("#no-results-msg", Static)
        no_msg.display = False

        # Build cluster list
        cluster_table = self.query_one("#cluster-table", DataTable)
        cluster_table.clear()
        cluster_table.display = True

        # Build a map of top scores per cluster from scored_pairs
        cluster_top_scores: dict[int, float] = {}
        # Map row_id -> cluster_id
        row_to_cluster: dict[int, int] = {}
        for cid, cinfo in result.clusters.items():
            for mid in cinfo["members"]:
                row_to_cluster[mid] = cid

        for id_a, id_b, score in result.scored_pairs:
            cid_a = row_to_cluster.get(id_a)
            if cid_a is not None:
                cluster_top_scores[cid_a] = max(
                    cluster_top_scores.get(cid_a, 0.0), score
                )

        # Only show multi-member clusters
        multi_clusters = [
            (cid, cinfo)
            for cid, cinfo in result.clusters.items()
            if cinfo["size"] > 1
        ]
        multi_clusters.sort(key=lambda x: x[1]["size"], reverse=True)

        for cid, cinfo in multi_clusters:
            top_score = cluster_top_scores.get(cid, 0.0)
            score_str = f"{top_score:.3f}"
            # Color code: green >0.9, yellow 0.7-0.9, red <0.7
            if top_score > 0.9:
                score_str = f"[green]{score_str}[/green]"
            elif top_score >= 0.7:
                score_str = f"[yellow]{score_str}[/yellow]"
            else:
                score_str = f"[red]{score_str}[/red]"
            cluster_table.add_row(str(cid), str(cinfo["size"]), score_str)

        # Clear detail
        detail_table = self.query_one("#detail-table", DataTable)
        detail_table.clear(columns=True)
        detail_table.display = False

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show cluster members when a cluster row is selected."""
        if self._result is None or self._data is None:
            return
        if event.data_table.id != "cluster-table":
            return

        # Get cluster ID from the selected row
        cluster_table = self.query_one("#cluster-table", DataTable)
        row_key = event.row_key
        row_data = cluster_table.get_row(row_key)
        try:
            cluster_id = int(row_data[0])
        except (ValueError, IndexError):
            return

        cluster_info = self._clusters.get(cluster_id)
        if cluster_info is None:
            return

        member_ids = cluster_info["members"]
        member_df = self._data.filter(pl.col("__row_id__").is_in(member_ids))

        # Show in detail table
        detail_table = self.query_one("#detail-table", DataTable)
        detail_table.clear(columns=True)
        detail_table.display = True

        # Add columns (skip internal columns)
        display_cols = [c for c in member_df.columns if not c.startswith("__")]
        for col in display_cols:
            detail_table.add_column(col)

        # Add rows
        for row in member_df.iter_rows(named=True):
            values = [str(row.get(c, "")) for c in display_cols]
            detail_table.add_row(*values)
