"""Export tab — save config and run jobs (placeholder)."""

from __future__ import annotations

from textual.widgets import Static


class ExportTab(Static):
    """Save config, run jobs, and export results. Placeholder for now."""

    def compose(self):
        yield Static(
            "[dim]Export tab — save and run coming soon...[/dim]\n\n"
            "This tab will allow you to:\n"
            "  - Save config to YAML\n"
            "  - Save as named preset\n"
            "  - Run full job with output options\n"
            "  - Select output format (CSV, Parquet, Excel)"
        )
