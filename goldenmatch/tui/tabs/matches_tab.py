"""Matches tab — match preview with drill-down (placeholder)."""

from __future__ import annotations

from textual.widgets import Static


class MatchesTab(Static):
    """Match preview with cluster and pair views. Placeholder for now."""

    def compose(self):
        yield Static(
            "[dim]Matches tab — match preview coming soon...[/dim]\n\n"
            "This tab will show:\n"
            "  - Cluster view with expandable members\n"
            "  - Pair view with per-field scores\n"
            "  - Color-coded match quality indicators"
        )
