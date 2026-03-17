"""Config tab — matchkey builder (placeholder)."""

from __future__ import annotations

from textual.widgets import Static


class ConfigTab(Static):
    """Matchkey builder with live feedback. Placeholder for now."""

    def compose(self):
        yield Static(
            "[dim]Config tab — matchkey builder coming soon...[/dim]\n\n"
            "This tab will allow you to:\n"
            "  - Add/remove matchkeys with a field picker\n"
            "  - Select transforms and scorers\n"
            "  - Adjust threshold with a live slider\n"
            "  - Configure blocking keys and golden rules"
        )
