"""Golden tab — golden record preview (placeholder)."""

from __future__ import annotations

from textual.widgets import Static


class GoldenTab(Static):
    """Golden record preview with per-field confidence. Placeholder for now."""

    def compose(self):
        yield Static(
            "[dim]Golden tab — golden record preview coming soon...[/dim]\n\n"
            "This tab will show:\n"
            "  - Merged golden records with confidence scores\n"
            "  - Per-field source attribution\n"
            "  - Click-to-inspect cluster drill-down"
        )
