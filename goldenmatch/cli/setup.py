"""CLI command for launching the setup wizard."""

from __future__ import annotations

import typer


def setup_cmd() -> None:
    """Launch the interactive setup wizard."""
    from goldenmatch.tui.screens.setup_wizard import SetupWizard
    from textual.app import App, ComposeResult

    class SetupApp(App):
        TITLE = "GoldenMatch Setup"
        CSS = """
        Screen { background: #1a1a2e; }
        Header { background: #d4a017; color: #1a1a2e; text-style: bold; }
        Footer { background: #16213e; }
        Footer > .footer--key { background: #d4a017 30%; color: #f0f0f0; }
        """

        def on_mount(self) -> None:
            self.push_screen(SetupWizard(), callback=self._on_done)

        def _on_done(self, result: str) -> None:
            self.exit()

    app = SetupApp()
    app.run()
