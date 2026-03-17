"""GoldenMatch Interactive TUI -- main Textual application."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, TabbedContent, TabPane
from textual import work

from goldenmatch.tui.sidebar import Sidebar
from goldenmatch.tui.tabs.config_tab import ConfigTab
from goldenmatch.tui.tabs.data_tab import DataTab
from goldenmatch.tui.tabs.export_tab import ExportTab
from goldenmatch.tui.tabs.golden_tab import GoldenTab
from goldenmatch.tui.tabs.matches_tab import MatchesTab


class GoldenMatchApp(App):
    """Interactive TUI for building GoldenMatch configs with live feedback."""

    TITLE = "GoldenMatch Interactive"

    CSS = """
    #sidebar {
        width: 28;
        background: $surface;
        border-right: solid $primary;
        padding: 1;
    }
    #main {
        width: 1fr;
    }
    .sidebar-section {
        margin-bottom: 1;
    }
    .sidebar-label {
        color: $text-muted;
        text-style: bold;
    }
    .stat-value {
        color: $success;
    }
    """

    BINDINGS = [
        Binding("f1", "help", "Help"),
        Binding("f2", "save_config", "Save Config"),
        Binding("f5", "run_sample", "Run Sample"),
        Binding("ctrl+r", "rerun", "Re-run"),
        Binding("ctrl+s", "save_preset", "Save Preset"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, files: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = files or []
        self.engine = None
        self.current_config = None
        self.last_result = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield Sidebar(id="sidebar")
            with Vertical(id="main"):
                with TabbedContent():
                    with TabPane("Data", id="tab-data"):
                        yield DataTab()
                    with TabPane("Config", id="tab-config"):
                        yield ConfigTab()
                    with TabPane("Matches", id="tab-matches"):
                        yield MatchesTab()
                    with TabPane("Golden", id="tab-golden"):
                        yield GoldenTab()
                    with TabPane("Export", id="tab-export"):
                        yield ExportTab()
        yield Footer()

    def on_mount(self) -> None:
        """Load files on app startup if paths were provided."""
        if self.file_paths:
            self.load_files(self.file_paths)

    def load_files(self, paths: list[str]) -> None:
        """Load data files via the MatchEngine and update the UI."""
        from goldenmatch.tui.engine import MatchEngine

        try:
            self.engine = MatchEngine(paths)
            sidebar = self.query_one(Sidebar)
            sidebar.update_file_info(self.engine)
            data_tab = self.query_one(DataTab)
            data_tab.show_profile(self.engine)
            # Update config tab with available columns
            config_tab = self.query_one(ConfigTab)
            config_tab.set_columns(self.engine.columns)
        except Exception as e:
            self.notify(f"Error loading files: {e}", severity="error")

    def on_config_tab_config_changed(self, event: ConfigTab.ConfigChanged) -> None:
        """Handle config changes from the Config tab."""
        self.current_config = event.config
        # Update sidebar
        sidebar = self.query_one(Sidebar)
        sidebar.update_config(event.config)
        # Run sample if engine is loaded
        if self.engine is not None:
            self.run_matching(event.config)

    @work(thread=True)
    def run_matching(self, config) -> None:
        """Run matching in a background thread."""
        if self.engine is None:
            return
        try:
            result = self.engine.run_sample(config, sample_size=1000)
            self.call_from_thread(self._on_matching_complete, result)
        except Exception as e:
            self.call_from_thread(
                self.notify, f"Matching error: {e}", severity="error"
            )

    def _on_matching_complete(self, result) -> None:
        """Called on the main thread when matching finishes."""
        self.last_result = result
        # Update sidebar stats
        sidebar = self.query_one(Sidebar)
        sidebar.update_stats(result.stats)
        # Update matches tab
        matches_tab = self.query_one(MatchesTab)
        matches_tab.update_results(result, self.engine.data)
        self.notify("Sample matching complete.", severity="information")

    def action_help(self) -> None:
        """Show help information."""
        self.notify(
            "F1:Help  F2:Save Config  F5:Run Sample  "
            "Ctrl+R:Re-run  Ctrl+S:Save Preset  Q:Quit",
            title="Key Bindings",
        )

    def action_save_config(self) -> None:
        """Save current config to YAML."""
        self.notify("Save config not yet implemented.", severity="warning")

    def action_run_sample(self) -> None:
        """Run matching on a sample of the data."""
        if self.engine is None:
            self.notify("No files loaded.", severity="warning")
            return
        if self.current_config is None:
            self.notify("No config set. Build a config in the Config tab first.", severity="warning")
            return
        self.notify("Running sample match...", severity="information")
        self.run_matching(self.current_config)

    def action_rerun(self) -> None:
        """Re-run with current config."""
        self.action_run_sample()

    def action_save_preset(self) -> None:
        """Save config as a named preset."""
        self.notify("Save preset not yet implemented.", severity="warning")
