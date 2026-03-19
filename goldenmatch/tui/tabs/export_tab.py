"""Export tab — save config, presets, and run full jobs."""

from __future__ import annotations

from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Select, Static, Switch


class ExportTab(Static):
    """Save config, run jobs, and export results."""

    DEFAULT_CSS = """
    ExportTab {
        height: 1fr;
        padding: 1;
    }
    .export-section {
        margin-bottom: 1;
    }
    .export-row {
        height: auto;
    }
    .switch-row {
        height: auto;
        margin-bottom: 0;
    }
    .switch-label {
        width: 20;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config = None

    def compose(self) -> ComposeResult:
        with Vertical():
            # ── Save Config ──
            yield Static("[bold]Save Config[/bold]", classes="export-section")
            with Horizontal(classes="export-row"):
                yield Input(value="goldenmatch.yaml", id="config-path")
                yield Button("Save", id="btn-save-config", variant="primary")
            yield Label("", id="save-status")

            yield Static("")

            # ── Save as Preset ──
            yield Static("[bold]Save as Preset[/bold]", classes="export-section")
            with Horizontal(classes="export-row"):
                yield Input(placeholder="preset name", id="preset-name")
                yield Button("Save Preset", id="btn-save-preset")
            yield Label("", id="preset-status")

            yield Static("")

            # ── Run Full Job ──
            yield Static("[bold]Run Full Job[/bold]", classes="export-section")
            with Horizontal(classes="export-row"):
                yield Static("Format: ", classes="switch-label")
                yield Select(
                    [(fmt, fmt) for fmt in ("csv", "parquet", "xlsx")],
                    value="csv",
                    id="output-format",
                )
            with Horizontal(classes="export-row"):
                yield Static("Output dir: ", classes="switch-label")
                yield Input(value=".", id="output-dir")

            # Output switches
            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="sw-golden")
                yield Static("Output golden records", classes="switch-label")
            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="sw-clusters")
                yield Static("Output clusters", classes="switch-label")
            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="sw-dupes")
                yield Static("Output duplicates", classes="switch-label")
            with Horizontal(classes="switch-row"):
                yield Switch(value=False, id="sw-unique")
                yield Static("Output unique records", classes="switch-label")
            with Horizontal(classes="switch-row"):
                yield Switch(value=True, id="sw-report")
                yield Static("Output report", classes="switch-label")
            with Horizontal(classes="switch-row"):
                yield Switch(value=False, id="sw-html-report-export")
                yield Static("HTML Report", classes="switch-label")
            with Horizontal(classes="switch-row"):
                yield Switch(value=False, id="sw-dashboard-export")
                yield Static("Before/After Dashboard", classes="switch-label")
            with Horizontal(classes="switch-row"):
                yield Switch(value=False, id="sw-graph-export")
                yield Static("Cluster Graph", classes="switch-label")

            yield Static("")
            yield Button("Run Full Job", id="btn-run-full", variant="warning")
            yield Label("", id="run-status")

    def set_config(self, config) -> None:
        """Store the current config for saving."""
        self._config = config

    def _save_config_to_path(self, path: str) -> Path:
        """Serialize and write the current config to a YAML file. Returns the Path."""
        if self._config is None:
            raise ValueError("No config available. Build a config in the Config tab first.")
        config_dict = self._config.model_dump(exclude_none=True)
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml.dump(config_dict, default_flow_style=False, sort_keys=False), encoding="utf-8")
        return dest

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks for save/preset/run."""
        if event.button.id == "btn-save-config":
            self._handle_save_config()
        elif event.button.id == "btn-save-preset":
            self._handle_save_preset()
        elif event.button.id == "btn-run-full":
            self._handle_run_full()

    def _handle_save_config(self) -> None:
        status = self.query_one("#save-status", Label)
        try:
            path_input = self.query_one("#config-path", Input)
            dest = self._save_config_to_path(path_input.value)
            status.update(f"[green]Saved to {dest}[/green]")
        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")

    def _handle_save_preset(self) -> None:
        status = self.query_one("#preset-status", Label)
        try:
            name_input = self.query_one("#preset-name", Input)
            name = name_input.value.strip()
            if not name:
                status.update("[red]Please enter a preset name.[/red]")
                return

            # Save config to a temp YAML first, then store as preset
            path_input = self.query_one("#config-path", Input)
            config_path = self._save_config_to_path(path_input.value)

            from goldenmatch.prefs.store import PresetStore
            store = PresetStore()
            dest = store.save(name, config_path)
            status.update(f"[green]Preset '{name}' saved to {dest}[/green]")
        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")

    def _handle_run_full(self) -> None:
        status = self.query_one("#run-status", Label)
        if self._config is None:
            status.update("[red]No config available. Build a config first.[/red]")
            return

        try:
            app = self.app
            if app.engine is None:
                status.update("[red]No data loaded.[/red]")
                return

            status.update("[yellow]Running full job...[/yellow]")

            # Gather output options
            fmt_select = self.query_one("#output-format", Select)
            dir_input = self.query_one("#output-dir", Input)
            output_options = {
                "format": fmt_select.value,
                "directory": dir_input.value,
                "output_golden": self.query_one("#sw-golden", Switch).value,
                "output_clusters": self.query_one("#sw-clusters", Switch).value,
                "output_dupes": self.query_one("#sw-dupes", Switch).value,
                "output_unique": self.query_one("#sw-unique", Switch).value,
                "output_report": self.query_one("#sw-report", Switch).value,
            }

            # Trigger via app
            app.run_full_job(self._config, output_options)
        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")
