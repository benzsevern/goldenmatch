"""Auto-config summary screen — first screen for zero-config mode."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Static


class AutoConfigScreen(Screen):
    """Summary screen showing auto-detected column configuration."""

    BINDINGS = [
        ("f5", "run", "Run"),
        ("e", "edit", "Edit Config"),
        ("s", "save", "Save Settings"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        file_name: str = "",
        row_count: int = 0,
        col_count: int = 0,
        column_profiles: list[dict] | None = None,
        blocking_info: str = "",
        threshold: float = 0.80,
        model_info: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.file_name = file_name
        self.row_count = row_count
        self.col_count = col_count
        self.column_profiles = column_profiles or []
        self.blocking_info = blocking_info
        self.threshold = threshold
        self.model_info = model_info

    def compose(self) -> ComposeResult:
        yield Header()
        with Center():
            with Vertical(id="autoconfig-card"):
                yield Static(
                    "[bold #d4a017]Auto-Detected Configuration[/]",
                    classes="autoconfig-title",
                )
                yield Static(
                    f"File: [bold]{self.file_name}[/]",
                    classes="autoconfig-info",
                )
                yield Static(
                    f"Records: [bold #2ecc71]{self.row_count:,}[/] │ "
                    f"Columns: [bold]{self.col_count}[/]",
                    classes="autoconfig-info",
                )
                yield Static(
                    "\n[bold #d4a017]Column Mapping[/]",
                    classes="autoconfig-info",
                )
                yield DataTable(id="column-table")
                yield Static(
                    f"\nBlocking: [bold]{self.blocking_info}[/]",
                    classes="autoconfig-info",
                )
                yield Static(
                    f"Threshold: [bold]{self.threshold}[/] (adaptive)",
                    classes="autoconfig-info",
                )
                yield Static(
                    f"Model: [bold]{self.model_info}[/]",
                    classes="autoconfig-info",
                )
                yield Static("")
                with Center(classes="autoconfig-buttons"):
                    yield Button("▶ Run", variant="primary", id="btn-run")
                    yield Button("Edit Config", id="btn-edit")
                    yield Button("Save Settings", id="btn-save")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the column table."""
        table = self.query_one("#column-table", DataTable)
        table.add_columns("Column", "Type", "Scorer", "Weight")

        for profile in self.column_profiles:
            col_name = profile.get("name", "")
            col_type = profile.get("type", "string")
            scorer = profile.get("scorer", "—")
            weight = profile.get("weight", "—")

            # Color-code by type
            type_colors = {
                "name": "#2ecc71",
                "email": "#3498db",
                "phone": "#9b59b6",
                "zip": "#e67e22",
                "address": "#1abc9c",
                "description": "#e74c3c",
                "identifier": "#8892a0",
                "string": "#f0f0f0",
            }
            color = type_colors.get(col_type, "#f0f0f0")
            type_display = f"[{color}]{col_type.title()}[/]"

            table.add_row(col_name, type_display, scorer, str(weight))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self.action_run()
        elif event.button.id == "btn-edit":
            self.action_edit()
        elif event.button.id == "btn-save":
            self.action_save()

    def action_run(self) -> None:
        """Dismiss screen and run matching."""
        self.dismiss("run")

    def action_edit(self) -> None:
        """Dismiss screen and show config tab."""
        self.dismiss("edit")

    def action_save(self) -> None:
        """Save settings then dismiss."""
        self.dismiss("save")

    def action_quit(self) -> None:
        self.app.exit()
