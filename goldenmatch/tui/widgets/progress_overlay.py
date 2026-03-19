"""Full-screen progress overlay for first-run matching."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Center, Vertical
from textual.widgets import ProgressBar, Static


PIPELINE_STAGES = [
    "Ingest",
    "Auto-fix",
    "Standardize",
    "Matchkeys",
    "Blocking",
    "Scoring",
    "Clustering",
    "Golden records",
]


class ProgressOverlay(Static):
    """Full-screen progress overlay showing pipeline stage progress."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_stage = 0
        self._stage_times: dict[int, float] = {}
        self._pairs_found = 0
        self._elapsed = 0.0

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="progress-box"):
                yield Static(
                    "[bold #d4a017]Matching in progress...[/]",
                    classes="progress-title",
                    id="progress-title",
                )
                yield ProgressBar(total=100, id="progress-bar")
                yield Static("", id="progress-stats")
                yield Static("", id="progress-pipeline")

    def update_progress(
        self,
        stage: int = 0,
        percent: float = 0.0,
        pairs: int = 0,
        elapsed: float = 0.0,
        stage_times: dict[int, float] | None = None,
    ) -> None:
        """Update progress display."""
        self._current_stage = stage
        self._pairs_found = pairs
        self._elapsed = elapsed
        if stage_times:
            self._stage_times = stage_times

        # Update progress bar
        try:
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.update(progress=percent)
        except Exception:
            pass

        # Update stats
        try:
            stage_name = PIPELINE_STAGES[stage] if stage < len(PIPELINE_STAGES) else "Processing"
            stats = self.query_one("#progress-stats", Static)
            stats.update(
                f"Stage: [bold #d4a017]{stage_name}[/]\n"
                f"Pairs: [bold #2ecc71]{pairs:,}[/] found\n"
                f"Elapsed: [bold]{elapsed:.1f}s[/]"
            )
        except Exception:
            pass

        # Update pipeline view
        try:
            pipeline = self.query_one("#progress-pipeline", Static)
            lines = ["\n[bold #8892a0]Pipeline:[/]"]
            for i, name in enumerate(PIPELINE_STAGES):
                time_str = f"  {self._stage_times.get(i, 0):.1f}s" if i in self._stage_times else ""
                if i < stage:
                    lines.append(f"  [#2ecc71]✓ {name}{time_str}[/]")
                elif i == stage:
                    lines.append(f"  [bold #d4a017]● {name}  {elapsed:.1f}s[/]")
                else:
                    lines.append(f"  [#8892a0 50%]○ {name}[/]")
            pipeline.update("\n".join(lines))
        except Exception:
            pass

    def set_complete(self) -> None:
        """Show completion state."""
        try:
            title = self.query_one("#progress-title", Static)
            title.update("[bold #2ecc71]Matching complete![/]")
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.update(progress=100)
        except Exception:
            pass
