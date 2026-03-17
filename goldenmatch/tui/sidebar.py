"""Sidebar widget showing config summary and live stats."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class Sidebar(Static):
    """Persistent sidebar showing config summary and live stats."""

    file_info: reactive[str] = reactive("")
    config_info: reactive[str] = reactive("")
    stats_info: reactive[str] = reactive("")

    def render(self) -> str:
        sections: list[str] = []
        sections.append("[bold cyan]Files[/bold cyan]")
        sections.append(self.file_info or "[dim]No files loaded[/dim]")
        sections.append("")
        sections.append("[bold cyan]Config[/bold cyan]")
        sections.append(self.config_info or "[dim]No config[/dim]")
        sections.append("")
        sections.append("[bold cyan]Stats[/bold cyan]")
        sections.append(self.stats_info or "[dim]Run to see stats[/dim]")
        return "\n".join(sections)

    def update_file_info(self, engine) -> None:
        """Update file information from a loaded MatchEngine."""
        self.file_info = (
            f"Records: [green]{engine.row_count:,}[/green]\n"
            f"Columns: [green]{len(engine.columns)}[/green]\n"
            f"Sources: {len(engine._files)}"
        )

    def update_config(self, config) -> None:
        """Update config summary from a GoldenMatchConfig."""
        if not config or not config.get_matchkeys():
            self.config_info = "[dim]No matchkeys[/dim]"
            return
        lines: list[str] = []
        for mk in config.get_matchkeys():
            lines.append(f"  [green]\u2713[/green] {mk.name}")
            if mk.threshold:
                lines.append(f"    threshold: {mk.threshold}")
        self.config_info = "\n".join(lines)

    def update_stats(self, stats) -> None:
        """Update live stats from an EngineStats result."""
        self.stats_info = (
            f"Clusters: [green]{stats.total_clusters}[/green]\n"
            f"Match Rate: [green]{stats.match_rate:.1%}[/green]\n"
            f"Singletons: {stats.singleton_count}\n"
            f"Max Size: {stats.max_cluster_size}"
        )
