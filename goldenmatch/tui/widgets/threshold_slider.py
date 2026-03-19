"""Live threshold slider widget with instant re-clustering."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Static


class ThresholdSlider(Static):
    """Inline threshold control with arrow key adjustment."""

    class ThresholdChanged(Message):
        def __init__(self, value: float):
            super().__init__()
            self.value = value

    def __init__(self, value: float = 0.80, step: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self._value = value
        self._step = step
        self._preview_clusters = 0

    @property
    def value(self) -> float:
        return self._value

    def render(self) -> str:
        bar_width = 20
        filled = int(self._value * bar_width)
        empty = bar_width - filled
        bar = "[#d4a017]" + "█" * filled + "[/][#8892a0]" + "░" * empty + "[/]"

        preview = ""
        if self._preview_clusters > 0:
            preview = f"  [#8892a0]~{self._preview_clusters:,} clusters[/]"

        return (
            f"[bold #d4a017]Threshold:[/] ◀ [bold]{self._value:.2f}[/] ▶  "
            f"{bar}{preview}  "
            f"[#8892a0][←/→ to adjust][/]"
        )

    def key_left(self) -> None:
        """Decrease threshold."""
        new_val = max(0.0, round(self._value - self._step, 2))
        if new_val != self._value:
            self._value = new_val
            self.refresh()
            self.post_message(self.ThresholdChanged(self._value))

    def key_right(self) -> None:
        """Increase threshold."""
        new_val = min(1.0, round(self._value + self._step, 2))
        if new_val != self._value:
            self._value = new_val
            self.refresh()
            self.post_message(self.ThresholdChanged(self._value))

    def set_preview(self, cluster_count: int) -> None:
        """Update the live preview cluster count."""
        self._preview_clusters = cluster_count
        self.refresh()
