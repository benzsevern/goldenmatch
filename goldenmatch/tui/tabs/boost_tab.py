"""Boost tab -- active learning with human-in-the-loop pair labeling."""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, DataTable, Static

import polars as pl

from goldenmatch.tui.engine import EngineResult

logger = logging.getLogger(__name__)


class BoostTab(Static):
    """Active learning: label borderline pairs to boost accuracy."""

    class BoostComplete(Message):
        """Posted when boost re-scoring is done."""

        def __init__(self, scored_pairs: list[tuple[int, int, float]]) -> None:
            super().__init__()
            self.scored_pairs = scored_pairs

    DEFAULT_CSS = """
    BoostTab {
        height: 1fr;
    }
    #boost-header {
        height: 3;
        padding: 0 2;
        color: #d4a017;
    }
    #boost-stats {
        height: 2;
        padding: 0 2;
        color: #8892a0;
    }
    #pair-display {
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
    }
    #pair-record-a {
        height: 45%;
        border: solid #2ecc71 40%;
    }
    #pair-record-b {
        height: 45%;
        border: solid #e74c3c 40%;
    }
    #pair-score-bar {
        height: 1;
        padding: 0 2;
        color: #d4a017;
    }
    #label-buttons {
        height: 3;
        align: center middle;
    }
    #label-buttons Button {
        margin: 0 2;
    }
    .boost-placeholder {
        padding: 2;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result: EngineResult | None = None
        self._data: pl.DataFrame | None = None
        self._display_cols: list[str] = []
        # Active learning state
        self._batch: list[int] = []  # indices into scored_pairs
        self._batch_pos: int = 0  # current position in batch
        self._labels: dict[int, bool] = {}  # pair_index -> True (match) / False (non-match)
        self._total_labeled: int = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "[bold #d4a017]Active Learning[/] -- Label borderline pairs to boost accuracy",
                id="boost-header",
            )
            yield Static(
                "[dim]Run matching first, then press Start to begin labeling.[/dim]",
                id="boost-stats",
            )
            yield Static(
                "[dim]Run matching from the Config tab to see results here.[/dim]",
                id="boost-placeholder",
                classes="boost-placeholder",
            )
            with Vertical(id="pair-display"):
                yield Static("", id="pair-score-bar")
                yield DataTable(id="pair-record-a")
                yield DataTable(id="pair-record-b")
            with Horizontal(id="label-buttons"):
                yield Button("Start Labeling", id="btn-start", variant="primary")
                yield Button("Match (y)", id="btn-match", variant="success")
                yield Button("Non-match (n)", id="btn-nonmatch", variant="error")
                yield Button("Skip (s)", id="btn-skip")
                yield Button("Apply Boost", id="btn-apply", variant="warning")

    def on_mount(self) -> None:
        # Hide pair display initially
        self.query_one("#pair-display").display = False
        self.query_one("#btn-match").display = False
        self.query_one("#btn-nonmatch").display = False
        self.query_one("#btn-skip").display = False
        self.query_one("#btn-apply").display = False

        # Setup tables
        for table_id in ("#pair-record-a", "#pair-record-b"):
            table = self.query_one(table_id, DataTable)
            table.cursor_type = "none"

    def update_results(self, result: EngineResult, data: pl.DataFrame) -> None:
        """Receive new matching results."""
        self._result = result
        self._data = data
        self._display_cols = [c for c in data.columns if not c.startswith("__")]
        self._labels.clear()
        self._batch.clear()
        self._batch_pos = 0
        self._total_labeled = 0

        n_pairs = len(result.scored_pairs)
        placeholder = self.query_one("#boost-placeholder")
        placeholder.display = False

        stats = self.query_one("#boost-stats")
        stats.update(
            f"[#8892a0]{n_pairs} scored pairs available. "
            f"Press [bold #d4a017]Start Labeling[/] to see the hardest borderline pairs.[/]"
        )

        # Reset UI
        self.query_one("#pair-display").display = False
        self.query_one("#btn-start").display = True
        self.query_one("#btn-match").display = False
        self.query_one("#btn-nonmatch").display = False
        self.query_one("#btn-skip").display = False
        self.query_one("#btn-apply").display = False

    def _select_borderline_batch(self, n: int = 10) -> list[int]:
        """Select the N most borderline pairs using active sampling.

        Prioritizes bottleneck pairs from low-confidence clusters -- these are
        the pairs where human labeling has the highest impact on accuracy.
        """
        if self._result is None:
            return []

        pairs = self._result.scored_pairs
        if not pairs:
            return []

        try:
            from goldenmatch.core.active_sampling import select_active_pairs
            indices = select_active_pairs(
                pairs,
                features=None,
                current_probs=None,
                labeled_indices=set(self._labels.keys()),
                n=n * 2,  # oversample, then re-rank
                strategy="combined",
            )
        except Exception as e:
            logger.warning("Active sampling failed, using score-based fallback: %s", e)
            import numpy as np
            scores = np.array([s for _, _, s in pairs])
            median = np.median(scores)
            dists = np.abs(scores - median)
            for idx in self._labels:
                dists[idx] = float("inf")
            indices = list(np.argsort(dists)[:n * 2])

        # Boost bottleneck pairs from low-confidence clusters to the front
        bottleneck_pairs = set()
        if self._result.clusters:
            for cinfo in self._result.clusters.values():
                bp = cinfo.get("bottleneck_pair")
                conf = cinfo.get("confidence", 1.0)
                if bp is not None and conf < 0.8:
                    bottleneck_pairs.add(bp)
                    bottleneck_pairs.add((bp[1], bp[0]))  # both orderings

        def _priority(idx: int) -> tuple:
            a, b, s = pairs[idx]
            is_bottleneck = (a, b) in bottleneck_pairs
            return (0 if is_bottleneck else 1, abs(s - 0.5))

        indices.sort(key=_priority)
        return indices[:n]

    def _show_current_pair(self) -> None:
        """Display the current pair for labeling."""
        if self._result is None or self._batch_pos >= len(self._batch):
            self._on_batch_complete()
            return

        pair_idx = self._batch[self._batch_pos]
        a, b, score = self._result.scored_pairs[pair_idx]

        # Update score bar
        score_bar = self.query_one("#pair-score-bar")
        progress = self._batch_pos + 1
        total = len(self._batch)
        score_bar.update(
            f"[bold #d4a017]Pair {progress}/{total}[/]  "
            f"Score: [bold]{score:.3f}[/]  "
            f"Labeled: [bold #2ecc71]{self._total_labeled}[/]  "
            f"| Is this a match?"
        )

        # Build row lookup
        row_a = self._data.filter(pl.col("__row_id__") == a)
        row_b = self._data.filter(pl.col("__row_id__") == b)

        # Populate tables
        for table_id, row_df, label in [
            ("#pair-record-a", row_a, "Record A"),
            ("#pair-record-b", row_b, "Record B"),
        ]:
            table = self.query_one(table_id, DataTable)
            table.clear(columns=True)
            table.add_column("Field")
            table.add_column(label)
            if row_df.height > 0:
                row = row_df.to_dicts()[0]
                for col in self._display_cols:
                    val = row.get(col, "")
                    table.add_row(col, str(val) if val is not None else "")

    def _on_batch_complete(self) -> None:
        """Called when all pairs in batch have been labeled."""
        self.query_one("#btn-match").display = False
        self.query_one("#btn-nonmatch").display = False
        self.query_one("#btn-skip").display = False

        score_bar = self.query_one("#pair-score-bar")
        if self._total_labeled > 0:
            score_bar.update(
                f"[bold #2ecc71]Batch complete![/]  "
                f"Labeled: [bold]{self._total_labeled}[/] pairs.  "
                f"Press [bold #d4a017]Apply Boost[/] to retrain, or [bold]Start Labeling[/] for more."
            )
            self.query_one("#btn-apply").display = True
        else:
            score_bar.update("[dim]No pairs labeled yet.[/dim]")

        self.query_one("#btn-start").display = True

    def _record_label(self, is_match: bool) -> None:
        """Record a label and advance to next pair."""
        if self._batch_pos < len(self._batch):
            pair_idx = self._batch[self._batch_pos]
            self._labels[pair_idx] = is_match
            self._total_labeled += 1
        self._batch_pos += 1
        self._show_current_pair()

    # ── Button handlers ──────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-start":
            self._start_labeling()
        elif event.button.id == "btn-match":
            self._record_label(True)
        elif event.button.id == "btn-nonmatch":
            self._record_label(False)
        elif event.button.id == "btn-skip":
            self._batch_pos += 1
            self._show_current_pair()
        elif event.button.id == "btn-apply":
            self._apply_boost()

    def _start_labeling(self) -> None:
        """Start a new labeling batch."""
        if self._result is None:
            return

        self._batch = self._select_borderline_batch(n=10)
        if not self._batch:
            stats = self.query_one("#boost-stats")
            stats.update("[yellow]No unlabeled pairs remaining.[/yellow]")
            return

        self._batch_pos = 0

        # Show pair display and label buttons
        self.query_one("#pair-display").display = True
        self.query_one("#btn-start").display = False
        self.query_one("#btn-match").display = True
        self.query_one("#btn-nonmatch").display = True
        self.query_one("#btn-skip").display = True
        self.query_one("#btn-apply").display = False

        self._show_current_pair()

    def _apply_boost(self) -> None:
        """Train classifier on collected labels and re-score pairs."""
        if self._result is None or not self._labels:
            return

        stats = self.query_one("#boost-stats")
        stats.update("[#d4a017]Training classifier on your labels...[/]")

        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from goldenmatch.core.boost import extract_feature_matrix

            pairs = self._result.scored_pairs
            columns = self._display_cols
            df = self._data

            # Extract features for labeled pairs
            labeled_indices = sorted(self._labels.keys())
            labeled_pairs = [pairs[i] for i in labeled_indices]
            y = np.array([1.0 if self._labels[i] else 0.0 for i in labeled_indices])

            X_labeled = extract_feature_matrix(labeled_pairs, df, columns)

            # Train logistic regression
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_labeled, y)

            # Score labeled set for feedback
            from sklearn.model_selection import cross_val_score
            if len(labeled_indices) >= 6:
                cv_f1 = cross_val_score(clf, X_labeled, y, cv=min(3, len(labeled_indices) // 2), scoring="f1").mean()
            else:
                cv_f1 = 0.0

            # Re-score ALL pairs
            X_all = extract_feature_matrix(pairs, df, columns)
            probs = clf.predict_proba(X_all)[:, 1]

            # Build new scored pairs with classifier probabilities
            new_pairs = [
                (a, b, float(prob))
                for (a, b, _), prob in zip(pairs, probs)
            ]

            stats.update(
                f"[bold #2ecc71]Boost applied![/]  "
                f"F1 (cross-val): [bold]{cv_f1:.1%}[/]  "
                f"Labels used: [bold]{len(labeled_indices)}[/]  "
                f"Re-scored [bold]{len(new_pairs)}[/] pairs."
            )

            # Post message so app can update results
            self.post_message(self.BoostComplete(new_pairs))

        except ImportError:
            stats.update("[red]scikit-learn required for boosting. pip install scikit-learn[/red]")
        except Exception as e:
            stats.update(f"[red]Boost error: {e}[/red]")
            logger.exception("Boost error")

    # ── Keyboard shortcuts ──────────────────────────────────────

    def key_y(self) -> None:
        """Label as match."""
        if self.query_one("#btn-match").display:
            self._record_label(True)

    def key_n(self) -> None:
        """Label as non-match."""
        if self.query_one("#btn-nonmatch").display:
            self._record_label(False)

    def key_s(self) -> None:
        """Skip current pair."""
        if self.query_one("#btn-skip").display:
            self._batch_pos += 1
            self._show_current_pair()
