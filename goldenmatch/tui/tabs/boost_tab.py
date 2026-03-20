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
        """Select N pairs stratified across the score range.

        Guarantees both high-score (likely match) and low-score (likely non-match)
        pairs so the classifier sees both classes. Prioritizes bottleneck pairs
        from low-confidence clusters.
        """
        if self._result is None:
            return []

        pairs = self._result.scored_pairs
        if not pairs:
            return []

        import numpy as np

        scores = np.array([s for _, _, s in pairs])
        labeled = set(self._labels.keys())

        # Stratified sampling across score buckets
        # Heavy weight on the decision boundary (0.7-0.95) where matches and non-matches mix
        buckets = [
            (0.95, 1.01, 2),   # very high -- likely matches, need some
            (0.90, 0.95, 2),   # high -- mix of match/non-match
            (0.85, 0.90, 2),   # borderline high
            (0.80, 0.85, 2),   # borderline
            (0.70, 0.80, 1),   # lower borderline
            (0.00, 0.70, 1),   # low -- likely non-matches, need some
        ]

        selected = []
        for lo, hi, count in buckets:
            bucket_idx = [i for i in range(len(scores)) if lo <= scores[i] < hi and i not in labeled]
            if bucket_idx:
                np.random.seed(42 + len(selected))
                chosen = list(np.random.choice(bucket_idx, size=min(count, len(bucket_idx)), replace=False))
                selected.extend(chosen)

        # Boost bottleneck pairs from low-confidence clusters
        bottleneck_pairs = set()
        if self._result.clusters:
            for cinfo in self._result.clusters.values():
                bp = cinfo.get("bottleneck_pair")
                conf = cinfo.get("confidence", 1.0)
                if bp is not None and conf < 0.8:
                    bottleneck_pairs.add(bp)
                    bottleneck_pairs.add((bp[1], bp[0]))

        def _priority(idx: int) -> tuple:
            a, b, s = pairs[idx]
            is_bottleneck = (a, b) in bottleneck_pairs
            return (0 if is_bottleneck else 1, -s)  # bottleneck first, then by score desc

        selected.sort(key=_priority)
        return selected[:n]

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
        """Train classifier on collected labels and rerank pairs.

        Uses the classifier to adjust scores near the decision boundary
        rather than replacing all scores. High-confidence matches and
        non-matches keep their original scores; only borderline pairs
        get the classifier's opinion blended in.
        """
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

            # Check we have both classes
            labeled_indices = sorted(self._labels.keys())
            y = np.array([1.0 if self._labels[i] else 0.0 for i in labeled_indices])
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos

            if n_pos == 0 or n_neg == 0:
                stats.update(
                    f"[yellow]Need labels from both classes. "
                    f"Got {n_pos} matches, {n_neg} non-matches. "
                    f"Label more pairs with both y and n.[/yellow]"
                )
                return

            labeled_pairs = [pairs[i] for i in labeled_indices]
            X_labeled = extract_feature_matrix(labeled_pairs, df, columns)

            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_labeled, y)

            # Re-score ALL pairs with reranking blend
            X_all = extract_feature_matrix(pairs, df, columns)
            probs = clf.predict_proba(X_all)[:, 1]

            # Blend: original score weighted by distance from decision boundary
            # Near the boundary (prob ~0.5): trust classifier more
            # Far from boundary (prob ~0 or ~1): trust original more
            new_pairs = []
            boosted = 0
            for i, (a, b, original_score) in enumerate(pairs):
                clf_prob = float(probs[i])
                # How confident is the classifier? 0 = uncertain, 1 = certain
                clf_confidence = abs(clf_prob - 0.5) * 2.0
                # Blend weight: trust classifier more when it's confident
                # and the original score is in the ambiguous range
                alpha = min(clf_confidence, 0.7)  # cap at 70% classifier influence
                blended = (1 - alpha) * original_score + alpha * clf_prob
                new_pairs.append((a, b, blended))
                if abs(blended - original_score) > 0.05:
                    boosted += 1

            # Quality check: does reranking help on the labeled pairs?
            # Count how many labeled pairs moved in the right direction
            improved = 0
            degraded = 0
            for idx in labeled_indices:
                a, b, orig = pairs[idx]
                blended_score = new_pairs[idx][2]
                is_match = self._labels[idx]
                if is_match:
                    # Match: higher score = better
                    if blended_score > orig:
                        improved += 1
                    elif blended_score < orig:
                        degraded += 1
                else:
                    # Non-match: lower score = better
                    if blended_score < orig:
                        improved += 1
                    elif blended_score > orig:
                        degraded += 1

            if degraded > improved and self._total_labeled >= 10:
                stats.update(
                    f"[bold yellow]Reranking may be hurting accuracy.[/]  "
                    f"Improved {improved}, degraded {degraded} of {len(labeled_indices)} labeled pairs.  "
                    f"Your data may need [bold]embedding fine-tuning[/] (LLM boost Level 2) "
                    f"rather than feature reranking. Try: [bold]goldenmatch dedupe --llm-boost[/]"
                )
            else:
                stats.update(
                    f"[bold #2ecc71]Boost applied![/]  "
                    f"Labels: [bold]{n_pos}[/] match + [bold]{n_neg}[/] non-match  "
                    f"Adjusted [bold]{boosted}[/] of {len(new_pairs)} pairs.  "
                    f"Quality: {improved} improved, {degraded} degraded."
                )

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
