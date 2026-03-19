"""Config tab -- matchkey builder with live feedback."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Label, Select, Static, Switch


# ── Available options ────────────────────────────────────────────────────────

COMPARISON_TYPES = [("Exact", "exact"), ("Weighted", "weighted")]

SCORER_OPTIONS = [
    ("jaro_winkler", "jaro_winkler"),
    ("levenshtein", "levenshtein"),
    ("exact", "exact"),
    ("token_sort", "token_sort"),
    ("soundex_match", "soundex_match"),
]

TRANSFORM_OPTIONS = [
    "lowercase",
    "uppercase",
    "strip",
    "strip_all",
    "soundex",
    "metaphone",
    "digits_only",
    "alpha_only",
    "normalize_whitespace",
]

STRATEGY_OPTIONS = [
    ("most_complete", "most_complete"),
    ("majority_vote", "majority_vote"),
    ("source_priority", "source_priority"),
    ("most_recent", "most_recent"),
    ("first_non_null", "first_non_null"),
]


# ── Field row widget ─────────────────────────────────────────────────────────


class FieldRow(Static):
    """A single field within a matchkey: column, transforms, scorer, weight."""

    def __init__(
        self,
        columns: list[str],
        show_scorer: bool = False,
        field_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._columns = columns
        self._show_scorer = show_scorer
        self._field_index = field_index

    def compose(self) -> ComposeResult:
        col_options = [(c, c) for c in self._columns]
        with Horizontal():
            yield Select(
                col_options,
                prompt="Column",
                id=f"field-col-{self._field_index}-{id(self)}",
            )
            yield Input(
                placeholder="transforms (comma-sep)",
                id=f"field-transforms-{self._field_index}-{id(self)}",
            )
            if self._show_scorer:
                yield Select(
                    SCORER_OPTIONS,
                    prompt="Scorer",
                    id=f"field-scorer-{self._field_index}-{id(self)}",
                )
                yield Input(
                    placeholder="weight",
                    value="1.0",
                    id=f"field-weight-{self._field_index}-{id(self)}",
                )
            yield Button(
                "X",
                variant="error",
                id=f"field-remove-{self._field_index}-{id(self)}",
            )


# ── Matchkey card widget ─────────────────────────────────────────────────────


class MatchkeyCard(Static):
    """UI card for a single matchkey definition."""

    DEFAULT_CSS = """
    MatchkeyCard {
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        columns: list[str],
        mk_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._columns = columns
        self._mk_index = mk_index
        self._field_counter = 0

    def compose(self) -> ComposeResult:
        yield Label(f"Matchkey #{self._mk_index + 1}")
        with Horizontal():
            yield Input(
                placeholder="matchkey name",
                value=f"mk_{self._mk_index + 1}",
                id=f"mk-name-{self._mk_index}",
            )
            yield Select(
                COMPARISON_TYPES,
                prompt="Type",
                value="exact",
                id=f"mk-type-{self._mk_index}",
            )
            yield Button(
                "Remove",
                variant="error",
                id=f"mk-remove-{self._mk_index}",
            )
        yield Vertical(id=f"mk-fields-{self._mk_index}")
        with Horizontal():
            yield Button(
                "+ Field",
                variant="primary",
                id=f"mk-add-field-{self._mk_index}",
            )
        # Threshold input -- only relevant for weighted
        yield Input(
            placeholder="threshold (e.g. 0.85)",
            value="0.85",
            id=f"mk-threshold-{self._mk_index}",
        )

    def on_mount(self) -> None:
        self._add_field_row()

    def _is_weighted(self) -> bool:
        try:
            sel = self.query_one(f"#mk-type-{self._mk_index}", Select)
            return sel.value == "weighted"
        except Exception:
            return False

    def _add_field_row(self) -> None:
        container = self.query_one(f"#mk-fields-{self._mk_index}", Vertical)
        row = FieldRow(
            columns=self._columns,
            show_scorer=self._is_weighted(),
            field_index=self._field_counter,
        )
        self._field_counter += 1
        container.mount(row)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith(f"mk-add-field-{self._mk_index}"):
            self._add_field_row()
            event.stop()
        elif btn_id.startswith("field-remove-"):
            # Remove the FieldRow ancestor of the button
            row = event.button.ancestors_with_self
            for ancestor in row:
                if isinstance(ancestor, FieldRow):
                    ancestor.remove()
                    break
            event.stop()

    def on_select_changed(self, event: Select.Changed) -> None:
        ctrl_id = str(event.control.id or "")
        if ctrl_id == f"mk-type-{self._mk_index}":
            # Rebuild fields when type changes
            container = self.query_one(f"#mk-fields-{self._mk_index}", Vertical)
            container.remove_children()
            self._field_counter = 0
            self._add_field_row()

    def get_config_dict(self) -> dict | None:
        """Extract matchkey config from current widget state."""
        try:
            name_input = self.query_one(f"#mk-name-{self._mk_index}", Input)
            type_select = self.query_one(f"#mk-type-{self._mk_index}", Select)
            name = name_input.value.strip() or f"mk_{self._mk_index + 1}"
            mk_type = type_select.value
            if mk_type is Select.BLANK:
                return None

            fields = []
            for field_row in self.query(FieldRow):
                col_sel = field_row.query(Select)
                col_val = None
                scorer_val = None
                for sel in col_sel:
                    sid = str(sel.id or "")
                    if "field-col" in sid and sel.value is not Select.BLANK:
                        col_val = sel.value
                    elif "field-scorer" in sid and sel.value is not Select.BLANK:
                        scorer_val = sel.value

                if col_val is None:
                    continue

                transforms_input = None
                weight_val = None
                for inp in field_row.query(Input):
                    iid = str(inp.id or "")
                    if "field-transforms" in iid:
                        transforms_input = inp.value.strip()
                    elif "field-weight" in iid:
                        weight_val = inp.value.strip()

                transforms = []
                if transforms_input:
                    transforms = [
                        t.strip() for t in transforms_input.split(",") if t.strip()
                    ]

                field_dict: dict = {"field": col_val, "transforms": transforms}
                if mk_type == "weighted":
                    field_dict["scorer"] = scorer_val or "jaro_winkler"
                    try:
                        field_dict["weight"] = float(weight_val or "1.0")
                    except ValueError:
                        field_dict["weight"] = 1.0
                fields.append(field_dict)

            if not fields:
                return None

            result: dict = {"name": name, "type": mk_type, "fields": fields}
            if mk_type == "weighted":
                threshold_input = self.query_one(
                    f"#mk-threshold-{self._mk_index}", Input
                )
                try:
                    result["threshold"] = float(threshold_input.value or "0.85")
                except ValueError:
                    result["threshold"] = 0.85
            return result
        except Exception:
            return None


# ── Main Config Tab ──────────────────────────────────────────────────────────


class ConfigTab(Static):
    """Matchkey builder with live feedback."""

    class ConfigChanged(Message):
        """Posted when config changes. Carries the built GoldenMatchConfig."""

        def __init__(self, config) -> None:
            self.config = config
            super().__init__()

    DEFAULT_CSS = """
    ConfigTab {
        height: 1fr;
    }
    #config-scroll {
        height: 1fr;
    }
    .config-section {
        margin-bottom: 1;
        padding: 1;
        border: solid $accent;
    }
    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mk_counter = 0
        self._columns: list[str] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="config-scroll"):
            # ── Matchkeys section ──
            with Vertical(classes="config-section"):
                yield Label("Matchkeys", classes="section-title")
                yield Vertical(id="matchkey-list")
                yield Button(
                    "+ Add Matchkey",
                    variant="success",
                    id="add-matchkey",
                )

            # ── Blocking section ──
            with Vertical(classes="config-section", id="blocking-section"):
                yield Label("Blocking Keys", classes="section-title")
                yield Input(
                    placeholder="blocking fields (comma-sep, e.g. zip,state)",
                    id="blocking-fields",
                )
                yield Select(
                    [
                        ("Static", "static"),
                        ("Adaptive", "adaptive"),
                        ("Sorted Neighborhood", "sorted_neighborhood"),
                        ("Multi-Pass", "multi_pass"),
                        ("ANN (Embeddings)", "ann"),
                        ("ANN Pairs", "ann_pairs"),
                        ("Canopy", "canopy"),
                    ],
                    prompt="Blocking Strategy",
                    value="static",
                    id="blocking-strategy",
                )

            # ── Golden rules section ──
            with Vertical(classes="config-section"):
                yield Label("Golden Rules", classes="section-title")
                yield Select(
                    STRATEGY_OPTIONS,
                    prompt="Default Strategy",
                    value="most_complete",
                    id="golden-strategy",
                )

            # ── Pipeline options ──
            with Vertical(classes="config-section"):
                yield Label("Pipeline Options", classes="section-title")
                with Horizontal():
                    yield Switch(value=True, id="sw-autofix")
                    yield Label("Auto-fix data quality")
                with Horizontal():
                    yield Switch(value=False, id="sw-autoblock")
                    yield Label("Auto-suggest blocking keys")
                with Horizontal():
                    yield Switch(value=False, id="sw-llm-boost")
                    yield Label("LLM Boost (requires API key)")
                with Horizontal():
                    yield Switch(value=False, id="sw-chunked")
                    yield Label("Large dataset mode (chunked)")
                with Horizontal():
                    yield Switch(value=False, id="sw-across-files")
                    yield Label("Match across files only")
                with Horizontal():
                    yield Switch(value=False, id="sw-anomalies")
                    yield Label("Anomaly detection")
                with Horizontal():
                    yield Switch(value=False, id="sw-preview")
                    yield Label("Merge preview (show changes before writing)")

            # ── Report options ──
            with Vertical(classes="config-section"):
                yield Label("Reports & Exports", classes="section-title")
                with Horizontal():
                    yield Switch(value=False, id="sw-html-report")
                    yield Label("HTML Report")
                with Horizontal():
                    yield Switch(value=False, id="sw-dashboard")
                    yield Label("Before/After Dashboard")
                with Horizontal():
                    yield Switch(value=False, id="sw-diff")
                    yield Label("CSV Diff (before/after changes)")
                with Horizontal():
                    yield Switch(value=False, id="sw-diff-html")
                    yield Label("HTML Diff (with highlighting)")
                with Horizontal():
                    yield Switch(value=False, id="sw-graph")
                    yield Label("Cluster Graph")

            # ── Apply button ──
            yield Button(
                "Apply Config",
                variant="primary",
                id="apply-config",
            )

    def set_columns(self, columns: list[str]) -> None:
        """Update available columns (called after engine loads data)."""
        self._columns = columns

    def on_mount(self) -> None:
        # Try to grab columns from engine if available
        app = self.app
        if hasattr(app, "engine") and app.engine is not None:
            self._columns = app.engine.columns

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "add-matchkey":
            self._add_matchkey()
            event.stop()
        elif btn_id.startswith("mk-remove-"):
            # Remove the matchkey card
            card = event.button.ancestors_with_self
            for ancestor in card:
                if isinstance(ancestor, MatchkeyCard):
                    ancestor.remove()
                    break
            event.stop()
        elif btn_id == "apply-config":
            self._build_and_post_config()
            event.stop()

    def _add_matchkey(self) -> None:
        # Re-fetch columns in case engine loaded after mount
        app = self.app
        if hasattr(app, "engine") and app.engine is not None:
            self._columns = app.engine.columns

        container = self.query_one("#matchkey-list", Vertical)
        card = MatchkeyCard(
            columns=self._columns,
            mk_index=self._mk_counter,
        )
        self._mk_counter += 1
        container.mount(card)

    def _build_and_post_config(self) -> None:
        """Build a GoldenMatchConfig from the current UI state and post it."""
        from goldenmatch.config.schemas import (
            BlockingConfig,
            BlockingKeyConfig,
            GoldenMatchConfig,
            GoldenRulesConfig,
            MatchkeyConfig,
            MatchkeyField,
        )

        matchkey_configs = []
        for card in self.query(MatchkeyCard):
            mk_dict = card.get_config_dict()
            if mk_dict is None:
                continue
            try:
                fields = [MatchkeyField(**f) for f in mk_dict.pop("fields")]
                mk = MatchkeyConfig(fields=fields, **mk_dict)
                matchkey_configs.append(mk)
            except Exception:
                continue

        if not matchkey_configs:
            self.notify("Add at least one valid matchkey.", severity="warning")
            return

        # ── Blocking ──
        blocking = None
        has_weighted = any(mk.type == "weighted" for mk in matchkey_configs)
        if has_weighted:
            blocking_input = self.query_one("#blocking-fields", Input)
            blocking_fields_str = blocking_input.value.strip()
            if blocking_fields_str:
                blocking_fields = [
                    f.strip() for f in blocking_fields_str.split(",") if f.strip()
                ]
                if blocking_fields:
                    strategy_sel = self.query_one("#blocking-strategy", Select)
                    strategy = strategy_sel.value if strategy_sel.value is not Select.BLANK else "static"
                    blocking = BlockingConfig(
                        keys=[BlockingKeyConfig(fields=blocking_fields)],
                        strategy=strategy,
                    )

            if blocking is None:
                self.notify(
                    "Weighted matchkeys require blocking fields.",
                    severity="warning",
                )
                return

        # ── Golden rules ──
        strategy_select = self.query_one("#golden-strategy", Select)
        strategy = strategy_select.value
        if strategy is Select.BLANK:
            strategy = "most_complete"
        golden_rules = GoldenRulesConfig(default_strategy=strategy)

        try:
            config = GoldenMatchConfig(
                matchkeys=matchkey_configs,
                blocking=blocking,
                golden_rules=golden_rules,
            )
        except Exception as e:
            self.notify(f"Config error: {e}", severity="error")
            return

        self.post_message(self.ConfigChanged(config))
