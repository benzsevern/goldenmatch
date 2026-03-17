# Engine + Preview Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MatchEngine (shared foundation for TUI and preview) and add `--preview` mode to the CLI for dry-run matching with Rich terminal output.

**Architecture:** MatchEngine wraps the existing pipeline into a clean API with sample extraction, scored-pairs caching, and re-clustering. Preview mode uses the engine to run on a sample and formats results with Rich tables. No Textual dependency.

**Tech Stack:** Python 3.11+, Polars, Rich, existing goldenmatch pipeline modules

**Spec:** `docs/superpowers/specs/2026-03-17-interactive-tui-design.md`

---

## Chunk 1: MatchEngine

### Task 1: EngineResult and EngineStats Dataclasses

**Files:**
- Create: `goldenmatch/tui/__init__.py`
- Create: `goldenmatch/tui/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests for dataclasses**

```python
# tests/test_engine.py
import pytest
import polars as pl
from goldenmatch.tui.engine import EngineResult, EngineStats


class TestEngineStats:
    def test_create(self):
        stats = EngineStats(
            total_records=1000,
            total_clusters=50,
            singleton_count=900,
            match_rate=0.05,
            cluster_sizes=[2, 3, 2, 5],
            avg_cluster_size=3.0,
            max_cluster_size=5,
            oversized_count=0,
        )
        assert stats.total_records == 1000
        assert stats.hit_rate is None

    def test_match_mode_stats(self):
        stats = EngineStats(
            total_records=500,
            total_clusters=0,
            singleton_count=0,
            match_rate=0.0,
            cluster_sizes=[],
            avg_cluster_size=0.0,
            max_cluster_size=0,
            oversized_count=0,
            hit_rate=0.7,
            avg_score=0.88,
        )
        assert stats.hit_rate == 0.7


class TestEngineResult:
    def test_create_dedupe(self):
        result = EngineResult(
            clusters={1: {"members": [0, 1], "size": 2, "oversized": False, "pair_scores": {}}},
            golden=None,
            unique=None,
            dupes=None,
            quarantine=None,
            matched=None,
            unmatched=None,
            scored_pairs=[(0, 1, 0.95)],
            stats=EngineStats(
                total_records=5, total_clusters=1, singleton_count=3,
                match_rate=0.2, cluster_sizes=[2], avg_cluster_size=2.0,
                max_cluster_size=2, oversized_count=0,
            ),
        )
        assert len(result.scored_pairs) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_engine.py -v`
Expected: FAIL — cannot import

- [ ] **Step 3: Implement dataclasses**

```python
# goldenmatch/tui/__init__.py
```

```python
# goldenmatch/tui/engine.py
"""MatchEngine — shared foundation for TUI and preview mode.

Wraps the existing pipeline modules into a clean API with sample
extraction, scored-pairs caching, and threshold re-clustering.
No Textual dependency — pure Python + Polars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig


@dataclass
class EngineStats:
    total_records: int
    total_clusters: int
    singleton_count: int
    match_rate: float
    cluster_sizes: list[int]
    avg_cluster_size: float
    max_cluster_size: int
    oversized_count: int
    hit_rate: float | None = None
    avg_score: float | None = None


@dataclass
class EngineResult:
    clusters: dict[int, dict]
    golden: pl.DataFrame | None
    unique: pl.DataFrame | None
    dupes: pl.DataFrame | None
    quarantine: pl.DataFrame | None
    matched: pl.DataFrame | None
    unmatched: pl.DataFrame | None
    scored_pairs: list[tuple[int, int, float]]
    stats: EngineStats
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/tui/ tests/test_engine.py
git commit -m "feat: EngineResult and EngineStats dataclasses"
```

---

### Task 2: MatchEngine — File Loading and Profiling

**Files:**
- Modify: `goldenmatch/tui/engine.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine.py`:

```python
from goldenmatch.tui.engine import MatchEngine


class TestMatchEngineLoad:
    def test_load_single_file(self, sample_csv):
        engine = MatchEngine([sample_csv])
        assert engine.row_count == 5
        assert "email" in engine.columns
        assert engine.profile is not None
        assert engine.profile["total_rows"] == 5

    def test_load_multiple_files(self, sample_csv, sample_csv_b):
        engine = MatchEngine([sample_csv, sample_csv_b])
        assert engine.row_count == 8

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MatchEngine([tmp_path / "missing.csv"])

    def test_columns_property(self, sample_csv):
        engine = MatchEngine([sample_csv])
        cols = engine.columns
        assert "first_name" in cols
        assert "last_name" in cols
        # Internal columns should not appear
        assert "__source__" not in cols
        assert "__row_id__" not in cols

    def test_sample_extraction(self, sample_csv):
        engine = MatchEngine([sample_csv])
        sample = engine.get_sample(3)
        assert isinstance(sample, pl.DataFrame)
        assert sample.height == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_engine.py::TestMatchEngineLoad -v`
Expected: FAIL

- [ ] **Step 3: Implement MatchEngine init, properties, get_sample**

Add to `goldenmatch/tui/engine.py`:

```python
from goldenmatch.core.ingest import load_file
from goldenmatch.core.profiler import profile_dataframe


class MatchEngine:
    """Wraps the pipeline into a clean API for the TUI and preview mode."""

    def __init__(self, files: list[Path | str]):
        self._files = [Path(f) for f in files]
        self._data: pl.DataFrame | None = None
        self._profile: dict | None = None
        self._last_result: EngineResult | None = None
        self._load()

    def _load(self) -> None:
        frames = []
        for f in self._files:
            lf = load_file(f)
            lf = lf.with_columns(pl.lit(f.stem).alias("__source__"))
            frames.append(lf.collect())
        combined = pl.concat(frames)
        # Add row IDs
        combined = combined.with_row_index("__row_id__").with_columns(
            pl.col("__row_id__").cast(pl.Int64)
        )
        self._data = combined
        # Profile without internal columns
        profile_cols = [c for c in combined.columns if not c.startswith("__")]
        self._profile = profile_dataframe(combined.select(profile_cols))

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def profile(self) -> dict:
        return self._profile

    @property
    def columns(self) -> list[str]:
        return [c for c in self._data.columns if not c.startswith("__")]

    @property
    def row_count(self) -> int:
        return self._data.height

    def get_sample(self, n: int) -> pl.DataFrame:
        if n >= self._data.height:
            return self._data
        return self._data.head(n)
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/tui/engine.py tests/test_engine.py
git commit -m "feat: MatchEngine file loading, profiling, and sampling"
```

---

### Task 3: MatchEngine — run_sample and run_full

**Files:**
- Modify: `goldenmatch/tui/engine.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_engine.py`:

```python
from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    OutputConfig, GoldenRulesConfig, GoldenFieldRule,
)


@pytest.fixture
def exact_email_config(tmp_path):
    return GoldenMatchConfig(
        matchkeys=[
            MatchkeyConfig(
                name="email_key",
                fields=[MatchkeyField(column="email", transforms=["lowercase"])],
                comparison="exact",
            )
        ],
        output=OutputConfig(format="csv", directory=str(tmp_path), run_name="test"),
        golden_rules=GoldenRulesConfig(
            default=GoldenFieldRule(strategy="most_complete"),
        ),
    )


class TestMatchEngineRunSample:
    def test_run_sample_dedupe(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_sample(exact_email_config, sample_size=5)
        assert isinstance(result, EngineResult)
        assert result.stats.total_records == 5
        assert result.stats.total_clusters >= 1
        assert len(result.scored_pairs) >= 1

    def test_run_sample_small(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_sample(exact_email_config, sample_size=3)
        assert result.stats.total_records == 3

    def test_scored_pairs_cached(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_sample(exact_email_config)
        assert engine._last_result is not None
        assert engine._last_result.scored_pairs == result.scored_pairs


class TestMatchEngineRecluster:
    def test_recluster_at_threshold(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        engine.run_sample(exact_email_config)
        # Exact matches always have score 1.0, so any threshold <= 1.0 gives same result
        stats = engine.recluster_at_threshold(1.0)
        assert isinstance(stats, EngineStats)
        assert stats.total_records > 0


class TestMatchEngineRunFull:
    def test_run_full(self, sample_csv, exact_email_config):
        engine = MatchEngine([sample_csv])
        result = engine.run_full(exact_email_config)
        assert result.stats.total_records == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_engine.py::TestMatchEngineRunSample -v`
Expected: FAIL

- [ ] **Step 3: Implement run_sample, run_full, recluster_at_threshold**

Add to MatchEngine class in `goldenmatch/tui/engine.py`:

```python
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.core.cluster import build_clusters
from goldenmatch.core.golden import build_golden_record
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.validate import ValidationRule, validate_dataframe

    def _compute_stats(self, clusters: dict, total_records: int) -> EngineStats:
        multi = {k: v for k, v in clusters.items() if v["size"] > 1}
        singletons = {k: v for k, v in clusters.items() if v["size"] == 1}
        cluster_sizes = [v["size"] for v in multi.values()]
        return EngineStats(
            total_records=total_records,
            total_clusters=len(multi),
            singleton_count=len(singletons),
            match_rate=len(multi) / total_records if total_records else 0.0,
            cluster_sizes=cluster_sizes,
            avg_cluster_size=sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
            max_cluster_size=max(cluster_sizes) if cluster_sizes else 0,
            oversized_count=sum(1 for v in multi.values() if v["oversized"]),
        )

    def _run_pipeline(self, df: pl.DataFrame, config: GoldenMatchConfig) -> EngineResult:
        """Core pipeline logic shared by run_sample and run_full."""
        matchkeys = config.get_matchkeys()

        # Auto-fix
        if config.validation and config.validation.auto_fix:
            df, _ = auto_fix_dataframe(df)

        # Validation
        quarantine = None
        if config.validation and config.validation.rules:
            rules = [
                ValidationRule(column=r.column, rule_type=r.rule_type, params=r.params, action=r.action)
                for r in config.validation.rules
            ]
            df, quarantine, _ = validate_dataframe(df, rules)

        # Standardization
        lf = df.lazy()
        if config.standardization and config.standardization.rules:
            lf = apply_standardization(lf, config.standardization.rules)

        # Matchkeys
        lf = compute_matchkeys(lf, matchkeys)
        df = lf.collect()

        # Score all pairs (cache ALL pairs before threshold for re-clustering)
        all_scored_pairs: list[tuple[int, int, float]] = []

        for mk in matchkeys:
            if mk.type == "exact":
                pairs = find_exact_matches(df.lazy(), mk)
                all_scored_pairs.extend(pairs)
            elif mk.type == "weighted" and config.blocking:
                blocks = build_blocks(df.lazy(), config.blocking)
                for block in blocks:
                    pairs = find_fuzzy_matches(block.df.collect(), mk)
                    all_scored_pairs.extend(pairs)

        # Cluster
        all_ids = df["__row_id__"].to_list()
        max_cs = 100
        if config.golden_rules and hasattr(config.golden_rules, "max_cluster_size"):
            max_cs = config.golden_rules.max_cluster_size
        clusters = build_clusters(all_scored_pairs, all_ids, max_cluster_size=max_cs)

        # Golden records
        golden_rules = config.golden_rules
        golden_rows = []
        if golden_rules:
            for cid, cinfo in clusters.items():
                if cinfo["size"] > 1 and not cinfo["oversized"]:
                    cluster_df = df.filter(pl.col("__row_id__").is_in(cinfo["members"]))
                    golden = build_golden_record(cluster_df, golden_rules)
                    row = {"__cluster_id__": cid}
                    row["__golden_confidence__"] = golden.get("__golden_confidence__", 0.0)
                    for col, val_info in golden.items():
                        if col in ("__cluster_id__", "__golden_confidence__"):
                            continue
                        if isinstance(val_info, dict) and "value" in val_info:
                            row[col] = val_info["value"]
                    golden_rows.append(row)
        golden_df = pl.DataFrame(golden_rows) if golden_rows else None

        # Classify
        dupe_ids = set()
        for cinfo in clusters.values():
            if cinfo["size"] > 1:
                dupe_ids.update(cinfo["members"])
        unique_ids = set(all_ids) - dupe_ids

        stats = self._compute_stats(clusters, len(df))

        return EngineResult(
            clusters=clusters,
            golden=golden_df,
            unique=df.filter(pl.col("__row_id__").is_in(list(unique_ids))),
            dupes=df.filter(pl.col("__row_id__").is_in(list(dupe_ids))),
            quarantine=quarantine,
            matched=None,
            unmatched=None,
            scored_pairs=all_scored_pairs,
            stats=stats,
        )

    def run_sample(self, config: GoldenMatchConfig, sample_size: int = 1000) -> EngineResult:
        sample = self.get_sample(sample_size)
        result = self._run_pipeline(sample, config)
        self._last_result = result
        return result

    def run_full(self, config: GoldenMatchConfig) -> EngineResult:
        result = self._run_pipeline(self._data, config)
        self._last_result = result
        return result

    def recluster_at_threshold(self, threshold: float) -> EngineStats:
        if self._last_result is None:
            raise RuntimeError("No previous run to recluster. Call run_sample first.")
        filtered = [(a, b, s) for a, b, s in self._last_result.scored_pairs if s >= threshold]
        all_ids = self._data["__row_id__"].to_list()[:self._last_result.stats.total_records]
        clusters = build_clusters(filtered, all_ids)
        return self._compute_stats(clusters, self._last_result.stats.total_records)
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `cd D:/show_case/goldenmatch && pytest --tb=short`
Expected: All PASS (331+ tests)

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/tui/engine.py tests/test_engine.py
git commit -m "feat: MatchEngine run_sample, run_full, and recluster_at_threshold"
```

---

## Chunk 2: Preview Mode

### Task 4: Preview Formatter

**Files:**
- Create: `goldenmatch/core/preview.py`
- Create: `tests/test_preview.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preview.py
import pytest
import polars as pl
from goldenmatch.core.preview import (
    format_preview_clusters,
    format_preview_golden,
    format_preview_stats,
    format_score_histogram,
)
from goldenmatch.tui.engine import EngineResult, EngineStats


@pytest.fixture
def sample_engine_result():
    clusters = {
        1: {"members": [0, 1], "size": 2, "oversized": False, "pair_scores": {(0, 1): 1.0}},
        2: {"members": [2, 3, 4], "size": 3, "oversized": False, "pair_scores": {(2, 3): 0.9, (3, 4): 0.85}},
        3: {"members": [5], "size": 1, "oversized": False, "pair_scores": {}},
    }
    golden = pl.DataFrame({
        "__cluster_id__": [1, 2],
        "__golden_confidence__": [1.0, 0.85],
        "name": ["John Smith", "Jane Doe"],
        "email": ["john@test.com", "jane@test.com"],
    })
    stats = EngineStats(
        total_records=6, total_clusters=2, singleton_count=1,
        match_rate=0.33, cluster_sizes=[2, 3],
        avg_cluster_size=2.5, max_cluster_size=3, oversized_count=0,
    )
    return EngineResult(
        clusters=clusters, golden=golden, unique=None, dupes=None,
        quarantine=None, matched=None, unmatched=None,
        scored_pairs=[(0, 1, 1.0), (2, 3, 0.9), (3, 4, 0.85)],
        stats=stats,
    )


class TestFormatPreviewStats:
    def test_returns_string(self, sample_engine_result):
        output = format_preview_stats(sample_engine_result.stats)
        assert isinstance(output, str)
        assert "6" in output  # total records
        assert "2" in output  # clusters


class TestFormatPreviewClusters:
    def test_returns_string(self, sample_engine_result):
        data = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3, 4, 5],
            "name": ["John", "John", "Jane", "Jane", "J Doe", "Bob"],
            "email": ["j@t.com", "j@t.com", "ja@t.com", "ja@t.com", "jd@t.com", "b@t.com"],
            "__source__": ["a"] * 6,
        })
        output = format_preview_clusters(sample_engine_result.clusters, data, max_clusters=10)
        assert isinstance(output, str)
        assert "Cluster" in output or "cluster" in output


class TestFormatPreviewGolden:
    def test_returns_string(self, sample_engine_result):
        output = format_preview_golden(sample_engine_result.golden, max_records=10)
        assert isinstance(output, str)
        assert "John Smith" in output or "john" in output.lower()


class TestFormatScoreHistogram:
    def test_returns_string(self):
        scores = [0.85, 0.9, 0.92, 0.95, 1.0, 1.0, 0.88, 0.91]
        output = format_score_histogram(scores)
        assert isinstance(output, str)
        assert len(output) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_preview.py -v`
Expected: FAIL

- [ ] **Step 3: Implement preview formatter**

```python
# goldenmatch/core/preview.py
"""Preview mode formatter — Rich terminal output for sample matching results.

No Textual dependency. Uses Rich tables, panels, and text formatting.
"""
from __future__ import annotations

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from goldenmatch.tui.engine import EngineResult, EngineStats


def format_preview_stats(stats: EngineStats) -> str:
    console = Console(record=True, width=100)
    table = Table(title="Preview Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Records", str(stats.total_records))
    table.add_row("Clusters (with duplicates)", str(stats.total_clusters))
    table.add_row("Singletons (unique)", str(stats.singleton_count))
    table.add_row("Match Rate", f"{stats.match_rate:.1%}")

    if stats.cluster_sizes:
        table.add_row("Avg Cluster Size", f"{stats.avg_cluster_size:.1f}")
        table.add_row("Max Cluster Size", str(stats.max_cluster_size))
    if stats.oversized_count:
        table.add_row("Oversized Clusters", str(stats.oversized_count))

    if stats.hit_rate is not None:
        table.add_row("Hit Rate", f"{stats.hit_rate:.1%}")
    if stats.avg_score is not None:
        table.add_row("Avg Match Score", f"{stats.avg_score:.3f}")

    console.print(table)
    return console.export_text()


def format_preview_clusters(
    clusters: dict[int, dict],
    data: pl.DataFrame,
    max_clusters: int = 10,
) -> str:
    console = Console(record=True, width=120)
    multi = {k: v for k, v in clusters.items() if v["size"] > 1}
    sorted_clusters = sorted(multi.items(), key=lambda x: x[1]["size"], reverse=True)[:max_clusters]

    if not sorted_clusters:
        console.print("[dim]No clusters found.[/dim]")
        return console.export_text()

    display_cols = [c for c in data.columns if not c.startswith("__")][:6]

    for cid, cinfo in sorted_clusters:
        member_df = data.filter(pl.col("__row_id__").is_in(cinfo["members"]))
        table = Table(title=f"Cluster {cid} ({cinfo['size']} records)", show_header=True)
        for col in display_cols:
            table.add_column(col)
        for row in member_df.select(display_cols).to_dicts():
            table.add_row(*[str(v) if v is not None else "" for v in row.values()])
        console.print(table)
        console.print()

    return console.export_text()


def format_preview_golden(
    golden: pl.DataFrame | None,
    max_records: int = 10,
) -> str:
    console = Console(record=True, width=120)

    if golden is None or golden.height == 0:
        console.print("[dim]No golden records generated.[/dim]")
        return console.export_text()

    display = golden.head(max_records)
    cols = [c for c in display.columns if c != "__cluster_id__"]

    table = Table(title=f"Golden Records (top {min(max_records, golden.height)})", show_header=True)
    table.add_column("Cluster", style="cyan")
    for col in cols:
        style = "green" if col == "__golden_confidence__" else None
        table.add_column(col, style=style)

    for row in display.to_dicts():
        values = [str(row.get("__cluster_id__", ""))]
        for col in cols:
            val = row.get(col)
            if col == "__golden_confidence__" and val is not None:
                values.append(f"{val:.2f}")
            else:
                values.append(str(val) if val is not None else "")
        table.add_row(*values)

    console.print(table)
    return console.export_text()


def format_score_histogram(scores: list[float], bins: int = 10) -> str:
    console = Console(record=True, width=80)

    if not scores:
        console.print("[dim]No scores to display.[/dim]")
        return console.export_text()

    min_s = min(scores)
    max_s = max(scores)
    if min_s == max_s:
        console.print(f"All scores: {min_s:.3f}")
        return console.export_text()

    bin_width = (max_s - min_s) / bins
    bin_counts = [0] * bins
    for s in scores:
        idx = min(int((s - min_s) / bin_width), bins - 1)
        bin_counts[idx] += 1

    max_count = max(bin_counts) if bin_counts else 1
    bar_width = 40

    console.print("[bold]Score Distribution[/bold]")
    for i, count in enumerate(bin_counts):
        low = min_s + i * bin_width
        high = low + bin_width
        bar_len = int((count / max_count) * bar_width) if max_count else 0
        bar = "█" * bar_len
        console.print(f"  {low:.2f}-{high:.2f} │ {bar} {count}")

    return console.export_text()
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_preview.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/preview.py tests/test_preview.py
git commit -m "feat: preview formatter with Rich tables for stats, clusters, golden, histogram"
```

---

### Task 5: CLI --preview Flag for Dedupe

**Files:**
- Modify: `goldenmatch/cli/dedupe.py`
- Create: `tests/test_cli_preview.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli_preview.py
import pytest
import yaml
from typer.testing import CliRunner
from goldenmatch.cli.main import app

runner = CliRunner()


@pytest.fixture
def preview_config(tmp_path, sample_csv):
    cfg_path = tmp_path / "goldenmatch.yaml"
    cfg_path.write_text(yaml.dump({
        "matchkeys": [{
            "name": "email_key",
            "fields": [{"column": "email", "transforms": ["lowercase"]}],
            "comparison": "exact",
        }],
        "output": {"format": "csv", "directory": str(tmp_path), "run_name": "preview_test"},
    }))
    return cfg_path


class TestDedupePreview:
    def test_preview_flag(self, sample_csv, preview_config):
        result = runner.invoke(app, [
            "dedupe", str(sample_csv),
            "--config", str(preview_config),
            "--preview",
        ], input="N\n")
        assert result.exit_code == 0
        assert "Preview Summary" in result.output or "Total Records" in result.output

    def test_preview_size(self, sample_csv, preview_config):
        result = runner.invoke(app, [
            "dedupe", str(sample_csv),
            "--config", str(preview_config),
            "--preview", "--preview-size", "3",
        ], input="N\n")
        assert result.exit_code == 0

    def test_no_preview_writes_files(self, sample_csv, preview_config, tmp_path):
        result = runner.invoke(app, [
            "dedupe", str(sample_csv),
            "--config", str(preview_config),
            "--output-report",
        ])
        assert result.exit_code == 0
        # Should have written files (non-preview mode)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cli_preview.py -v`
Expected: FAIL

- [ ] **Step 3: Add --preview flags to dedupe CLI**

Read and modify `goldenmatch/cli/dedupe.py`:
- Add `preview: bool = typer.Option(False, "--preview", help="Preview results without writing files")`
- Add `preview_size: int = typer.Option(10000, "--preview-size", help="Number of records to sample for preview")`
- Add `preview_random: bool = typer.Option(False, "--preview-random", help="Use random sampling instead of first N")`
- When `preview=True`: use MatchEngine, run_sample, print formatted results, prompt "Run full job now? [y/N]"

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cli_preview.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/cli/dedupe.py tests/test_cli_preview.py
git commit -m "feat: --preview flag for dedupe CLI with sample matching"
```

---

### Task 6: CLI --preview Flag for Match

**Files:**
- Modify: `goldenmatch/cli/match.py`
- Modify: `tests/test_cli_preview.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_cli_preview.py`:

```python
class TestMatchPreview:
    def test_preview_flag(self, sample_csv, sample_csv_b, preview_config):
        result = runner.invoke(app, [
            "match", str(sample_csv),
            "--against", str(sample_csv_b),
            "--config", str(preview_config),
            "--preview",
        ], input="N\n")
        assert result.exit_code == 0
```

- [ ] **Step 2: Add --preview to match CLI**

Same pattern as dedupe: add preview/preview_size/preview_random flags to `goldenmatch/cli/match.py`.

- [ ] **Step 3: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cli_preview.py -v`
Expected: All PASS

- [ ] **Step 4: Run full suite**

Run: `cd D:/show_case/goldenmatch && pytest --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/cli/match.py tests/test_cli_preview.py
git commit -m "feat: --preview flag for match CLI"
```

---

### Task 7: Add interactive command stub + dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `goldenmatch/cli/main.py`

- [ ] **Step 1: Add textual dependency to pyproject.toml**

Add `"textual>=1.0"` to the dependencies list in pyproject.toml.

- [ ] **Step 2: Add interactive command stub to main.py**

```python
@app.command("interactive")
def interactive_cmd(
    files: list[str] = typer.Argument(..., help="File(s) to load"),
) -> None:
    """Launch the interactive TUI for building configs with live feedback."""
    console.print("[yellow]Interactive TUI coming soon. Use --preview for now.[/yellow]")
```

- [ ] **Step 3: Re-install and run full suite**

Run: `cd D:/show_case/goldenmatch && pip install -e ".[dev]" && pytest --tb=short`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml goldenmatch/cli/main.py
git commit -m "feat: add textual dependency and interactive command stub"
```
