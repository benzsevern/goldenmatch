# GoldenMatch Interactive TUI + Preview Mode — Design Specification

## Overview

Two new features that solve the "blind batch" problem — users currently configure, run, inspect files, edit YAML, and repeat with no live feedback.

1. **`goldenmatch interactive [files...]`** — A full-screen Textual TUI for building configs with live match feedback
2. **`goldenmatch dedupe --preview` / `goldenmatch match --preview`** — Dry-run mode that shows sample results in the terminal without writing files

## Goals

- Eliminate the config-run-inspect-edit loop by providing live feedback as the user builds their config
- Make matchkey tuning (thresholds, scorers, transforms) intuitive with instant sample results
- Reduce the learning curve by showing what each setting does in real time
- Provide a quick validation step (preview) before committing to full batch runs
- Keep the existing batch CLI fully functional — interactive and preview are additive

## Non-Goals

- Web UI or browser-based interface
- Real-time streaming / incremental matching
- Multi-user collaboration
- Replacing the existing batch CLI

## Technology Stack

- **TUI Framework**: Textual (by Textualize, built on Rich)
- **Async Workers**: Textual's Worker system for background sample matching
- **Existing Pipeline**: Reuse all core modules (ingest, matchkey, blocker, scorer, cluster, golden, profiler, standardize, autofix, validate)

## TUI Layout

VS Code-style hybrid with persistent sidebar and tabbed main panel:

```
┌─────────────────────────────────────────────────────────────┐
│  GoldenMatch Interactive                    [?] Help  [Q]uit│
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│  SIDEBAR     │  MAIN PANEL                                  │
│              │                                              │
│  Config      │  Changes based on active tab:                │
│  ────────    │                                              │
│  Matchkeys:  │  [Data] [Config] [Matches] [Golden] [Export] │
│   ✓ email    │                                              │
│   ✓ name_zip │  ┌──────────────────────────────────────┐    │
│              │  │                                      │    │
│  Threshold:  │  │  (tab content area)                  │    │
│   0.85       │  │                                      │    │
│              │  │                                      │    │
│  Stats       │  │                                      │    │
│  ────────    │  │                                      │    │
│  Records: 5K │  │                                      │    │
│  Clusters:142│  │                                      │    │
│  Match%: 8.2 │  │                                      │    │
│              │  └──────────────────────────────────────┘    │
├──────────────┴──────────────────────────────────────────────┤
│  Status: Ready │ F1:Help F2:Save F5:Run  Tab:Switch        │
└─────────────────────────────────────────────────────────────┘
```

## Tab Views

### Data Tab

File browser and profiler integration:
- Load files by path, see column list with profiler stats (null%, suspected type, sample values)
- Apply column maps interactively (select source column, type target name)
- Toggle auto-fix and standardization with before/after preview of sample rows
- Shows file metadata: encoding detected, delimiter, row count, column count

### Config Tab

Matchkey builder with live feedback:
- Add/remove matchkeys with a field picker widget
- Select transforms and scorers from dropdown lists
- Adjust threshold with a slider — live counter updates in sidebar: "At 0.85: ~142 clusters | At 0.80: ~198 clusters"
- Configure blocking keys (field selection + transforms)
- Golden rules per field with conditional inputs:
  - Strategy dropdown (most_complete, majority_vote, source_priority, most_recent, first_non_null)
  - If `most_recent` selected: date_column dropdown appears (populated from available columns)
  - If `source_priority` selected: sortable list of source names appears (drag to reorder priority)
  - Default strategy selector for unconfigured fields
  - max_cluster_size input
- Each change triggers a background sample re-run

### Matches Tab

Match preview with drill-down:
- Cluster view: list of clusters with size, shows members side by side when expanded
- Pair view: individual record pairs with per-field scores
- Color-coded scores: green (>0.9 strong match), yellow (0.7-0.9 borderline), red (below threshold but close)
- Runs on sample data (default 1000 records)

### Golden Tab

Golden record preview:
- Shows merged golden records with per-field confidence scores
- Per-field highlighting: which source/record won, confidence level
- Click/select a golden record to see the cluster it was built from
- Overall record confidence displayed

## Interactive Mode: Dedupe vs. Match

The `interactive` command supports both workflows. The user selects the mode in the Data tab after loading files.

### Dedupe Mode

Default when files are loaded without specifying target/reference roles. All files are combined and matched against each other. The Matches tab shows clusters, the Golden tab shows merged records.

### Match Mode

Activated when the user designates one file as "target" and one or more as "reference" in the Data tab (via a role dropdown per file). The TUI adapts:

- **Config Tab**: Same matchkey builder. Blocking and golden rules sections are hidden (golden records don't apply in match mode).
- **Matches Tab**: Shows a match pair list instead of clusters. Each row is a target record with its best (or all) reference match(es). Columns: target fields, ref fields, match score. Color-coded by score.
- **Golden Tab**: Replaced with a **Coverage Tab** showing hit rate, score distribution, and unmatched targets.
- **Sidebar stats**: Shows "Matched: 350/500 (70%)" instead of cluster counts.

The `goldenmatch interactive` command itself takes files positionally. Mode is selected in the UI, not the CLI:

```bash
goldenmatch interactive customers.csv prospects.csv reference.csv
```

### Export Tab

Save and run:
- Save config to YAML file (with file path input)
- Save as named preset
- Run full job with output option checkboxes (golden, clusters, dupes, unique, report)
- Preview vs. full run toggle
- Output format selector (CSV, Parquet, Excel)
- Output directory path input

## Sidebar (Persistent)

Always visible on the left, shows:
- **Config summary**: list of configured matchkeys with checkmarks, current threshold
- **Live stats**: estimated clusters, match rate, records loaded — updates when config changes
- **File info**: number of records, columns, source names

Stats update asynchronously after each config change via the background matching engine.

## Key Bindings

| Key | Action |
|-----|--------|
| Tab / Shift+Tab | Switch between tabs |
| F1 | Help overlay with command reference |
| F2 | Save config to YAML |
| F5 | Run matching on sample |
| Ctrl+R | Re-run with current config |
| Ctrl+S | Save config as preset |
| Q / Ctrl+Q | Quit (with unsaved changes prompt) |

## Live Feedback Engine

The core innovation. When the user changes config (adds a matchkey, adjusts threshold, changes transforms):

1. A 300ms debounce timer starts
2. If no further changes within 300ms, a background Worker is dispatched
3. The worker runs the full pipeline on a sample (default 1000 records)
4. Results are posted back to the UI via Textual's message system
5. Sidebar stats, Matches tab, and Golden tab update reactively

If the user changes config while a run is in progress, the current worker is cancelled and a new one starts after the debounce.

**Cancellation is cooperative.** Textual's `Worker.cancel()` sets a flag but does not interrupt blocking code. Since the pipeline is synchronous Polars (CPU-bound in a thread), the engine checks `worker.is_cancelled` between pipeline steps (after ingest, after scoring, after clustering) and bails out early if cancelled. For a 1000-record sample this is usually fast enough that cancellation latency is imperceptible.

**Threshold slider optimization.** The dual-threshold display ("At 0.85: ~142 clusters | At 0.80: ~198 clusters") does NOT re-run the full pipeline twice. Instead, the engine caches all scored pairs from the most recent sample run. Re-clustering at a different threshold only requires re-running the cheap Union-Find step on the cached pairs with a new threshold filter. This makes slider interaction feel instant.

### Engine API

`run_sample` and `run_full` are **synchronous** methods. The TUI dispatches them via Textual's `self.run_worker(engine.run_sample, thread=True)` which runs them in a background thread. This works because Polars releases the GIL during computation.

```python
@dataclass
class EngineResult:
    """Result from a matching run."""
    # Core outputs (from existing pipeline)
    clusters: dict[int, dict]               # cluster_id -> {members, size, oversized, pair_scores}
    golden: pl.DataFrame | None             # merged golden records with confidence
    unique: pl.DataFrame | None             # records with no matches
    dupes: pl.DataFrame | None              # records that had duplicates
    quarantine: pl.DataFrame | None         # rows that failed validation
    matched: pl.DataFrame | None            # for match mode: matched target+ref pairs
    unmatched: pl.DataFrame | None          # for match mode: unmatched targets

    # Scored pairs cache (for threshold re-clustering without re-scoring)
    scored_pairs: list[tuple[int, int, float]]  # (id_a, id_b, score) — all pairs above blocking, before threshold

    # Derived stats (computed by MatchEngine from pipeline output)
    stats: EngineStats

@dataclass
class EngineStats:
    total_records: int
    total_clusters: int                     # clusters with size > 1
    singleton_count: int                    # clusters with size == 1
    match_rate: float                       # total_clusters / total_records
    cluster_sizes: list[int]                # sizes of multi-member clusters
    avg_cluster_size: float
    max_cluster_size: int
    oversized_count: int
    # Match mode only
    hit_rate: float | None = None           # matched / total_targets
    avg_score: float | None = None

class MatchEngine:
    def __init__(self, files: list[Path]):
        """Load files and profile them."""

    def run_sample(self, config: GoldenMatchConfig, sample_size: int = 1000) -> EngineResult:
        """Run matching pipeline on a sample. Synchronous — call via Worker thread."""

    def run_full(self, config: GoldenMatchConfig) -> EngineResult:
        """Run full pipeline. Synchronous — call via Worker thread."""

    def recluster_at_threshold(self, threshold: float) -> EngineStats:
        """Re-run clustering on cached scored_pairs with a new threshold.
        Fast operation (Union-Find only, no re-scoring). Returns updated stats."""

    @property
    def profile(self) -> dict:
        """File profile from the profiler."""

    @property
    def columns(self) -> list[str]:
        """Available column names."""

    @property
    def row_count(self) -> int:
        """Total rows across all loaded files."""
```

**Sample size rationale.** Interactive mode defaults to 1000 records (needs sub-second response for slider/config changes). Preview mode defaults to 10,000 records (one-shot run, can take a few seconds). Both are configurable.

## Preview Mode (Batch CLI)

```bash
goldenmatch dedupe data.csv --config config.yaml --preview
goldenmatch dedupe data.csv --config config.yaml --preview --preview-size 5000
goldenmatch match targets.csv --against refs.csv --config config.yaml --preview
```

Behavior:
- Runs the full pipeline on a sample (default 10,000 records, configurable via `--preview-size`)
- Prints results to terminal using Rich instead of writing files:
  - Summary stats table (same metrics as --output-report)
  - Top 10 clusters with members shown side by side in a Rich table
  - Top 10 golden records with per-field confidence scores
  - Score distribution (text-based histogram)
  - Quarantine summary if validation rules are configured
- Ends with prompt: "Run full job now? [y/N]"
  - If "y", runs the full pipeline and writes output files
  - If "N", exits

### Preview Sampling Strategy

- Default: first N records (deterministic, reproducible)
- `--preview-random`: random sample instead (better for large files with sorted data)
- Shows warning if sample is small relative to file: "Previewing 10,000 of 5,000,000 records. Results may not be representative."

## Architecture

### New Files

```
goldenmatch/
├── tui/
│   ├── __init__.py
│   ├── app.py              # Main Textual App class, layout, key bindings
│   ├── sidebar.py          # Config summary + live stats sidebar widget
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── data_tab.py     # File browser, profiler, column mapping
│   │   ├── config_tab.py   # Matchkey builder, threshold slider, blocking, golden rules
│   │   ├── matches_tab.py  # Cluster/pair view with color-coded scores
│   │   ├── golden_tab.py   # Golden record preview with confidence
│   │   └── export_tab.py   # Save config, run job, preview toggle
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── field_picker.py # Column selector with transform/scorer dropdowns
│   │   ├── threshold.py    # Threshold slider with live cluster count
│   │   ├── score_table.py  # Color-coded match pair table
│   │   └── profile_card.py # Per-column profiler stats card
│   └── engine.py           # Background sample matching engine (async worker)
├── cli/
│   ├── main.py             # Add "interactive" command
│   ├── dedupe.py           # Add --preview, --preview-size, --preview-random flags
│   └── match.py            # Add --preview, --preview-size, --preview-random flags
└── core/
    └── preview.py          # Preview mode logic: sample, run, format Rich output
```

### Dependencies

Add to pyproject.toml:
```toml
"textual>=1.0",
```

### Module Responsibilities

**`tui/engine.py`** — Wraps existing pipeline modules into a simple API for the TUI. Handles file loading, profiling, sample extraction, and matching. Does not depend on Textual — pure Python + Polars.

**`tui/app.py`** — Main Textual App. Defines the layout (sidebar + tabbed main panel), key bindings, and message routing between tabs and the engine.

**`tui/sidebar.py`** — Reactive widget that displays config summary and live stats. Subscribes to config change and engine result messages.

**`tui/tabs/*.py`** — Each tab is a Textual Widget that manages its own UI and communicates with the engine via the app's message system.

**`tui/widgets/*.py`** — Reusable UI components used across tabs.

**`core/preview.py`** — Standalone module for preview mode. Takes a config and file paths, runs the pipeline on a sample, and returns Rich renderables (tables, panels) for terminal output. No Textual dependency.

## Error Handling & Edge Cases

- **Large files in interactive mode**: Sample matching runs on first 1000 records by default. Status bar shows "Sampled: 1K of 5M". User can adjust sample size in Data tab settings.
- **No matchkeys configured yet**: Matches tab shows placeholder message "Add a matchkey in the Config tab to see matches here." Stats show dashes.
- **File load failures**: Data tab shows the error inline with a retry option. Does not crash the TUI.
- **Background worker conflicts**: Config changes cancel in-progress sample runs and start new ones (debounced 300ms).
- **Terminal too small**: Minimum 80x24. Below that, display a centered "Terminal too small (need 80x24)" message.
- **Preview on huge files**: Shows warning when sample is <1% of total records. Suggests increasing --preview-size.
- **Unsaved changes on quit**: Prompt "You have unsaved config changes. Save before quitting? [Y/n/cancel]"

## Testing Strategy

### Unit Tests
- Engine: load files, run_sample, run_full — verify results match existing pipeline output
- Preview: formatting functions produce correct Rich output
- Widget state: field picker, threshold slider state management

### Snapshot Tests
- Textual provides `pilot` for simulated app testing
- Render each tab with fixture data, compare screenshots
- Test key binding responses

### Integration Tests
- Load fixture files, build config through engine API, verify sample results
- Preview mode end-to-end: run CLI with --preview, capture output, verify structure

## Future Considerations (Out of Scope for V1)

- Undo/redo for config changes
- Drag-and-drop file loading
- Config diff view (compare two presets)
- Export match results directly from TUI
- Plugin system for custom widgets
