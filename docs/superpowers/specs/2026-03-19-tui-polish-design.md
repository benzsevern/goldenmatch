# TUI Polish — Design Specification

## Overview

Comprehensive polish pass on the GoldenMatch TUI: gold/amber visual theme, auto-config summary screen for zero-config first impressions, staged progress indicators, split-view matches tab, and power-user keyboard shortcuts with live threshold tuning.

## Goals

- Professional first impression when running `goldenmatch dedupe file.csv`
- Gold/amber branded theme replacing Textual defaults
- Auto-config summary screen before tabs appear
- Full-screen progress on first run, footer bar on re-runs
- Split-view matches tab with cluster master-detail
- Keyboard shortcuts for all common operations
- Live threshold slider with instant re-clustering

## Non-Goals

- Complete TUI rewrite (polish existing structure)
- Mobile/responsive design
- Custom Textual widget library

---

## Feature 1: Gold/Amber Theme

### Colors

| Role | Value | Usage |
|------|-------|-------|
| Background | `#1a1a2e` | App background |
| Surface | `#16213e` | Panels, sidebar, cards |
| Primary | `#d4a017` | Gold accent, active tab, buttons |
| Success | `#2ecc71` | Matched, confirmed |
| Warning | `#e67e22` | Conflicts, amber highlights |
| Error | `#e74c3c` | Rejected, errors |
| Text | `#f0f0f0` | Primary text |
| Text muted | `#8892a0` | Secondary text, labels |
| Border | `#d4a017 40%` | Subtle gold borders |

### Branded Elements

- **Header**: "⚡ GoldenMatch" in gold with version number
- **Footer**: Gold-tinted key hints
- **Active tab**: Gold underline, inactive tabs muted
- **Buttons**: Gold background with dark text for primary actions
- **Focus ring**: Gold outline on focused widgets

### Implementation

Custom CSS in `app.py` replacing the current minimal styles. Use Textual's CSS variable system (`$primary`, `$surface`, etc.) for consistency.

---

## Feature 2: Auto-Config Summary Screen

### Design

New screen `goldenmatch/tui/screens/autoconfig_screen.py` shown as the first screen when no config is provided.

### Layout

```
┌─────────────────────────────────────────────────────┐
│  ⚡ GoldenMatch                                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Auto-Detected Configuration                        │
│                                                     │
│  File: customers.csv                                │
│  Records: 12,847 │ Columns: 8 │ Est. dupes: ~15%   │
│                                                     │
│  Column Mapping                                     │
│  ┌──────────┬──────────┬──────────────┬───────┐     │
│  │ Column   │ Type     │ Scorer       │ Weight│     │
│  ├──────────┼──────────┼──────────────┼───────┤     │
│  │ name     │ Name     │ ensemble     │ 1.0   │     │
│  │ email    │ Email    │ exact        │ 1.0   │     │
│  │ phone    │ Phone    │ exact        │ 0.8   │     │
│  │ zip      │ Zip      │ exact        │ 0.5   │     │
│  └──────────┴──────────┴──────────────┴───────┘     │
│                                                     │
│  Blocking: multi_pass (soundex + substring + token) │
│  Threshold: 0.80 (adaptive)                         │
│  Model: all-MiniLM-L6-v2 (auto, <50K rows)         │
│                                                     │
│  [▶ Run]  [Edit Config]  [Save Settings]            │
│                                                     │
│  F5: Run  │  E: Edit  │  S: Save  │  Q: Quit       │
└─────────────────────────────────────────────────────┘
```

### Behavior

- **Run (F5)**: Transitions to full-screen progress, then tabbed results
- **Edit Config (E)**: Transitions to tabbed view with Config tab focused
- **Save Settings (S)**: Saves to `.goldenmatch.yaml`, then runs
- Column types are clickable — Enter cycles through alternatives
- Shown only on first launch without config; subsequent runs with saved config skip to tabs

### Column Type Cycling

When a column row is focused, Enter/Space cycles:
```
name → email → phone → zip → address → string → (skip) → name
```
Scorer and weight update automatically based on the selected type.

---

## Feature 3: Progress Views

### First-Run Progress (Full-Screen)

Shown when no previous results exist.

```
              Matching in progress...

       ████████████████░░░░░░░░░░  62%

       Stage    Scoring fuzzy pairs
       Pairs    1,247 found
       Blocks   342 / 512
       Elapsed  4.2s

       Pipeline:
       ✓ Ingest          0.1s
       ✓ Auto-fix        0.3s
       ✓ Standardize     0.2s
       ✓ Matchkeys       0.1s
       ● Scoring         4.2s
       ○ Clustering
       ○ Golden records
```

Pipeline stages: ✓ completed, ● in-progress, ○ pending. Elapsed time per completed stage.

### Re-Run Progress (Footer Bar)

Shown when re-running with existing results visible.

```
Re-running... ████████░░░░ 45% │ Scoring │ 2.1s
```

Non-intrusive. Old results remain visible until new results replace them.

### Implementation

- New widget `goldenmatch/tui/widgets/progress_overlay.py` for full-screen
- New widget `goldenmatch/tui/widgets/progress_bar.py` for footer bar
- MatchEngine emits progress events via callback

---

## Feature 4: Matches Split View

### Layout

```
┌──────────────────────┬──────────────────────────────────┐
│ Clusters (1,247)     │ Cluster #42                      │
│                      │                                  │
│ Sort: [Score ▼]      │ Score: 0.94 avg │ 3 records      │
│                      │                                  │
│ ▸ #42  3rec  0.94   │ Golden Record                    │
│   #107 2rec  0.91   │ ┌────────────────────────────┐   │
│   #23  4rec  0.88   │ │ name: John Smith           │   │
│   ...               │ │ email: john@test.com       │   │
│                      │ └────────────────────────────┘   │
│ Filter: [________]   │                                  │
│                      │ Members                          │
│ ☐ Show singletons   │ ┌────────┬──────────┬──────────┐ │
│                      │ │ John S │ john@t.. │ 10001    │ │
│ Total: 1,247 clusters│ │ Jon S  │ jon@t..  │ 10001    │ │
│ Records: 3,891 dupes │ │ J.Smith│ js@t..   │ 10001    │ │
│                      │ └────────┴──────────┴──────────┘ │
└──────────────────────┴──────────────────────────────────┘
```

### Interactions

- Arrow keys navigate cluster list
- Enter selects cluster for detail view
- `/` focuses filter box (search by record values)
- `S` cycles sort order (score → size → ID)
- Diff highlighting: differing fields in amber, matching fields in green
- Cluster list shows: cluster ID, record count, average score
- Detail panel shows: golden record, member table with scores

### Implementation

Replace current `MatchesTab` with split-panel layout using Textual's `Horizontal` container. Left panel is a `ListView` of clusters, right panel is a `Container` with golden record card and member table.

---

## Feature 5: Keyboard Shortcuts

### Global

| Key | Action |
|---|---|
| `1-5` | Jump to tab (Data, Config, Matches, Golden, Export) |
| `F5` | Run / re-run with current config |
| `F2` | Save config to `.goldenmatch.yaml` |
| `Ctrl+E` | Quick export (default format to default dir) |
| `Ctrl+R` | Re-run matching |
| `Tab` / `Shift+Tab` | Cycle between tabs |
| `Q` | Quit |
| `?` | Show keyboard shortcut overlay |

### Matches Tab

| Key | Action |
|---|---|
| `↑/↓` | Navigate cluster list |
| `Enter` | Select cluster for detail |
| `/` | Focus filter box |
| `S` | Sort cycle |

### Config Tab — Threshold Slider

```
Threshold: ◀ 0.80 ▶   [←/→ to adjust by 0.05]
Live preview: ~1,247 clusters at this threshold
```

Left/right arrows adjust by 0.05. Uses `recluster_at_threshold()` from MatchEngine for instant feedback without re-running the full pipeline.

### Shortcut Overlay

Pressing `?` shows a modal overlay listing all shortcuts, dismissible with Escape or `?` again.

---

## File Structure

```
goldenmatch/tui/
├── app.py                          # Modified: theme CSS, screen routing
├── sidebar.py                      # Modified: gold theme
├── engine.py                       # Modified: progress callbacks
├── screens/
│   └── autoconfig_screen.py        # NEW: auto-config summary screen
├── widgets/
│   ├── progress_overlay.py         # NEW: full-screen progress
│   ├── progress_bar.py             # NEW: footer progress bar
│   ├── threshold_slider.py         # NEW: live threshold widget
│   └── shortcut_overlay.py         # NEW: ? keyboard help modal
└── tabs/
    ├── data_tab.py                 # Existing
    ├── config_tab.py               # Modified: column type cycling
    ├── matches_tab.py              # Modified: split view
    ├── golden_tab.py               # Existing
    └── export_tab.py               # Existing
```

## Rollout Plan

1. **Phase 1: Theme** — Gold/amber CSS, branded header/footer
2. **Phase 2: Auto-Config Screen** — Summary card with column table, Run/Edit/Save buttons
3. **Phase 3: Progress Views** — Full-screen overlay + footer bar, progress callbacks in engine
4. **Phase 4: Matches Split View** — Master-detail layout, cluster list, detail panel
5. **Phase 5: Keyboard Shortcuts** — All keybindings, threshold slider, shortcut overlay

## Testing Strategy

### TUI Tests (pytest-asyncio)

- `test_autoconfig_screen.py` — screen renders, button actions, column cycling
- `test_progress.py` — overlay shows/hides, stage transitions, percentage updates
- `test_matches_split.py` — cluster selection, sort cycling, filter, detail panel updates
- `test_shortcuts.py` — key bindings trigger correct actions, overlay toggle
- `test_theme.py` — app renders without errors, CSS applies

Use `app.run_test()` with pilot for all TUI tests.
