---
layout: default
title: Interactive TUI
nav_order: 6
---

# Interactive TUI

GoldenMatch includes a gold-themed terminal UI built with Textual. Launch it with `goldenmatch interactive` or `goldenmatch dedupe` (zero-config mode opens the TUI automatically).

```bash
goldenmatch interactive customers.csv
goldenmatch interactive customers.csv --config config.yaml
```

---

## 6 Tabs

The TUI has 6 tabs, accessible via keyboard shortcuts `1` through `6`.

### Tab 1: Data

Data profiling view. Shows:
- Record count, column names, data types
- Null percentages, unique value counts
- Sample values for each column
- Auto-detected column types (name, email, phone, zip, address)

### Tab 2: Config

Configuration editor. Shows:
- Auto-detected matchkeys, blocking strategy, and golden rules
- Edit thresholds, scorers, and weights inline
- Save configuration to YAML
- Run/Edit/Save buttons

### Tab 3: Matches

Split-view match results:
- **Left panel**: cluster list with IDs, sizes, and confidence scores
- **Right panel**: golden record + individual member details for selected cluster
- Live threshold slider: arrow keys adjust threshold in 0.05 increments with instant cluster count preview
- Click any cluster to inspect member records and field-level scores

### Tab 4: Golden

Golden record viewer:
- Canonical merged records
- Merge strategy applied to each field
- Source provenance for each value
- Export golden records to CSV

### Tab 5: Boost

Active learning for accuracy improvement:
- Shows borderline pairs (near the threshold)
- Label with keyboard: `y` (match), `n` (no match), `s` (skip)
- After labeling ~10 pairs, trains a LogisticRegression classifier
- Reclassifies all borderline pairs with the trained model
- Shows before/after accuracy comparison

### Tab 6: Export

Output options:
- Export golden records, duplicates, unique records
- Choose format: CSV or Parquet
- Export lineage JSON
- Generate HTML report

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `1`--`6` | Jump to tab (Data, Config, Matches, Golden, Boost, Export) |
| `F5` | Run the pipeline |
| `?` | Show all keyboard shortcuts |
| `Ctrl+E` | Export results |
| `Left`/`Right` | Adjust threshold (Matches tab) |
| `y` | Label pair as match (Boost tab) |
| `n` | Label pair as non-match (Boost tab) |
| `s` | Skip pair (Boost tab) |
| `q` | Quit |

---

## Pipeline progress

On first run, the TUI shows a full-screen progress view:
- Stage tracker with status indicators (done / running / pending)
- Progress bar for the current stage
- Record count and timing

On re-runs (e.g., after threshold adjustment), progress appears in the footer bar.

---

## Auto-config summary

When launched without a config file, the first screen shows:
- Detected columns and their inferred types
- Suggested scorers for each column
- Proposed blocking strategy
- Three options: **Run** (accept and run), **Edit** (modify config), **Save** (write YAML)

---

## TUI with config file

```bash
goldenmatch interactive customers.csv --config config.yaml
```

When a config file is provided, the TUI skips auto-detection and uses the specified matchkeys, blocking, and golden rules.

---

## Screenshots

The TUI uses a dark theme with gold accents. Key views:

- **Data tab**: column profiling with null rates and type detection
- **Matches tab**: split-view cluster browser with threshold slider
- **Golden tab**: merged records with merge provenance
