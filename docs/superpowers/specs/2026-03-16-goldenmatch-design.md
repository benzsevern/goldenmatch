# GoldenMatch — Design Specification

## Overview

GoldenMatch is a Python CLI tool for high-performance record matching and deduplication. It supports two primary modes — **dedupe** (find and merge duplicates within or across files) and **list-match** (compare target lists against reference files). Built on Polars for millions-of-records performance, with configurable matchkeys via YAML, persistent session preferences, and golden record creation with per-field confidence scoring.

## Goals

- Deduplicate or match records across CSV, Excel (.xlsx), and Parquet files
- Handle millions of records without running out of memory
- Provide configurable, composable matchkeys with both exact and fuzzy matching
- Generate golden (merged) records with per-field confidence scores
- Persist matchkey configurations across sessions as named presets
- Offer an interactive wizard to help users build YAML configs
- Produce configurable output: golden records, clusters, duplicates, uniques, reports

## Non-Goals

- GUI or web interface (CLI only)
- Real-time / streaming ingestion (batch processing only)
- Database connectors (file-based input/output only)
- ML-based entity resolution (rule-based + similarity scoring only)

## Technology Stack

- **Language**: Python 3.11+
- **Data engine**: Polars (lazy evaluation, columnar, multi-threaded)
- **CLI**: Typer + Rich (pretty output, progress bars, tables)
- **Config**: YAML via PyYAML, validated with Pydantic
- **Fuzzy matching**: rapidfuzz (Jaro-Winkler, Levenshtein, etc.)
- **Phonetic**: jellyfish (Soundex, Metaphone)
- **Packaging**: pyproject.toml, installable via pip

## Architecture

```
goldenmatch/
├── pyproject.toml
├── README.md
├── goldenmatch/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py           # Top-level Typer app: dedupe, match, config, init
│   │   ├── dedupe.py         # Dedupe mode subcommands and options
│   │   └── match.py          # List-match mode subcommands and options
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ingest.py         # File loading (CSV, Excel, Parquet) via Polars
│   │   ├── matchkey.py       # Matchkey builder: normalize, transform, concatenate
│   │   ├── blocker.py        # Blocking strategies to avoid N-squared comparisons
│   │   ├── scorer.py         # Similarity scoring: exact, fuzzy, phonetic
│   │   ├── cluster.py        # Transitive closure to group matched records
│   │   └── golden.py         # Golden record builder with field-level confidence
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py         # YAML config loading + Pydantic validation
│   │   ├── wizard.py         # Interactive config builder (guided YAML creation)
│   │   └── schemas.py        # Pydantic models for all config structures
│   ├── output/
│   │   ├── __init__.py
│   │   ├── writer.py         # Output file generation (CSV, Excel, Parquet)
│   │   └── report.py         # Match summary statistics and reporting
│   ├── prefs/
│   │   ├── __init__.py
│   │   └── store.py          # Persistent session preferences (~/.goldenmatch/presets/)
│   └── utils/
│       ├── __init__.py
│       └── transforms.py     # Field transforms: substring, upper, soundex, normalize
└── tests/
    ├── __init__.py
    ├── conftest.py            # Shared fixtures, sample data generators
    ├── test_ingest.py
    ├── test_matchkey.py
    ├── test_blocker.py
    ├── test_scorer.py
    ├── test_cluster.py
    ├── test_golden.py
    ├── test_config.py
    ├── test_wizard.py
    ├── test_cli_dedupe.py
    ├── test_cli_match.py
    └── test_prefs.py
```

## Pipeline Flow

The matching pipeline executes in a fixed sequence:

```
1. INGEST      Load files into Polars LazyFrames, validate columns
2. TRANSFORM   Apply per-field transforms to build matchkey values (Polars expressions, vectorized)
3. BLOCK       Group records into blocks using blocking keys (Polars group_by, stays in Polars)
4. COMPARE     Within each block, generate candidate pairs and score them:
                 - Exact matchkeys: Polars inner join on concatenated key (stays in Polars)
                 - Fuzzy matchkeys: Collect block into Python, score with rapidfuzz, return pairs
5. THRESHOLD   Filter candidate pairs to those meeting the matchkey threshold
6. CLUSTER     Union-Find over all surviving pairs to form clusters (dedupe) or join results (list-match)
7. GOLDEN      For dedupe: merge cluster members into golden records with confidence scores
8. OUTPUT      Write configured output files incrementally
```

**Critical boundary — Polars vs. Python**: Steps 1-3 and exact comparisons in step 4 run entirely in Polars (lazy, multi-threaded, vectorized). Fuzzy comparisons in step 4 require materializing each block into Python and calling rapidfuzz. This is the primary performance bottleneck. The blocker's job is to keep blocks small enough that this materialization is fast. If any single block exceeds `max_block_size` (default: 5000), it is logged as a warning and optionally split using a secondary blocking key or skipped with a flag.

**Blocking is required when any matchkey uses fuzzy comparison.** If the user defines a fuzzy matchkey but no blocking config, the tool will error with a message explaining why blocking is necessary and suggesting a blocking key based on the matchkey fields.

**Blocking is optional for exact-only matchkeys.** Exact matchkeys use Polars joins directly, which are O(n) with hash joins and do not require blocking.

## Core Concepts

### Matchkeys

A matchkey is a composite key built from one or more transformed fields. Multiple matchkeys can be defined — records that match on ANY matchkey are considered potential duplicates (OR logic across keys). Each matchkey can use exact or fuzzy comparison.

**Exact matchkeys**: All fields are transformed and concatenated into a single composite string. Records with identical composite keys are matches. The `threshold` field is ignored (always 1.0). These run as Polars hash joins.

**Fuzzy (weighted) matchkeys**: Each field is scored independently using its configured scorer, then aggregated into a composite score. Records above the threshold are matches. These require blocking.

**Weighted score aggregation formula**: For a fuzzy matchkey with N fields:

```
score = sum(field_score_i * weight_i for non-null fields) / sum(weight_i for non-null fields)
```

- Weights are not required to sum to 1.0 — they are relative.
- If a field is null in either record of the pair, that field is excluded from both the numerator and denominator. This means null fields do not penalize the score; the score is based only on fields that can be compared.
- If ALL fields are null for one or both records, the pair scores 0.0.

```yaml
matchkeys:
  - name: "name_zip"
    description: "Last name prefix + ZIP prefix"
    fields:
      - column: last_name
        transforms: [lowercase, strip, "substring:0:5"]
      - column: zip
        transforms: ["substring:0:3"]
    comparison: exact
    # threshold is ignored for exact; always 1.0

  - name: "fuzzy_name_email"
    description: "Fuzzy full name + exact email"
    fields:
      - column: full_name
        transforms: [lowercase, strip]
        scorer: jaro_winkler
        weight: 0.6
      - column: email
        transforms: [lowercase, strip]
        scorer: exact
        weight: 0.4
    comparison: weighted
    threshold: 0.85
```

### Available Transforms

| Transform | Description | Example |
|-----------|-------------|---------|
| `lowercase` | Convert to lowercase | "SMITH" -> "smith" |
| `uppercase` | Convert to uppercase | "smith" -> "SMITH" |
| `strip` | Remove leading/trailing whitespace | " smith " -> "smith" |
| `strip_all` | Remove all whitespace | "s m i t h" -> "smith" |
| `substring:start:end` | Extract substring | "smith"[:3] -> "smi" |
| `soundex` | Soundex phonetic code | "smith" -> "S530" |
| `metaphone` | Metaphone phonetic code | "smith" -> "SM0" |
| `digits_only` | Remove non-digit characters | "555-1234" -> "5551234" |
| `alpha_only` | Remove non-alpha characters | "O'Brien" -> "OBrien" |
| `normalize_whitespace` | Collapse multiple spaces to one | "John  Smith" -> "John Smith" |

### Available Scorers

| Scorer | Description | Use Case |
|--------|-------------|----------|
| `exact` | Binary 0 or 1 | IDs, codes, emails |
| `jaro_winkler` | Edit distance, prefix-weighted | Names |
| `levenshtein` | Edit distance, normalized 0-1 | General strings |
| `token_sort` | Token-sorted ratio | Names with reordering |
| `soundex_match` | Phonetic code comparison | Name variants |

### Blocking

To handle millions of records without O(n-squared) pairwise comparisons, records are grouped into blocks. Only records sharing the same block key are compared.

```yaml
blocking:
  - key_fields:
      - column: last_name
        transforms: [lowercase, "substring:0:3"]
      - column: state
        transforms: [uppercase]
```

Multiple blocking keys can be defined (union of blocks). A record can appear in multiple blocks.

**Max block size**: Default 5000 records. Blocks exceeding this are logged as warnings. Configurable via `max_block_size` in the blocking config. Oversized blocks can optionally be skipped with `skip_oversized: true`. See the complete YAML schema for the full blocking config structure.

**Performance target**: With blocking, a 5-million-record file with average block size of 50 should complete in minutes, not hours.

### Golden Record Rules

When duplicates are found in dedupe mode, fields are merged into a golden record using configurable per-field strategies.

```yaml
golden_rules:
  default:
    strategy: most_complete      # fallback for unconfigured fields

  email:
    strategy: most_recent
    date_column: updated_at
    source_priority: [crm_export, legacy_db]

  address:
    strategy: source_priority
    source_priority: [crm_export, legacy_db]

  phone:
    strategy: most_complete

  specialty:
    strategy: majority_vote
```

#### Strategies

| Strategy | Logic |
|----------|-------|
| `most_recent` | Non-null value from the row with the latest date (requires `date_column`) |
| `source_priority` | First non-null value from the preferred source order |
| `most_complete` | Longest non-null string value |
| `majority_vote` | Most common value across duplicates |
| `first_non_null` | First non-null value by source order (CLI file order), then row order within file |

#### Confidence Scoring

Each field in the golden record gets a confidence score (0.0 - 1.0) computed per-strategy:

| Strategy | Confidence Formula |
|----------|-------------------|
| `most_recent` | 1.0 if only one record has the latest date; 0.5 if tied on date (falls back to source_priority) |
| `source_priority` | 1.0 if highest-priority source has a value; decreases by 0.1 per priority level fallen through |
| `most_complete` | 1.0 if winner is strictly longer than all others; 0.7 if tied with another value of same length |
| `majority_vote` | `count_of_winning_value / total_non_null_records_in_cluster` |
| `first_non_null` | 0.6 always (low confidence — ordering is arbitrary; prefer other strategies) |

The overall record confidence is the mean of all field confidences.

Fields where all cluster members agree get an automatic 1.0 regardless of strategy.

### Clustering (Dedupe Mode)

The clustering algorithm is **Union-Find (disjoint set union)** with path compression and union by rank:

1. Every record starts as its own cluster.
2. For each surviving candidate pair (above threshold), union the two records.
3. After all pairs are processed, find the root of each record to determine cluster membership.

**Safeguards**:
- **Max cluster size**: Default 100. If a cluster exceeds this, it is flagged in the report as `oversized_cluster` and the golden record is not created (the raw cluster is still output). Configurable via `max_cluster_size` in the config.
- **Cluster IDs**: Monotonically increasing integers starting at 1.

### Matching (List-Match Mode)

List-match does **not** use clustering. Instead, each target record is compared against reference records within its block, and results are returned as:

- **Default (1:best)**: Each target gets its single best-scoring match above threshold. Ties are broken by: (1) highest composite score, (2) first file in `--against` order, (3) first row in file.
- **`--match-mode all`**: Each target gets ALL reference matches above threshold, with scores. Output has one row per target-reference pair.
- **`--match-mode none`**: No joining — just flag targets as matched/unmatched with a boolean column.

Golden record creation is **not available** in list-match mode. The output joins target fields with reference fields side-by-side for matched pairs.

## Complete YAML Config Schema

Below is a fully annotated example config showing all top-level sections and defaults:

```yaml
# goldenmatch.yaml — complete example

# --- Input settings (optional, can also be passed via CLI args) ---
input:
  files:
    - path: crm_export.csv
      source_name: crm              # Used in golden_rules source_priority
      delimiter: ","                 # Default: ","
      encoding: utf-8               # Default: utf-8
      sheet: null                    # For .xlsx files only
    - path: legacy_db.parquet
      source_name: legacy

# --- Mode: "dedupe" or "match" (optional, inferred from CLI subcommand) ---
mode: dedupe

# --- Matchkeys (required) ---
matchkeys:
  - name: "name_zip"
    description: "Last name prefix + ZIP prefix"
    fields:
      - column: last_name
        transforms: [lowercase, strip, "substring:0:5"]
      - column: zip
        transforms: ["substring:0:3"]
    comparison: exact

  - name: "fuzzy_name_email"
    description: "Fuzzy full name + exact email"
    fields:
      - column: full_name
        transforms: [lowercase, strip]
        scorer: jaro_winkler
        weight: 0.6
      - column: email
        transforms: [lowercase, strip]
        scorer: exact
        weight: 0.4
    comparison: weighted
    threshold: 0.85

# --- Blocking (required if any matchkey uses fuzzy comparison) ---
blocking:
  max_block_size: 5000             # Default: 5000. Warn if exceeded.
  skip_oversized: false            # Default: false. If true, skip oversized blocks.
  keys:
    - key_fields:
        - column: last_name
          transforms: [lowercase, "substring:0:3"]
        - column: state
          transforms: [uppercase]

# --- Golden record rules (dedupe mode only, optional) ---
golden_rules:
  max_cluster_size: 100            # Default: 100. Clusters above this are flagged, not merged.
  default:
    strategy: most_complete        # Fallback for fields without explicit rules

  email:
    strategy: most_recent
    date_column: updated_at
  address:
    strategy: source_priority
    source_priority: [crm, legacy]
  phone:
    strategy: most_complete
  specialty:
    strategy: majority_vote

# --- Output preferences (optional, overridden by CLI flags) ---
output:
  format: csv                      # csv | xlsx | parquet. Default: csv
  directory: ./results             # Default: current directory
  run_name: null                   # Default: timestamp. Set for deterministic naming.

# --- List-match settings (match mode only, optional) ---
match_settings:
  match_mode: best                 # best | all | none. Default: best
```

**Required fields**: Only `matchkeys` is strictly required. Everything else has defaults or is inferred from CLI arguments. If `--config` is omitted from the CLI, the tool looks for `goldenmatch.yaml` in the current directory and fails with a helpful message if not found.

## Modes

### Dedupe Mode

Find duplicates within a single file or across multiple files.

```bash
# Single file
goldenmatch dedupe data.csv --config match_config.yaml

# Multiple files, find dupes across all files combined
goldenmatch dedupe file1.csv file2.csv --config match_config.yaml

# Multiple files, only match across files (not within)
goldenmatch dedupe file1.csv file2.csv --config match_config.yaml --across-files-only
```

**Output flags** (combinable):

| Flag | Description |
|------|-------------|
| `--output-golden` | Merged golden records with confidence scores |
| `--output-clusters` | All records grouped by cluster ID with match scores |
| `--output-dupes` | Only the duplicate records (excludes survivors) |
| `--output-unique` | Records that had no matches |
| `--output-all` | Golden + clusters + confidence + unique |
| `--output-report` | Summary: total records, clusters found, match rate, size distribution |

**Output format**: `--format csv` (default), `--format xlsx`, `--format parquet`

**Output directory**: `--output-dir ./results/` (default: current directory)

### List-Match Mode

Compare a target list against one or more reference files.

```bash
# Single reference
goldenmatch match targets.csv --against reference.csv --config match_config.yaml

# Multiple references
goldenmatch match targets.csv --against ref1.csv ref2.csv --config match_config.yaml
```

**Output flags** (combinable):

| Flag | Description |
|------|-------------|
| `--output-matched` | Target records that found matches, joined with matched reference records |
| `--output-unmatched` | Target records with no match |
| `--output-scores` | All comparisons with similarity scores (even below threshold) |
| `--output-all` | Matched + unmatched + scores |
| `--output-report` | Summary: hit rate, score distribution, match counts per reference |

## Config Wizard

`goldenmatch init` provides an interactive, guided experience for building a YAML config.

### Flow

1. **File selection**: "What files are you working with?" — user provides paths, tool auto-detects columns and shows a preview (first 5 rows via Rich table).
2. **Mode selection**: "Dedupe or list-match?"
3. **Matchkey building**: "Which fields should form your matchkey?" — displays column list, user selects. For each field:
   - Suggests transforms based on field name heuristics (e.g., name fields get `soundex` suggestion, zip fields get `substring` suggestion)
   - Asks exact vs. fuzzy, and which scorer
4. **Blocking key**: Suggests a blocking key based on the matchkey fields. User can accept or customize.
5. **Threshold**: "How strict should matching be?" — offers presets (strict: 0.95, moderate: 0.85, loose: 0.70) or custom.
6. **Golden rules** (dedupe only): For each field, asks preferred merge strategy. Suggests defaults based on field name.
7. **Output preferences**: Which outputs to generate by default.
8. **Save**: Writes to `goldenmatch.yaml` in current directory. Optionally saves as a named preset.

### Smart Suggestions

The wizard uses field name heuristics to suggest appropriate transforms and scorers:

- Fields containing "name" -> suggest `lowercase, strip, jaro_winkler`
- Fields containing "zip" or "postal" -> suggest `substring:0:5, exact`
- Fields containing "email" -> suggest `lowercase, strip, exact`
- Fields containing "phone" -> suggest `digits_only, exact`
- Fields containing "address" -> suggest `lowercase, normalize_whitespace, token_sort`

## Session Preferences

Stored in `~/.goldenmatch/presets/` as named YAML files.

```bash
goldenmatch config save my_medical_dedup      # Save current config as preset
goldenmatch config load my_medical_dedup      # Load preset to ./goldenmatch.yaml
goldenmatch config list                       # List all saved presets
goldenmatch config delete my_medical_dedup    # Delete a preset
goldenmatch config show my_medical_dedup      # Display preset contents
```

Presets are full YAML configs. Loading a preset copies it to the working directory where it can be further customized.

## Performance Design

### Memory Management
- **Polars lazy evaluation**: Scan files without loading entirely into memory. Predicates push down to file readers.
- **Streaming output**: Write results incrementally rather than holding all results in memory.
- **Chunked processing**: For files exceeding available memory, process blocks in chunks.

### Computation
- **Blocking**: Reduces comparison space from O(n-squared) to O(n * average_block_size).
- **Polars parallelism**: All expression-based transforms run multi-threaded by default.
- **Vectorized operations**: Matchkey computation and exact comparisons use Polars native expressions, not row-by-row Python.
- **Lazy fuzzy scoring**: Only compute expensive fuzzy scores for records that pass blocking.

### Benchmarks (targets)
- 100K records, single matchkey, exact: < 5 seconds
- 1M records, single matchkey with blocking, exact: < 30 seconds
- 5M records, single matchkey with blocking, fuzzy: < 5 minutes

## Output Formats

All output files follow a consistent naming convention:

```
{run_name}_{output_type}.{format}
```

Example: `my_dedup_golden.csv`, `my_dedup_clusters.parquet`, `my_dedup_report.csv`

The run name defaults to a timestamp but can be set via `--run-name`.

## Error Handling

- **Missing columns**: If a configured column doesn't exist in the input file, fail fast with a clear error listing available columns.
- **Type mismatches**: If a transform expects a string but gets a number, auto-cast with a warning.
- **Empty files**: Warn and skip, don't crash.
- **Corrupt rows**: Log and skip with a count of skipped rows in the report.
- **No matches found**: Produce empty output files with headers, plus a report noting 0 matches.

## Logging

GoldenMatch uses Python's `logging` module writing to stderr (so stdout is reserved for piping output).

**Verbosity flags**:

| Flag | Level | What gets logged |
|------|-------|-----------------|
| `--quiet` / `-q` | WARNING | Only warnings and errors |
| (default) | INFO | Progress milestones: files loaded, blocks created, pairs scored, clusters formed, output written |
| `--verbose` / `-v` | DEBUG | Per-block stats, per-matchkey pair counts, skipped rows, timing per pipeline stage |

**Structured output**: When `--log-file path.log` is provided, logs are written as JSON lines (timestamp, level, message, context) for machine parsing. stderr output remains human-readable.

**Progress bars**: Long-running operations (blocking, scoring, clustering) display Rich progress bars in the terminal. Suppressed when `--quiet` is set or when stdout is not a TTY.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Config error (invalid YAML, missing required fields, validation failure) |
| 2 | Input error (file not found, missing columns, unsupported format) |
| 3 | Runtime error (unexpected failure during processing) |

## Source Name Mapping

When using multi-file dedupe with golden rules that reference `source_priority`, files must have source names. These can be provided in three ways (in priority order):

1. **Config file**: The `input.files[].source_name` field (see complete YAML schema above)
2. **CLI flag**: `goldenmatch dedupe file1.csv:crm file2.csv:legacy --config ...` (colon-separated name suffix)
3. **Filename inference**: If neither is provided, the source name defaults to the filename stem (e.g., `crm_export.csv` becomes `crm_export`)

## Testing Strategy

- **Unit tests**: Each module (ingest, matchkey, blocker, scorer, cluster, golden) tested independently with small synthetic datasets.
- **Integration tests**: End-to-end CLI tests for both modes with fixture files.
- **Performance tests**: Benchmarks with 100K and 1M row synthetic datasets to catch regressions.
- **Config validation tests**: Malformed YAML, missing fields, invalid transforms.
- **Wizard tests**: Mock stdin to test interactive flows.

## Future Considerations (Out of Scope for V1)

- Database connectors (Snowflake, PostgreSQL)
- Web UI / dashboard for reviewing match results
- ML-based scoring (learned similarity models)
- REST API mode
- Plugin system for custom transforms and scorers
