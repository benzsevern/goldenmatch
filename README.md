# GoldenMatch

**High-performance CLI for record deduplication, list matching, and golden record creation.**

Built with Polars, RapidFuzz, and Typer for fast, configurable entity resolution workflows.

<!-- Badges -->
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Installation

```bash
pip install goldenmatch
```

For development:

```bash
git clone https://github.com/benzsevern/goldenmatch.git
cd goldenmatch
pip install -e ".[dev]"
```

## Quick Start

### Dedupe Mode

Find and merge duplicate records across one or more files:

```bash
goldenmatch dedupe customers.csv \
  --config config.yaml \
  --output-all \
  --output-dir results/
```

### Match Mode

Match a target file against one or more reference files:

```bash
goldenmatch match targets.csv \
  --against reference.csv \
  --config config.yaml \
  --output-all \
  --output-dir results/
```

## Config File Reference

GoldenMatch is driven by a YAML config file that defines matchkeys, blocking, and golden record strategies.

```yaml
# Exact matchkey: records sharing the same transformed value are duplicates
matchkeys:
  - name: exact_email
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]

  # Weighted matchkey: fuzzy scoring across multiple fields
  - name: fuzzy_name_zip
    type: weighted
    threshold: 0.85
    fields:
      - field: first_name
        transforms: [lowercase, strip]
        scorer: jaro_winkler
        weight: 0.4
      - field: last_name
        transforms: [lowercase, strip]
        scorer: jaro_winkler
        weight: 0.4
      - field: zip
        transforms: [strip]
        scorer: exact
        weight: 0.2

# Blocking reduces comparison space for weighted matchkeys
blocking:
  keys:
    - fields: [zip]
      transforms: [strip]
    - fields: [last_name]
      transforms: [lowercase, strip, soundex]
  max_block_size: 5000
  skip_oversized: false

# Golden record: how to merge duplicate clusters into a single record
golden_rules:
  default_strategy: most_complete
  field_rules:
    email:
      strategy: majority_vote
    first_name:
      strategy: source_priority
      source_priority: [crm, marketing, web]

# Output settings
output:
  directory: ./output
  format: csv
  run_name: my_run
```

## Available Transforms

Transforms are applied to field values before comparison. They can be chained in order.

| Transform                | Description                            |
| ------------------------ | -------------------------------------- |
| `lowercase`              | Convert to lowercase                   |
| `uppercase`              | Convert to uppercase                   |
| `strip`                  | Remove leading/trailing whitespace     |
| `strip_all`              | Remove all whitespace                  |
| `normalize_whitespace`   | Collapse runs of whitespace to single space |
| `digits_only`            | Keep only numeric characters           |
| `alpha_only`             | Keep only alphabetic characters        |
| `soundex`                | Apply Soundex phonetic encoding        |
| `metaphone`              | Apply Metaphone phonetic encoding      |
| `substring:<start>:<end>`| Extract a substring by index           |

## Available Scorers

Scorers compute similarity between two field values in weighted matchkeys.

| Scorer          | Description                                    |
| --------------- | ---------------------------------------------- |
| `exact`         | 1.0 if values are identical, 0.0 otherwise     |
| `jaro_winkler`  | Jaro-Winkler string similarity (0.0 to 1.0)   |
| `levenshtein`   | Normalized Levenshtein similarity (0.0 to 1.0) |
| `token_sort`    | Token sort ratio via RapidFuzz (0.0 to 1.0)   |
| `soundex_match` | 1.0 if Soundex codes match, 0.0 otherwise      |

## Golden Record Strategies

When duplicates are found, GoldenMatch merges each cluster into a single golden record using per-field strategies.

| Strategy          | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `most_complete`   | Pick the longest (most complete) non-null value       |
| `majority_vote`   | Pick the most frequent value across the cluster       |
| `source_priority` | Pick from a ranked list of source labels              |
| `most_recent`     | Pick from the row with the latest date                |
| `first_non_null`  | Pick the first non-null value encountered             |

## CLI Reference

### `goldenmatch dedupe`

Run deduplication on one or more input files.

```
goldenmatch dedupe FILE [FILE ...] [OPTIONS]
```

| Flag                 | Description                              |
| -------------------- | ---------------------------------------- |
| `--config`, `-c`     | Path to YAML config file (required)      |
| `--output-golden`    | Output golden records                    |
| `--output-clusters`  | Output cluster membership                |
| `--output-dupes`     | Output duplicate records                 |
| `--output-unique`    | Output unique (non-duplicate) records    |
| `--output-all`       | Enable all output types                  |
| `--output-report`    | Print summary report to console          |
| `--across-files-only`| Only match across different source files |
| `--output-dir`       | Output directory                         |
| `--format`, `-f`     | Output format: `csv` or `parquet`        |
| `--run-name`         | Prefix for output filenames              |
| `--verbose`, `-v`    | Verbose output                           |
| `--quiet`, `-q`      | Suppress console output                  |

### `goldenmatch match`

Match a target file against one or more reference files.

```
goldenmatch match TARGET [OPTIONS]
```

| Flag                | Description                              |
| ------------------- | ---------------------------------------- |
| `--against`, `-a`   | Reference file(s) (required, repeatable) |
| `--config`, `-c`    | Path to YAML config file (required)      |
| `--output-matched`  | Output matched records                   |
| `--output-unmatched`| Output unmatched records                 |
| `--output-scores`   | Output score details                     |
| `--output-all`      | Enable all output types                  |
| `--output-report`   | Print summary report to console          |
| `--match-mode`      | `best` (default) or `all`               |
| `--output-dir`      | Output directory                         |
| `--format`, `-f`    | Output format: `csv` or `parquet`        |
| `--run-name`        | Prefix for output filenames              |
| `--verbose`, `-v`   | Verbose output                           |
| `--quiet`, `-q`     | Suppress console output                  |

### `goldenmatch config`

Manage saved config presets.

```
goldenmatch config save <name> <config_path>
goldenmatch config load <name> [--dest <path>]
goldenmatch config list
goldenmatch config delete <name>
goldenmatch config show <name>
```

### `goldenmatch init`

Launch the interactive config wizard to generate a YAML config file.

```
goldenmatch init [--output <path>]
```

## Performance Notes

- **Polars**: All data loading and transformation runs on Polars LazyFrames for memory-efficient, multi-threaded execution.
- **Blocking**: Weighted/fuzzy matchkeys require blocking keys to avoid O(n^2) comparisons. Choose blocking fields that group likely matches together (e.g., zip code, phonetic last name).
- **Max block size**: Oversized blocks (default > 5,000 records) can be skipped to prevent runaway comparisons.
- **Output formats**: Parquet output is significantly smaller and faster to write than CSV for large result sets.
- **Scalability**: Tested on datasets with 100k+ records. For millions of records, use tight blocking keys and exact matchkeys where possible.

## License

MIT
