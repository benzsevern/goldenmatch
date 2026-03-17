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
  # strategy: static | adaptive | sorted_neighborhood | multi_pass | ann | canopy

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
| `token_sort`             | Sort tokens alphabetically (order-invariant) |
| `first_token`            | Extract first word                     |
| `last_token`             | Extract last word                      |
| `substring:<start>:<end>`| Extract a substring by index           |
| `qgram:<N>`              | Character n-grams (sorted, top 5)      |

## Available Scorers

Scorers compute similarity between two field values in weighted matchkeys.

| Scorer          | Description                                    |
| --------------- | ---------------------------------------------- |
| `exact`         | 1.0 if values are identical, 0.0 otherwise     |
| `jaro_winkler`  | Jaro-Winkler string similarity (0.0 to 1.0)   |
| `levenshtein`   | Normalized Levenshtein similarity (0.0 to 1.0) |
| `token_sort`    | Token sort ratio via RapidFuzz (0.0 to 1.0)   |
| `soundex_match` | 1.0 if Soundex codes match, 0.0 otherwise      |
| `embedding`     | Cosine similarity of sentence-transformer embeddings (0.0 to 1.0) |

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

## Blocking Strategies

GoldenMatch supports multiple blocking strategies to balance recall and performance:

| Strategy | Description |
|----------|-------------|
| `static` | Group by blocking key (default) |
| `adaptive` | Static + recursive sub-blocking for oversized blocks |
| `sorted_neighborhood` | Sliding window over sorted records |
| `multi_pass` | Union of blocks from multiple passes (best for noisy data) |
| `ann` | Approximate nearest neighbor via FAISS on sentence-transformer embeddings |
| `canopy` | TF-IDF canopy clustering with loose/tight thresholds |

### Multi-Pass Blocking (Noisy Data)

```yaml
blocking:
  strategy: multi_pass
  passes:
    - key_fields: [name]
      transforms: [lowercase, "substring:0:8"]
    - key_fields: [name]
      transforms: [lowercase, token_sort, "substring:0:10"]
    - key_fields: [name]
      transforms: [lowercase, soundex]
```

### Embedding-Based Features

Install optional dependencies for semantic matching:

```bash
pip install goldenmatch[embeddings]
```

**Embedding scorer** — cosine similarity of sentence-transformer embeddings for semantic matching:

```yaml
matchkeys:
  - name: semantic_product
    type: weighted
    threshold: 0.75
    fields:
      - field: title
        scorer: embedding
        model: all-MiniLM-L6-v2
        weight: 0.6
      - field: manufacturer
        scorer: exact
        weight: 0.4
```

**ANN blocking** — approximate nearest neighbor blocking for large datasets:

```yaml
blocking:
  strategy: ann
  ann_column: title
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20
```

## Performance

### 1M Record Benchmark

GoldenMatch processes 1 million records in **~15 seconds** (exact matching, full pipeline including auto-fix, standardization, matching, clustering, and golden record generation):

| Stage | Time | % |
|-------|------|---|
| Ingest | 0.30s | 2% |
| Auto-fix | 2.29s | 15% |
| Standardize | 2.34s | 15% |
| Matchkeys | 0.17s | 1% |
| **Matching** | **0.29s** | **2%** |
| Clustering | 7.25s | 48% |
| Golden records | 2.51s | 17% |
| **Total** | **15.15s** | |

Results: 138,730 duplicate clusters found (175,602 pairs) with **100% precision** and **100% recall** against known ground truth.

### Leipzig Benchmark Results

Evaluated against the standard [University of Leipzig entity resolution benchmark datasets](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution):

| Dataset | Strategy | Precision | Recall | F1 | Time |
|---------|----------|-----------|--------|-----|------|
| **DBLP-ACM** (2.6K vs 2.3K) | exact title | 88.5% | 88.3% | 88.4% | 0.2s |
| **DBLP-ACM** | fuzzy title+authors+year | **97.0%** | **96.9%** | **97.0%** | 0.9s |
| **DBLP-ACM** | cascaded exact+fuzzy | 87.6% | 98.1% | 92.5% | 1.2s |
| **DBLP-Scholar** (2.6K vs 64K) | exact title | 76.7% | 47.8% | 58.9% | 0.5s |
| **DBLP-Scholar** | fuzzy title+year | 37.0% | 77.7% | 50.1% | 7.9s |
| **Abt-Buy** (1K vs 1K) | exact name | 100.0% | 0.9% | 1.8% | 0.0s |
| **Abt-Buy** | fuzzy name (token sort) | 46.7% | 31.0% | 37.3% | 0.3s |
| **Amazon-Google** (1.4K vs 3.2K) | exact title | 43.1% | 3.4% | 6.3% | 0.1s |
| **Amazon-Google** | fuzzy title+mfr | 32.2% | 26.3% | 29.0% | 0.8s |

**DBLP-ACM** achieves 97% F1 — competitive with published results. The e-commerce datasets (Abt-Buy, Amazon-Google) are semantic matching problems where the same product has different names across sources; the `embedding` scorer with ANN blocking is available for these use cases.

### Performance Notes

- **Polars**: All data loading and transformation runs on Polars with native expressions for maximum throughput.
- **Exact matching**: Uses Polars self-join (hash-based, O(n)) instead of Python pairwise comparison.
- **Fuzzy matching**: Uses vectorized `rapidfuzz.process.cdist` for NxN score matrices in C.
- **Embedding matching**: Sentence-transformer embeddings with FAISS ANN indexing for semantic similarity.
- **Blocking**: Six strategies available — static, adaptive, sorted neighborhood, multi-pass, ANN, and canopy clustering.
- **Cascading**: Exact matchkeys run first; matched pairs are excluded from expensive fuzzy comparisons.
- **Output formats**: CSV, Parquet, and Excel supported.

## License

MIT
