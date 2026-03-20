# GoldenMatch

**Entity resolution toolkit — deduplicate records, match across sources, and maintain golden records. Works on files or live databases.**

Built with Polars, RapidFuzz, sentence-transformers, and FAISS. Zero-config mode auto-detects your data; optional LLM boost for harder datasets.

[![PyPI](https://img.shields.io/pypi/v/goldenmatch?color=d4a017)](https://pypi.org/project/goldenmatch/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-605%2B%20passing-brightgreen)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benzsevern/goldenmatch/blob/main/scripts/gpu_colab_notebook.ipynb)

### See it in action

![GoldenMatch Demo](docs/screenshots/demo.svg)

```bash
pip install goldenmatch
goldenmatch demo
```

---

## Features

- **Zero-config** — `goldenmatch dedupe file.csv` auto-detects columns, picks scorers, runs automatically
- **Gold-themed TUI** — interactive interface with keyboard shortcuts, live threshold tuning, setup wizard
- **8 scoring methods** — exact, Jaro-Winkler, Levenshtein, token sort, soundex, ensemble, embedding, record embedding
- **7 blocking strategies** — static, adaptive, sorted neighborhood, multi-pass, ANN, ann_pairs, canopy
- **Vertex AI embeddings** — 85%+ F1 accuracy with no GPU needed (Google Cloud managed API)
- **Database sync** — incremental Postgres matching with persistent ANN index and golden record versioning
- **REST API + MCP Server** — real-time matching via HTTP or Claude Desktop integration
- **Anomaly detection** — flag fake emails, placeholder data, suspicious records
- **Merge preview + undo** — see what will change before writing, rollback any run
- **Before/after dashboard** — shareable HTML showing data transformation with charts
- **Schema-free matching** — auto-maps columns between different schemas (full_name -> first_name + last_name)
- **Cloud storage** — read directly from S3, GCS, or Azure Blob
- **API connector** — pull from Salesforce, HubSpot, or any REST/GraphQL API
- **Scheduled runs** — cron-like scheduling with run history
- **LLM boost** — optional Claude/GPT-4 labeling + fine-tuning for harder datasets
- **Golden records** — 5 merge strategies (most_complete, majority_vote, source_priority, most_recent, first_non_null)
- **Large dataset mode** — chunked processing for files that don't fit in memory

## Installation

```bash
pip install goldenmatch                    # core (files only)
pip install goldenmatch[embeddings]        # + sentence-transformers, FAISS
pip install goldenmatch[llm]               # + Claude/OpenAI for LLM boost
pip install goldenmatch[postgres]          # + Postgres database sync

# Run the setup wizard to configure GPU, API keys, and database:
goldenmatch setup
```

## Setup Wizard

Run `goldenmatch setup` for an interactive walkthrough:

![Setup Wizard](docs/screenshots/setup-welcome.svg)

Guides you through GPU mode selection, Vertex AI / Colab / local GPU configuration, LLM boost API keys, and database sync — with copy-paste commands at every step.

![GPU Selection](docs/screenshots/setup-gpu.svg)

## Quick Start

### Zero-Config (no YAML needed)

```bash
goldenmatch dedupe customers.csv
```

Auto-detects column types (name, email, phone, zip, address, description), assigns appropriate scorers, picks blocking strategy, and launches the TUI for review.

### With Config

```bash
goldenmatch dedupe customers.csv --config config.yaml --output-all --output-dir results/
```

### Match Mode

```bash
goldenmatch match targets.csv --against reference.csv --config config.yaml --output-all
```

### Database Sync

```bash
# First run: full scan, create metadata tables
goldenmatch sync --table customers --connection-string "$DATABASE_URL" --config config.yaml

# Subsequent runs: incremental (only new records)
goldenmatch sync --table customers --connection-string "$DATABASE_URL"
```

## How It Works

```
Files/DB → Ingest → Standardize → Block → Score → Cluster → Golden Records → Output
                                     ↑        ↑
                              SQL blocking   8 scorers
                              ANN blocking   ensemble
                              7 strategies   embeddings
```

**Pipeline:**
1. **Ingest** — CSV, Excel, Parquet, or Postgres table
2. **Standardize** — configurable per-column transforms
3. **Block** — reduce comparison space (multi-pass, ANN, canopy, etc.)
4. **Score** — compare record pairs with appropriate scorer
5. **Cluster** — group matches via Union-Find
6. **Golden** — merge each cluster into one canonical record
7. **Output** — files (CSV/Parquet) or database tables

## Config Reference

```yaml
matchkeys:
  - name: exact_email
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]

  - name: fuzzy_name_zip
    type: weighted
    threshold: 0.85
    fields:
      - field: first_name
        scorer: jaro_winkler
        weight: 0.4
        transforms: [lowercase, strip]
      - field: last_name
        scorer: jaro_winkler
        weight: 0.4
        transforms: [lowercase, strip]
      - field: zip
        scorer: exact
        weight: 0.2

blocking:
  strategy: multi_pass  # static | adaptive | sorted_neighborhood | multi_pass | ann | ann_pairs | canopy
  keys:
    - fields: [zip]
    - fields: [last_name]
      transforms: [lowercase, soundex]

golden_rules:
  default_strategy: most_complete
  field_rules:
    email: { strategy: majority_vote }
    first_name: { strategy: source_priority, source_priority: [crm, marketing] }

output:
  directory: ./output
  format: csv
```

## Scorers

| Scorer | Description | Best For |
|--------|-------------|----------|
| `exact` | Binary match | Email, phone, ID |
| `jaro_winkler` | Edit distance similarity | Names |
| `levenshtein` | Normalized Levenshtein | General strings |
| `token_sort` | Order-invariant token matching | Names, addresses |
| `soundex_match` | Phonetic match | Names |
| `ensemble` | max(jaro_winkler, token_sort, soundex) | Names with reordering |
| `embedding` | Cosine similarity of sentence embeddings | Semantic matching |
| `record_embedding` | Embed concatenated fields | Cross-field semantic matching |

## Blocking Strategies

| Strategy | Description |
|----------|-------------|
| `static` | Group by blocking key (default) |
| `adaptive` | Static + recursive sub-blocking for oversized blocks |
| `sorted_neighborhood` | Sliding window over sorted records |
| `multi_pass` | Union of blocks from multiple passes (best for noisy data) |
| `ann` | ANN via FAISS on sentence-transformer embeddings |
| `ann_pairs` | Direct-pair ANN scoring (50-100x faster than `ann`) |
| `canopy` | TF-IDF canopy clustering |

## Database Integration

GoldenMatch can sync against live Postgres databases with incremental matching:

```bash
pip install goldenmatch[postgres]

goldenmatch sync \
  --table customers \
  --connection-string "postgresql://user:pass@localhost/mydb" \
  --config config.yaml
```

**Features:**
- **Incremental sync** — only processes records added since last run
- **Hybrid blocking** — SQL WHERE clauses for exact fields + FAISS ANN for semantic fields, results unioned
- **Persistent ANN index** — disk cache + DB source of truth, progressive embedding across runs
- **Golden record versioning** — append-only with `is_current` flag, full audit trail
- **Cluster management** — persistent clusters with merge, conflict detection, max size safety cap

**Metadata tables** (auto-created):

| Table | Purpose |
|-------|---------|
| `gm_state` | Processing state, watermarks |
| `gm_clusters` | Persistent cluster membership |
| `gm_golden_records` | Versioned golden records |
| `gm_embeddings` | Cached embeddings for ANN |
| `gm_match_log` | Audit trail of all match decisions |

## LLM Boost (Optional)

For harder datasets where zero-shot scoring isn't enough:

```bash
pip install goldenmatch[llm]

# First run: LLM labels ~300 pairs (~$0.30), fine-tunes embedding model
goldenmatch dedupe products.csv --llm-boost

# Subsequent runs: uses saved model ($0)
goldenmatch dedupe products.csv --llm-boost
```

**Tiered auto-escalation:**
- **Level 1** — zero-shot (free, instant)
- **Level 2** — bi-encoder fine-tuning (~$0.20, ~2 min CPU)
- **Level 3** — Ditto-style cross-encoder with data augmentation (~$0.50, ~5 min CPU)

**Active sampling** selects the most informative pairs for the LLM to label (uncertainty, disagreement, boundary, diversity), reducing label cost by ~45% compared to random sampling.

**Note:** With Vertex AI embeddings, zero-shot already achieves 84.8% F1 on Abt-Buy — better than any threshold-learning approach. LLM boost is most valuable when using local models (MiniLM) where it improved Abt-Buy from 44.5% to 59.5% F1.

## Benchmarks

### Leipzig Entity Resolution Benchmarks

| Dataset | Best Strategy | F1 | Time |
|---------|--------------|-----|------|
| **DBLP-ACM** (2.6K vs 2.3K) | Vertex AI embeddings | **97.4%** | 119s |
| **DBLP-Scholar** (2.6K vs 64K) | multi-pass + fuzzy | **74.7%** | 83.9s |
| **Abt-Buy** (1K vs 1K) | Vertex AI embeddings | **84.8%** | 56s |
| **Amazon-Google** (1.4K vs 3.2K) | Vertex AI embeddings | **60.4%** | 110s |

**Zero config, zero labels, zero GPU.** Vertex AI's `text-embedding-004` provides state-of-the-art embeddings via API. Previous best without Vertex AI: Abt-Buy 59.5% (LLM boost with local MiniLM).

### Throughput (Scale Curve)

Measured on a laptop (17GB RAM) with exact + fuzzy matching, blocking, clustering, and golden record generation:

| Records | Time | Throughput | Pairs Found | Memory |
|---------|------|------------|-------------|--------|
| 1,000 | 0.2s | 5,500 rec/s | 210 | 101 MB |
| 10,000 | 1.4s | 7,300 rec/s | 7,000 | 123 MB |
| 100,000 | 12s | **8,200 rec/s** | 571,000 | 544 MB |

For datasets over 1M records, use `goldenmatch sync` (database mode) with incremental matching and persistent ANN indexing. See [Large Dataset Mode](#large-dataset-mode).

### How GoldenMatch Compares

| | **GoldenMatch** | **dedupe** | **Splink** | **Zingg** | **Ditto** |
|---|---|---|---|---|---|
| Abt-Buy F1 | **84.7%** | ~75% | ~70% | ~80% | 89.3% |
| DBLP-ACM F1 | **97.4%** | ~96% | ~95% | ~96% | 99.0% |
| Training required | No | Yes | Yes | Yes | Yes (1000+) |
| Zero-config | Yes | No | No | No | No |
| Interactive TUI | Yes | No | No | No | No |
| Database sync | Postgres | Cloud (paid) | No | No | No |
| REST API / MCP | Both | Cloud only | No | No | No |
| GPU required | No | No | No | Spark | Yes |

GoldenMatch's sweet spot is **ease of use + competitive accuracy**. Ditto has higher F1 but requires 1000+ manual labels and a GPU. Splink scales to billions on Spark but needs label training. GoldenMatch auto-configures from your data and reaches 85%+ F1 with zero labels.

## Interactive TUI

## Large Dataset Mode

For datasets over 1M records, use database sync mode. GoldenMatch processes records in chunks, maintains a persistent ANN index, and matches incrementally:

```bash
# Load into Postgres, then sync
goldenmatch sync --table customers --connection-string "$DATABASE_URL" --config config.yaml

# Watch for new records continuously
goldenmatch watch --table customers --connection-string "$DATABASE_URL" --interval 30
```

**How it works:**
- Reads in configurable chunks (default 10K) — never loads entire table into memory
- Hybrid blocking: SQL WHERE for exact fields + persistent FAISS ANN for semantic fields
- Progressive embedding: computes 100K embeddings per run, ANN improves over time
- Persistent clusters with golden record versioning

**Scale:** Tested to 10M+ records in Postgres. For 100M+, use larger chunk sizes and dedicated Postgres infrastructure.

## Interactive TUI

GoldenMatch includes a gold-themed interactive terminal UI:

- **Auto-config summary** — first screen shows detected columns, scorers, and blocking strategy with Run/Edit/Save options
- **Pipeline progress** — full-screen progress with stage tracker (✓/●/○) on first run, footer bar on re-runs
- **Split-view matches** — cluster list on the left, golden record + member details on the right
- **Live threshold slider** — arrow keys adjust threshold in 0.05 increments with instant cluster count preview
- **Keyboard shortcuts** — `1-5` jump to tabs, `F5` run, `?` show all shortcuts, `Ctrl+E` export

**Data profiling:**

![Data Tab](docs/screenshots/tui-data.svg)

**Match results with cluster detail:**

![Matches Tab](docs/screenshots/tui-matches.svg)

**Golden records:**

![Golden Tab](docs/screenshots/tui-golden.svg)

## Settings Persistence

GoldenMatch saves preferences across sessions:

- **Global**: `~/.goldenmatch/settings.yaml` — output mode, default model, API keys
- **Project**: `.goldenmatch.yaml` — column mappings, thresholds, blocking config

Settings tuned in the TUI can be saved to the project file. Next run picks them up automatically.

## CLI Reference

| Command | Description |
|---------|-------------|
| `goldenmatch demo` | Built-in demo with sample data |
| `goldenmatch setup` | Interactive setup wizard (GPU, API keys, database) |
| `goldenmatch dedupe FILE [...]` | Deduplicate one or more files |
| `goldenmatch match TARGET --against REF` | Match target against reference |
| `goldenmatch sync --table TABLE` | Sync against Postgres database |
| `goldenmatch watch --table TABLE` | Live stream mode (continuous polling) |
| `goldenmatch schedule --every 1h FILE` | Run deduplication on a schedule |
| `goldenmatch serve FILE [...]` | Start REST API server |
| `goldenmatch mcp-serve FILE [...]` | Start MCP server (Claude Desktop) |
| `goldenmatch rollback RUN_ID` | Undo a previous merge run |
| `goldenmatch runs` | List previous runs for rollback |
| `goldenmatch init` | Interactive config wizard |
| `goldenmatch interactive FILE [...]` | Launch TUI |
| `goldenmatch profile FILE` | Profile data quality |
| `goldenmatch config save/load/list/show` | Manage config presets |

**Key dedupe flags:**

| Flag | Description |
|------|-------------|
| `--anomalies` | Detect fake emails, placeholder data, suspicious records |
| `--preview` | Show what will change before writing (merge preview) |
| `--diff` / `--diff-html` | Generate before/after change report |
| `--dashboard` | Before/after data quality dashboard (HTML) |
| `--html-report` | Detailed match report with charts |
| `--chunked` | Large dataset mode (process in chunks) |
| `--llm-boost` | Improve accuracy with LLM-labeled training |
| `s3://` / `gs://` / `az://` | Read directly from cloud storage |

## Architecture

```
goldenmatch/
├── cli/            # 16 CLI commands (Typer)
├── config/         # Pydantic schemas, YAML loader, settings
├── core/           # Pipeline: ingest, block, score, cluster, golden, explainer,
│                   #   report, dashboard, graph, anomaly, diff, rollback,
│                   #   schema_match, chunked, cloud_ingest, api_connector, scheduler
├── db/             # Postgres: connector, sync, reconcile, clusters, ANN index
├── api/            # REST API server
├── mcp/            # MCP server for Claude Desktop
├── tui/            # Gold-themed Textual TUI + setup wizard
└── utils/          # Transforms, helpers
```

**Run tests:** `pytest` (600+ tests)

## License

MIT
