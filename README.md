# GoldenMatch

**Entity resolution toolkit — deduplicate records, match across sources, and maintain golden records. Works on files or live databases.**

Built with Polars, RapidFuzz, sentence-transformers, and FAISS. Zero-config mode auto-detects your data; optional LLM boost for harder datasets.

[![PyPI](https://img.shields.io/pypi/v/goldenmatch?color=d4a017)](https://pypi.org/project/goldenmatch/)
[![Downloads](https://static.pepy.tech/badge/goldenmatch)](https://pepy.tech/project/goldenmatch)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-855%20passing-brightgreen)
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
- **10+ scoring methods** — exact, Jaro-Winkler, Levenshtein, token sort, soundex, ensemble, embedding, record embedding, dice, jaccard + plugin extensible
- **8+ blocking strategies** — static, adaptive, sorted neighborhood, multi-pass, ANN, ann_pairs, canopy, **learned** (data-driven predicate selection)
- **Fellegi-Sunter probabilistic matching** — EM-trained m/u probabilities, automatic threshold estimation, comparison vectors with 2/3-level agreement
- **Vertex AI embeddings** — 85%+ F1 accuracy with no GPU needed (Google Cloud managed API)
- **Database sync** — incremental Postgres matching with persistent ANN index and golden record versioning
- **REST API + MCP Server** — real-time matching via HTTP or Claude Desktop (12 tools: match, unmerge, explain, config advisor, etc.)
- **Review queue** — REST endpoint surfaces borderline pairs for data steward approval/rejection
- **Lineage tracking** — every merge decision saved to a JSON sidecar with per-field score breakdown
- **Daemon mode** — `goldenmatch watch --daemon` runs as a service with health endpoint and PID file
- **Anomaly detection** — flag fake emails, placeholder data, suspicious records
- **Merge preview + undo** — see what will change before writing, rollback any run or unmerge individual records
- **Active learning boost** — label 10 borderline pairs in the TUI, instantly retrain a classifier for 99% accuracy
- **Cluster confidence scoring** — weakly-connected clusters flagged with bottleneck pair identification
- **Single-record matching** — `match_one` primitive for streaming: embed, query ANN, score, return matches
- **Privacy-preserving matching** — bloom filter transforms + Dice/Jaccard scoring for fuzzy matching on encrypted PII
- **Before/after dashboard** — shareable HTML showing data transformation with charts
- **Schema-free matching** — auto-maps columns between different schemas (full_name -> first_name + last_name)
- **Cloud storage** — read directly from S3, GCS, or Azure Blob
- **API connector** — pull from Salesforce, HubSpot, or any REST/GraphQL API
- **Scheduled runs** — cron-like scheduling with run history
- **LLM scorer with budget controls** — GPT-4o-mini scores borderline pairs, boosting product matching from 44.5% to **66.3% F1** (precision 35%→95%) for just $0.04. Budget caps, model tiering, graceful degradation
- **LLM boost** — optional Claude/GPT-4 labeling + fine-tuning for harder datasets
- **Golden records** — 5 merge strategies (most_complete, majority_vote, source_priority, most_recent, first_non_null)
- **Parallel fuzzy scoring** — blocks scored concurrently via thread pool with intra-field early termination
- **Cross-encoder reranking** — re-score borderline pairs with a pre-trained cross-encoder for higher precision
- **Auto-select blocking** — histogram analysis picks the best blocking key automatically
- **Dynamic block splitting** — oversized blocks auto-split by highest-cardinality column (zero config)
- **Large dataset mode** — chunked processing for files that don't fit in memory
- **Plugin architecture** — extend with custom scorers, transforms, connectors, and golden strategies via pip-installable plugins
- **Enterprise connectors** — Snowflake, Databricks, BigQuery, HubSpot, Salesforce (optional deps)
- **DuckDB backend** — out-of-core processing for 10M+ records without Spark
- **Natural language explainability** — template-based per-pair and per-cluster explanations at zero LLM cost
- **Streaming / CDC mode** — incremental record matching with micro-batch or immediate processing
- **Multi-table graph ER** — match across entity types with cross-relationship evidence propagation
- **7 domain packs** — pre-built YAML rulebooks for electronics, software, healthcare, financial, real estate, people, retail
- **Evaluation CLI** — `goldenmatch evaluate` reports precision/recall/F1 against ground truth CSV
- **Incremental matching** — `goldenmatch incremental` matches new CSV records against an existing base dataset
- **GitHub Actions "Try It"** — zero-install demo via `workflow_dispatch` (paste a CSV URL, get results)
- **Codespaces ready** — one-click dev environment with `.devcontainer` config
- **dbt integration** — `dbt-goldenmatch` package for DuckDB-based entity resolution in dbt pipelines

## Installation

```bash
pip install goldenmatch                    # core (files only)
pip install goldenmatch[embeddings]        # + sentence-transformers, FAISS
pip install goldenmatch[llm]               # + Claude/OpenAI for LLM boost
pip install goldenmatch[postgres]          # + Postgres database sync
pip install goldenmatch[snowflake]        # + Snowflake connector
pip install goldenmatch[bigquery]         # + BigQuery connector
pip install goldenmatch[databricks]       # + Databricks connector
pip install goldenmatch[salesforce]       # + Salesforce connector
pip install goldenmatch[duckdb]           # + DuckDB backend

# Run the setup wizard to configure GPU, API keys, and database:
goldenmatch setup
```

## Setup Wizard

Run `goldenmatch setup` for an interactive walkthrough:

![Setup Wizard](docs/screenshots/setup-welcome.svg)

Guides you through GPU mode selection, Vertex AI / Colab / local GPU configuration, LLM boost API keys, and database sync — with copy-paste commands at every step.

![GPU Selection](docs/screenshots/setup-gpu.svg)

## Benchmarks (v0.3.1)

Tested on [Leipzig benchmark datasets](https://dbs.uni-leipzig.de/research/projects/object-matching/benchmark-datasets-for-entity-resolution) (DBLP-ACM, Abt-Buy).

### Accuracy

| Dataset | Strategy | Precision | Recall | F1 | Cost |
|---------|----------|-----------|--------|-----|------|
| DBLP-ACM (bibliographic) | Weighted fuzzy | 97.2% | 97.1% | **97.2%** | $0 |
| DBLP-ACM | Fellegi-Sunter (opt-in) | 98.8% | 57.6% | 72.8% | $0 |
| DBLP-ACM | Learned blocking | 97.6% | 96.3% | 96.9% | $0 |
| Abt-Buy (product) | Embedding + ANN | 35.5% | 59.4% | 44.5% | $0 |
| Abt-Buy | Model extraction + emb | 39.3% | 71.0% | 50.6% | $0 |
| Abt-Buy | **Domain + emb + LLM** | **94.8%** | **58.3%** | **72.2%** | **$0.04** |
| Amazon-Google (software) | emb+ANN + LLM | 63.3% | 35.2% | **45.3%** | $0.02 |

### Speed

| Records | Time | Throughput | Memory |
|---------|------|-----------|--------|
| 1,000 | 0.15s | 6,667 rec/s | 101 MB |
| 10,000 | 1.67s | 5,975 rec/s | 123 MB |
| 100,000 | 12.78s | **7,823 rec/s** | 546 MB |

Measured on a laptop (Windows 11, Python 3.12, 16GB RAM) with fuzzy + exact + golden record pipeline.

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
                              SQL blocking   10 scorers
                              ANN blocking   ensemble
                              7 strategies   embeddings
                                             parallel blocks
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
    rerank: true             # re-score borderline pairs with cross-encoder
    rerank_band: 0.1         # pairs within threshold +/- 0.1 get reranked
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

  - name: semantic
    type: weighted
    threshold: 0.80
    fields:
      - columns: [title, authors, venue]
        scorer: record_embedding
        weight: 1.0
        column_weights: {title: 2.0, authors: 1.0, venue: 0.5}  # bias embedding toward title

llm_scorer:
  enabled: true              # score borderline pairs with GPT/Claude
  auto_threshold: 0.95       # auto-accept pairs above this
  candidate_lo: 0.75         # LLM scores pairs in [0.75, 0.95]
  # provider: openai         # auto-detected from OPENAI_API_KEY
  # model: gpt-4o-mini       # default, cheapest option

blocking:
  strategy: adaptive         # static | adaptive | sorted_neighborhood | multi_pass | ann | ann_pairs | canopy
  auto_select: true          # auto-pick best key by histogram analysis
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
| `dice` | Dice coefficient on bloom filters | Privacy-preserving matching |
| `jaccard` | Jaccard similarity on bloom filters | Privacy-preserving matching |

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
| `learned` | Data-driven predicate selection (auto-discovers blocking rules) |

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

**Note:** LLM boost is most valuable for product matching with local models (MiniLM) where it improved Abt-Buy from 44.5% to 59.5% F1. For structured data (names, addresses, bibliographic), fuzzy matching alone achieves 97%+ F1.

## Benchmarks

### Leipzig Entity Resolution Benchmarks

| Dataset | Best Strategy | F1 | Cost |
|---------|--------------|-----|------|
| **DBLP-ACM** (2.6K vs 2.3K) | multi-pass + fuzzy | **97.2%** | $0 |
| **DBLP-Scholar** (2.6K vs 64K) | multi-pass + fuzzy | **74.7%** | $0 |
| **Abt-Buy** (1K vs 1K) | Vertex AI + GPT-4o-mini scorer | **81.7%** | ~$0.74 |
| **Abt-Buy** (zero-shot) | Vertex AI embeddings | **62.8%** | ~$0.05 |
| **Amazon-Google** (1.4K vs 3.2K) | Vertex AI + reranking | **44.0%** | ~$0.10 |

**Structured data (names, addresses, bibliographic):** RapidFuzz multi-pass fuzzy matching at 97.2% — zero cost, zero labels. **Product matching:** Vertex AI embeddings for candidate generation + GPT-4o-mini scorer for borderline pairs achieves 81.7% at ~$0.74 total cost.

### Throughput (Scale Curve)

Measured on a laptop (17GB RAM) with exact + fuzzy matching, blocking, clustering, and golden record generation:

| Records | Time | Throughput | Pairs Found | Memory |
|---------|------|------------|-------------|--------|
| 1,000 | 0.2s | 5,500 rec/s | 210 | 101 MB |
| 10,000 | 1.4s | 7,300 rec/s | 7,000 | 123 MB |
| 100,000 | 12s | **8,200 rec/s** | 571,000 | 544 MB |

**Fuzzy matching speedup:** Parallel block scoring + intra-field early termination reduced 100K fuzzy matching from ~100s to **~39s** (2.5x) through the pipeline. The 1M exact-only benchmark runs in **7.8s**.

For datasets over 1M records, use `goldenmatch sync` (database mode) with incremental matching and persistent ANN indexing. See [Large Dataset Mode](#large-dataset-mode).

### How GoldenMatch Compares

| | **GoldenMatch** | **dedupe** | **Splink** | **Zingg** | **Ditto** |
|---|---|---|---|---|---|
| Abt-Buy F1 | **81.7%** | ~75% | ~70% | ~80% | 89.3% |
| DBLP-ACM F1 | **97.2%** | ~96% | ~95% | ~96% | 99.0% |
| Training required | No | Yes | Yes | Yes | Yes (1000+) |
| Zero-config | Yes | No | No | No | No |
| Interactive TUI | Yes | No | No | No | No |
| Database sync | Postgres | Cloud (paid) | No | No | No |
| REST API / MCP | Both | Cloud only | No | No | No |
| GPU required | No | No | No | Spark | Yes |

GoldenMatch's sweet spot is **ease of use + competitive accuracy**. On bibliographic matching (DBLP-ACM), GoldenMatch hits 97.2% with zero config. On product matching (Abt-Buy), the LLM scorer reaches 81.7% — within 8pts of Ditto's 89.3%, but with zero training labels and no GPU. Ditto requires 1000+ hand-labeled pairs and a GPU.

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
- **Keyboard shortcuts** — `1-6` jump to tabs (Data, Config, Matches, Golden, Boost, Export), `F5` run, `?` show all shortcuts, `Ctrl+E` export

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
| `goldenmatch watch --table TABLE` | Live stream mode (continuous polling, `--daemon` for service mode) |
| `goldenmatch schedule --every 1h FILE` | Run deduplication on a schedule |
| `goldenmatch serve FILE [...]` | Start REST API server |
| `goldenmatch mcp-serve FILE [...]` | Start MCP server (Claude Desktop) |
| `goldenmatch rollback RUN_ID` | Undo a previous merge run |
| `goldenmatch unmerge RECORD_ID` | Remove a record from its cluster |
| `goldenmatch runs` | List previous runs for rollback |
| `goldenmatch init` | Interactive config wizard |
| `goldenmatch interactive FILE [...]` | Launch TUI |
| `goldenmatch profile FILE` | Profile data quality |
| `goldenmatch evaluate FILE --gt GT.csv` | Evaluate matching against ground truth |
| `goldenmatch incremental BASE --new NEW` | Match new records against existing base |
| `goldenmatch analyze-blocking FILE` | Analyze data and suggest blocking strategies |
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
| `--daemon` | Run watch mode as a background service with health endpoint |
| `s3://` / `gs://` / `az://` | Read directly from cloud storage |

## Architecture

```
goldenmatch/
├── cli/            # 19 CLI commands (Typer)
├── config/         # Pydantic schemas, YAML loader, settings
├── core/           # Pipeline: ingest, block, score, cluster, golden, explainer,
│                   #   report, dashboard, graph, anomaly, diff, rollback,
│                   #   schema_match, chunked, cloud_ingest, api_connector, scheduler,
│                   #   llm_scorer, lineage, match_one, evaluate, gpu, vertex_embedder,
│                   #   probabilistic, learned_blocking, streaming, graph_er, domain
├── domains/        # 7 built-in YAML domain packs (electronics, software, healthcare, ...)
├── plugins/        # Plugin system (scorers, transforms, connectors, golden strategies)
├── connectors/     # Enterprise connectors (Snowflake, Databricks, BigQuery, HubSpot, Salesforce)
├── backends/       # DuckDB backend for out-of-core processing
├── db/             # Postgres: connector, sync, reconcile, clusters, ANN index
├── api/            # REST API server
├── mcp/            # MCP server for Claude Desktop
├── tui/            # Gold-themed Textual TUI + setup wizard
└── utils/          # Transforms, helpers
```

**Run tests:** `pytest` (855 tests)

## License

MIT
