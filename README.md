# GoldenMatch

**Entity resolution toolkit — deduplicate records, match across sources, and maintain golden records. Works on files or live databases.**

Built with Polars, RapidFuzz, sentence-transformers, and FAISS. Zero-config mode auto-detects your data; optional LLM boost for harder datasets.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-605%2B%20passing-brightgreen)

---

## Features

- **Zero-config** — `goldenmatch dedupe file.csv` auto-detects columns, picks scorers, shows auto-config summary
- **Gold-themed TUI** — professional interactive interface with keyboard shortcuts, live threshold tuning, split-view results
- **8 scoring methods** — exact, Jaro-Winkler, Levenshtein, token sort, soundex, ensemble, embedding, record embedding
- **7 blocking strategies** — static, adaptive, sorted neighborhood, multi-pass, ANN, ann_pairs, canopy
- **Database sync** — incremental matching against Postgres with persistent ANN index and golden record versioning
- **LLM boost** — optional Claude/GPT-4 labeling + sentence-transformer fine-tuning for harder datasets
- **Golden records** — 5 merge strategies (most_complete, majority_vote, source_priority, most_recent, first_non_null)

## Installation

```bash
pip install goldenmatch                    # core (files only)
pip install goldenmatch[embeddings]        # + sentence-transformers, FAISS
pip install goldenmatch[llm]               # + Claude/OpenAI for LLM boost
pip install goldenmatch[postgres]          # + Postgres database sync
```

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

Best result: **Abt-Buy 59.5% F1** (up from 44.5% zero-shot) with 300 LLM labels and optimal train/score split.

## Benchmarks

### Leipzig Entity Resolution Benchmarks

| Dataset | Best Strategy | F1 | Time |
|---------|--------------|-----|------|
| **DBLP-ACM** (2.6K vs 2.3K) | multi-pass + fuzzy | **97.2%** | 2.7s |
| **DBLP-Scholar** (2.6K vs 64K) | multi-pass + fuzzy | **74.7%** | 83.9s |
| **Abt-Buy** (1K vs 1K) | LLM boost (optimal) | **59.5%** | 7 min |
| **Abt-Buy** | rec_emb + ann_pairs | 44.5% | 0.1s |
| **Amazon-Google** (1.4K vs 3.2K) | rec_emb + ann_pairs | **40.5%** | 0.3s |

### 1M Record Benchmark

1 million records deduplicated in **~15 seconds** on a laptop (exact matching, full pipeline).

## Interactive TUI

GoldenMatch includes a gold-themed interactive terminal UI:

- **Auto-config summary** — first screen shows detected columns, scorers, and blocking strategy with Run/Edit/Save options
- **Pipeline progress** — full-screen progress with stage tracker (✓/●/○) on first run, footer bar on re-runs
- **Split-view matches** — cluster list on the left, golden record + member details on the right
- **Live threshold slider** — arrow keys adjust threshold in 0.05 increments with instant cluster count preview
- **Keyboard shortcuts** — `1-5` jump to tabs, `F5` run, `?` show all shortcuts, `Ctrl+E` export

```
┌─────────────────────────────────────────────────┐
│  ⚡ GoldenMatch                                 │
├──────────────────┬──────────────────────────────┤
│ Clusters (1,247) │ Cluster #42 — 3 records      │
│                  │                              │
│ ▸ #42  3r  0.94 │ Golden: John Smith            │
│   #107 2r  0.91 │         john@test.com         │
│   #23  4r  0.88 │                              │
│   ...            │ Members:                      │
│                  │  John Smith  john@test.com    │
│                  │  Jon Smith   jon@test.com     │
│                  │  J. Smith    js@test.com      │
├──────────────────┴──────────────────────────────┤
│ F5:Run │ 1-5:Tabs │ ?:Help │ Q:Quit            │
└─────────────────────────────────────────────────┘
```

## Settings Persistence

GoldenMatch saves preferences across sessions:

- **Global**: `~/.goldenmatch/settings.yaml` — output mode, default model, API keys
- **Project**: `.goldenmatch.yaml` — column mappings, thresholds, blocking config

Settings tuned in the TUI can be saved to the project file. Next run picks them up automatically.

## CLI Reference

| Command | Description |
|---------|-------------|
| `goldenmatch dedupe FILE [...]` | Deduplicate one or more files |
| `goldenmatch match TARGET --against REF` | Match target against reference |
| `goldenmatch sync --table TABLE --connection-string URL` | Sync against database |
| `goldenmatch init` | Interactive config wizard |
| `goldenmatch config save/load/list/delete/show` | Manage config presets |
| `goldenmatch profile FILE` | Profile data quality |
| `goldenmatch interactive FILE [...]` | Launch TUI |

## Architecture

```
goldenmatch/
├── cli/            # Typer CLI commands (dedupe, match, sync)
├── config/         # Pydantic schemas, YAML loader, settings persistence
├── core/           # Pipeline modules (ingest, block, score, cluster, golden)
├── db/             # Database integration (connector, blocking, sync, reconcile)
├── tui/            # Textual TUI + MatchEngine
└── utils/          # Transforms, helpers
```

**605+ tests** covering all modules. Run with `pytest`.

## License

MIT
