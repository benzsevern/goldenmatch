<!-- mcp-name: io.github.benzsevern/goldenmatch -->
# GoldenMatch

**Find duplicate records in 30 seconds. No rules to write, no models to train.**
Built by [Ben Severn](https://bensevern.dev).

![GoldenMatch Demo](docs/screenshots/demo.svg)

```bash
# Python
pip install goldenmatch

# TypeScript / Node.js
npm install goldenmatch
```

```bash
goldenmatch dedupe customers.csv
```

[![PyPI](https://img.shields.io/pypi/v/goldenmatch?color=d4a017)](https://pypi.org/project/goldenmatch/)
[![npm](https://img.shields.io/npm/v/goldenmatch?color=339933&label=npm)](https://www.npmjs.com/package/goldenmatch)
[![CI](https://github.com/benzsevern/goldenmatch/actions/workflows/ci.yml/badge.svg)](https://github.com/benzsevern/goldenmatch/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/benzsevern/goldenmatch/graph/badge.svg)](https://codecov.io/gh/benzsevern/goldenmatch)
[![Downloads](https://static.pepy.tech/badge/goldenmatch/month)](https://pepy.tech/project/goldenmatch)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-benzsevern.github.io%2Fgoldenmatch-d4a017)](https://benzsevern.github.io/goldenmatch/)
[![DQBench ER](https://img.shields.io/badge/DQBench%20ER-95.30-gold)](https://github.com/benzsevern/dqbench)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benzsevern/goldenmatch/blob/main/scripts/gpu_colab_notebook.ipynb)

> **v1.5.0 is out** — auto-config now runs a preflight + postflight verification layer. Bibliographic and domain-extracted schemas no longer crash under zero-config, remote-asset scorers are demoted by default, and every `DedupeResult` carries an inspectable `postflight_report`. See [Auto-Config Verification](#auto-config-verification-v150).

---

## Why GoldenMatch?

- **Zero-config** — auto-detects columns, picks scorers, and runs. No training data needed
- **97.2% F1** on DBLP-ACM out of the box. [DQBench ER score: 95.30](https://github.com/benzsevern/dqbench)
- **Privacy-preserving** — match across organizations without sharing raw data (PPRL, 92.4% F1)
- **30 MCP tools** — use from Claude Desktop, Claude Code, or any AI assistant ([Smithery](https://smithery.ai/servers/benzsevern/goldenmatch))
- **Production-ready** — Postgres sync, daemon mode, lineage tracking, review queues

### Choose your path

| I want to... | Go here |
|--------------|---------|
| Deduplicate a CSV right now | [Quick Start](https://benzsevern.github.io/goldenmatch/quick-start) |
| Use from Claude Desktop / AI assistant | [MCP Server](https://benzsevern.github.io/goldenmatch/mcp) |
| Build AI agents that deduplicate | [ER Agent (A2A)](https://benzsevern.github.io/goldenmatch/agent) |
| Write Python code | [Python API](https://benzsevern.github.io/goldenmatch/python-api) |
| Write TypeScript / Node.js | [TypeScript API](https://benzsevern.github.io/goldenmatch/typescript) |
| Deploy to Vercel Edge / Cloudflare Workers | [TypeScript API](https://benzsevern.github.io/goldenmatch/typescript) |
| Use the interactive TUI | [TUI Guide](https://benzsevern.github.io/goldenmatch/tui) |

---

<details>
<summary><strong>All features</strong> (click to expand)</summary>

### Matching
- **10+ scoring methods** — exact, Jaro-Winkler, Levenshtein, token sort, soundex, ensemble, embedding, record embedding, dice, jaccard + plugin extensible
- **8+ blocking strategies** — static, adaptive, sorted neighborhood, multi-pass, ANN, ann_pairs, canopy, **learned** (data-driven predicate selection)
- **Fellegi-Sunter probabilistic matching** — EM-trained m/u probabilities, automatic threshold estimation
- **LLM scorer with budget controls** — GPT-4o-mini scores borderline pairs for just $0.04. Budget caps, model tiering, graceful degradation
- **Cross-encoder reranking** — re-score borderline pairs with a pre-trained cross-encoder for higher precision
- **Schema-free matching** — auto-maps columns between different schemas (full_name -> first_name + last_name)

### Data Quality
- **GoldenCheck integration** — `pip install goldenmatch[quality]` adds data quality scanning (encoding, Unicode, format validation)
- **GoldenFlow transforms** — `pip install goldenmatch[transform]` normalizes phone numbers, dates, categorical spelling
- **Anomaly detection** — flag fake emails, placeholder data, suspicious records

### Golden Records
- **5 merge strategies** — most_complete, majority_vote, source_priority, most_recent, first_non_null
- **Quality-weighted survivorship** — fields scored by source quality from GoldenCheck
- **Field-level provenance** — tracks which source row contributed each field
- **Cluster quality scoring** — clusters labeled `strong`/`weak`/`split`; oversized clusters auto-split via MST

### Privacy
- **PPRL multi-party linkage** — match across organizations without sharing raw data (92.4% F1 on FEBRL4)
- **PPRL auto-configuration** — profiles your data and picks optimal fields, bloom filter parameters, and threshold

### Integration
- **REST API + MCP Server** — 30 tools for matching, explaining, reviewing, data quality, and transforms
- **A2A Agent** — 10 skills for AI-to-AI autonomous entity resolution
- **Database sync** — incremental Postgres matching with persistent ANN index
- **Enterprise connectors** — Snowflake, Databricks, BigQuery, HubSpot, Salesforce
- **DuckDB backend** — out-of-core processing for 10M+ records without Spark
- **Ray distributed backend** — scale to 50M+ records with `pip install goldenmatch[ray]`
- **dbt integration** — `dbt-goldenmatch` package for DuckDB-based ER in dbt pipelines

### Developer Experience
- **Gold-themed TUI** — interactive interface with keyboard shortcuts, live threshold tuning
- **Active learning boost** — label 10 borderline pairs in the TUI, retrain a classifier for 99% accuracy
- **Review queue** — REST endpoint surfaces borderline pairs for data steward approval
- **Merge preview + undo** — rollback any run or unmerge individual records
- **Lineage tracking** — every merge decision saved with per-field score breakdown
- **Natural language explainability** — template-based per-pair and per-cluster explanations at zero LLM cost
- **Evaluation CLI** — `goldenmatch evaluate` reports precision/recall/F1 against ground truth
- **7 domain packs** — electronics, software, healthcare, financial, real estate, people, retail
- **Plugin architecture** — extend with custom scorers, transforms, connectors via pip
- **Streaming / CDC mode** — incremental record matching with micro-batch or immediate processing
- **GitHub Actions "Try It"** — zero-install demo via `workflow_dispatch`
- **Codespaces ready** — one-click dev environment

</details>

## TypeScript / Node.js

GoldenMatch ships an npm package with full feature parity — same scorers, clustering, golden records, and YAML configs.

```bash
npm install goldenmatch
```

```typescript
import { dedupe } from "goldenmatch";

const rows = [
  { id: 1, name: "John Smith", email: "john@example.com", zip: "12345" },
  { id: 2, name: "Jon Smith",  email: "john@example.com", zip: "12345" },
  { id: 3, name: "Jane Doe",   email: "jane@example.com", zip: "54321" },
];

const result = dedupe(rows, {
  fuzzy: { name: 0.85 },
  blocking: ["zip"],
  threshold: 0.85,
});

console.log(result.stats);  // { totalRecords: 3, totalClusters: 2, ... }
```

- **Edge-safe core** — runs in browsers, Vercel Edge Runtime, Cloudflare Workers, Deno
- **Feature parity** with Python: fuzzy scorers, probabilistic Fellegi-Sunter, PPRL, graph ER, LLM reranking, MCP/REST/A2A servers, 11+ CLI commands, interactive TUI
- **478 tests, strict TypeScript** (`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`)
- **Zero-dep install** works — optional peer deps unlock native paths (hnswlib-node, @huggingface/transformers for ONNX cross-encoder, piscina for worker threads, pg/duckdb/snowflake for data connectors)

Full docs: [benzsevern.github.io/goldenmatch/typescript](https://benzsevern.github.io/goldenmatch/typescript)
See [packages/goldenmatch-js/examples/](packages/goldenmatch-js/examples/) for 10+ usage examples.

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
pip install goldenmatch[quality]          # + GoldenCheck data quality scanning

# Run the setup wizard to configure GPU, API keys, and database:
goldenmatch setup
```

## Python API

GoldenMatch exposes 95 functions and classes from a single import. See [examples/](examples/) for complete runnable scripts.

```python
import goldenmatch as gm
```

### Quick Start

```python
import goldenmatch as gm

# Deduplicate a CSV (zero-config)
result = gm.dedupe("customers.csv")

# Exact + fuzzy matching
result = gm.dedupe("customers.csv", exact=["email"], fuzzy={"name": 0.85, "zip": 0.95})
result.golden.write_csv("deduped.csv")
print(result)  # DedupeResult(records=5000, clusters=847, match_rate=12.0%)

# Match across files
result = gm.match("new_customers.csv", "master.csv", fuzzy={"name": 0.85})
result.to_csv("matches.csv")

# With YAML config
result = gm.dedupe("data.csv", config="config.yaml")

# With LLM scorer for product matching
result = gm.dedupe("products.csv", fuzzy={"title": 0.80}, llm_scorer=True)

# With Ray backend for large datasets
result = gm.dedupe("huge.parquet", exact=["email"], backend="ray")
```

### Auto-Config Verification (v1.5.0)

Zero-config used to crash on bibliographic and domain-extracted schemas — auto-config would emit a matchkey referencing `__title_key__` without enabling `config.domain`, and the pipeline would raise `ValueError: Missing required columns`. v1.5.0 closes the gap with a preflight + postflight verification layer that runs automatically around `auto_configure_df`.

**Preflight** (`gm.preflight`) runs 6 checks at the end of `auto_configure_df`:

- column resolution (auto-repairs missing domain-extracted columns by enabling `config.domain`)
- cardinality bounds on exact matchkeys (drops near-unique and near-constant keys)
- block-size sanity (flags blocks that would stall the scorer)
- remote-asset demotion (any `embedding`, `record_embedding`, or cross-encoder rerank is demoted unless you pass `allow_remote_assets=True`)
- confidence-gated weight capping (low-confidence fields cap at weight 0.3)

Unrepairable issues raise `ConfigValidationError` with the full `PreflightReport` attached as `err.report`. Repaired issues stay on the report as `findings` with `repaired=True`.

**Postflight** (`gm.postflight`) runs 4 signals after scoring, before clustering:

- score-distribution histogram + bimodality detection (auto-nudges threshold on clear bimodality)
- blocking-recall estimate (gated at 10K+ rows)
- preliminary cluster sizes + oversized-cluster bottleneck pair
- threshold-band overlap percentage (advises `--llm-auto` when overlap > 20% and LLM is off)

The report attaches to `DedupeResult.postflight_report` / `MatchResult.postflight_report`.

```python
import goldenmatch as gm
import polars as pl

df = pl.read_csv("bibliography.csv")

# Zero-config -- preflight + postflight run automatically
result = gm.dedupe_df(df)

# Inspect the preflight report (private-by-convention underscore)
for finding in result.config._preflight_report.findings:
    print(f"[{finding.severity}] {finding.check}: {finding.message}")

# Inspect postflight signals (public)
sig = result.postflight_report.signals
print(f"Scored {sig['total_pairs_scored']} pairs")
print(f"Threshold overlap: {sig['threshold_overlap_pct']:.1%}")
print(f"Oversized clusters: {len(sig['oversized_clusters'])}")
```

**Offline by default.** Remote-asset scorers are demoted unless you opt in:

```python
cfg = gm.auto_configure_df(df, allow_remote_assets=True)  # loads cross-encoder etc.
```

**Strict mode for parity runs.** `strict=True` still computes postflight signals and emits advisories, but skips threshold adjustments — use it for DQBench, regression suites, and any reproducible output:

```python
cfg = gm.auto_configure_df(df, strict=True)
```

**New classifier smarts in v1.5.0:**

- Columns with cardinality ≥ 0.95 are classified as `identifier`, not `phone` / `zip` / `numeric`.
- New `year` col_type routes to blocking, not scoring.
- New `multi_name` col_type handles comma/semicolon-delimited author-style fields.
- Low-confidence fields (< 0.5) cap at weight 0.3.

See `examples/verification_inspection.py` and `examples/strict_mode_parity.py` for runnable walkthroughs.

### Privacy-Preserving Linkage

```python
import goldenmatch as gm

# Auto-configured PPRL (picks fields and threshold automatically)
result = gm.pprl_link("hospital_a.csv", "hospital_b.csv")
print(f"Found {result['match_count']} matches across {len(result['clusters'])} clusters")

# Manual field selection
result = gm.pprl_link("party_a.csv", "party_b.csv",
    fields=["first_name", "last_name", "dob", "zip"],
    threshold=0.85, security_level="high")

# Auto-config analysis
config = gm.pprl_auto_config(df)
print(config.recommended_fields)  # ['first_name', 'last_name', 'zip_code', 'birth_year']
```

### Evaluate Accuracy

```python
import goldenmatch as gm

# Measure precision/recall/F1 against ground truth
metrics = gm.evaluate("data.csv", config="config.yaml", ground_truth="gt.csv")
print(f"F1: {metrics['f1']:.1%}, Precision: {metrics['precision']:.1%}")

# Evaluate programmatically
result = gm.evaluate_pairs(predicted_pairs, ground_truth_set)
print(result.f1)
```

### Build Configs Programmatically

```python
import goldenmatch as gm

# Auto-generate config from data
config = gm.auto_configure([("data.csv", "source")])

# Or build manually
config = gm.GoldenMatchConfig(
    matchkeys=[
        gm.MatchkeyConfig(name="exact_email", type="exact",
            fields=[gm.MatchkeyField(field="email", transforms=["lowercase"])]),
        gm.MatchkeyConfig(name="fuzzy_name", type="weighted", threshold=0.85,
            fields=[
                gm.MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7),
                gm.MatchkeyField(field="zip", scorer="exact", weight=0.3),
            ]),
    ],
    blocking=gm.BlockingConfig(strategy="learned"),
    llm_scorer=gm.LLMScorerConfig(enabled=True, mode="cluster"),
    backend="ray",
)
```

### Streaming / Incremental

```python
import goldenmatch as gm

# Match a single new record against existing data
matches = gm.match_one(new_record, existing_df, matchkey)

# Stream processor for continuous matching
processor = gm.StreamProcessor(df, config)
matches = processor.process_record(new_record)
```

### Advanced Features

```python
import goldenmatch as gm

# Domain extraction
rulebooks = gm.discover_rulebooks()  # 7 built-in packs
enhanced_df, low_conf = gm.extract_with_rulebook(df, "title", rulebooks["electronics"])

# Fellegi-Sunter probabilistic
em_result = gm.train_em(df, matchkey, n_sample_pairs=10000)
pairs = gm.score_probabilistic(block_df, matchkey, em_result)

# Explain a match decision
explanation = gm.explain_pair(record_a, record_b, matchkey)

# Cluster operations
gm.unmerge_record(record_id, clusters)  # Remove from cluster
gm.unmerge_cluster(cluster_id, clusters)  # Shatter to singletons

# Data quality
df, fixes = gm.auto_fix_dataframe(df)
anomalies = gm.detect_anomalies(df)
column_map = gm.auto_map_columns(df_a, df_b)  # Schema matching

# Graph ER (multi-table)
clusters = gm.run_graph_er(entities, relationships)
```

## Setup Wizard

Run `goldenmatch setup` for an interactive walkthrough:

![Setup Wizard](docs/screenshots/setup-welcome.svg)

Guides you through GPU mode selection, Vertex AI / Colab / local GPU configuration, LLM boost API keys, and database sync — with copy-paste commands at every step.

![GPU Selection](docs/screenshots/setup-gpu.svg)

## Why GoldenMatch?

| | GoldenMatch | [dedupe](https://github.com/dedupeio/dedupe) | [recordlinkage](https://github.com/J535D165/recordlinkage) | [Zingg](https://github.com/zinggAI/zingg) | [Splink](https://github.com/moj-analytical-services/splink) |
|---|---|---|---|---|---|
| Zero-config mode | Yes | No (requires training) | No (manual config) | No (Spark required) | No (SQL config) |
| Fuzzy + probabilistic + LLM | All three | Probabilistic only | Probabilistic only | ML-based | Probabilistic only |
| Privacy-preserving (PPRL) | Built-in (92.4% F1) | No | No | No | No |
| Interactive TUI | Yes | No | No | No | No |
| Golden record synthesis | 5 strategies | No | No | No | No |
| MCP server (AI integration) | Yes (30 tools) | No | No | No | No |
| Database sync | Postgres + DuckDB | No | No | No | Spark/DuckDB |
| Single `pip install` | Yes | Yes | Yes | No (Java/Spark) | Yes |
| Polars-native | Yes | No (pandas) | No (pandas) | No (Spark) | Yes (DuckDB) |

GoldenMatch is the only tool that combines zero-config operation, probabilistic matching, LLM scoring, privacy-preserving linkage, and golden record synthesis in a single Python package.

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
5. **Cluster** — group matches via Union-Find; auto-split oversized clusters via MST; assign quality labels (`strong`/`weak`/`split`)
6. **Golden** — merge each cluster into one canonical record using quality-weighted survivorship (5 strategies); track field-level provenance
7. **Output** — files (CSV/Parquet) or database tables + lineage JSON sidecar with provenance

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
  auto_split: true                  # Auto-split oversized clusters via MST
  quality_weighting: true           # Use GoldenCheck quality scores in survivorship
  weak_cluster_threshold: 0.3       # Edge gap threshold for confidence downgrade
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

## SQL Extensions

Use GoldenMatch directly from PostgreSQL or DuckDB:

```sql
-- PostgreSQL
CREATE EXTENSION goldenmatch_pg;
SELECT goldenmatch.goldenmatch_dedupe_table('customers', '{"exact": ["email"]}');
SELECT goldenmatch.goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler');
```

```bash
# DuckDB
pip install goldenmatch-duckdb
```

```python
import duckdb, goldenmatch_duckdb
con = duckdb.connect()
goldenmatch_duckdb.register(con)
con.sql("SELECT goldenmatch_score('John Smith', 'Jon Smyth', 'jaro_winkler')")
```

See [goldenmatch-extensions](https://github.com/benzsevern/goldenmatch-extensions) for installation and full documentation.

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

**Iterative calibration:** When many borderline pairs exist, iterative calibration samples ~100 pairs per round, learns the optimal threshold via grid search, and applies it to all candidates — typically converging in 2-3 rounds.

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

**Equipment data (401K rows):** 27,937 clusters, 384,650 matched, 323s. LLM calibration learned threshold from 200 pairs (~$0.01). ANN fallback created 363 sub-blocks from 15 oversized blocks.

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

### Library Comparison (v1.2.7)

Head-to-head against Splink, Dedupe, and RecordLinkage on two datasets. GoldenMatch uses explicit config, zero training data.

**Febrl (5,000 synthetic PII records, 6,538 true pairs):**

| Library | Precision | Recall | F1 | Time |
|---|---|---|---|---|
| Splink | 1.000 | 0.995 | 0.998 | 2.0s |
| **GoldenMatch** | 1.000 | 0.943 | **0.971** | 6.8s |
| Dedupe | 1.000 | 0.865 | 0.928 | 7.2s |
| RecordLinkage | 0.999 | 0.733 | 0.845 | 2.2s |

**DBLP-ACM (4,910 bibliographic records, 2,224 true matches):**

| Library | Precision | Recall | F1 | Time |
|---|---|---|---|---|
| RecordLinkage | 0.888 | 0.961 | 0.923 | 13.0s |
| **GoldenMatch** | 0.891 | 0.945 | **0.918** | 6.2s |
| Dedupe | 0.604 | 0.936 | 0.734 | 10.5s |
| Splink | 0.646 | 0.834 | 0.728 | 3.4s |

**Key takeaway:** GoldenMatch is the most consistent performer — top-2 F1 on both datasets with zero training data. Splink dominates structured PII but struggles on non-PII. RecordLinkage wins DBLP-ACM but lags on PII.

<details>
<summary>Febrl explicit config example</summary>

```python
config = GoldenMatchConfig(
    blocking=BlockingConfig(
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["surname"], transforms=["soundex"]),
            BlockingKeyConfig(fields=["given_name"], transforms=["soundex"]),
            BlockingKeyConfig(fields=["postcode"], transforms=[]),
            BlockingKeyConfig(fields=["date_of_birth"], transforms=[]),
        ],
        max_block_size=500, skip_oversized=True,
    ),
    matchkeys=[MatchkeyConfig(
        name="person", type="weighted", threshold=0.7,
        fields=[
            MatchkeyField(field="given_name", scorer="jaro_winkler", weight=2.0, transforms=["lowercase", "strip"]),
            MatchkeyField(field="surname", scorer="jaro_winkler", weight=2.0, transforms=["lowercase", "strip"]),
            MatchkeyField(field="date_of_birth", scorer="exact", weight=1.5),
            MatchkeyField(field="address_1", scorer="token_sort", weight=1.0, transforms=["lowercase", "strip"]),
            MatchkeyField(field="postcode", scorer="exact", weight=0.5),
        ],
    )],
)
result = goldenmatch.dedupe_df(df, config=config)
```

</details>

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
| `goldenmatch label FILE --config --gt` | Interactively label pairs to build ground truth CSV |
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

## Remote MCP Server

GoldenMatch is available as a hosted MCP server on [Smithery](https://smithery.ai/servers/benzsevern/goldenmatch) — connect from any MCP client without installing anything.

**Claude Desktop / Claude Code:**
```json
{
  "mcpServers": {
    "goldenmatch": {
      "url": "https://goldenmatch-mcp-production.up.railway.app/mcp/"
    }
  }
}
```

**Local server** (if you prefer to run locally):
```bash
pip install goldenmatch[mcp]
goldenmatch mcp-serve data.csv
```

30 tools available: deduplicate files, match records, explain decisions, review borderline pairs, privacy-preserving linkage, configure rules, scan data quality, run transforms, and synthesize golden records.

## Architecture

```
goldenmatch/
├── cli/            # 21 CLI commands (Typer)
│                   #   Python API: 95 public exports from `import goldenmatch as gm`
│                   #   -- every feature accessible without knowing internal module structure
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

**Run tests:** `pytest` (924 tests)

## Part of the Golden Suite

| Tool | Purpose | Install |
|------|---------|---------|
| [GoldenCheck](https://github.com/benzsevern/goldencheck) | Validate & profile data quality | `pip install goldencheck` |
| [GoldenFlow](https://github.com/benzsevern/goldenflow) | Transform & standardize data | `pip install goldenflow` |
| [GoldenMatch](https://github.com/benzsevern/goldenmatch) | Deduplicate & match records | `pip install goldenmatch` |
| [GoldenPipe](https://github.com/benzsevern/goldenpipe) | Orchestrate the full pipeline | `pip install goldenpipe` |

## What's New in v1.4.0

- **Scoring & survivorship quality** -- MST-based cluster auto-splitting at weakest edges, cluster quality labels (strong/weak/split), quality-weighted survivorship strategies using GoldenCheck scores, field-level provenance tracking.
- **Smart auto-config** -- auto-config now profiles cleaned data (after GoldenCheck/GoldenFlow), detects data domains and extracts identifiers, selects learned blocking for large datasets, enables reranking for multi-field matchkeys, adjusts thresholds from data quality.
- **GoldenFlow integration** -- optional data transformation step in the pipeline. Phone normalization, date standardization, categorical correction. `pip install goldenmatch[transform]`.
- **`llm_auto` flag** -- `dedupe_df(df, llm_auto=True)` auto-enables LLM scorer ($0.05 budget cap) and memory store when API key detected.

## What's New in v1.3.0

- **CCMS cluster comparison** -- compare two clustering outcomes without ground truth using the Case Count Metric System (Talburt et al.). Classifies each cluster as unchanged, merged, partitioned, or overlapping. Includes Talburt-Wang Index (TWI) for normalized similarity.
- **Parameter sensitivity analysis** -- sweep threshold, blocking, or matchkey parameters across a range and compare each run against a baseline. `stability_report()` identifies optimal value ranges. Failed sweep points are logged and skipped, preserving partial results.
- **New CLI commands** -- `goldenmatch compare-clusters` for ad-hoc comparison, `goldenmatch sensitivity` for automated parameter tuning.
- **New Python API** -- `compare_clusters()`, `CompareResult`, `run_sensitivity()`, `SensitivityResult`, `SweepParam` exported from `goldenmatch`.

## What's New in v1.2.7

- **Auto-config cardinality guards** — three new guards prevent auto-config failures on edge-case data:
  - Blocking: excludes near-unique columns (cardinality_ratio >= 0.95)
  - Matchkeys: skips exact matchkeys for low-cardinality columns (cardinality_ratio < 0.01)
  - Description columns: routes long text to fuzzy matching (token_sort) alongside embedding
- **Library comparison benchmarks** — head-to-head results against Splink, Dedupe, and RecordLinkage on Febrl (0.971 F1) and DBLP-ACM (0.918 F1). GoldenMatch is the most consistent performer across data types.

## What's New in v1.2.6

- **Iterative LLM calibration** — instead of scoring all candidates, calibrates the decision threshold from ~200 sampled pairs. Typically converges in 2-3 rounds at negligible cost (~$0.01 on a 401K-row equipment dataset).
- **ANN hybrid blocking** — oversized blocks that exceed the max block size now fall back to embedding-based ANN sub-blocking automatically, keeping blocks tractable without manual tuning.
- **Auto-config classification fixes** — improved heuristics for ID and price fields, utility-based field ranking to select better blocking keys, and LLM-assisted classification for ambiguous column names.

## Author

[Ben Severn](https://bensevern.dev)

## License

MIT
