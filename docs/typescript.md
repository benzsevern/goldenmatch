---
layout: default
title: TypeScript API
nav_order: 4.5
---

# TypeScript / Node.js

GoldenMatch is also published as an npm package with full feature parity with the Python toolkit.

```bash
npm install goldenmatch
```

---

## Quick Start

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

console.log(result.stats);
```

---

## Two Entrypoints

The package ships with two separate entry points so the core stays edge-safe and dependency-free:

- **`goldenmatch`** — edge-safe core. Works in browsers, Cloudflare Workers, Vercel Edge Runtime, Deno, Bun, and Node.
- **`goldenmatch/node`** — adds Node-only features: file I/O (CSV, JSON), HTTP servers, DB connectors.

```typescript
// Edge-safe core (pure TS, no Node APIs)
import { dedupe, match, scoreStrings, applyTransforms } from "goldenmatch";

// Node-only additions
import { readFile, writeCsv, dedupeFile, startApiServer } from "goldenmatch/node";
```

---

## Core API

### dedupe(rows, options)

Deduplicate an array of rows.

```typescript
interface DedupeOptions {
  config?: GoldenMatchConfig;
  exact?: readonly string[];
  fuzzy?: Record<string, number>;
  blocking?: readonly string[];
  threshold?: number;
  llmScorer?: boolean;
}

interface DedupeResult {
  goldenRecords: readonly Row[];
  clusters: ReadonlyMap<number, ClusterInfo>;
  dupes: readonly Row[];
  unique: readonly Row[];
  stats: DedupeStats;
  scoredPairs: readonly ScoredPair[];
  config: GoldenMatchConfig;
}
```

### match(target, reference, options)

Match target records against a reference dataset. Returns matched pairs with confidence scores.

### scoreStrings(a, b, scorer?)

Score similarity between two strings. Available scorers:
`exact`, `jaro_winkler`, `levenshtein`, `token_sort`, `soundex_match`, `dice`, `jaccard`, `ensemble`.

```typescript
import { scoreStrings } from "goldenmatch";

const score = scoreStrings("MARTHA", "MARHTA", "jaro_winkler");
// 0.9611
```

### applyTransforms(value, transforms)

Apply a chain of normalization transforms to a value.

```typescript
import { applyTransforms } from "goldenmatch";

applyTransforms("  John Q. Smith  ", ["strip", "lowercase", "alpha_only"]);
// "johnqsmith"
```

---

## Scorers

All scorers implement the same interface as Python `goldenmatch.core.scorer`:

| Scorer | Use case |
|---|---|
| **jaro_winkler** | Short strings (names). `MARTHA`/`MARHTA` -> 0.9611 |
| **levenshtein** | Normalized edit distance |
| **token_sort** | Word reordering tolerant (rapidfuzz-compatible) |
| **soundex_match** | Phonetic matching (1.0 if same code) |
| **ensemble** | Weighted combination of jaro_winkler + levenshtein + token_sort + dice |
| **dice**, **jaccard** | Set-based similarity for hex-encoded bloom filters (PPRL) |
| **embedding** | Cosine similarity of embeddings |
| **record_embedding** | Cosine similarity across whole records |

---

## Blocking Strategies

- `static` — single blocking key with transforms
- `multi_pass` — multiple blocking keys, union of blocks
- `sorted_neighborhood` — sliding window over sorted data
- `adaptive` — static + auto-split oversized blocks
- `ann` — approximate nearest neighbor (requires `hnswlib-node` peer dep)
- `canopy` — TF-IDF canopy clustering
- `learned` — data-driven predicate selection

---

## Golden Record Strategies

- `most_complete` — pick longest string
- `majority_vote` — pick most frequent
- `source_priority` — pick first non-null from priority list
- `most_recent` — pick value with most recent date
- `first_non_null` — pick first non-null

---

## Transforms

Applied at matchkey time. Same names as the Python toolkit:

`lowercase`, `uppercase`, `strip`, `strip_all`, `soundex`, `metaphone`,
`digits_only`, `alpha_only`, `normalize_whitespace`, `token_sort`,
`first_token`, `last_token`, `substring:start:end`, `qgram:n`.

---

## CLI

The npm package ships a `goldenmatch-js` binary:

```bash
# Dedupe a CSV
npx goldenmatch-js dedupe data.csv --output golden.csv

# Score two strings
npx goldenmatch-js score "MARTHA" "MARHTA" --scorer jaro_winkler
# jaro_winkler: 0.9611

# Match two datasets
npx goldenmatch-js match target.csv reference.csv -o matched.csv

# Profile a dataset
npx goldenmatch-js profile data.csv

# Launch interactive TUI (requires ink peer deps)
npx goldenmatch-js tui data.csv
```

---

## Servers

### MCP server (Claude Desktop / Claude Code)

```bash
npx goldenmatch-js mcp-serve
```

Exposes 19 MCP tools over JSON-RPC on stdio.

### REST API server

```bash
npx goldenmatch-js serve --port 8000
```

Endpoints: `/health`, `/dedupe`, `/match`, `/score`, `/explain`, `/profile`, `/clusters`, `/reviews`.

### A2A agent server

```bash
npx goldenmatch-js agent-serve --port 8200
```

Agent card at `/.well-known/agent.json` advertises 10 skills.

### Interactive TUI

```bash
npx goldenmatch-js tui
```

Requires the Ink peer deps (see below).

---

## Optional Peer Dependencies

All peer deps are optional. Install only what you need:

| Peer dep | Unlocks |
|---|---|
| `yaml` | YAML config file loading |
| `hnswlib-node` | Sub-linear ANN blocking (vs brute-force) |
| `@huggingface/transformers` | ONNX cross-encoder reranking (MiniLM) |
| `piscina` | Worker-thread parallel block scoring |
| `ink`, `react`, `ink-table`, `ink-select-input`, `ink-text-input`, `ink-spinner`, `ink-gradient` | Interactive TUI |
| `pg` | Postgres connector + sync |
| `@duckdb/node-api` | DuckDB connector |
| `snowflake-sdk` | Snowflake connector |
| `@google-cloud/bigquery` | BigQuery connector |
| `@databricks/sql` | Databricks connector |

---

## Advanced Features

- **Probabilistic matching** — Fellegi-Sunter with Splink-style EM
- **PPRL** — Privacy-preserving record linkage with SHA-256 bloom filters (3 security levels: standard, high, paranoid)
- **Graph ER** — Multi-table entity resolution with evidence propagation
- **Streaming** — Incremental single-record matching
- **Memory** — Persistent corrections + threshold learning
- **Sensitivity analysis** — Parameter sweep with CCMS / TWI cluster comparison
- **Lineage tracking** — Full provenance per field per golden record

---

## Examples

See [`packages/goldenmatch-js/examples/`](https://github.com/benzsevern/goldenmatch/tree/main/packages/goldenmatch-js/examples) for 11 full end-to-end TypeScript examples covering dedupe, match, PPRL, streaming, graph ER, Fellegi-Sunter, and more.

---

## Source

- **npm**: [https://www.npmjs.com/package/goldenmatch](https://www.npmjs.com/package/goldenmatch)
- **GitHub**: [https://github.com/benzsevern/goldenmatch/tree/main/packages/goldenmatch-js](https://github.com/benzsevern/goldenmatch/tree/main/packages/goldenmatch-js)

---

## Comparison With Python

| Feature | Python | TypeScript |
|---|---|---|
| Core matching | Polars + rapidfuzz | Pure TS |
| Fellegi-Sunter | Yes | Yes |
| PPRL | SHA-256 | SHA-256 (interop verified byte-for-byte) |
| Graph ER | Yes | Yes |
| LLM scorer | Yes | Yes (via fetch, edge-safe) |
| Cross-encoder | sentence-transformers | @huggingface/transformers (ONNX) |
| ANN blocking | FAISS | hnswlib-node |
| Parallel scoring | Threads + Ray | piscina worker threads |
| Interactive UI | Textual TUI | Ink TUI |
| MCP server | 30 tools | 19 tools |
| REST API | Yes | Yes |
| A2A server | Yes | Yes |
| YAML configs | Yes | Yes (round-trippable) |
| Edge-safe core | No | Yes |
