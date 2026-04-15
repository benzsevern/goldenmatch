# GoldenMatch (TypeScript)

**Entity resolution toolkit for Node.js and edge runtimes. Deduplicate, match, and create golden records — in TypeScript.**

```bash
npm install goldenmatch
```

[![npm](https://img.shields.io/npm/v/goldenmatch?color=d4a017)](https://www.npmjs.com/package/goldenmatch)
[![Node](https://img.shields.io/node/v/goldenmatch?color=339933)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/benzsevern/goldenmatch/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-590%20passing-brightgreen)](https://github.com/benzsevern/goldenmatch/tree/main/packages/goldenmatch-js/tests)

---

## Why this port?

- **Edge-safe core** — the matching engine runs in browsers, Workers, Vercel Edge Runtime, Deno
- **Pure TypeScript** — no native dependencies required; peer deps unlock performance (hnswlib, ONNX, piscina)
- **Feature parity with Python goldenmatch** — same scorers, same clustering, same YAML configs
- **590 tests, strict TypeScript** — `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`

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
// { totalRecords: 3, totalClusters: 2, matchRate: 0.67, ... }

for (const record of result.goldenRecords) {
  console.log(record);
}
```

## Auto-Config Verification (v0.3)

Auto-generated configs are now checked both before the pipeline runs and after
scoring finishes, so you get actionable diagnostics instead of silent failures
on edge-case data.

### Preflight — six static checks

When you call `autoConfigureRows(rows)`, the returned config ships with a
`_preflightReport` summarising six config-time checks:

1. **missing_column** — matchkey/blocking references a column not in the data
2. **cardinality_high** — a column is near-unique (poor blocking signal)
3. **cardinality_low** — a column has too few distinct values to discriminate
4. **block_size** — a blocking key would produce oversized blocks
5. **remote_asset** — a scorer requires a model download (gated offline)
6. **weight_confidence** — a weighted matchkey's weights look unbalanced

Many findings trigger **auto-repairs** (field dropped, scorer swapped,
weight clamped). `hasErrors === true` on unrepairable errors raises
`ConfigValidationError` with the full report attached.

```ts
import { autoConfigureRows, ConfigValidationError } from "goldenmatch";

const cfg = autoConfigureRows(rows);
for (const f of cfg._preflightReport!.findings) {
  console.log(`[${f.severity}] ${f.check}/${f.subject}: ${f.message}`);
}
```

Defaults are **offline-safe**: remote-asset scorers (cross-encoder, remote
embeddings) are dropped unless you opt in with `allowRemoteAssets: true`.

### Postflight — four runtime signals

Inside `dedupe()` / `match()`, after scoring but before clustering, the
pipeline computes four signals attached as `result.postflightReport`:

1. **scoreHistogram** — 100-bin pair-score distribution
2. **blockSizePercentiles** + **preliminaryClusterSizes** — p50/p95/p99/max
3. **thresholdOverlapPct** — fraction of pairs near the current threshold
4. **oversizedClusters** — components above size limit, with bottleneck pair

If the score distribution is clearly bimodal, postflight proposes a
threshold adjustment. In **strict mode** (`autoConfigureRows(rows, { strict: true })`
or manual `_strictAutoconfig: true`) the signals are still emitted but the
threshold is never touched — use this for reproducible CI pipelines.

See `examples/verificationInspection.ts` and `examples/strictModeParity.ts`
for runnable demos.

## Three entrypoints

```typescript
import { dedupe, match, scoreStrings } from "goldenmatch";         // edge-safe core
import { readFile, writeCsv } from "goldenmatch/node";              // Node-only file I/O
// CLI: `npx goldenmatch-js dedupe data.csv --output golden.csv`
```

## Feature matrix

### Scoring algorithms
- Exact, Jaro-Winkler, Levenshtein, Token-Sort, Soundex, Dice, Jaccard, Ensemble
- Probabilistic (Fellegi-Sunter with Splink-style EM)
- LLM scorer (OpenAI/Anthropic via fetch — edge-safe)
- Cross-encoder reranking (via @huggingface/transformers)

### Blocking strategies
- Static, multi-pass, sorted-neighborhood, adaptive
- ANN (approximate nearest neighbor via hnswlib-node peer dep or brute-force)
- Canopy (TF-IDF)
- Learned (data-driven predicate selection)

### Golden record strategies
- most_complete, majority_vote, source_priority, most_recent, first_non_null
- Full provenance tracking

### Pipeline features
- PPRL (privacy-preserving record linkage, 3 security levels with HMAC-SHA256)
- Graph ER (multi-table entity resolution with evidence propagation)
- Sensitivity analysis (parameter sweep with CCMS/TWI)
- Streaming (incremental single-record matching)
- Memory (persistent corrections + threshold learning)
- Review queue (human-in-the-loop)

## Optional peer deps

Zero-dep install works. These unlock advanced paths:

| Peer dep | What it enables |
|---|---|
| `yaml` | YAML config file loading |
| `hnswlib-node` | True sub-linear ANN blocking (vs brute-force) |
| `@huggingface/transformers` | ONNX cross-encoder reranking (MiniLM) |
| `piscina` | Worker-thread parallel block scoring |
| `ink` + `react` | Interactive terminal UI |
| `ink-table`, `ink-select-input`, `ink-text-input`, `ink-spinner`, `ink-gradient` | Richer TUI widgets |
| `pg` | Postgres connector + sync |
| `@duckdb/node-api` | DuckDB connector |
| `snowflake-sdk`, `@google-cloud/bigquery`, `@databricks/sql` | Cloud warehouse connectors |

## Servers

```bash
# MCP server (for Claude Desktop / Code)
npx goldenmatch-js mcp-serve

# REST API
npx goldenmatch-js serve --port 8000

# A2A agent server
npx goldenmatch-js agent-serve --port 8200

# Interactive TUI
npx goldenmatch-js tui data.csv
```

## CLI commands

```
goldenmatch-js dedupe <files...>    Deduplicate records
goldenmatch-js match <target> <ref> Match target against reference
goldenmatch-js score <a> <b>        Score similarity between two strings
goldenmatch-js info                 Show scorers, strategies, transforms
goldenmatch-js profile <file>       Profile a dataset
goldenmatch-js demo                 Run a quick demo on synthetic data
goldenmatch-js mcp-serve            Start MCP server (stdio)
goldenmatch-js serve                Start REST API
goldenmatch-js agent-serve          Start A2A agent
goldenmatch-js tui                  Interactive terminal UI
```

## Examples

See [`examples/`](./examples) for 10+ full examples covering basic dedupe, CSV pipelines,
probabilistic matching (Fellegi-Sunter), PPRL, streaming, LLM scoring, explanations, and evaluation.

## Documentation

Full docs: https://benzsevern.github.io/goldenmatch/typescript

## License

MIT. See [LICENSE](https://github.com/benzsevern/goldenmatch/blob/main/LICENSE).
