# Porting Golden Suite to TypeScript — Playbook

> Master reference for porting the Golden Suite (GoldenCheck, GoldenFlow, GoldenMatch) from Python to TypeScript. Written so a fresh Claude Code session can pick up any repo and port it with full context.

## Completed Ports

| Repo | Python LOC | TS LOC | TS Files | Status |
|------|-----------|--------|----------|--------|
| **GoldenCheck** | ~11,700 (86 files) | ~10,000 (72 files) | `packages/goldencheck-js/` | Done |
| **GoldenFlow** | ~5,350 (61 files) | ~5,234 (54 files) | `packages/goldenflow-js/` | Done |
| **GoldenMatch** | ~59,500 (282 files) | — | `packages/goldenmatch-js/` | **Next** |

## The Pattern (proven across GoldenCheck + GoldenFlow)

```
repo-root/
├── goldenmatch/              # Python package (existing, untouched)
├── packages/goldenmatch-js/  # TypeScript port (new)
│   ├── package.json          # npm: "goldenmatch"
│   ├── tsconfig.json         # Strict TypeScript
│   ├── tsup.config.ts        # Build: 4 entry points, dual ESM+CJS
│   ├── vitest.config.ts      # Test runner
│   ├── src/
│   │   ├── index.ts          # Re-exports core
│   │   ├── cli.ts            # Commander.js CLI
│   │   ├── core/             # Edge-safe (browsers, Workers, Edge Runtime)
│   │   │   ├── index.ts      # Public API surface
│   │   │   ├── types.ts      # All interfaces, enums, factory functions
│   │   │   ├── data.ts       # TabularData (Polars replacement)
│   │   │   └── ...           # Module-per-module port
│   │   └── node/             # Node 20+ only
│   │       ├── index.ts      # Re-exports core + Node-only
│   │       ├── reader.ts     # CSV/Parquet file reading
│   │       └── ...           # MCP, TUI, DB, etc.
│   └── tests/
│       ├── unit/             # Per-module tests
│       ├── parity/           # Python-TS equivalence tests
│       └── smoke.test.ts     # Basic import sanity
├── package.json              # Root orchestrator (NOT a workspace)
└── scripts/
    └── gen_parity_goldens.py # Python generates golden outputs for TS parity tests
```

---

## Config Files (copy from GoldenFlow, adjust names)

### package.json
```json
{
  "name": "goldenmatch",
  "type": "module",
  "exports": {
    ".": { "types": "./dist/index.d.ts", "import": "./dist/index.js", "require": "./dist/index.cjs" },
    "./core": { "types": "./dist/core/index.d.ts", "import": "./dist/core/index.js", "require": "./dist/core/index.cjs" },
    "./node": { "types": "./dist/node/index.d.ts", "import": "./dist/node/index.js", "require": "./dist/node/index.cjs" }
  },
  "bin": { "goldenmatch-js": "./dist/cli.cjs" },
  "engines": { "node": ">=20" },
  "dependencies": { "commander": "^13.0.0" },
  "peerDependencies": { "yaml": "*" },
  "peerDependenciesMeta": { "yaml": { "optional": true } },
  "devDependencies": { "@types/node": "^20.0.0", "rimraf": "^5.0.0", "tsup": "^8.5.1", "typescript": "^5.4.0", "vitest": "^4.1.0", "yaml": "^2.7.0" }
}
```

### tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2022", "module": "ESNext", "moduleResolution": "Bundler",
    "lib": ["ES2022"], "strict": true, "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true, "noImplicitOverride": true,
    "noFallthroughCasesInSwitch": true, "forceConsistentCasingInFileNames": true,
    "esModuleInterop": true, "resolveJsonModule": true, "isolatedModules": true,
    "skipLibCheck": true, "declaration": true, "declarationMap": true,
    "sourceMap": true, "outDir": "dist", "types": ["node"]
  },
  "include": ["src/**/*.ts", "tests/**/*.ts", "*.config.ts"],
  "exclude": ["dist", "node_modules"]
}
```

### tsup.config.ts
```typescript
import { defineConfig } from "tsup";
export default defineConfig({
  entry: { index: "src/index.ts", "core/index": "src/core/index.ts", "node/index": "src/node/index.ts", cli: "src/cli.ts" },
  format: ["esm", "cjs"], dts: true, sourcemap: true, clean: true, target: "node20", splitting: false, treeshake: true,
});
```

### Root package.json
```json
{ "private": true, "scripts": { "install:js": "npm --prefix packages/goldenmatch-js install", "build:js": "npm --prefix packages/goldenmatch-js run build", "test:js": "npm --prefix packages/goldenmatch-js run test", "typecheck:js": "npm --prefix packages/goldenmatch-js run typecheck" } }
```

---

## Hard-Won Lessons (MUST follow — learned from GoldenCheck + GoldenFlow ports)

### Edge-safety rules for `src/core/`
- **NEVER** import `node:fs`, `node:path`, `node:http`, or use `process.on()`
- **NEVER** use `require()` — only `import` (tsup handles CJS conversion)
- If a module needs Node APIs, put it in `src/node/` — this was a bug in the GoldenFlow port that had to be fixed during review (history.ts was in core with `require()`)
- LLM providers: use raw `fetch()` — edge-safe, no SDK needed

### Error handling rules
- **NEVER** use bare `catch {}` — always log: `catch (e) { console.warn("...", e instanceof Error ? e.message : String(e)); }`
- MCP server `handleTool` must wrap ALL tool cases in try-catch that returns JSON error — this was a critical bug in GoldenFlow's first draft
- Config loader must throw on invalid YAML (non-object), not silently return empty config
- LLM corrector must log HTTP status on `!resp.ok` and validate JSON response structure
- REST API server must validate paths to prevent traversal (resolve + check startsWith cwd)

### CSV parser rules
- **NEVER** coerce leading-zero strings to numbers: `"01234"` (zip code) must stay a string
- Guard: `if (raw.length > 1 && raw[0] === "0" && raw[1] !== ".") return raw;`
- Handle quoted fields with commas, escaped quotes (`""` inside quoted field)
- Return `null` for empty CSV fields

### Date handling
- **Always use UTC** methods (`getUTCFullYear`, `getUTCMonth`, `getUTCDate`, etc.)
- `new Date(val)` parses into local timezone — extract with UTC methods for consistent behavior across environments
- Use millisecond arithmetic for date_shift (`d.getTime() + days * 86_400_000`), not `setDate()`

### Profiler / TabularData
- `TabularData.column()` uses `isNullish()` which converts "N/A"→null — this inflates null counts
- Add `rawColumn()` that preserves original values for profiling
- Transforms must receive raw values (not pre-filtered), so the transform engine should NOT use `toColumnValue()` when extracting column values for transforms
- Row comparison after transform: compare `row[column] ?? null` vs `newValues[i]`, not `toColumnValue(row[column])` which masks changes

### Math/numeric rules
- **NEVER** `Math.min(...array)` or `Math.max(...array)` — crashes on >65K elements. Use a for-loop.
- PRNG is Mulberry32 (NOT Python's Mersenne Twister) — sampling results will differ

### Type system rules
- `Row = Readonly<Record<string, unknown>>` — accept `unknown` values, coerce in getters
- All interfaces are `readonly` — use spread `{ ...obj, ...overrides }` for updates
- `exactOptionalPropertyTypes: true` means `field?: T` requires explicit `undefined`
- Export types with `export type { ... }` for tree-shaking

### Security
- REST API and MCP server must sanitize file paths: `resolve(path)` then `startsWith(cwd)`
- Never expose internal filesystem error messages to API clients

### Publishing checklist
1. `npm run typecheck` — 0 errors
2. `npm run test` — all pass
3. `npm run build` — clean ESM + CJS + .d.ts
4. `git tag goldenmatch-js-v0.1.0 && git push origin goldenmatch-js-v0.1.0`
5. Requires `NPM_TOKEN` GitHub secret

---

## GoldenMatch-Specific Porting Guide

### Scope Assessment

GoldenMatch is **~10x larger** than GoldenFlow (~59,500 LOC vs ~5,350). A full port is NOT practical in one session. **Recommended: phased approach.**

### Phase 1: Core Matching Engine (MVP)

Target: `dedupe_df()` and `match_df()` working with exact + fuzzy matching.

| Python module | TS location | Priority | Notes |
|---|---|---|---|
| `config/schemas.py` | `src/core/types.ts` | P0 | GoldenMatchConfig, MatchkeyConfig, BlockingConfig |
| `core/scorer.py` | `src/core/scorer.ts` | P0 | Field scorers: jaro_winkler, levenshtein, token_sort, exact |
| `core/matchkey.py` | `src/core/matchkey.ts` | P0 | Matchkey builder + transforms |
| `core/blocker.py` | `src/core/blocker.ts` | P0 | Static blocking (hash-based grouping) |
| `core/cluster.py` | `src/core/cluster.ts` | P0 | Union-Find + MST splitting |
| `core/pipeline.py` | `src/core/pipeline.ts` | P0 | Orchestrator: block → score → cluster |
| `core/golden.py` | `src/core/golden.ts` | P0 | Golden record builder (merge strategies) |
| `_api.py` | `src/core/api.ts` | P0 | High-level `dedupe()`, `match()` functions |

**Key algorithm to implement:**

```
Input rows
  → build blocking keys (hash on transformed fields)
  → group into blocks (Map<string, Row[]>)
  → for each block: score all pairs (NxN with field-level scoring)
  → filter pairs above threshold
  → cluster matched pairs (Union-Find)
  → optionally split weak clusters (MST bottleneck)
  → build golden records (most_complete / majority_vote / recency)
  → return DedupeResult
```

**Fuzzy matching (edge-safe):** Implement Jaro-Winkler and Levenshtein in pure TS (no native deps). RapidFuzz is Python-only. Reference implementations:
- Jaro-Winkler: ~40 lines, well-documented algorithm
- Levenshtein ratio: `1 - (distance / max(len_a, len_b))` — GoldenFlow already has this in `auto-correct.ts`

### Phase 2: Config & CLI

| Python module | TS location | Notes |
|---|---|---|
| `config/loader.py` | `src/core/config/loader.ts` | YAML → GoldenMatchConfig |
| `config/wizard.py` | `src/node/init-wizard.ts` | Interactive config generation |
| `core/autoconfig.py` | `src/core/autoconfig.ts` | Auto-generate config from data |
| `cli/main.py` + subcommands | `src/cli.ts` | Commander.js (dedupe, match, demo, profile, etc.) |
| `connectors/` | `src/node/connectors/` | CSV/JSON file I/O (reuse GoldenFlow pattern) |

### Phase 3: Probabilistic + LLM

| Python module | TS location | Notes |
|---|---|---|
| `core/probabilistic.py` | `src/core/probabilistic.ts` | Fellegi-Sunter EM training |
| `core/llm_scorer.py` | `src/core/llm/scorer.ts` | Edge-safe via fetch() |
| `core/llm_cluster.py` | `src/core/llm/cluster.ts` | In-context clustering |
| `core/llm_budget.py` | `src/core/llm/budget.ts` | Cost tracking |
| `core/explain.py` | `src/core/explain.ts` | NL pair/cluster explanations |

### Phase 4: Advanced Features

| Python module | TS location | Notes |
|---|---|---|
| `core/ann_blocker.py` | `src/core/ann-blocker.ts` | Needs FAISS.js or similar |
| `core/learned_blocking.py` | `src/core/learned-blocking.ts` | Decision tree predicates |
| `core/cross_encoder.py` | `src/core/cross-encoder.ts` | Needs ONNX Runtime or Transformers.js |
| `core/embedder.py` | `src/core/embedder.ts` | Sentence embeddings |
| `pprl/` | `src/core/pprl/` | Privacy-preserving record linkage |
| `core/graph_er.py` | `src/core/graph-er.ts` | Multi-table ER |
| `core/memory/` | `src/core/memory/` | Learning memory |
| `core/review_queue.py` | `src/core/review-queue.ts` | Human-in-the-loop |

### Phase 5: Infrastructure

| Python module | TS location | Notes |
|---|---|---|
| `mcp/server.py` | `src/node/mcp/server.ts` | 30 MCP tools |
| `a2a/server.py` | `src/node/a2a/server.ts` | Agent-to-Agent |
| `api/server.py` | `src/node/api/server.ts` | REST API |
| `tui/` | `src/node/tui/` | Terminal UI (ink or blessed) |
| `db/` | `src/node/db/` | Postgres, DuckDB connectors |
| `output/` | `src/core/output/` | Report generation |
| `backends/` | `src/node/backends/` | Ray → Worker threads, DuckDB → WASM |

### Phase 6: Cloud Connectors (optional)

| Python module | TS location | Notes |
|---|---|---|
| `connectors/snowflake.py` | `src/node/connectors/snowflake.ts` | Snowflake SDK |
| `connectors/bigquery.py` | `src/node/connectors/bigquery.ts` | BigQuery client |
| `connectors/databricks.py` | `src/node/connectors/databricks.ts` | Databricks SQL |
| `connectors/salesforce.py` | `src/node/connectors/salesforce.ts` | Simple Salesforce |

---

## GoldenMatch Types to Port

```typescript
// src/core/types.ts — key interfaces

interface DedupeResult {
  readonly goldenRecords: readonly Row[];
  readonly clusters: ReadonlyMap<number, Cluster>;
  readonly dupes: readonly Row[];
  readonly unique: readonly Row[];
  readonly stats: DedupeStats;
  readonly scoredPairs: readonly ScoredPair[];
  readonly config: GoldenMatchConfig;
}

interface MatchResult {
  readonly matched: readonly Row[];
  readonly unmatched: readonly Row[];
  readonly stats: MatchStats;
}

interface Cluster {
  readonly members: readonly number[];
  readonly size: number;
  readonly pairScores: ReadonlyMap<string, number>; // "id_a:id_b" → score
  readonly confidence: number;
  readonly quality: "strong" | "weak" | "split";
}

interface ScoredPair {
  readonly idA: number;
  readonly idB: number;
  readonly score: number;
  readonly fieldScores: Readonly<Record<string, number>>;
}

interface GoldenMatchConfig {
  readonly matchkeys: readonly MatchkeyConfig[];
  readonly blocking: BlockingConfig;
  readonly threshold: number;
  readonly goldenRules: GoldenRulesConfig | null;
  readonly llmScorer: LLMScorerConfig | null;
}

interface MatchkeyConfig {
  readonly fields: readonly MatchkeyField[];
  readonly type: "exact" | "weighted" | "probabilistic";
}

interface MatchkeyField {
  readonly field: string;
  readonly transforms: readonly string[];
  readonly scorer: string; // "jaro_winkler", "levenshtein", "exact", etc.
  readonly weight: number;
}

interface BlockingConfig {
  readonly strategy: "static" | "adaptive" | "sorted_neighborhood" | "multi_pass";
  readonly keys: readonly string[];
  readonly maxBlockSize: number;
}
```

---

## Parallelization Strategy for GoldenMatch

Given the size (~59K LOC), use aggressive parallelization:

**Wave 1 (foundation — 3 parallel agents):**
1. Types + TabularData + config schema/loader
2. Fuzzy scorers (Jaro-Winkler, Levenshtein, token_sort, exact, ensemble)
3. Matchkey builder + transforms

**Wave 2 (core engine — 3 parallel agents):**
1. Blocker (static + adaptive)
2. Cluster (Union-Find + MST)
3. Golden record builder (merge strategies)

**Wave 3 (orchestration — 2 parallel agents):**
1. Pipeline orchestrator + high-level API (`dedupe()`, `match()`)
2. Config autoconfig + explain

**Wave 4 (integrations — 3 parallel agents):**
1. CLI (Commander.js, 15+ commands)
2. MCP server (30 tools) + REST API
3. LLM scorer + budget + cluster

**Wave 5 (advanced — deferred):**
- Probabilistic, ANN, embeddings, cross-encoder, PPRL, graph ER, memory, TUI, DB, cloud connectors

---

## Estimated Effort

| Phase | Python LOC | Expected TS LOC | Priority |
|-------|-----------|-----------------|----------|
| Phase 1: Core engine | ~5,000 | ~4,000 | Must-have |
| Phase 2: Config & CLI | ~3,000 | ~2,500 | Must-have |
| Phase 3: Probabilistic + LLM | ~4,000 | ~3,500 | Should-have |
| Phase 4: Advanced features | ~15,000 | ~12,000 | Nice-to-have |
| Phase 5: Infrastructure | ~10,000 | ~8,000 | Nice-to-have |
| Phase 6: Cloud connectors | ~5,000 | ~4,000 | Defer |
| **Total (all phases)** | **~42,000** | **~34,000** | |

**Recommended MVP (Phases 1-2):** ~6,500 TS LOC, achievable in one session. This gives you `dedupe()`, `match()`, config, CLI, and file I/O — enough to be a usable npm package.

---

## Reference Implementations

When in doubt, look at the completed ports:

- **GoldenCheck JS:** `D:\show_case\goldencheck\packages\goldencheck-js\`
- **GoldenFlow JS:** `D:\show_case\goldenflow\packages\goldenflow-js\`

Key files to reference:
- `types.ts` — how Python dataclasses/Pydantic → TS interfaces + factory functions
- `data.ts` — TabularData (Polars replacement)
- `transforms/registry.ts` — decorator-based registration pattern
- `engine/transformer.ts` — the orchestrator pattern
- `node/connectors/file.ts` — CSV parser with type coercion (leading-zero safe)
- `node/mcp/server.ts` — MCP tool definitions + handler with try-catch + path sanitization
- `cli.ts` — Commander.js CLI with async command handlers

---

## GitHub Setup for GoldenMatch

```bash
# After porting, same pattern as GoldenFlow:
gh auth switch --user benzsevern

# Set NPM_TOKEN (reuse the same token from goldenflow repo)
# gh secret set NPM_TOKEN --repo benzsevern/goldenmatch --body "<your-npm-token>"

# Branch, commit, PR, merge
git checkout -b ts-port
# ... do the work ...
git push -u origin ts-port
gh pr create --title "feat: TypeScript port of GoldenMatch" ...
gh pr merge N --squash

# Tag and release
git tag goldenmatch-js-v0.1.0
git push origin goldenmatch-js-v0.1.0
gh release create v2.x.x --title "vX.Y.Z — TypeScript Port" ...

# Update repo metadata
gh repo edit benzsevern/goldenmatch --description "Entity resolution toolkit — deduplicate, match, and create golden records. Python & TypeScript."
gh repo edit benzsevern/goldenmatch --add-topic typescript --add-topic nodejs --add-topic npm

# Switch back
gh auth switch --user benzsevern-mjh
```
