# GoldenMatch TypeScript Examples

Each example is a standalone `.ts` file runnable with:

```bash
npx tsx examples/<name>.ts
```

| # | Example | What it shows |
|---|---------|---------------|
| 01 | `01-basic-dedupe.ts` | Dedupe an in-memory array with exact + fuzzy matchkeys |
| 02 | `02-match-two-datasets.ts` | Match target records against a reference dataset |
| 03 | `03-csv-file-pipeline.ts` | Read CSV -> dedupe -> write golden records |
| 04 | `04-string-scoring.ts` | Compare every scorer on the same string pairs |
| 05 | `05-custom-config.ts` | Build/save/load a full GoldenMatchConfig (YAML) |
| 06 | `06-probabilistic-fs.ts` | Fellegi-Sunter with EM training |
| 07 | `07-pprl-privacy.ts` | Privacy-preserving record linkage (3 security levels) |
| 08 | `08-streaming.ts` | Incremental streaming -- add records one at a time |
| 09 | `09-llm-scorer.ts` | LLM scorer for borderline pairs (needs OPENAI_API_KEY) |
| 10 | `10-explain.ts` | Template NL explanation of a pair match |
| 11 | `11-evaluate.ts` | Evaluate against ground truth (precision/recall/F1) |
| 12 | `verificationInspection.ts` | Inspect preflight findings + postflight signals |
| 13 | `strictModeParity.ts` | Use `_strictAutoconfig` to disable runtime threshold shifts |

## Running

From the repo root:

```bash
cd packages/goldenmatch-js
npm install
npx tsx examples/01-basic-dedupe.ts
```

Or install `tsx` globally:

```bash
npm install -g tsx
tsx examples/01-basic-dedupe.ts
```

## Optional peer deps by example

Most examples use only the core package. A few need optional peer deps:

| Example | Peer dep required |
|---------|-------------------|
| `05-custom-config.ts` (YAML save/load path only) | `yaml` |
| `09-llm-scorer.ts` | none (uses `fetch`); needs `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in env |

Install them on demand:

```bash
npm install yaml
```
