/**
 * LLM scorer: send borderline pairs to OpenAI/Anthropic for a YES/NO
 * verdict, with a hard budget cap.
 *
 * Flow:
 *   - score >= autoThreshold: auto-accept (promoted to 1.0, no LLM call)
 *   - candidateLo <= score < autoThreshold: ask the LLM
 *   - score < candidateLo: left as-is
 *
 * Run: OPENAI_API_KEY=sk-... npx tsx examples/09-llm-scorer.ts
 */
import {
  llmScorePairs,
  makeScoredPair,
  type LLMScorerConfig,
  type Row,
  type ScoredPair,
} from "goldenmatch";

// Borderline pairs from a prior dedupe() run (scores 0.65 - 0.92)
const rows: Row[] = [
  { __row_id__: 0, name: "Apple Inc",          description: "Consumer electronics manufacturer" },
  { __row_id__: 1, name: "Apple Incorporated", description: "Maker of iPhones and Macs" },
  { __row_id__: 2, name: "Apple Orchard Co",   description: "Fruit grower in Washington State" },
  { __row_id__: 3, name: "Microsoft Corp",     description: "Software company, maker of Windows" },
  { __row_id__: 4, name: "Microsoft",          description: "Cloud + software + Xbox" },
];

const candidatePairs: ScoredPair[] = [
  makeScoredPair(0, 1, 0.87), // borderline -- ask LLM
  makeScoredPair(0, 2, 0.72), // borderline -- ask LLM (but almost certainly not same)
  makeScoredPair(3, 4, 0.93), // auto-accept (>= 0.90)
];

const config: LLMScorerConfig = {
  enabled: true,
  provider: "openai",               // or "anthropic"
  model: "gpt-4o-mini",
  autoThreshold: 0.90,
  candidateLo: 0.60,
  candidateHi: 0.90,
  batchSize: 10,
  maxWorkers: 4,
  mode: "pairwise",
  budget: {
    maxCostUsd: 0.05,
    maxCalls: 50,
    warnAtPct: 0.8,
  },
};

const apiKey = process.env["OPENAI_API_KEY"];
if (!apiKey) {
  console.warn("OPENAI_API_KEY not set. Running in no-op mode (candidates pass through).\n");
}

const result = await llmScorePairs(candidatePairs, rows, config, apiKey);

console.log("After LLM scoring:");
for (const p of result.pairs) {
  console.log(`  (${p.idA}, ${p.idB})  score=${p.score.toFixed(2)}`);
}

if (result.budget) {
  console.log("\nBudget usage:");
  console.log(`  Calls:     ${result.budget.calls}`);
  console.log(`  Cost USD:  ${result.budget.costUsd.toFixed(4)}`);
  console.log(`  Tokens:    ${result.budget.inputTokens} in / ${result.budget.outputTokens} out`);
} else {
  console.log("\nNo budget info (no LLM calls made).");
}

/**
 * Python -> TS differences:
 *   - Python `llm_score_pairs()` is sync-looking via asyncio; TS `llmScorePairs()` is
 *     an async function returning a Promise.
 *   - API key is passed explicitly (no env var magic); pass `process.env.OPENAI_API_KEY`
 *     at the call site. Works on edge runtimes that expose `fetch`.
 */
