/**
 * scorer.ts — LLM scorer for borderline record pairs.
 * Ports `goldenmatch/core/llm_scorer.py`.
 *
 * Three-tier decision:
 *   score >= autoThreshold        -> auto-accept (promote to 1.0)
 *   candidateLo <= score < hi     -> send to LLM
 *   score < candidateLo           -> keep original score (never demoted)
 *
 * Edge-safe: uses `fetch()` (global on Node 20+/edge runtimes).
 * No `node:` imports.
 */

import type { Row, ScoredPair, LLMScorerConfig } from "../types.js";
import { makeScoredPair } from "../types.js";
import { BudgetTracker, countTokensApprox } from "./budget.js";
import type { BudgetSnapshot } from "./budget.js";

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

export interface LLMScoreResult {
  readonly pairs: readonly ScoredPair[];
  readonly budget: BudgetSnapshot | null;
}

export interface LLMCallResult {
  readonly decisions: ReadonlyMap<number, boolean>;
  readonly inputTokens: number;
  readonly outputTokens: number;
}

type Provider = "openai" | "anthropic";

// ---------------------------------------------------------------------------
// Provider detection
// ---------------------------------------------------------------------------

/**
 * Pick a provider based on config + key heuristics.
 * OpenAI keys start with `sk-` (or `sk-proj-`); Anthropic keys with `sk-ant-`.
 */
function detectProvider(apiKey?: string, configProvider?: string): Provider {
  if (configProvider === "openai" || configProvider === "anthropic") {
    return configProvider;
  }
  if (apiKey?.startsWith("sk-ant-")) return "anthropic";
  return "openai";
}

function defaultModel(provider: Provider): string {
  return provider === "openai" ? "gpt-4o-mini" : "claude-haiku-4-5-20251001";
}

// ---------------------------------------------------------------------------
// Prompt construction
// ---------------------------------------------------------------------------

/** Pull non-internal fields from a Row into a compact display string. */
function summariseRow(row: Row, cols: readonly string[]): string {
  const parts: string[] = [];
  for (const c of cols) {
    const v = row[c];
    if (v === null || v === undefined || v === "") continue;
    parts.push(`${c}: ${String(v)}`);
  }
  return parts.join(" | ").slice(0, 200);
}

/** Build the batch prompt for a list of candidate pairs. */
function buildBatchPrompt(
  batch: readonly ScoredPair[],
  rowById: ReadonlyMap<number, Row>,
  cols: readonly string[],
): string {
  const lines: string[] = [
    "For each numbered pair, answer YES if they are the same entity/product, " +
      "NO if they are different. Respond with just the number and YES/NO, one per line.",
    "",
  ];
  batch.forEach((pair, k) => {
    const rowA = rowById.get(pair.idA) ?? {};
    const rowB = rowById.get(pair.idB) ?? {};
    const textA = summariseRow(rowA, cols);
    const textB = summariseRow(rowB, cols);
    lines.push(`${k + 1}. A: ${textA}`);
    lines.push(`   B: ${textB}`);
  });
  return lines.join("\n");
}

/** Parse a batch YES/NO response into a decision list aligned to batch. */
function parseBatchResponse(answer: string, batchSize: number): boolean[] {
  const decisions: boolean[] = [];
  const lines = answer.split(/\r?\n/);
  for (const raw of lines) {
    const line = raw.trim().toUpperCase();
    if (!line) continue;
    if (line.includes("YES")) decisions.push(true);
    else if (line.includes("NO")) decisions.push(false);
    if (decisions.length >= batchSize) break;
  }
  // Pad with `false` so callers can align by index.
  while (decisions.length < batchSize) decisions.push(false);
  return decisions;
}

// ---------------------------------------------------------------------------
// Provider calls (fetch-based, edge-safe)
// ---------------------------------------------------------------------------

async function callOpenAI(
  prompt: string,
  apiKey: string,
  model: string,
  maxTokens: number,
): Promise<{ text: string; inputTokens: number; outputTokens: number }> {
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      max_tokens: maxTokens,
    }),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new LLMHttpError(resp.status, `OpenAI ${resp.status}: ${body.slice(0, 200)}`);
  }
  const data = (await resp.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
    usage?: { prompt_tokens?: number; completion_tokens?: number };
  };
  const text = data.choices?.[0]?.message?.content?.trim() ?? "";
  return {
    text,
    inputTokens: data.usage?.prompt_tokens ?? 0,
    outputTokens: data.usage?.completion_tokens ?? 0,
  };
}

async function callAnthropic(
  prompt: string,
  apiKey: string,
  model: string,
  maxTokens: number,
): Promise<{ text: string; inputTokens: number; outputTokens: number }> {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "content-type": "application/json",
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new LLMHttpError(resp.status, `Anthropic ${resp.status}: ${body.slice(0, 200)}`);
  }
  const data = (await resp.json()) as {
    content?: Array<{ text?: string }>;
    usage?: { input_tokens?: number; output_tokens?: number };
  };
  const text = data.content?.[0]?.text?.trim() ?? "";
  return {
    text,
    inputTokens: data.usage?.input_tokens ?? 0,
    outputTokens: data.usage?.output_tokens ?? 0,
  };
}

/** Error thrown by provider helpers when the HTTP call fails. */
export class LLMHttpError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "LLMHttpError";
  }
}

// ---------------------------------------------------------------------------
// Batch orchestration
// ---------------------------------------------------------------------------

function* batchify<T>(items: readonly T[], size: number): Generator<T[]> {
  const step = Math.max(1, size);
  for (let i = 0; i < items.length; i += step) {
    yield items.slice(i, i + step);
  }
}

async function scoreBatch(
  batch: readonly ScoredPair[],
  rowById: ReadonlyMap<number, Row>,
  cols: readonly string[],
  provider: Provider,
  model: string,
  apiKey: string,
  budget: BudgetTracker,
): Promise<LLMCallResult> {
  const prompt = buildBatchPrompt(batch, rowById, cols);
  const estIn = countTokensApprox(prompt);
  const estOut = batch.length * 10;

  if (!budget.canSend(estIn)) {
    return { decisions: new Map(), inputTokens: 0, outputTokens: 0 };
  }

  try {
    const { text, inputTokens, outputTokens } =
      provider === "openai"
        ? await callOpenAI(prompt, apiKey, model, batch.length * 10)
        : await callAnthropic(prompt, apiKey, model, batch.length * 10);

    budget.record(inputTokens || estIn, outputTokens || estOut, model);

    const decisions = parseBatchResponse(text, batch.length);
    const out = new Map<number, boolean>();
    batch.forEach((pair, k) => {
      out.set(pairIndex(pair), decisions[k] ?? false);
    });
    return { decisions: out, inputTokens, outputTokens };
  } catch (err) {
    if (err instanceof LLMHttpError) {
      // Graceful degradation: caller keeps original fuzzy scores.
      return { decisions: new Map(), inputTokens: 0, outputTokens: 0 };
    }
    // Unknown error — also degrade gracefully.
    return { decisions: new Map(), inputTokens: 0, outputTokens: 0 };
  }
}

/** A stable numeric key for a pair, used as a Map index. */
function pairIndex(pair: ScoredPair): number {
  // Cantor pairing on the canonical (min,max) ids.
  const a = Math.min(pair.idA, pair.idB);
  const b = Math.max(pair.idA, pair.idB);
  return ((a + b) * (a + b + 1)) / 2 + b;
}

// ---------------------------------------------------------------------------
// Public: llmScorePairs
// ---------------------------------------------------------------------------

/**
 * Score borderline pairs with an LLM. Never demotes: pairs the LLM rejects
 * keep their original fuzzy score. Pairs the LLM confirms are promoted to 1.0.
 *
 * When no `apiKey` is available, degrades gracefully and returns the input.
 */
export async function llmScorePairs(
  pairs: readonly ScoredPair[],
  rows: readonly Row[],
  config: LLMScorerConfig,
  apiKey?: string,
): Promise<LLMScoreResult> {
  const budget = new BudgetTracker(
    config.budget ?? {},
    config.model ?? "gpt-4o-mini",
  );

  if (pairs.length === 0) {
    return { pairs: [], budget: budget.snapshot() };
  }

  const provider = detectProvider(apiKey, config.provider);
  const model = config.model ?? defaultModel(provider);

  // Display columns: everything not prefixed with `__`.
  const cols = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!k.startsWith("__")) cols.add(k);
    }
  }
  const displayCols = [...cols];

  const rowById = new Map<number, Row>();
  for (const r of rows) {
    const id = r["__row_id__"];
    if (typeof id === "number") rowById.set(id, r);
  }

  // Three-tier partition.
  const autoAccept: ScoredPair[] = [];
  const candidates: ScoredPair[] = [];
  const below: ScoredPair[] = [];
  for (const p of pairs) {
    if (p.score >= config.autoThreshold) autoAccept.push(p);
    else if (p.score >= config.candidateLo) candidates.push(p);
    else below.push(p);
  }

  // Build result scaffold: auto-accept promoted to 1.0, below untouched.
  const resultPairs: ScoredPair[] = [];
  for (const p of autoAccept) {
    resultPairs.push(makeScoredPair(p.idA, p.idB, 1.0));
  }
  for (const p of below) {
    resultPairs.push(p);
  }

  // If no API key, pass candidates through unchanged.
  if (!apiKey) {
    resultPairs.push(...candidates);
    return { pairs: resultPairs, budget: budget.snapshot() };
  }

  // Batch LLM scoring for candidates.
  const batchSize = Math.max(1, config.batchSize || 20);
  const llmDecisions = new Map<number, boolean>();
  for (const batch of batchify(candidates, batchSize)) {
    if (!budget.canProceed()) break;
    const res = await scoreBatch(
      batch,
      rowById,
      displayCols,
      provider,
      model,
      apiKey,
      budget,
    );
    res.decisions.forEach((v, k) => llmDecisions.set(k, v));
  }

  // Merge candidates: promote YES to 1.0, keep NO/unscored at original score.
  for (const p of candidates) {
    const decision = llmDecisions.get(pairIndex(p));
    if (decision === true) {
      resultPairs.push(makeScoredPair(p.idA, p.idB, 1.0));
    } else {
      resultPairs.push(p);
    }
  }

  return { pairs: resultPairs, budget: budget.snapshot() };
}

// ---------------------------------------------------------------------------
// Public: scoreStringsWithLlm (single-pair helper)
// ---------------------------------------------------------------------------

/**
 * Ask the LLM a single yes/no question about two strings. Returns 1.0
 * for yes, 0.0 for no, and 0.0 on any error (graceful).
 */
export async function scoreStringsWithLlm(
  a: string,
  b: string,
  config: LLMScorerConfig,
  apiKey?: string,
): Promise<{ score: number; budget: BudgetSnapshot; error?: string }> {
  const budget = new BudgetTracker(
    config.budget ?? {},
    config.model ?? "gpt-4o-mini",
  );
  if (!apiKey) return { score: 0, budget: budget.snapshot() };

  const provider = detectProvider(apiKey, config.provider);
  const model = config.model ?? defaultModel(provider);

  const prompt =
    "Are these two values referring to the same entity? Answer YES or NO.\n" +
    `A: ${a}\nB: ${b}`;

  try {
    const { text, inputTokens, outputTokens } =
      provider === "openai"
        ? await callOpenAI(prompt, apiKey, model, 10)
        : await callAnthropic(prompt, apiKey, model, 10);
    budget.record(inputTokens, outputTokens, model);
    const upper = text.trim().toUpperCase();
    const score = upper.includes("YES") ? 1.0 : 0.0;
    return { score, budget: budget.snapshot() };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    // eslint-disable-next-line no-console
    console.warn("scoreStringsWithLlm failed:", message);
    // Return score=0 (treats as "not matched") but surface the error so
    // operators can distinguish HTTP failures from genuine LLM "no" answers.
    return { score: 0, budget: budget.snapshot(), error: message };
  }
}

// Re-export budget types for convenience.
export type { BudgetSnapshot } from "./budget.js";
