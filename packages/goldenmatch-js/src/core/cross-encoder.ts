/**
 * cross-encoder.ts — LLM-based pair reranking ("cross-encoder lite").
 *
 * The Python port uses ONNX/sentence-transformers cross-encoders, which need
 * native deps. This edge-safe variant performs zero-shot reranking by asking
 * an LLM (OpenAI / Anthropic) for a 0..1 match score on borderline pairs.
 *
 * - Borderline pairs are identified by `band` around the matchkey threshold
 *   and/or top-N highest fuzzy scores below the auto-accept cutoff.
 * - The combined score is `0.5 * original + 0.5 * rerank` by default.
 * - Budget tracking uses BudgetTracker; on any HTTP failure we degrade to
 *   the original score for that pair.
 *
 * Edge-safe: uses global `fetch()` only.
 */

import type { BudgetConfig, MatchkeyConfig, Row, ScoredPair } from "./types.js";
import { BudgetTracker, countTokensApprox } from "./llm/budget.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type CrossEncoderProvider = "openai" | "anthropic";

export type CrossEncoderReranker = "llm" | "cross-encoder";

export interface CrossEncoderModelOptions {
  /** HuggingFace model id. Default "Xenova/ms-marco-MiniLM-L-6-v2". */
  readonly model?: string;
  /** Execution device. Default "cpu". */
  readonly device?: "cpu" | "webgpu";
  /** Use quantized weights (q8). Default true. */
  readonly quantized?: boolean;
}

export interface CrossEncoderOptions {
  /**
   * Reranker backend. `"llm"` (default) uses OpenAI/Anthropic.
   * `"cross-encoder"` loads `@huggingface/transformers` and runs a real
   * ONNX cross-encoder model locally; falls back to LLM on load/inference
   * failure.
   */
  readonly reranker?: CrossEncoderReranker;
  readonly provider?: CrossEncoderProvider;
  readonly model?: string;
  readonly apiKey?: string;
  /** Device for cross-encoder model (when reranker="cross-encoder"). */
  readonly device?: "cpu" | "webgpu";
  /** Use quantized cross-encoder weights (q8). Default true. */
  readonly quantized?: boolean;
  /** Re-rank pairs scoring within `band` of `mk.threshold` (default 0.1). */
  readonly band?: number;
  /** Maximum number of pairs to rerank, ranked highest to lowest. */
  readonly topN?: number;
  /** Weight given to the LLM rerank vs the original score. Default 0.5. */
  readonly rerankWeight?: number;
  /** Budget cap shared with regular LLM scoring. */
  readonly budget?: BudgetConfig;
  /** Optional override for retry attempts. Default 2. */
  readonly maxRetries?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readEnv(key: string): string | undefined {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.[key];
}

function detectProvider(
  apiKey: string | undefined,
  configProvider: CrossEncoderProvider | undefined,
): CrossEncoderProvider {
  if (configProvider) return configProvider;
  if (apiKey?.startsWith("sk-ant-")) return "anthropic";
  return "openai";
}

function defaultModel(provider: CrossEncoderProvider): string {
  return provider === "openai" ? "gpt-4o-mini" : "claude-haiku-4-5-20251001";
}

async function sleep(ms: number): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, ms));
}

function summariseRow(row: Row, fields: readonly string[]): string {
  const parts: string[] = [];
  for (const f of fields) {
    const v = row[f];
    if (v === null || v === undefined || v === "") continue;
    parts.push(`${f}: ${String(v)}`);
  }
  return parts.join(" | ").slice(0, 300);
}

function buildPrompt(
  rowA: Row,
  rowB: Row,
  fields: readonly string[],
): string {
  return [
    "Are these two records the same real-world entity?",
    'Answer with strict JSON: {"score": <number 0..1>}.',
    "1.0 = certainly the same. 0.0 = certainly different.",
    "",
    `A: ${summariseRow(rowA, fields)}`,
    `B: ${summariseRow(rowB, fields)}`,
  ].join("\n");
}

/** Extract a 0..1 score from an LLM response. Tolerates loose formatting. */
function parseScore(text: string): number | null {
  const trimmed = text.trim();
  // Try strict JSON first.
  try {
    const obj = JSON.parse(trimmed) as { score?: unknown };
    if (typeof obj.score === "number" && Number.isFinite(obj.score)) {
      return Math.min(1, Math.max(0, obj.score));
    }
  } catch {
    // fall through
  }
  // Try to find a JSON object inside.
  const match = trimmed.match(/\{[^}]*"score"\s*:\s*([0-9.]+)[^}]*\}/);
  if (match) {
    const v = parseFloat(match[1]!);
    if (Number.isFinite(v)) return Math.min(1, Math.max(0, v));
  }
  // Last resort: a bare number.
  const num = parseFloat(trimmed);
  if (Number.isFinite(num)) return Math.min(1, Math.max(0, num));
  return null;
}

// ---------------------------------------------------------------------------
// Provider calls
// ---------------------------------------------------------------------------

interface CallResult {
  readonly text: string;
  readonly inputTokens: number;
  readonly outputTokens: number;
}

async function callOpenAI(
  prompt: string,
  apiKey: string,
  model: string,
  maxRetries: number,
): Promise<CallResult> {
  return await callWithRetry(
    "https://api.openai.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        temperature: 0,
        max_tokens: 32,
        response_format: { type: "json_object" },
      }),
    },
    maxRetries,
    (data) => {
      const d = data as {
        choices?: Array<{ message?: { content?: string } }>;
        usage?: { prompt_tokens?: number; completion_tokens?: number };
      };
      return {
        text: d.choices?.[0]?.message?.content?.trim() ?? "",
        inputTokens: d.usage?.prompt_tokens ?? 0,
        outputTokens: d.usage?.completion_tokens ?? 0,
      };
    },
  );
}

async function callAnthropic(
  prompt: string,
  apiKey: string,
  model: string,
  maxRetries: number,
): Promise<CallResult> {
  return await callWithRetry(
    "https://api.anthropic.com/v1/messages",
    {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model,
        max_tokens: 32,
        messages: [{ role: "user", content: prompt }],
      }),
    },
    maxRetries,
    (data) => {
      const d = data as {
        content?: Array<{ text?: string }>;
        usage?: { input_tokens?: number; output_tokens?: number };
      };
      return {
        text: d.content?.[0]?.text?.trim() ?? "",
        inputTokens: d.usage?.input_tokens ?? 0,
        outputTokens: d.usage?.output_tokens ?? 0,
      };
    },
  );
}

async function callWithRetry(
  url: string,
  init: RequestInit,
  maxRetries: number,
  parse: (data: unknown) => CallResult,
): Promise<CallResult> {
  let lastErr: unknown = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const resp = await fetch(url, init);
      if (resp.status === 429 || (resp.status >= 500 && resp.status < 600)) {
        if (attempt < maxRetries) {
          await sleep(500 * Math.pow(2, attempt));
          continue;
        }
      }
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new CrossEncoderHttpError(
          resp.status,
          `${resp.status}: ${body.slice(0, 200)}`,
        );
      }
      const data = await resp.json();
      return parse(data);
    } catch (err) {
      lastErr = err;
      if (err instanceof CrossEncoderHttpError) throw err;
      if (attempt < maxRetries) {
        await sleep(500 * Math.pow(2, attempt));
        continue;
      }
    }
  }
  throw new CrossEncoderHttpError(0, `Network error: ${String(lastErr)}`);
}

export class CrossEncoderHttpError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "CrossEncoderHttpError";
  }
}

// ---------------------------------------------------------------------------
// CrossEncoderModel — local ONNX cross-encoder via @huggingface/transformers
// ---------------------------------------------------------------------------

/**
 * Optional local cross-encoder backed by @huggingface/transformers (ONNX).
 *
 * Kept optional so goldenmatch-js stays edge-safe / zero-deps by default.
 * The peer dependency must be installed explicitly:
 *   npm install @huggingface/transformers
 *
 * Typical usage is indirect: pass `reranker: "cross-encoder"` to
 * `rerankPair` / `rerankTopPairs` and a shared model instance is cached.
 */
export class CrossEncoderModel {
  // Using `any` to avoid a compile-time dep on the optional package.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private pipelineFn: any = null;
  private loading: Promise<void> | null = null;

  constructor(private readonly options: CrossEncoderModelOptions = {}) {}

  private async ensureLoaded(): Promise<void> {
    if (this.pipelineFn) return;
    if (this.loading) return this.loading;
    this.loading = (async () => {
      try {
        // Dynamic import so tsup/bundlers treat this as optional.
        const modName = "@huggingface/transformers";
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const transformers: any = await import(/* @vite-ignore */ modName);
        const pipeline = transformers.pipeline ?? transformers.default?.pipeline;
        if (typeof pipeline !== "function") {
          throw new Error("pipeline() export not found on @huggingface/transformers");
        }
        this.pipelineFn = await pipeline(
          "text-classification",
          this.options.model ?? "Xenova/ms-marco-MiniLM-L-6-v2",
          {
            device: this.options.device ?? "cpu",
            dtype: this.options.quantized !== false ? "q8" : "fp32",
          },
        );
      } catch (err) {
        this.loading = null;
        throw new Error(
          "'@huggingface/transformers' is required for the cross-encoder reranker. " +
            "Install: npm install @huggingface/transformers. " +
            "Original error: " +
            (err instanceof Error ? err.message : String(err)),
        );
      }
    })();
    return this.loading;
  }

  /** Score a single text pair. Returns a [0,1] relevance probability. */
  async score(textA: string, textB: string): Promise<number> {
    await this.ensureLoaded();
    const result = await this.pipelineFn({ text: textA, text_pair: textB });
    // Result may be an array (batch of 1) or a single object. Defensive.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const r: any = Array.isArray(result) ? result[0] : result;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const logits: any = r?.logits;
    const raw: unknown =
      r?.score ??
      (Array.isArray(logits) ? logits[0] : undefined) ??
      (logits && typeof logits === "object" && "data" in logits
        ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (logits as any).data?.[0]
        : undefined) ??
      r?.value ??
      0;
    const raw_n = typeof raw === "number" && Number.isFinite(raw) ? raw : 0;
    // text-classification pipelines already return a probability in [0,1].
    if (raw_n >= 0 && raw_n <= 1) return raw_n;
    // Sigmoid fallback for raw logits.
    return 1 / (1 + Math.exp(-raw_n));
  }

  /**
   * Score a batch of text pairs. Currently calls `score` serially —
   * transformers.js v3 batching APIs vary across versions, so we stay
   * conservative. Still avoids any LLM HTTP round-trips.
   */
  async scoreBatch(
    pairs: ReadonlyArray<readonly [string, string]>,
  ): Promise<number[]> {
    await this.ensureLoaded();
    const scores: number[] = [];
    for (const [a, b] of pairs) {
      scores.push(await this.score(a, b));
    }
    return scores;
  }
}

// Shared model cache keyed by model/device/quantized tuple. Most callers
// reuse defaults, so this effectively reuses a single instance per process.
let _modelCache: CrossEncoderModel | null = null;
let _modelCacheKey: string | null = null;

function getCrossEncoderModel(options: CrossEncoderOptions): CrossEncoderModel {
  const key = [
    options.model ?? "",
    options.device ?? "",
    options.quantized === undefined ? "" : String(options.quantized),
  ].join("|");
  if (_modelCache && _modelCacheKey === key) return _modelCache;
  const modelOpts: {
    model?: string;
    device?: "cpu" | "webgpu";
    quantized?: boolean;
  } = {};
  if (options.model !== undefined) modelOpts.model = options.model;
  if (options.device !== undefined) modelOpts.device = options.device;
  if (options.quantized !== undefined) modelOpts.quantized = options.quantized;
  _modelCache = new CrossEncoderModel(modelOpts);
  _modelCacheKey = key;
  return _modelCache;
}

/** Test hook: reset the cached model instance. */
export function _resetCrossEncoderModelCache(): void {
  _modelCache = null;
  _modelCacheKey = null;
}

function rowToText(row: Row, fields: readonly string[]): string {
  const parts: string[] = [];
  for (const f of fields) {
    const v = row[f];
    if (v === null || v === undefined || v === "") continue;
    parts.push(`${f}: ${String(v)}`);
  }
  return parts.join(" | ");
}

// ---------------------------------------------------------------------------
// Single-pair rerank
// ---------------------------------------------------------------------------

/**
 * Ask the LLM for a single 0..1 match score for two rows.
 *
 * Returns `NaN` when the call fails or the response is unparseable so
 * callers can fall back to the original score.
 */
export async function rerankPair(
  rowA: Row,
  rowB: Row,
  fields: readonly string[],
  options: CrossEncoderOptions = {},
): Promise<number> {
  // Cross-encoder fast path: real ONNX model via @huggingface/transformers.
  // On any failure (package missing, model load failure, inference error)
  // we fall back to the LLM path so callers keep getting a score.
  if (options.reranker === "cross-encoder") {
    try {
      const model = getCrossEncoderModel(options);
      const textA = rowToText(rowA, fields);
      const textB = rowToText(rowB, fields);
      return await model.score(textA, textB);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        "cross-encoder reranker failed, falling back to LLM:",
        err instanceof Error ? err.message : String(err),
      );
      // fall through to LLM path
    }
  }

  const apiKey = options.apiKey ?? readEnv("OPENAI_API_KEY") ?? readEnv("ANTHROPIC_API_KEY");
  if (!apiKey) return NaN;

  const provider = detectProvider(apiKey, options.provider);
  const model = options.model ?? defaultModel(provider);
  const maxRetries = options.maxRetries ?? 2;

  const prompt = buildPrompt(rowA, rowB, fields);

  try {
    const res =
      provider === "openai"
        ? await callOpenAI(prompt, apiKey, model, maxRetries)
        : await callAnthropic(prompt, apiKey, model, maxRetries);
    const score = parseScore(res.text);
    return score ?? NaN;
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(
      "cross-encoder LLM score failed:",
      err instanceof Error ? err.message : String(err),
    );
    return NaN;
  }
}

// ---------------------------------------------------------------------------
// Batch rerank top pairs
// ---------------------------------------------------------------------------

/**
 * Rerank borderline pairs via LLM. Pairs outside the borderline band are
 * returned unchanged. Pairs the LLM can't score (HTTP error, parse fail,
 * budget exhausted) keep their original score.
 *
 * Combine rule: `final = (1 - w) * original + w * rerank`, with `w = rerankWeight`.
 *
 * Pairs whose final score falls below `mk.threshold` are dropped from the
 * result, matching the Python "rerank then re-filter" behaviour.
 */
export async function rerankTopPairs(
  pairs: readonly ScoredPair[],
  rows: readonly Row[],
  mk: MatchkeyConfig,
  options: CrossEncoderOptions = {},
): Promise<readonly ScoredPair[]> {
  if (pairs.length === 0) return [];

  const useCrossEncoder = options.reranker === "cross-encoder";
  const apiKey = options.apiKey ?? readEnv("OPENAI_API_KEY") ?? readEnv("ANTHROPIC_API_KEY");
  // When neither backend is available, pass pairs through unchanged.
  if (!useCrossEncoder && !apiKey) return pairs;

  const provider = apiKey ? detectProvider(apiKey, options.provider) : "openai";
  const model = options.model ?? defaultModel(provider);
  const maxRetries = options.maxRetries ?? 2;
  const band = options.band ?? 0.1;
  const weight = options.rerankWeight ?? 0.5;
  const threshold = mk.threshold ?? 0.85;

  // Build row lookup.
  const rowById = new Map<number, Row>();
  for (const r of rows) {
    const id = r["__row_id__"];
    if (typeof id === "number") rowById.set(id, r);
  }
  const fieldNames = mk.fields.map((f) => f.field);

  // Identify borderline candidates: within `band` below threshold.
  // We rerank pairs whose original score sits in [threshold - band, 1.0].
  const lo = threshold - band;
  const candidatesIdx: number[] = [];
  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i]!.score >= lo) candidatesIdx.push(i);
  }
  // Sort candidates by score descending.
  candidatesIdx.sort((a, b) => pairs[b]!.score - pairs[a]!.score);

  // Cap to topN if configured.
  const limit = options.topN ?? candidatesIdx.length;
  const targets = candidatesIdx.slice(0, Math.max(0, limit));

  const budget = new BudgetTracker(options.budget ?? {}, model);

  // Cross-encoder fast path. Try loading the model once up-front; on any
  // failure we fall back to LLM scoring (if apiKey present) or give up.
  let ceModel: CrossEncoderModel | null = null;
  let ceFailed = false;
  if (useCrossEncoder) {
    try {
      ceModel = getCrossEncoderModel(options);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        "cross-encoder load failed, falling back to LLM:",
        err instanceof Error ? err.message : String(err),
      );
      ceFailed = true;
    }
  }

  const newScores = new Map<number, number>();
  let loggedLlmError = false;
  for (const idx of targets) {
    const pair = pairs[idx]!;
    const rowA = rowById.get(pair.idA);
    const rowB = rowById.get(pair.idB);
    if (!rowA || !rowB) continue;

    // Prefer cross-encoder if requested and available.
    if (ceModel && !ceFailed) {
      try {
        const score = await ceModel.score(
          rowToText(rowA, fieldNames),
          rowToText(rowB, fieldNames),
        );
        const combined = (1 - weight) * pair.score + weight * score;
        newScores.set(idx, Math.min(1, Math.max(0, combined)));
        continue;
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn(
          "cross-encoder inference failed, falling back to LLM for remaining pairs:",
          err instanceof Error ? err.message : String(err),
        );
        ceFailed = true;
        // fall through to LLM branch
      }
    }

    // LLM branch.
    if (!apiKey) continue;
    if (!budget.canProceed()) break;

    const prompt = buildPrompt(rowA, rowB, fieldNames);
    const estIn = countTokensApprox(prompt);
    if (!budget.canSend(estIn)) break;

    try {
      const res =
        provider === "openai"
          ? await callOpenAI(prompt, apiKey, model, maxRetries)
          : await callAnthropic(prompt, apiKey, model, maxRetries);
      budget.record(res.inputTokens || estIn, res.outputTokens || 16, model);
      const llmScore = parseScore(res.text);
      if (llmScore === null) continue;
      const combined = (1 - weight) * pair.score + weight * llmScore;
      newScores.set(idx, Math.min(1, Math.max(0, combined)));
    } catch (err) {
      // Degrade gracefully: keep original score for this pair.
      if (!loggedLlmError) {
        // eslint-disable-next-line no-console
        console.warn(
          "rerank LLM call failed for pair; keeping original score. First error:",
          err instanceof Error ? err.message : String(err),
        );
        loggedLlmError = true;
      }
      continue;
    }
  }

  // Rebuild output, dropping pairs whose new score falls under threshold.
  const out: ScoredPair[] = [];
  for (let i = 0; i < pairs.length; i++) {
    const original = pairs[i]!;
    const reranked = newScores.get(i);
    const finalScore = reranked ?? original.score;
    if (finalScore < threshold) continue;
    out.push({ idA: original.idA, idB: original.idB, score: finalScore });
  }
  return out;
}
