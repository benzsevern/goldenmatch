/**
 * budget.ts — LLM budget tracking: cost accounting, model tiering,
 * and graceful degradation. Ports `goldenmatch/core/llm_budget.py`.
 *
 * Edge-safe: no `node:` imports, no `process`.
 */

import type { BudgetConfig } from "../types.js";

// ---------------------------------------------------------------------------
// Model pricing (per 1M tokens, USD)
//
// The Python reference uses per-1K tokens; we use per-1M internally because
// that matches vendor pricing pages. The `estimateCost` math divides by
// 1_000_000, so the final numbers match either convention.
// ---------------------------------------------------------------------------

const PRICING: Record<string, { readonly input: number; readonly output: number }> = {
  "gpt-4o-mini": { input: 0.15, output: 0.6 },
  "gpt-4o": { input: 2.5, output: 10.0 },
  "gpt-4-turbo": { input: 10.0, output: 30.0 },
  "claude-3-5-haiku-latest": { input: 0.8, output: 4.0 },
  "claude-3-5-sonnet-latest": { input: 3.0, output: 15.0 },
  "claude-haiku-4-5-20251001": { input: 0.8, output: 4.0 },
  "claude-sonnet-4-20250514": { input: 3.0, output: 15.0 },
  "claude-opus-4-6": { input: 15.0, output: 75.0 },
};

/** Default pricing when a model isn't in the table. */
const DEFAULT_PRICING = { input: 1.0, output: 4.0 };

// ---------------------------------------------------------------------------
// Snapshot type
// ---------------------------------------------------------------------------

export interface BudgetSnapshot {
  readonly calls: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly costUsd: number;
  readonly model: string;
  readonly modelsUsed: Readonly<Record<string, number>>;
  readonly remainingCalls: number | null;
  readonly remainingUsd: number | null;
  readonly pctUsed: number;
  readonly exhausted: boolean;
}

// ---------------------------------------------------------------------------
// BudgetTracker
// ---------------------------------------------------------------------------

/**
 * Tracks LLM token usage, cost, and enforces budget limits.
 *
 * Mirrors `goldenmatch.core.llm_budget.BudgetTracker`. No thread lock
 * is needed here — the edge runtime is single-threaded per request.
 */
export class BudgetTracker {
  private _calls = 0;
  private _inputTokens = 0;
  private _outputTokens = 0;
  private _costUsd = 0;
  private _escalationCost = 0;
  private readonly _modelsUsed: Record<string, number> = {};

  constructor(
    private readonly config: BudgetConfig = {},
    public readonly model: string = "gpt-4o-mini",
  ) {}

  // ──────────────────────────────────────────────────────────
  // Cost estimation
  // ──────────────────────────────────────────────────────────

  /** Estimate the cost of a hypothetical call (USD). */
  estimateCost(
    inputTokens: number,
    outputTokens: number,
    model?: string,
  ): number {
    const m = model ?? this.model;
    const p = PRICING[m] ?? DEFAULT_PRICING;
    return (
      (inputTokens / 1_000_000) * p.input +
      (outputTokens / 1_000_000) * p.output
    );
  }

  // ──────────────────────────────────────────────────────────
  // Recording usage
  // ──────────────────────────────────────────────────────────

  /** Record usage from a completed API call. */
  record(inputTokens: number, outputTokens: number, model?: string): void {
    const m = model ?? this.model;
    const cost = this.estimateCost(inputTokens, outputTokens, m);
    this._calls += 1;
    this._inputTokens += inputTokens;
    this._outputTokens += outputTokens;
    this._costUsd += cost;
    this._modelsUsed[m] = (this._modelsUsed[m] ?? 0) + 1;
    if (this.config.escalationModel && m === this.config.escalationModel) {
      this._escalationCost += cost;
    }
  }

  // ──────────────────────────────────────────────────────────
  // Budget checks
  // ──────────────────────────────────────────────────────────

  /**
   * Return true if another call can proceed without exceeding the budget.
   * If `estimatedCost` is provided, checks whether the projected total stays
   * under `maxCostUsd`.
   */
  canProceed(estimatedCost?: number): boolean {
    if (this.config.maxCalls !== undefined && this._calls >= this.config.maxCalls) {
      return false;
    }
    if (
      this.config.maxCostUsd !== undefined &&
      this._costUsd >= this.config.maxCostUsd
    ) {
      return false;
    }
    if (
      estimatedCost !== undefined &&
      this.config.maxCostUsd !== undefined &&
      this._costUsd + estimatedCost > this.config.maxCostUsd
    ) {
      return false;
    }
    return true;
  }

  /**
   * Estimate whether a batch of a given token size can be sent.
   * Mirrors Python's `can_send(estimated_tokens)`.
   */
  canSend(estimatedTokens: number): boolean {
    if (!this.canProceed()) return false;
    if (this.config.maxCostUsd === undefined) return true;
    const est = this.estimateCost(estimatedTokens, 0, this.model);
    return this._costUsd + est <= this.config.maxCostUsd;
  }

  /**
   * Pick a model based on a pair score and escalation config.
   * Returns `escalationModel` when the score is in the escalation band
   * and the escalation sub-budget hasn't been exhausted.
   */
  selectModel(pairScore: number, defaultModel: string): string {
    if (!this.config.escalationModel) return defaultModel;
    const band = this.config.escalationBand;
    if (band === undefined || band.length < 2) return defaultModel;
    const lo = band[0]!;
    const hi = band[1]!;
    if (pairScore < lo || pairScore > hi) return defaultModel;

    if (
      this.config.maxCostUsd !== undefined &&
      this.config.escalationBudgetPct !== undefined
    ) {
      const maxEscalation =
        this.config.maxCostUsd * (this.config.escalationBudgetPct / 100);
      if (this._escalationCost >= maxEscalation) return defaultModel;
    }
    return this.config.escalationModel;
  }

  // ──────────────────────────────────────────────────────────
  // Accessors
  // ──────────────────────────────────────────────────────────

  get costUsd(): number {
    return Math.round(this._costUsd * 1e6) / 1e6;
  }

  get calls(): number {
    return this._calls;
  }

  get inputTokens(): number {
    return this._inputTokens;
  }

  get outputTokens(): number {
    return this._outputTokens;
  }

  get exhausted(): boolean {
    if (
      this.config.maxCostUsd !== undefined &&
      this._costUsd >= this.config.maxCostUsd
    ) {
      return true;
    }
    if (
      this.config.maxCalls !== undefined &&
      this._calls >= this.config.maxCalls
    ) {
      return true;
    }
    return false;
  }

  /** Return a snapshot of the current budget state. */
  snapshot(): BudgetSnapshot {
    const maxCalls = this.config.maxCalls;
    const maxCost = this.config.maxCostUsd;

    const remainingCalls =
      maxCalls !== undefined ? Math.max(0, maxCalls - this._calls) : null;
    const remainingUsd =
      maxCost !== undefined ? Math.max(0, maxCost - this._costUsd) : null;

    let pctUsed = 0;
    if (maxCost !== undefined && maxCost > 0) {
      pctUsed = Math.min(100, (this._costUsd / maxCost) * 100);
    } else if (maxCalls !== undefined && maxCalls > 0) {
      pctUsed = Math.min(100, (this._calls / maxCalls) * 100);
    }

    return {
      calls: this._calls,
      inputTokens: this._inputTokens,
      outputTokens: this._outputTokens,
      costUsd: this.costUsd,
      model: this.model,
      modelsUsed: { ...this._modelsUsed },
      remainingCalls,
      remainingUsd: remainingUsd === null ? null : Math.round(remainingUsd * 1e6) / 1e6,
      pctUsed: Math.round(pctUsed * 10) / 10,
      exhausted: this.exhausted,
    };
  }
}

// ---------------------------------------------------------------------------
// Token counting
// ---------------------------------------------------------------------------

/**
 * Rough token count approximation.
 * Rule of thumb: ~4 chars per token for English text.
 */
export function countTokensApprox(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

/** Return the pricing table (read-only) for inspection/tests. */
export function getPricing(): Readonly<
  Record<string, { readonly input: number; readonly output: number }>
> {
  return PRICING;
}
