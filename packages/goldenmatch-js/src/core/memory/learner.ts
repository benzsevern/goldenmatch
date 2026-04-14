/**
 * memory/learner.ts — Threshold tuning & weight learning from corrections.
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/core/memory/learner.py. Given ≥10 corrections, sweep
 * thresholds and pick the one maximizing F1 on the correction set. Given
 * ≥50 corrections with per-field subscores, fit a simple logistic-
 * regression-like weight update.
 */

import type { LearningConfig, MatchkeyConfig } from "../types.js";
import type { Correction } from "./store.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LearnedParams {
  readonly threshold?: number;
  readonly fieldWeights?: Readonly<Record<string, number>>;
  readonly correctionCount: number;
}

/**
 * Per-correction subscores. When present, keys correspond to matchkey field
 * names and values are in [0,1] representing each field's contribution.
 * The learner uses these only when ≥ weightsMinCorrections samples include
 * them.
 */
export interface CorrectionSubscores {
  readonly pairKey: string; // "minId|maxId"
  readonly subscores: Readonly<Record<string, number>>;
}

// ---------------------------------------------------------------------------
// Learner
// ---------------------------------------------------------------------------

const DEFAULT_LEARNING_CONFIG: LearningConfig = {
  thresholdMinCorrections: 10,
  weightsMinCorrections: 50,
};

export class MemoryLearner {
  constructor(
    private readonly config: LearningConfig = DEFAULT_LEARNING_CONFIG,
  ) {}

  /**
   * Tune threshold and (optionally) field weights from corrections.
   *
   * Threshold tuning: sweep 0.5..0.95 in 0.05 steps, compute F1 using each
   * correction's stored `score` vs its verdict. Returns the threshold with
   * the best F1 (ties break toward higher threshold for precision).
   *
   * Field weights: requires subscores. Fits a tiny gradient update that
   * nudges weights toward better discrimination of match / no_match.
   */
  learn(
    corrections: readonly Correction[],
    baseline: MatchkeyConfig,
    subscores?: readonly CorrectionSubscores[],
  ): LearnedParams {
    const result: {
      threshold?: number;
      fieldWeights?: Record<string, number>;
      correctionCount: number;
    } = { correctionCount: corrections.length };

    if (corrections.length >= this.config.thresholdMinCorrections) {
      const tuned = tuneThreshold(corrections);
      if (tuned !== null) result.threshold = tuned;
    }

    if (
      subscores &&
      corrections.length >= this.config.weightsMinCorrections &&
      subscores.length >= this.config.weightsMinCorrections
    ) {
      const learnedWeights = tuneWeights(corrections, subscores, baseline);
      if (learnedWeights) result.fieldWeights = learnedWeights;
    }

    return result;
  }
}

// ---------------------------------------------------------------------------
// Threshold tuning
// ---------------------------------------------------------------------------

/**
 * Sweep thresholds in [0.5, 0.95] step 0.05 and pick one maximizing F1.
 * Returns null if corrections cannot produce a meaningful F1 (e.g. all
 * same verdict).
 */
function tuneThreshold(corrections: readonly Correction[]): number | null {
  const positives = corrections.filter((c) => c.verdict === "match");
  const negatives = corrections.filter((c) => c.verdict === "no_match");
  if (positives.length === 0 || negatives.length === 0) return null;

  let bestThreshold = 0.85;
  let bestF1 = -1;

  for (let t = 0.5; t <= 0.95 + 1e-9; t += 0.05) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (const c of corrections) {
      const predicted = c.score >= t;
      if (c.verdict === "match") {
        if (predicted) tp++;
        else fn++;
      } else {
        if (predicted) fp++;
      }
    }
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 =
      precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    if (f1 > bestF1 || (f1 === bestF1 && t > bestThreshold)) {
      bestF1 = f1;
      bestThreshold = t;
    }
  }

  return Number(bestThreshold.toFixed(3));
}

// ---------------------------------------------------------------------------
// Weight tuning (simple gradient pass)
// ---------------------------------------------------------------------------

function sigmoid(x: number): number {
  if (x >= 0) {
    const ex = Math.exp(-x);
    return 1 / (1 + ex);
  }
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

function tuneWeights(
  corrections: readonly Correction[],
  subscores: readonly CorrectionSubscores[],
  baseline: MatchkeyConfig,
): Record<string, number> | null {
  const subByPair = new Map<string, Record<string, number>>();
  for (const s of subscores) {
    subByPair.set(s.pairKey, { ...s.subscores });
  }

  // Collect field list from baseline matchkey.
  const fields = baseline.fields.map((f) => f.field);
  if (fields.length === 0) return null;

  // Initialize weights from baseline.
  const weights = new Map<string, number>();
  for (const f of baseline.fields) weights.set(f.field, f.weight);

  // Build training set: for each correction we find its subscores.
  type Sample = { y: number; x: Record<string, number> };
  const samples: Sample[] = [];
  for (const c of corrections) {
    const [a, b] = c.rowIdA < c.rowIdB ? [c.rowIdA, c.rowIdB] : [c.rowIdB, c.rowIdA];
    const key = `${a}|${b}`;
    const sub = subByPair.get(key);
    if (!sub) continue;
    samples.push({
      y: c.verdict === "match" ? 1 : 0,
      x: sub,
    });
  }
  if (samples.length < 10) return null;

  const learningRate = 0.1;
  const iterations = 50;
  for (let iter = 0; iter < iterations; iter++) {
    const grad = new Map<string, number>();
    for (const f of fields) grad.set(f, 0);

    for (const sample of samples) {
      let z = 0;
      for (const f of fields) {
        const w = weights.get(f) ?? 0;
        const x = sample.x[f] ?? 0;
        z += w * x;
      }
      const pred = sigmoid(z);
      const err = pred - sample.y;
      for (const f of fields) {
        const x = sample.x[f] ?? 0;
        grad.set(f, (grad.get(f) ?? 0) + err * x);
      }
    }

    for (const f of fields) {
      const g = (grad.get(f) ?? 0) / samples.length;
      const w = weights.get(f) ?? 0;
      weights.set(f, w - learningRate * g);
    }
  }

  // Re-normalize weights so they sum to 1 (matchkey weights must average
  // out to the original budget; keep same total).
  const originalTotal = baseline.fields.reduce((acc, f) => acc + f.weight, 0);
  const newTotal = fields.reduce((acc, f) => acc + Math.max(0, weights.get(f) ?? 0), 0);
  if (newTotal <= 0) return null;
  const scale = originalTotal / newTotal;

  const out: Record<string, number> = {};
  for (const f of fields) {
    const w = Math.max(0, weights.get(f) ?? 0) * scale;
    out[f] = Number(w.toFixed(4));
  }
  return out;
}
