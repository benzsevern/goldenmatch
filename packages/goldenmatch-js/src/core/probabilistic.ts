/**
 * probabilistic.ts — Fellegi-Sunter probabilistic matching with EM-trained
 * parameters. Ports `goldenmatch/core/probabilistic.py` (discrete path).
 *
 * Implements:
 * - Comparison vectors (2/3/N-level field agreements)
 * - Splink-style EM: u estimated from random pairs (fixed), m trained via EM
 * - Blocking fields get fixed neutral priors
 * - Match weights as log2(m/u) log-likelihood ratios, normalized to [0,1]
 *
 * Edge-safe: no `node:` imports, no numpy. Uses typed arrays where helpful.
 */

import type { Row, MatchkeyConfig, MatchkeyField, ScoredPair } from "./types.js";
import { makeScoredPair } from "./types.js";
import { scoreField, asString } from "./scorer.js";
import { applyTransforms } from "./transforms.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EMOptions {
  readonly maxIterations?: number;
  readonly convergence?: number;
  readonly blockingFields?: readonly string[];
  readonly seed?: number;
  readonly nSamplePairs?: number;
}

export interface EMResult {
  /** P(level | match) per field. */
  readonly m: Readonly<Record<string, readonly number[]>>;
  /** P(level | non-match) per field. */
  readonly u: Readonly<Record<string, readonly number[]>>;
  /** log2(m / u) per level per field. Score weights. */
  readonly matchWeights: Readonly<Record<string, readonly number[]>>;
  /** Estimated p(match) in the sampled population. */
  readonly proportionMatched: number;
  readonly iterations: number;
  readonly converged: boolean;
}

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift32) — avoids relying on Math.random's seedability
// ---------------------------------------------------------------------------

function makeRng(seed: number): () => number {
  let x = seed | 0 || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    // Return in [0, 1): divide by 2^32 (not 2^32-1) so the value cannot reach 1.0.
    return (x >>> 0) / 0x100000000;
  };
}

// ---------------------------------------------------------------------------
// Field levels helper
// ---------------------------------------------------------------------------

function fieldLevels(f: MatchkeyField): number {
  return f.levels ?? 2;
}

function fieldPartialThreshold(f: MatchkeyField): number {
  return f.partialThreshold ?? 0.7;
}

// ---------------------------------------------------------------------------
// Public: buildComparisonVector
// ---------------------------------------------------------------------------

/**
 * Build a comparison vector: one integer level per field.
 *   levels=2: 0=disagree, 1=agree
 *   levels=3: 0=disagree, 1=partial, 2=agree (>= 0.95)
 *   levels=N: evenly spaced thresholds k/N for k in 1..N-1
 */
export function buildComparisonVector(
  rowA: Row,
  rowB: Row,
  fields: readonly MatchkeyField[],
): readonly number[] {
  const levels: number[] = [];
  for (const f of fields) {
    let valA = asString(rowA[f.field]);
    let valB = asString(rowB[f.field]);
    if (f.transforms.length > 0) {
      valA = applyTransforms(valA, f.transforms);
      valB = applyTransforms(valB, f.transforms);
    }
    const s = scoreField(valA, valB, f.scorer);
    const n = fieldLevels(f);
    const partial = fieldPartialThreshold(f);

    if (s === null) {
      levels.push(0);
      continue;
    }

    if (n === 2) {
      levels.push(s >= partial ? 1 : 0);
    } else if (n === 3) {
      if (s >= 0.95) levels.push(2);
      else if (s >= partial) levels.push(1);
      else levels.push(0);
    } else {
      let level = 0;
      for (let k = 1; k < n; k++) {
        if (s >= k / n) level = k;
      }
      levels.push(level);
    }
  }
  return levels;
}

// ---------------------------------------------------------------------------
// Random-pair sampling (used for u estimation)
// ---------------------------------------------------------------------------

function samplePairs(
  rows: readonly Row[],
  nPairs: number,
  rand: () => number,
): Array<readonly [number, number]> {
  const ids: number[] = [];
  for (const r of rows) {
    const id = r["__row_id__"];
    if (typeof id === "number") ids.push(id);
  }
  if (ids.length < 2) return [];

  const maxPossible = (ids.length * (ids.length - 1)) / 2;
  if (maxPossible <= nPairs) {
    const out: Array<readonly [number, number]> = [];
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        out.push([ids[i]!, ids[j]!] as const);
      }
    }
    return out;
  }

  const seen = new Set<string>();
  const pairs: Array<readonly [number, number]> = [];
  const maxAttempts = nPairs * 10;
  let attempts = 0;
  while (pairs.length < nPairs && attempts < maxAttempts) {
    attempts++;
    const i = Math.floor(rand() * ids.length);
    let j = Math.floor(rand() * ids.length);
    if (j === i) j = (j + 1) % ids.length;
    const a = Math.min(ids[i]!, ids[j]!);
    const b = Math.max(ids[i]!, ids[j]!);
    const key = `${a}:${b}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pairs.push([a, b] as const);
  }
  return pairs;
}

function buildComparisonMatrix(
  pairs: ReadonlyArray<readonly [number, number]>,
  rowById: ReadonlyMap<number, Row>,
  fields: readonly MatchkeyField[],
): number[][] {
  const out: number[][] = [];
  for (const [a, b] of pairs) {
    const rowA = rowById.get(a) ?? {};
    const rowB = rowById.get(b) ?? {};
    const vec = buildComparisonVector(rowA, rowB, fields);
    out.push([...vec]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Public: trainEM
// ---------------------------------------------------------------------------

/**
 * Splink-style EM training:
 *   1. Estimate u from random pairs (fixed throughout).
 *   2. Train m via EM starting from exponential priors.
 *   3. Blocking fields bypass EM and receive fixed neutral u + linear weights.
 */
export function trainEM(
  rows: readonly Row[],
  mk: MatchkeyConfig,
  options?: EMOptions,
): EMResult {
  // Probabilistic-only parameters; fall through to defaults for other variants.
  const emIterations =
    mk.type === "probabilistic" ? mk.emIterations : undefined;
  const convergenceThreshold =
    mk.type === "probabilistic" ? mk.convergenceThreshold : undefined;
  const maxIterations = options?.maxIterations ?? emIterations ?? 20;
  const convergence = options?.convergence ?? convergenceThreshold ?? 0.001;
  const blockingFields = new Set(options?.blockingFields ?? []);
  const seed = options?.seed ?? 42;
  const nSamplePairs = options?.nSamplePairs ?? 10000;

  const fields = mk.fields;
  if (fields.length === 0) return fallbackResult(mk);

  const rand = makeRng(seed);
  const rowById = new Map<number, Row>();
  for (const r of rows) {
    const id = r["__row_id__"];
    if (typeof id === "number") rowById.set(id, r);
  }

  // Step 1: u from random pairs.
  const sampleForU = samplePairs(rows, Math.min(nSamplePairs, 5000), rand);
  if (sampleForU.length < 10) return fallbackResult(mk);
  const uMatrix = buildComparisonMatrix(sampleForU, rowById, fields);

  const u: Record<string, number[]> = {};
  fields.forEach((f, j) => {
    const n = fieldLevels(f);
    const counts = new Array<number>(n).fill(0);
    for (const row of uMatrix) {
      const lvl = row[j]!;
      if (lvl >= 0 && lvl < n) counts[lvl]! += 1;
    }
    const total = counts.reduce((a, b) => a + b, 0) + n * 1e-6;
    u[f.field] = counts.map((c) => (c + 1e-6) / total);
  });

  // Blocking fields get neutral u.
  for (const f of fields) {
    if (blockingFields.has(f.field)) {
      const n = fieldLevels(f);
      if (n === 2) u[f.field] = [0.5, 0.5];
      else u[f.field] = [0.34, 0.33, ...new Array<number>(n - 2).fill(0.33 / Math.max(1, n - 2))];
    }
  }

  // Step 2: m priors (exponential: highest level gets most mass).
  const m: Record<string, number[]> = {};
  for (const f of fields) {
    const n = fieldLevels(f);
    const raw: number[] = [];
    for (let k = 0; k < n; k++) raw.push(2 ** k);
    const sum = raw.reduce((a, b) => a + b, 0);
    m[f.field] = raw.map((r) => r / sum);
  }

  // Use the same random-pair matrix for EM. In Python, blocked pairs are
  // preferred when available; we don't have blocks in this entry point, so
  // we train on the random sample (the fallback path).
  const compMatrix = uMatrix;
  const nPairs = compMatrix.length;

  let pMatch = 0.02;
  let converged = false;
  let iterations = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;
    const oldM: Record<string, number[]> = {};
    for (const k of Object.keys(m)) oldM[k] = [...m[k]!];

    // E-step.
    const posteriors = new Float64Array(nPairs);
    for (let i = 0; i < nPairs; i++) {
      let logM = Math.log(Math.max(pMatch, 1e-10));
      let logU = Math.log(Math.max(1 - pMatch, 1e-10));
      for (let j = 0; j < fields.length; j++) {
        const f = fields[j]!;
        const level = compMatrix[i]![j]!;
        const mProb = Math.max(m[f.field]![level] ?? 1e-10, 1e-10);
        const uProb = Math.max(u[f.field]![level] ?? 1e-10, 1e-10);
        logM += Math.log(mProb);
        logU += Math.log(uProb);
      }
      const maxLog = Math.max(logM, logU);
      const em = Math.exp(logM - maxLog);
      const eu = Math.exp(logU - maxLog);
      posteriors[i] = em / (em + eu);
    }

    // M-step (m only).
    let totalMatch = 0;
    for (let i = 0; i < nPairs; i++) totalMatch += posteriors[i]!;
    pMatch = Math.max(totalMatch / nPairs, 1e-6);

    for (let j = 0; j < fields.length; j++) {
      const f = fields[j]!;
      if (blockingFields.has(f.field)) continue;
      const n = fieldLevels(f);
      const newM = new Array<number>(n).fill(0);
      for (let i = 0; i < nPairs; i++) {
        const level = compMatrix[i]![j]!;
        if (level >= 0 && level < n) newM[level]! += posteriors[i]!;
      }
      const denom = totalMatch + n * 1e-6;
      for (let k = 0; k < n; k++) {
        newM[k] = (newM[k]! + 1e-6) / denom;
      }
      m[f.field] = newM;
    }

    // Convergence.
    let maxDelta = 0;
    for (const f of fields) {
      if (blockingFields.has(f.field)) continue;
      const n = fieldLevels(f);
      for (let k = 0; k < n; k++) {
        const d = Math.abs(m[f.field]![k]! - oldM[f.field]![k]!);
        if (d > maxDelta) maxDelta = d;
      }
    }
    if (maxDelta < convergence) {
      converged = true;
      break;
    }
  }

  // Match weights = log2(m/u), with fixed linear weights for blocking fields.
  const matchWeights: Record<string, number[]> = {};
  for (const f of fields) {
    const n = fieldLevels(f);
    if (blockingFields.has(f.field)) {
      const w: number[] = [];
      for (let k = 0; k < n; k++) {
        w.push(n > 1 ? -3.0 + (6.0 * k) / (n - 1) : 3.0);
      }
      matchWeights[f.field] = w;
      continue;
    }
    const w: number[] = [];
    for (let k = 0; k < n; k++) {
      const mVal = Math.max(m[f.field]![k]!, 1e-10);
      const uVal = Math.max(u[f.field]![k]!, 1e-10);
      w.push(Math.log2(mVal / uVal));
    }
    matchWeights[f.field] = w;
  }

  return {
    m: m as Readonly<Record<string, readonly number[]>>,
    u: u as Readonly<Record<string, readonly number[]>>,
    matchWeights: matchWeights as Readonly<Record<string, readonly number[]>>,
    proportionMatched: pMatch,
    iterations,
    converged,
  };
}

// ---------------------------------------------------------------------------
// Public: scoreProbabilistic
// ---------------------------------------------------------------------------

export interface ProbScoreOptions {
  readonly excludePairs?: ReadonlySet<string>;
  readonly threshold?: number;
}

/**
 * Score all pairs in a block using F-S match weights.
 * Returns normalized scores in [0,1] (weight sum mapped to 0-1 via min/max).
 * Pairs below threshold are filtered out.
 */
export function scoreProbabilistic(
  rows: readonly Row[],
  mk: MatchkeyConfig,
  em: EMResult,
  options?: ProbScoreOptions,
): ScoredPair[] {
  const fields = mk.fields;
  if (fields.length === 0) return [];

  const excludePairs = options?.excludePairs ?? new Set<string>();
  const linkThreshold =
    mk.type === "probabilistic" ? mk.linkThreshold : undefined;
  const threshold = options?.threshold ?? linkThreshold ?? 0.5;

  // Min/max possible weight totals for normalization.
  let maxWeight = 0;
  let minWeight = 0;
  for (const f of fields) {
    const w = em.matchWeights[f.field];
    if (!w || w.length === 0) continue;
    maxWeight += Math.max(...w);
    minWeight += Math.min(...w);
  }
  const weightRange = maxWeight - minWeight;

  const rowIds: number[] = [];
  const rowLookup: Row[] = [];
  for (const r of rows) {
    const id = r["__row_id__"];
    if (typeof id === "number") {
      rowIds.push(id);
      rowLookup.push(r);
    }
  }

  const results: ScoredPair[] = [];
  for (let i = 0; i < rowIds.length; i++) {
    for (let j = i + 1; j < rowIds.length; j++) {
      const a = Math.min(rowIds[i]!, rowIds[j]!);
      const b = Math.max(rowIds[i]!, rowIds[j]!);
      const key = `${a}:${b}`;
      if (excludePairs.has(key)) continue;

      const vec = buildComparisonVector(rowLookup[i]!, rowLookup[j]!, fields);

      let total = 0;
      for (let k = 0; k < fields.length; k++) {
        const f = fields[k]!;
        const level = vec[k]!;
        const w = em.matchWeights[f.field];
        if (!w) continue;
        total += w[level] ?? 0;
      }

      const normalized =
        weightRange > 0 ? (total - minWeight) / weightRange : 0.5;

      if (normalized >= threshold) {
        results.push(
          makeScoredPair(a, b, Math.round(normalized * 10000) / 10000),
        );
      }
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Public: scoreProbabilisticPair (single-pair variant for match_one use)
// ---------------------------------------------------------------------------

export function scoreProbabilisticPair(
  rowA: Row,
  rowB: Row,
  mk: MatchkeyConfig,
  em: EMResult,
): number {
  const fields = mk.fields;
  if (fields.length === 0) return 0.5;

  let maxWeight = 0;
  let minWeight = 0;
  for (const f of fields) {
    const w = em.matchWeights[f.field];
    if (!w || w.length === 0) continue;
    maxWeight += Math.max(...w);
    minWeight += Math.min(...w);
  }
  const weightRange = maxWeight - minWeight;
  if (weightRange <= 0) return 0.5;

  const vec = buildComparisonVector(rowA, rowB, fields);
  let total = 0;
  for (let k = 0; k < fields.length; k++) {
    const f = fields[k]!;
    const level = vec[k]!;
    const w = em.matchWeights[f.field];
    if (!w) continue;
    total += w[level] ?? 0;
  }
  return (total - minWeight) / weightRange;
}

// ---------------------------------------------------------------------------
// Fallback result for tiny datasets
// ---------------------------------------------------------------------------

function fallbackResult(mk: MatchkeyConfig): EMResult {
  const m: Record<string, number[]> = {};
  const u: Record<string, number[]> = {};
  const w: Record<string, number[]> = {};
  for (const f of mk.fields) {
    const n = fieldLevels(f);
    if (n === 2) {
      m[f.field] = [0.1, 0.9];
      u[f.field] = [0.9, 0.1];
      w[f.field] = [Math.log2(0.1 / 0.9), Math.log2(0.9 / 0.1)];
    } else if (n === 3) {
      m[f.field] = [0.05, 0.15, 0.8];
      u[f.field] = [0.8, 0.15, 0.05];
      w[f.field] = [
        Math.log2(0.05 / 0.8),
        Math.log2(0.15 / 0.15),
        Math.log2(0.8 / 0.05),
      ];
    } else {
      // Uniform fallback.
      const mv = new Array<number>(n).fill(1 / n);
      const uv = new Array<number>(n).fill(1 / n);
      m[f.field] = mv;
      u[f.field] = uv;
      w[f.field] = new Array<number>(n).fill(0);
    }
  }
  return {
    m,
    u,
    matchWeights: w,
    proportionMatched: 0.05,
    iterations: 0,
    converged: false,
  };
}
