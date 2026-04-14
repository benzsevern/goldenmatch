/**
 * evaluate.ts — Precision/recall/F1 evaluation against ground truth.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/evaluate.py.
 */

import type { Row, ScoredPair, ClusterInfo } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EvalResult {
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
  readonly truePositives: number;
  readonly falsePositives: number;
  readonly falseNegatives: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function canonicalPair(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

function toPairSet(
  pairs: readonly (readonly [number, number])[],
): Set<string> {
  const out = new Set<string>();
  for (const [a, b] of pairs) {
    if (a === b) continue;
    out.add(canonicalPair(a, b));
  }
  return out;
}

function computeMetrics(
  tp: number,
  fp: number,
  fn: number,
): EvalResult {
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 =
    precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return {
    precision,
    recall,
    f1,
    truePositives: tp,
    falsePositives: fp,
    falseNegatives: fn,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Evaluate predicted pairs against ground-truth pairs.
 *
 * Pairs are treated as unordered (canonicalized to min:max).
 */
export function evaluatePairs(
  predictedPairs: readonly ScoredPair[],
  groundTruthPairs: readonly (readonly [number, number])[],
): EvalResult {
  const truth = toPairSet(groundTruthPairs);
  const predicted = new Set<string>();
  for (const p of predictedPairs) {
    if (p.idA === p.idB) continue;
    predicted.add(canonicalPair(p.idA, p.idB));
  }

  let tp = 0;
  let fp = 0;
  for (const key of predicted) {
    if (truth.has(key)) tp++;
    else fp++;
  }
  let fn = 0;
  for (const key of truth) {
    if (!predicted.has(key)) fn++;
  }

  return computeMetrics(tp, fp, fn);
}

/**
 * Evaluate clusters against ground-truth pairs.
 *
 * Expands each cluster's members into the full set of intra-cluster pairs and
 * compares that set to the ground truth.
 */
export function evaluateClusters(
  clusters: ReadonlyMap<number, ClusterInfo>,
  groundTruthPairs: readonly (readonly [number, number])[],
  _allIds: readonly number[],
): EvalResult {
  const predicted = new Set<string>();
  for (const info of clusters.values()) {
    const members = info.members;
    if (members.length < 2) continue;
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        predicted.add(canonicalPair(members[i]!, members[j]!));
      }
    }
  }

  const truth = toPairSet(groundTruthPairs);

  let tp = 0;
  let fp = 0;
  for (const key of predicted) {
    if (truth.has(key)) tp++;
    else fp++;
  }
  let fn = 0;
  for (const key of truth) {
    if (!predicted.has(key)) fn++;
  }

  return computeMetrics(tp, fp, fn);
}

/**
 * Extract ground truth pairs from a list of rows containing two id columns.
 *
 * Numeric strings are parsed to integers. Rows with missing/unparseable ids
 * are skipped.
 */
export function loadGroundTruthPairs(
  rows: readonly Row[],
  idColA: string,
  idColB: string,
): (readonly [number, number])[] {
  const out: [number, number][] = [];
  for (const row of rows) {
    const rawA = row[idColA];
    const rawB = row[idColB];
    if (rawA === null || rawA === undefined) continue;
    if (rawB === null || rawB === undefined) continue;
    const a = typeof rawA === "number" ? rawA : Number(rawA);
    const b = typeof rawB === "number" ? rawB : Number(rawB);
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    out.push([a, b]);
  }
  return out;
}
