/**
 * explain.ts — Natural language explanations for match decisions.
 * Ports `goldenmatch/core/explain.py` (+ parts of `explainer.py`).
 *
 * Template-based, zero LLM cost. Produces human-readable summaries of why
 * two records matched, plus cluster-level summaries.
 *
 * Edge-safe: no `node:` imports.
 */

import type {
  Row,
  MatchkeyConfig,
  MatchkeyField,
  ClusterInfo,
} from "./types.js";
import { scoreField, asString } from "./scorer.js";
import { pairKey } from "./cluster.js";
import { applyTransforms } from "./transforms.js";

// ---------------------------------------------------------------------------
// Score descriptors
// ---------------------------------------------------------------------------

const SCORE_DESCRIPTORS: ReadonlyArray<readonly [number, string]> = [
  [0.95, "identical"],
  [0.85, "very similar"],
  [0.7, "similar"],
  [0.5, "somewhat similar"],
  [0.3, "weakly similar"],
  [0.0, "different"],
];

const SCORER_NAMES: Readonly<Record<string, string>> = {
  jaro_winkler: "string similarity",
  levenshtein: "edit distance",
  token_sort: "token similarity",
  soundex_match: "phonetic match",
  exact: "exact match",
  ensemble: "best-of-multiple",
  dice: "Dice coefficient",
  jaccard: "Jaccard similarity",
  embedding: "semantic similarity",
  record_embedding: "record similarity",
};

function describeScore(score: number): string {
  for (const [threshold, desc] of SCORE_DESCRIPTORS) {
    if (score >= threshold) return desc;
  }
  return "different";
}

function describeScorer(name: string): string {
  return SCORER_NAMES[name] ?? name;
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface FieldScoreDetail {
  readonly field: string;
  readonly scorer: string;
  readonly valueA: string | null;
  readonly valueB: string | null;
  readonly score: number | null;
  readonly weight: number;
  readonly diffType: "identical" | "similar" | "different" | "missing" | "unknown";
}

export interface PairExplanation {
  readonly score: number;
  readonly fieldScores: Readonly<Record<string, number | null>>;
  readonly explanation: string;
  readonly confidence: "high" | "medium" | "low";
  readonly reasoning: readonly string[];
  readonly details: readonly FieldScoreDetail[];
}

export interface ClusterExplanation {
  readonly clusterId: number;
  readonly size: number;
  readonly confidence: number;
  readonly quality: string;
  readonly summary: string;
  readonly strongestField: string | null;
  readonly weakestLink: readonly [number, number] | null;
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function fmtVal(v: string | null): string {
  if (v === null || v === undefined) return "[null]";
  const s = String(v).trim();
  if (s.length > 40) return s.slice(0, 37) + "...";
  return s;
}

function classifyDiff(
  score: number | null,
): "identical" | "similar" | "different" | "missing" {
  if (score === null) return "missing";
  if (score >= 0.99) return "identical";
  if (score >= 0.7) return "similar";
  return "different";
}

function confidenceBand(score: number): "high" | "medium" | "low" {
  if (score >= 0.9) return "high";
  if (score >= 0.75) return "medium";
  return "low";
}

// ---------------------------------------------------------------------------
// Per-field scoring (used by both pair and cluster explanation)
// ---------------------------------------------------------------------------

function scoreFieldDetail(
  rowA: Row,
  rowB: Row,
  field: MatchkeyField,
): FieldScoreDetail {
  const rawA = asString(rowA[field.field]);
  const rawB = asString(rowB[field.field]);
  const valA = applyTransforms(rawA, field.transforms);
  const valB = applyTransforms(rawB, field.transforms);
  const score = scoreField(valA, valB, field.scorer);
  return {
    field: field.field,
    scorer: field.scorer,
    valueA: valA,
    valueB: valB,
    score,
    weight: field.weight,
    diffType: classifyDiff(score),
  };
}

function aggregateScore(details: readonly FieldScoreDetail[]): number {
  let weightedSum = 0;
  let weightSum = 0;
  for (const d of details) {
    if (d.score === null) continue;
    weightedSum += d.score * d.weight;
    weightSum += d.weight;
  }
  return weightSum === 0 ? 0 : weightedSum / weightSum;
}

// ---------------------------------------------------------------------------
// Public: explainPair
// ---------------------------------------------------------------------------

/**
 * Produce an NL explanation for why two rows match (or don't), using the
 * scorers and weights defined by the matchkey config.
 */
export function explainPair(
  rowA: Row,
  rowB: Row,
  mk: MatchkeyConfig,
): PairExplanation {
  const details = mk.fields.map((f) => scoreFieldDetail(rowA, rowB, f));
  const overall = aggregateScore(details);

  // Sort by contribution (score * weight) descending.
  const sorted = [...details].sort((a, b) => {
    const aw = (a.score ?? 0) * a.weight;
    const bw = (b.score ?? 0) * b.weight;
    return bw - aw;
  });

  // Build per-field phrases.
  const reasoning: string[] = [];
  let weakest: FieldScoreDetail | null = null;
  let weakestScore = 1.0;

  for (const d of sorted) {
    if (d.score !== null && d.score < weakestScore) {
      weakestScore = d.score;
      weakest = d;
    }

    const scorerDesc = describeScorer(d.scorer);
    if (d.diffType === "missing") {
      reasoning.push(`${d.field} missing on one side`);
    } else if (d.diffType === "identical" || (d.score ?? 0) >= 0.99) {
      reasoning.push(`${d.field} match exactly (${fmtVal(d.valueA)})`);
    } else if ((d.score ?? 0) >= 0.8) {
      reasoning.push(
        `${d.field} are ${describeScore(d.score!)} ` +
          `(${fmtVal(d.valueA)} ~ ${fmtVal(d.valueB)}, ` +
          `${scorerDesc} ${d.score!.toFixed(2)})`,
      );
    } else if ((d.score ?? 0) > 0) {
      reasoning.push(
        `${d.field} differ ` +
          `(${fmtVal(d.valueA)} vs ${fmtVal(d.valueB)}, ` +
          `${scorerDesc} ${d.score!.toFixed(2)})`,
      );
    } else {
      reasoning.push(
        `${d.field} do not match ` +
          `(${fmtVal(d.valueA)} vs ${fmtVal(d.valueB)})`,
      );
    }
  }

  // Build top-line explanation.
  const overallDesc = describeScore(overall);
  const header = `Match (${overallDesc}, score ${overall.toFixed(2)}):`;
  const body = reasoning.join("; ");
  const weakestNote =
    weakest && weakestScore < 0.8 ? ` Weakest signal: ${weakest.field}.` : "";
  const explanation = `${header} ${body}.${weakestNote}`.replace(/\s+/g, " ").trim();

  // Field scores map.
  const fieldScores: Record<string, number | null> = {};
  for (const d of details) fieldScores[d.field] = d.score;

  return {
    score: overall,
    fieldScores,
    explanation,
    confidence: confidenceBand(overall),
    reasoning,
    details,
  };
}

// ---------------------------------------------------------------------------
// Public: explainCluster
// ---------------------------------------------------------------------------

/**
 * Produce a template summary for a cluster: size, confidence, weakest link.
 * Mirrors `explain_cluster_nl` in Python.
 */
export function explainCluster(
  clusterId: number,
  cluster: ClusterInfo,
  rows: readonly Row[],
  mk: MatchkeyConfig,
): ClusterExplanation {
  const size = cluster.size;
  const confidence = cluster.confidence;
  const pairScores = cluster.pairScores;

  if (size <= 1) {
    return {
      clusterId,
      size,
      confidence,
      quality: cluster.clusterQuality,
      summary: "Singleton cluster with 1 record.",
      strongestField: null,
      weakestLink: null,
    };
  }

  // Score statistics.
  const scores: number[] = [];
  pairScores.forEach((s) => scores.push(s));
  const minScore = scores.length > 0 ? Math.min(...scores) : 0;
  const maxScore = scores.length > 0 ? Math.max(...scores) : 0;
  const avgScore =
    scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

  const parts: string[] = [];
  parts.push(
    `Cluster of ${size} records ` +
      `(confidence ${confidence.toFixed(2)}, ` +
      `scores ${minScore.toFixed(2)}-${maxScore.toFixed(2)}, ` +
      `avg ${avgScore.toFixed(2)}).`,
  );

  if (cluster.bottleneckPair !== null) {
    const [a, b] = cluster.bottleneckPair;
    const bpScore = pairScores.get(pairKey(a, b)) ?? 0;
    parts.push(
      `Weakest link: records ${a} and ${b} (score ${bpScore.toFixed(2)}).`,
    );
  }

  if (cluster.oversized) {
    parts.push("WARNING: cluster exceeds max size limit.");
  }

  // Identify the strongest field by averaging per-field scores across member pairs.
  const strongestField = computeStrongestField(cluster, rows, mk);

  return {
    clusterId,
    size,
    confidence,
    quality: cluster.clusterQuality,
    summary: parts.join(" "),
    strongestField,
    weakestLink: cluster.bottleneckPair,
  };
}

function computeStrongestField(
  cluster: ClusterInfo,
  rows: readonly Row[],
  mk: MatchkeyConfig,
): string | null {
  if (mk.fields.length === 0) return null;

  const rowById = new Map<number, Row>();
  for (const r of rows) {
    const id = r["__row_id__"];
    if (typeof id === "number") rowById.set(id, r);
  }

  const fieldSums: Record<string, { sum: number; count: number }> = {};
  for (const f of mk.fields) {
    fieldSums[f.field] = { sum: 0, count: 0 };
  }

  // Sample every pair in the cluster.
  const members = cluster.members;
  for (let i = 0; i < members.length; i++) {
    for (let j = i + 1; j < members.length; j++) {
      const rowA = rowById.get(members[i]!);
      const rowB = rowById.get(members[j]!);
      if (!rowA || !rowB) continue;
      for (const f of mk.fields) {
        const d = scoreFieldDetail(rowA, rowB, f);
        if (d.score === null) continue;
        const entry = fieldSums[f.field]!;
        entry.sum += d.score;
        entry.count += 1;
      }
    }
  }

  let best: string | null = null;
  let bestAvg = -1;
  for (const [name, { sum, count }] of Object.entries(fieldSums)) {
    if (count === 0) continue;
    const avg = sum / count;
    if (avg > bestAvg) {
      bestAvg = avg;
      best = name;
    }
  }
  return best;
}
