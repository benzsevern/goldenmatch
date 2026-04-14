/**
 * pprl/protocol.ts — Privacy-preserving record linkage.
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/pprl/protocol.py. Encodes both datasets as bloom
 * filters (CLKs) over the selected fields, then scores pairs via Dice or
 * Jaccard similarity. Two protocol stubs are surfaced: trusted third
 * party (no crypto beyond the bloom filter itself) and a simple SMC
 * sketch that adds a salt per party before encoding.
 */

import type { Row } from "../types.js";
import { applyTransform } from "../transforms.js";
import { diceCoefficient } from "../scorer.js";
import { profileRows, type ColumnProfile } from "../profiler.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PPRLConfig {
  readonly fields: readonly string[];
  readonly securityLevel: "standard" | "high" | "paranoid";
  readonly protocol: "trusted_third_party" | "smc";
  readonly threshold: number;
  /** Optional salt used with "high"/"paranoid" levels. */
  readonly salt?: string;
}

export interface PPRLMatch {
  readonly idA: number;
  readonly idB: number;
  readonly score: number;
}

export interface PPRLResult {
  readonly matches: readonly PPRLMatch[];
  readonly stats: Readonly<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function rowString(row: Row, fields: readonly string[]): string {
  const parts: string[] = [];
  for (const f of fields) {
    const v = row[f];
    if (v === null || v === undefined) continue;
    const s = typeof v === "string" ? v : String(v);
    const normalized = applyTransform(s, "lowercase") ?? s;
    const cleaned = applyTransform(normalized, "normalize_whitespace") ?? normalized;
    parts.push(cleaned);
  }
  return parts.join(" ");
}

function encodeRow(row: Row, config: PPRLConfig): string {
  const value = rowString(row, config.fields);
  if (value.length === 0) return "";
  const transformName =
    config.salt && (config.securityLevel === "high" || config.securityLevel === "paranoid")
      ? `bloom_filter:${config.securityLevel}:${config.salt}`
      : `bloom_filter:${config.securityLevel}`;
  return applyTransform(value, transformName) ?? "";
}

// ---------------------------------------------------------------------------
// Core linkage
// ---------------------------------------------------------------------------

/**
 * Encode both row sets as bloom filters and emit pair matches above the
 * configured threshold.
 */
export function runPPRL(
  rowsA: readonly Row[],
  rowsB: readonly Row[],
  config: PPRLConfig,
): PPRLResult {
  const encodedA: string[] = rowsA.map((r) => encodeRow(r, config));
  const encodedB: string[] = rowsB.map((r) => encodeRow(r, config));

  const matches: PPRLMatch[] = [];
  let compared = 0;

  for (let i = 0; i < encodedA.length; i++) {
    const a = encodedA[i]!;
    if (a.length === 0) continue;
    for (let j = 0; j < encodedB.length; j++) {
      const b = encodedB[j]!;
      if (b.length === 0) continue;
      compared++;
      const score = diceCoefficient(a, b);
      if (score >= config.threshold) {
        matches.push({ idA: i, idB: j, score });
      }
    }
  }

  return {
    matches,
    stats: {
      protocol: config.protocol,
      securityLevel: config.securityLevel,
      comparedPairs: compared,
      matchCount: matches.length,
      threshold: config.threshold,
      fields: config.fields,
    },
  };
}

// ---------------------------------------------------------------------------
// Auto-config
// ---------------------------------------------------------------------------

const MIN_LENGTH = 3;
const MAX_LENGTH = 15;
const MAX_FIELDS = 4;
const MIN_THRESHOLD = 0.85;

/**
 * Auto-pick PPRL parameters for the given dataset pair. Penalizes
 * near-unique fields (IDs), over-long fields, and high-null fields.
 */
export function autoConfigurePPRL(
  rowsA: readonly Row[],
  rowsB: readonly Row[],
): PPRLConfig {
  const profileA = profileRows(rowsA);
  const profileB = profileRows(rowsB);

  const commonCols = new Set<string>();
  for (const c of profileA.columns) {
    if (profileB.byName[c.name]) commonCols.add(c.name);
  }

  interface Candidate {
    readonly name: string;
    readonly score: number;
  }

  const candidates: Candidate[] = [];
  for (const name of commonCols) {
    const pa = profileA.byName[name];
    const pb = profileB.byName[name];
    if (!pa || !pb) continue;

    const nullRate = Math.max(pa.nullRate, pb.nullRate);
    if (nullRate > 0.3) continue;

    const avgLen = (pa.avgLength + pb.avgLength) / 2;
    if (avgLen < MIN_LENGTH) continue;
    if (avgLen > MAX_LENGTH) continue;

    // Penalize near-unique fields (likely IDs)
    const card = Math.max(pa.cardinalityRatio, pb.cardinalityRatio);
    if (card > 0.95) continue;

    // Score: prefer moderate cardinality, low nulls, moderate length.
    const lenPenalty = Math.abs(avgLen - 8) / 8;
    const score = (1 - nullRate) * (1 - Math.abs(card - 0.5)) * (1 - lenPenalty);

    candidates.push({ name, score });
  }

  candidates.sort((a, b) => b.score - a.score);
  const fields = candidates.slice(0, MAX_FIELDS).map((c) => c.name);

  return {
    fields,
    securityLevel: "standard",
    protocol: "trusted_third_party",
    threshold: MIN_THRESHOLD,
  };
}

// ---------------------------------------------------------------------------
// Protocol wrappers (API-parity stubs)
// ---------------------------------------------------------------------------

/**
 * Trusted-third-party linkage: both parties ship encoded CLKs to a
 * trusted intermediary that runs the similarity scoring. Same mechanics
 * as `runPPRL`, but callsite is semantically distinct.
 */
export function linkTrustedThirdParty(
  rowsA: readonly Row[],
  rowsB: readonly Row[],
  config: PPRLConfig,
): PPRLResult {
  return runPPRL(rowsA, rowsB, { ...config, protocol: "trusted_third_party" });
}

/**
 * Secure-multiparty-computation linkage (simplified): each party salts
 * its inputs with a shared secret. Requires a non-empty `salt` in config
 * and a "high"/"paranoid" security level.
 */
export function linkSMC(
  rowsA: readonly Row[],
  rowsB: readonly Row[],
  config: PPRLConfig,
): PPRLResult {
  if (!config.salt || config.salt.length === 0) {
    throw new Error("SMC protocol requires a non-empty `salt`");
  }
  if (config.securityLevel === "standard") {
    throw new Error("SMC protocol requires securityLevel of 'high' or 'paranoid'");
  }
  return runPPRL(rowsA, rowsB, { ...config, protocol: "smc" });
}

// Re-export profile type for consumers that want it alongside.
export type { ColumnProfile };
