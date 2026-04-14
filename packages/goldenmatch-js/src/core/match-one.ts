/**
 * match-one.ts — Single-record matching primitive.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/match_one.py.
 */

import type { Row, MatchkeyConfig } from "./types.js";
import { scorePair, asString } from "./scorer.js";
import { applyTransforms } from "./transforms.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MatchOneHit {
  readonly rowId: number;
  readonly score: number;
}

// ---------------------------------------------------------------------------
// matchOne
// ---------------------------------------------------------------------------

/**
 * Match a single record against a dataset using a weighted matchkey.
 *
 * Threshold defaults to 0 (return everything). For exact matchkeys use
 * {@link findExactMatchesOne}.
 *
 * Returns hits sorted by descending score. Rows are expected to carry
 * `__row_id__`.
 */
export function matchOne(
  record: Row,
  rows: readonly Row[],
  mk: MatchkeyConfig,
): readonly MatchOneHit[] {
  const threshold = mk.threshold ?? 0;
  const matches: MatchOneHit[] = [];
  for (const row of rows) {
    const score = scorePair(record, row, mk.fields);
    if (score >= threshold) {
      matches.push({ rowId: row["__row_id__"] as number, score });
    }
  }
  matches.sort((a, b) => b.score - a.score);
  return matches;
}

// ---------------------------------------------------------------------------
// findExactMatchesOne
// ---------------------------------------------------------------------------

/**
 * Find exact matches for a single record against a dataset.
 *
 * Builds the composite matchkey for the probe record, then scans the rows
 * and returns any that share the same composite key (score 1.0). Null
 * transformed fields disqualify the comparison.
 */
export function findExactMatchesOne(
  record: Row,
  rows: readonly Row[],
  mk: MatchkeyConfig,
): readonly MatchOneHit[] {
  // Build composite key for probe
  const probeParts: string[] = [];
  for (const f of mk.fields) {
    const t = applyTransforms(asString(record[f.field]), f.transforms);
    if (t === null) return [];
    probeParts.push(t);
  }
  const probeKey = probeParts.join("\x00");

  const hits: MatchOneHit[] = [];
  for (const row of rows) {
    const parts: string[] = [];
    let hasNull = false;
    for (const f of mk.fields) {
      const t = applyTransforms(asString(row[f.field]), f.transforms);
      if (t === null) {
        hasNull = true;
        break;
      }
      parts.push(t);
    }
    if (hasNull) continue;
    if (parts.join("\x00") === probeKey) {
      hits.push({ rowId: row["__row_id__"] as number, score: 1.0 });
    }
  }
  return hits;
}
