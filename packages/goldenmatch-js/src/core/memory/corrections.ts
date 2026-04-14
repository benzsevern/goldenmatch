/**
 * memory/corrections.ts — Apply stored corrections to scored pairs.
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/core/memory/corrections.py. A correction is only
 * applied if both rows still hash to the values seen when the correction
 * was recorded (dual-hash staleness detection).
 */

import type { Row, ScoredPair } from "../types.js";
import type { Correction, MemoryStore } from "./store.js";

// ---------------------------------------------------------------------------
// Row hashing
// ---------------------------------------------------------------------------

/**
 * Deterministic FNV-1a 32-bit hash. Matches store-side hashing so
 * corrections can survive serialization/round-trips.
 */
function hashString(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16);
}

/** Hash of a row across its non-internal fields (sorted, stringified). */
export function hashRow(row: Row): string {
  const keys = Object.keys(row)
    .filter((k) => !k.startsWith("__"))
    .sort();
  const parts: string[] = [];
  for (const k of keys) {
    const v = row[k];
    const s = v === null || v === undefined ? "\u0000null" : String(v);
    parts.push(`${k}=${s}`);
  }
  return hashString(parts.join("|"));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pairKey(a: number, b: number): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

function getRowId(row: Row): number | null {
  const raw = row["__row_id__"];
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Stored-correction metadata
// ---------------------------------------------------------------------------

export interface StoredRowHashes {
  readonly rowIdAHash: string;
  readonly rowIdBHash: string;
}

/**
 * A caller can either provide a per-correction hash map (populated at
 * collection time) or ask applyCorrections to compute current hashes alone
 * — in which case staleness detection is a no-op (hashes always match).
 */
export interface ApplyCorrectionsOptions {
  readonly originalHashes?: ReadonlyMap<string, StoredRowHashes>;
  /** When a correction matches, clamp pair score to this value. Default 1.0 for match, 0.0 for no_match. */
  readonly matchScore?: number;
  readonly noMatchScore?: number;
}

// ---------------------------------------------------------------------------
// Apply corrections
// ---------------------------------------------------------------------------

/**
 * Apply user corrections stored in `store` to a list of scored pairs.
 *
 * For each correction:
 *   - Find the pair (idA,idB) in the scored_pairs list.
 *   - If caller supplied original hashes, compare them against a fresh
 *     hash of the current row. Mismatch => stale, skip.
 *   - Otherwise apply the verdict:
 *       "match"    -> score clamped to matchScore (default 1.0)
 *       "no_match" -> score clamped to noMatchScore (default 0.0)
 *
 * Returns the modified pairs plus counts of applied / stale corrections.
 */
export function applyCorrections(
  pairs: readonly ScoredPair[],
  rows: readonly Row[],
  store: MemoryStore,
  options?: ApplyCorrectionsOptions,
): { pairs: readonly ScoredPair[]; applied: number; stale: number } {
  const matchScore = options?.matchScore ?? 1.0;
  const noMatchScore = options?.noMatchScore ?? 0.0;

  // Build index: rowId -> Row for current-state hashing.
  const rowById = new Map<number, Row>();
  for (const r of rows) {
    const id = getRowId(r);
    if (id !== null) rowById.set(id, r);
  }

  // Index corrections by canonical pair key.
  const byPair = new Map<string, Correction>();
  for (const c of store.list()) {
    const key = pairKey(c.rowIdA, c.rowIdB);
    const existing = byPair.get(key);
    // Keep the highest-trust correction per pair (most recent on tie).
    if (
      existing === undefined ||
      c.trust > existing.trust ||
      (c.trust === existing.trust && c.timestamp > existing.timestamp)
    ) {
      byPair.set(key, c);
    }
  }

  let applied = 0;
  let stale = 0;
  const out: ScoredPair[] = [];

  for (const pair of pairs) {
    const key = pairKey(pair.idA, pair.idB);
    const correction = byPair.get(key);
    if (!correction) {
      out.push(pair);
      continue;
    }

    // Dual-hash staleness check (if caller populated `originalHashes`).
    if (options?.originalHashes) {
      const stored = options.originalHashes.get(key);
      if (stored) {
        const rowA = rowById.get(pair.idA);
        const rowB = rowById.get(pair.idB);
        if (!rowA || !rowB) {
          stale++;
          out.push(pair);
          continue;
        }
        const currentA = hashRow(rowA);
        const currentB = hashRow(rowB);
        const match =
          (currentA === stored.rowIdAHash && currentB === stored.rowIdBHash) ||
          (currentA === stored.rowIdBHash && currentB === stored.rowIdAHash);
        if (!match) {
          stale++;
          out.push(pair);
          continue;
        }
      }
    }

    applied++;
    out.push({
      idA: pair.idA,
      idB: pair.idB,
      score: correction.verdict === "match" ? matchScore : noMatchScore,
    });
  }

  return { pairs: out, applied, stale };
}
