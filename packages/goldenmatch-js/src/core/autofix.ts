/**
 * autofix.ts — Lightweight row auto-fix utilities.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/autofix.py. Trims whitespace, nulls empty strings,
 * and converts common "no value" tokens to null.
 */

import type { Row } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AutoFixLog {
  readonly fixType: string;
  readonly column: string;
  readonly affectedRows: number;
}

export interface AutoFixResult {
  readonly rows: Row[];
  readonly log: AutoFixLog[];
}

// ---------------------------------------------------------------------------
// Tokens treated as null
// ---------------------------------------------------------------------------

const NULL_TOKENS: ReadonlySet<string> = new Set([
  "n/a",
  "na",
  "none",
  "null",
  "nil",
  "unknown",
  "unk",
  "-",
  "--",
  "?",
]);

function isNullToken(s: string): boolean {
  const lower = s.trim().toLowerCase();
  if (lower.length === 0) return true;
  return NULL_TOKENS.has(lower);
}

// ---------------------------------------------------------------------------
// autoFixRows
// ---------------------------------------------------------------------------

/**
 * Apply conservative fixes row-by-row:
 * - trim string values
 * - convert empty strings and common "no value" tokens to null
 *
 * Internal columns (prefix `__`) are preserved unchanged.
 */
export function autoFixRows(rows: readonly Row[]): AutoFixResult {
  const out: Row[] = [];
  const trimCounts = new Map<string, number>();
  const nullCounts = new Map<string, number>();

  for (const row of rows) {
    const fixed: Record<string, unknown> = {};
    let changed = false;
    for (const [key, value] of Object.entries(row)) {
      if (key.startsWith("__")) {
        fixed[key] = value;
        continue;
      }
      if (typeof value === "string") {
        const trimmed = value.trim();
        if (trimmed !== value) {
          trimCounts.set(key, (trimCounts.get(key) ?? 0) + 1);
          changed = true;
        }
        if (isNullToken(trimmed)) {
          fixed[key] = null;
          nullCounts.set(key, (nullCounts.get(key) ?? 0) + 1);
          changed = true;
        } else {
          fixed[key] = trimmed;
        }
      } else {
        fixed[key] = value;
      }
    }
    out.push(changed ? (fixed as Row) : row);
  }

  const log: AutoFixLog[] = [];
  for (const [col, n] of trimCounts) {
    log.push({ fixType: "trim_whitespace", column: col, affectedRows: n });
  }
  for (const [col, n] of nullCounts) {
    log.push({ fixType: "null_empty_or_token", column: col, affectedRows: n });
  }

  return { rows: out, log };
}
