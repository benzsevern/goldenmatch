/**
 * matchkey.ts — Matchkey builder for GoldenMatch-JS.
 * Edge-safe: no `node:` imports, pure TypeScript only.
 *
 * Ports matchkey building from goldenmatch/core/matchkey.py.
 * In Python this uses Polars expressions; here we work with Row arrays.
 */

import type { Row, MatchkeyConfig } from "./types.js";
import { applyTransforms } from "./transforms.js";

// ---------------------------------------------------------------------------
// computeMatchkeyValue — build a matchkey value for a single row
// ---------------------------------------------------------------------------

/**
 * Build a composite matchkey value for a single row.
 *
 * For each field in the matchkey config:
 * 1. Read the raw value from the row
 * 2. Apply the field's transform chain
 * 3. Concatenate all parts with "||" separator
 *
 * Returns `null` if any field value is null/undefined or transforms to null.
 */
export function computeMatchkeyValue(
  row: Row,
  mk: MatchkeyConfig,
): string | null {
  const parts: string[] = [];
  for (const f of mk.fields) {
    const raw = row[f.field];
    if (raw === null || raw === undefined) return null;
    const val = applyTransforms(String(raw), f.transforms);
    if (val === null) return null;
    parts.push(val);
  }
  return parts.join("||");
}

// ---------------------------------------------------------------------------
// computeMatchkeys — add matchkey columns to all rows
// ---------------------------------------------------------------------------

/**
 * Add matchkey columns to rows. For each matchkey `mk`, adds a column
 * `__mk_{mk.name}__` with the computed matchkey value.
 *
 * Returns new row objects (does not mutate originals).
 */
export function computeMatchkeys(
  rows: readonly Row[],
  matchkeys: readonly MatchkeyConfig[],
): Row[] {
  return rows.map((row) => {
    const extra: Record<string, unknown> = {};
    for (const mk of matchkeys) {
      extra[`__mk_${mk.name}__`] = computeMatchkeyValue(row, mk);
    }
    return { ...row, ...extra };
  });
}

// ---------------------------------------------------------------------------
// addRowIds — add sequential __row_id__ column
// ---------------------------------------------------------------------------

/**
 * Add `__row_id__` column as sequential integers starting from `offset`.
 *
 * Returns new row objects (does not mutate originals).
 */
export function addRowIds(rows: readonly Row[], offset: number = 0): Row[] {
  return rows.map((row, i) => ({
    ...row,
    __row_id__: offset + i,
  }));
}

// ---------------------------------------------------------------------------
// addSourceColumn — add __source__ column
// ---------------------------------------------------------------------------

/**
 * Add `__source__` column with the given source name to every row.
 *
 * Returns new row objects (does not mutate originals).
 */
export function addSourceColumn(
  rows: readonly Row[],
  sourceName: string,
): Row[] {
  return rows.map((row) => ({
    ...row,
    __source__: sourceName,
  }));
}
