/**
 * ingest.ts — Edge-safe data-shape transforms for row arrays.
 * Edge-safe: no `node:` imports, no file I/O.
 *
 * Ports goldenmatch/core/ingest.py minus file loading — callers are
 * expected to bring already-parsed rows (JSON, fetched CSV, etc.).
 */

import type { Row } from "./types.js";

// ---------------------------------------------------------------------------
// Column renaming
// ---------------------------------------------------------------------------

/**
 * Rename columns according to a {oldName: newName} map.
 *
 * Keys missing from the map are passed through untouched. If a rename
 * would collide with an existing column, the mapped column wins
 * (mirroring Polars behavior).
 */
export function applyColumnMap(
  rows: readonly Row[],
  columnMap: Readonly<Record<string, string>>,
): Row[] {
  if (Object.keys(columnMap).length === 0) {
    return rows.map((r) => ({ ...r }));
  }
  return rows.map((row) => {
    const newRow: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(row)) {
      const newKey = columnMap[key] ?? key;
      newRow[newKey] = val;
    }
    return newRow as Row;
  });
}

// ---------------------------------------------------------------------------
// Column validation
// ---------------------------------------------------------------------------

/**
 * Ensure every required column exists on the first row of `rows`.
 * No-ops on empty input.
 */
export function validateColumns(
  rows: readonly Row[],
  required: readonly string[],
): void {
  if (rows.length === 0) return;
  const first = rows[0]!;
  const missing = required.filter((c) => !(c in first));
  if (missing.length > 0) {
    throw new Error(`Required columns missing: ${missing.join(", ")}`);
  }
}

// ---------------------------------------------------------------------------
// Row concat (union of schemas)
// ---------------------------------------------------------------------------

/**
 * Concatenate several row arrays. Unioned schema: any column present in
 * any input appears in the output; missing values become null.
 */
export function concatRows(rowsArrays: readonly (readonly Row[])[]): Row[] {
  const allKeys = new Set<string>();
  let totalLen = 0;
  for (const arr of rowsArrays) {
    totalLen += arr.length;
    for (const row of arr) {
      for (const k of Object.keys(row)) allKeys.add(k);
    }
  }

  const out: Row[] = new Array(totalLen);
  let idx = 0;
  for (const arr of rowsArrays) {
    for (const row of arr) {
      const merged: Record<string, unknown> = {};
      for (const k of allKeys) {
        merged[k] = k in row ? row[k] : null;
      }
      out[idx++] = merged as Row;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Source tagging
// ---------------------------------------------------------------------------

/**
 * Add a `__source__` column to each row. Useful when concatenating rows
 * from multiple datasets and downstream logic needs to know the origin.
 */
export function tagSource(rows: readonly Row[], source: string): Row[] {
  return rows.map((row) => ({ ...row, __source__: source }));
}

/**
 * Add a `__row_id__` column if missing. IDs are assigned sequentially
 * starting at `startAt` (default 0).
 */
export function assignRowIds(rows: readonly Row[], startAt: number = 0): Row[] {
  return rows.map((row, i) => ({
    ...row,
    __row_id__: row["__row_id__"] ?? startAt + i,
  }));
}
