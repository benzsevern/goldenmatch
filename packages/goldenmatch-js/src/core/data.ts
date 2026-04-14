/**
 * data.ts — TabularData, edge-safe Polars replacement.
 * Wraps readonly Row[] with column operations, joins, groupBy, sampling.
 * No Node.js imports, no `process`.
 */

import type { ColumnValue, Row } from "./types.js";

// ---------------------------------------------------------------------------
// Null handling
// ---------------------------------------------------------------------------

/** Strings treated as null (case-insensitive, trimmed). */
const NULL_STRINGS = new Set([
  "",
  "null",
  "none",
  "nan",
  "n/a",
  "na",
  "nil",
  "#n/a",
  "missing",
  "undefined",
]);

/** Returns true for null, undefined, NaN, and null-ish string sentinels. */
export function isNullish(v: unknown): v is null | undefined {
  if (v === null || v === undefined) return true;
  if (typeof v === "string") return NULL_STRINGS.has(v.toLowerCase().trim());
  if (typeof v === "number") return Number.isNaN(v);
  return false;
}

/** Normalize an unknown value to ColumnValue (string | number | boolean | null). */
export function toColumnValue(v: unknown): ColumnValue {
  if (isNullish(v)) return null;
  if (typeof v === "string") return v;
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v;
  return String(v);
}

// ---------------------------------------------------------------------------
// Mulberry32 seedable PRNG (NOT Mersenne Twister)
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// TabularData
// ---------------------------------------------------------------------------

export class TabularData {
  private readonly _rows: readonly Row[];
  private _columnCache = new Map<string, readonly ColumnValue[]>();

  constructor(rows: readonly Row[]) {
    this._rows = rows;
  }

  // ---- Getters ----

  get rows(): readonly Row[] {
    return this._rows;
  }

  get columns(): readonly string[] {
    if (this._rows.length === 0) return [];
    return Object.keys(this._rows[0]!);
  }

  get rowCount(): number {
    return this._rows.length;
  }

  // ---- Column access ----

  /** Get column values with null coercion (N/A, NaN, etc. become null). */
  column(name: string): readonly ColumnValue[] {
    const cached = this._columnCache.get(name);
    if (cached) return cached;
    const values = this._rows.map((r) => toColumnValue(r[name]));
    this._columnCache.set(name, values);
    return values;
  }

  /** Raw column access -- preserves original values without null coercion.
   *  Use for profiling where "N/A" should remain a string, not become null. */
  rawColumn(name: string): readonly ColumnValue[] {
    return this._rows.map((r) => {
      const v = r[name];
      if (v === null || v === undefined) return null;
      if (
        typeof v === "string" ||
        typeof v === "number" ||
        typeof v === "boolean"
      )
        return v;
      return String(v);
    });
  }

  // ---- Null helpers ----

  nullCount(col: string): number {
    let count = 0;
    for (const v of this.column(col)) {
      if (v === null) count++;
    }
    return count;
  }

  dropNulls(col: string): ColumnValue[] {
    return this.column(col).filter(
      (v): v is Exclude<ColumnValue, null> => v !== null,
    );
  }

  // ---- Aggregation ----

  nUnique(col: string): number {
    const set = new Set<ColumnValue>();
    for (const v of this.dropNulls(col)) set.add(v);
    return set.size;
  }

  valueCounts(col: string): Map<ColumnValue, number> {
    const map = new Map<ColumnValue, number>();
    for (const v of this.dropNulls(col)) {
      map.set(v, (map.get(v) ?? 0) + 1);
    }
    return map;
  }

  /** MUST use loop -- Math.min(...array) crashes on >65K elements. */
  min(col: string): number | null {
    const nums = this.numericValues(col);
    if (nums.length === 0) return null;
    let m = nums[0]!;
    for (let i = 1; i < nums.length; i++) {
      if (nums[i]! < m) m = nums[i]!;
    }
    return m;
  }

  /** MUST use loop -- Math.max(...array) crashes on >65K elements. */
  max(col: string): number | null {
    const nums = this.numericValues(col);
    if (nums.length === 0) return null;
    let m = nums[0]!;
    for (let i = 1; i < nums.length; i++) {
      if (nums[i]! > m) m = nums[i]!;
    }
    return m;
  }

  mean(col: string): number | null {
    const nums = this.numericValues(col);
    if (nums.length === 0) return null;
    let sum = 0;
    for (const n of nums) sum += n;
    return sum / nums.length;
  }

  std(col: string): number | null {
    const nums = this.numericValues(col);
    if (nums.length < 2) return null;
    const avg = this.mean(col)!;
    let sumSq = 0;
    for (const n of nums) sumSq += (n - avg) ** 2;
    return Math.sqrt(sumSq / (nums.length - 1));
  }

  // ---- Filtering, mapping, slicing ----

  filter(predicate: (row: Row) => boolean): TabularData {
    return new TabularData(this._rows.filter(predicate));
  }

  map(fn: (row: Row, index: number) => Row): TabularData {
    return new TabularData(this._rows.map(fn));
  }

  slice(start: number, end?: number): TabularData {
    return new TabularData(this._rows.slice(start, end));
  }

  // ---- Column projection ----

  /** Keep only the named columns. */
  select(cols: readonly string[]): TabularData {
    const colSet = new Set(cols);
    const rows = this._rows.map((r) => {
      const out: Record<string, unknown> = {};
      for (const c of colSet) {
        if (c in r) out[c] = r[c];
      }
      return out as Row;
    });
    return new TabularData(rows);
  }

  /** Drop the named columns. */
  drop(cols: readonly string[]): TabularData {
    const dropSet = new Set(cols);
    const rows = this._rows.map((r) => {
      const out: Record<string, unknown> = {};
      for (const k of Object.keys(r)) {
        if (!dropSet.has(k)) out[k] = r[k];
      }
      return out as Row;
    });
    return new TabularData(rows);
  }

  // ---- Column mutation ----

  /** Return a new TabularData with an added (or replaced) column. */
  addColumn(name: string, values: readonly ColumnValue[]): TabularData {
    if (values.length !== this._rows.length) {
      throw new Error(
        `addColumn: values length (${values.length}) != row count (${this._rows.length})`,
      );
    }
    const rows = this._rows.map((r, i) => ({
      ...r,
      [name]: values[i],
    })) as Row[];
    return new TabularData(rows);
  }

  /** Add a sequential row index column (like Polars with_row_index). */
  withRowIndex(name = "__row_id__", offset = 0): TabularData {
    const rows = this._rows.map(
      (r, i) =>
        ({
          [name]: i + offset,
          ...r,
        }) as Row,
    );
    return new TabularData(rows);
  }

  // ---- Group by ----

  /** Group rows by a column, returning Map<stringKey, TabularData>. */
  groupBy(key: string): Map<string, TabularData> {
    const groups = new Map<string, Row[]>();
    for (const row of this._rows) {
      const v = toColumnValue(row[key]);
      const k = v === null ? "__null__" : String(v);
      let arr = groups.get(k);
      if (!arr) {
        arr = [];
        groups.set(k, arr);
      }
      arr.push(row);
    }
    const result = new Map<string, TabularData>();
    for (const [k, rows] of groups) {
      result.set(k, new TabularData(rows));
    }
    return result;
  }

  // ---- Join ----

  /**
   * Inner join with another TabularData on a shared column.
   * Columns from `other` get a suffix to avoid collisions.
   */
  join(
    other: TabularData,
    on: string,
    suffix = "_right",
  ): TabularData {
    // Build index on other
    const otherIndex = new Map<string, Row[]>();
    for (const row of other._rows) {
      const v = toColumnValue(row[on]);
      const k = v === null ? "__null__" : String(v);
      let arr = otherIndex.get(k);
      if (!arr) {
        arr = [];
        otherIndex.set(k, arr);
      }
      arr.push(row);
    }

    const otherCols = other.columns.filter((c) => c !== on);
    const result: Row[] = [];

    for (const leftRow of this._rows) {
      const v = toColumnValue(leftRow[on]);
      const k = v === null ? "__null__" : String(v);
      const matches = otherIndex.get(k);
      if (!matches) continue;

      for (const rightRow of matches) {
        const merged: Record<string, unknown> = { ...leftRow };
        for (const c of otherCols) {
          const key = c in leftRow ? `${c}${suffix}` : c;
          merged[key] = rightRow[c];
        }
        result.push(merged as Row);
      }
    }

    return new TabularData(result);
  }

  // ---- Sampling ----

  /** Fisher-Yates partial shuffle with seedable PRNG. */
  sample(n: number, seed = 42): TabularData {
    if (n >= this._rows.length) return this;
    const rng = mulberry32(seed);
    const indices = Array.from({ length: this._rows.length }, (_, i) => i);
    // Partial Fisher-Yates: shuffle last n elements
    for (let i = indices.length - 1; i > 0 && indices.length - 1 - i < n; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j]!, indices[i]!];
    }
    const sampled = indices.slice(indices.length - n).map((i) => this._rows[i]!);
    return new TabularData(sampled);
  }

  // ---- Sorting ----

  /** Sort by a column (ascending). Nulls sort last. */
  sortBy(col: string): TabularData {
    const sorted = [...this._rows].sort((a, b) => {
      const va = toColumnValue(a[col]);
      const vb = toColumnValue(b[col]);
      if (va === null && vb === null) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      if (typeof va === "number" && typeof vb === "number") return va - vb;
      return String(va).localeCompare(String(vb));
    });
    return new TabularData(sorted);
  }

  // ---- Unique ----

  /** Return rows with unique values in the given column (keeps first occurrence). */
  unique(col: string): TabularData {
    const seen = new Set<string>();
    const result: Row[] = [];
    for (const row of this._rows) {
      const v = toColumnValue(row[col]);
      const k = v === null ? "__null__" : String(v);
      if (!seen.has(k)) {
        seen.add(k);
        result.push(row);
      }
    }
    return new TabularData(result);
  }

  // ---- Serialization ----

  /** Return rows as plain dicts. */
  toDicts(): Row[] {
    return [...this._rows];
  }

  // ---- Numeric / string helpers ----

  numericValues(col: string): number[] {
    const result: number[] = [];
    for (const v of this.column(col)) {
      if (typeof v === "number" && Number.isFinite(v)) {
        result.push(v);
      }
    }
    return result;
  }

  stringValues(col: string): string[] {
    const result: string[] = [];
    for (const v of this.column(col)) {
      if (typeof v === "string") result.push(v);
    }
    return result;
  }

  // ---- Static constructors ----

  /** Create from an array of row dicts. */
  static fromDicts(rows: readonly Row[]): TabularData {
    return new TabularData(rows);
  }

  /** Create from column-oriented data: {col: values[]}. */
  static fromColumns(
    cols: Readonly<Record<string, readonly ColumnValue[]>>,
  ): TabularData {
    const colNames = Object.keys(cols);
    if (colNames.length === 0) return new TabularData([]);

    const len = cols[colNames[0]!]!.length;
    for (const name of colNames) {
      if (cols[name]!.length !== len) {
        throw new Error(
          `fromColumns: column "${name}" length (${cols[name]!.length}) != expected (${len})`,
        );
      }
    }

    const rows: Row[] = [];
    for (let i = 0; i < len; i++) {
      const row: Record<string, unknown> = {};
      for (const name of colNames) {
        row[name] = cols[name]![i];
      }
      rows.push(row as Row);
    }
    return new TabularData(rows);
  }
}
