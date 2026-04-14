/**
 * profiler.ts — Lightweight per-column data profiler.
 * Edge-safe: no `node:` imports.
 *
 * Ports parts of goldenmatch/core/profiler.py that autoconfig relies on.
 */

import type { Row } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ColumnType =
  | "email"
  | "phone"
  | "zip"
  | "date"
  | "name"
  | "geo"
  | "id"
  | "numeric"
  | "text";

export interface ColumnProfile {
  readonly name: string;
  readonly nullRate: number;
  readonly nullCount: number;
  readonly totalCount: number;
  readonly distinctCount: number;
  readonly cardinalityRatio: number;
  readonly inferredType: ColumnType;
  readonly avgLength: number;
  readonly maxLength: number;
  readonly sampleValues: readonly string[];
}

export interface DatasetProfile {
  readonly rowCount: number;
  readonly columns: readonly ColumnProfile[];
  readonly byName: Readonly<Record<string, ColumnProfile>>;
}

// ---------------------------------------------------------------------------
// Regex heuristics
// ---------------------------------------------------------------------------

const EMAIL_VALUE_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const PHONE_STRIP_RE = /[()\-+.\s]/g;
const DATE_VALUE_RES: readonly RegExp[] = [
  /^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$/,
  /^\d{4}[/\-]\d{1,2}[/\-]\d{1,2}$/,
  /^\d{1,2}\s[A-Za-z]+\s\d{2,4}$/,
];
const ZIP_VALUE_RE = /^\d{5}(-?\d{4})?$/;
const NAME_VALUE_RE = /^[A-Za-z][A-Za-z \-']{0,28}[A-Za-z]$|^[A-Za-z]{2,3}$/;

// ---------------------------------------------------------------------------
// Per-column profiling
// ---------------------------------------------------------------------------

function toStringOrNull(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string") {
    const t = value.trim();
    return t.length === 0 ? null : t;
  }
  return String(value);
}

function guessType(values: readonly string[], columnName: string): ColumnType {
  if (values.length === 0) return "text";
  const n = values.length;
  const lname = columnName.toLowerCase();

  // Email: >60% look like addresses
  const emailCount = values.reduce(
    (acc, v) => acc + (EMAIL_VALUE_RE.test(v) ? 1 : 0),
    0,
  );
  if (emailCount / n > 0.6) return "email";

  // Phone
  let phoneCount = 0;
  for (const v of values) {
    const stripped = v.replace(PHONE_STRIP_RE, "");
    if (/^\d+$/.test(stripped) && stripped.length >= 7 && stripped.length <= 15) {
      phoneCount++;
    }
  }
  if (phoneCount / n > 0.6) return "phone";

  // Zip: 5 or 9 digits (with optional dash)
  const zipCount = values.reduce(
    (acc, v) => acc + (ZIP_VALUE_RE.test(v) ? 1 : 0),
    0,
  );
  if (zipCount / n > 0.6) return "zip";

  // Date
  let dateCount = 0;
  for (const v of values) {
    if (DATE_VALUE_RES.some((re) => re.test(v))) dateCount++;
  }
  if (dateCount / n > 0.6) return "date";

  // Geographic columns by name + short text values
  if (/^(city|state|county|country|region|province)/i.test(lname)) return "geo";
  if (/city_desc|state_cd|country_code|state_code/i.test(lname)) return "geo";

  // Identifier columns by name
  if (/^id$|_id$|uuid|guid/i.test(lname)) return "id";

  // Name: >60% match alpha-name pattern
  const nameCount = values.reduce(
    (acc, v) => acc + (NAME_VALUE_RE.test(v) ? 1 : 0),
    0,
  );
  if (nameCount / n > 0.6) return "name";

  // Numeric
  let numericCount = 0;
  for (const v of values) {
    if (/^-?\d+(\.\d+)?$/.test(v)) numericCount++;
  }
  if (numericCount / n > 0.8) return "numeric";

  return "text";
}

function profileColumn(name: string, rawValues: readonly unknown[]): ColumnProfile {
  const totalCount = rawValues.length;
  let nullCount = 0;
  const nonNull: string[] = [];
  for (const v of rawValues) {
    const s = toStringOrNull(v);
    if (s === null) nullCount++;
    else nonNull.push(s);
  }

  const distinct = new Set(nonNull);
  const distinctCount = distinct.size;
  const cardinalityRatio = totalCount > 0 ? distinctCount / totalCount : 0;

  let totalLen = 0;
  let maxLen = 0;
  for (const v of nonNull) {
    totalLen += v.length;
    if (v.length > maxLen) maxLen = v.length;
  }
  const avgLength = nonNull.length > 0 ? totalLen / nonNull.length : 0;
  const nullRate = totalCount > 0 ? nullCount / totalCount : 0;

  // Sample values (first 5 unique)
  const sampleValues: string[] = [];
  for (const v of distinct) {
    sampleValues.push(v);
    if (sampleValues.length >= 5) break;
  }

  // Subsample for type guessing for performance
  const sampleForType = nonNull.length > 500 ? nonNull.slice(0, 500) : nonNull;
  const inferredType = guessType(sampleForType, name);

  return {
    name,
    nullRate,
    nullCount,
    totalCount,
    distinctCount,
    cardinalityRatio,
    inferredType,
    avgLength,
    maxLength: maxLen,
    sampleValues,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Profile all columns of a row array. */
export function profileRows(rows: readonly Row[]): DatasetProfile {
  if (rows.length === 0) {
    return { rowCount: 0, columns: [], byName: {} };
  }

  // Collect column names from all rows (not just first)
  const colSet = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!k.startsWith("__")) colSet.add(k);
    }
  }
  const columns = [...colSet];

  const profiles: ColumnProfile[] = [];
  const byName: Record<string, ColumnProfile> = {};
  for (const col of columns) {
    const values = rows.map((r) => r[col]);
    const profile = profileColumn(col, values);
    profiles.push(profile);
    byName[col] = profile;
  }

  return {
    rowCount: rows.length,
    columns: profiles,
    byName,
  };
}
