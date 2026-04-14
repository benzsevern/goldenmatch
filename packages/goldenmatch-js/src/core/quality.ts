/**
 * quality.ts — Lightweight quality scan stub.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports a subset of goldenmatch/core/quality.py. The Python version
 * integrates with GoldenCheck; this port only provides the interface and a
 * handful of basic heuristics that are safe to run client-side.
 */

import type { Row, QualityConfig } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type QualitySeverity = "info" | "warn" | "error";

export interface QualityFinding {
  readonly column: string;
  readonly issue: string;
  readonly severity: QualitySeverity;
  readonly affectedRows: number;
  readonly sampleValues: readonly unknown[];
}

export interface QualityRunResult {
  readonly rows: readonly Row[];
  readonly findings: readonly QualityFinding[];
}

// ---------------------------------------------------------------------------
// Pattern detectors
// ---------------------------------------------------------------------------

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const DIGITS_RE = /^\d+$/;
const DATE_PATTERNS: readonly RegExp[] = [
  /^\d{4}-\d{2}-\d{2}$/, // ISO
  /^\d{1,2}\/\d{1,2}\/\d{2,4}$/, // US
  /^\d{1,2}-\d{1,2}-\d{2,4}$/,
  /^\d{8}$/, // yyyymmdd
];

function collectColumns(rows: readonly Row[]): string[] {
  const cols = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!key.startsWith("__")) cols.add(key);
    }
  }
  return [...cols];
}

function asStr(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return null;
}

// ---------------------------------------------------------------------------
// scanQuality
// ---------------------------------------------------------------------------

/**
 * Run a few cheap heuristics across the dataset: high null rate, low
 * cardinality, inconsistent date format, obviously malformed emails.
 */
export function scanQuality(
  rows: readonly Row[],
  _config?: QualityConfig,
): QualityFinding[] {
  const findings: QualityFinding[] = [];
  if (rows.length === 0) return findings;

  const total = rows.length;
  const columns = collectColumns(rows);

  for (const col of columns) {
    let nullCount = 0;
    let emailLike = 0;
    let malformedEmail = 0;
    let dateLike = 0;
    const dateFormatsSeen = new Set<number>();
    const nonNullSamples: unknown[] = [];
    const distinct = new Set<string>();

    for (const row of rows) {
      const raw = row[col];
      if (raw === null || raw === undefined || raw === "") {
        nullCount++;
        continue;
      }
      if (nonNullSamples.length < 5) nonNullSamples.push(raw);
      const s = asStr(raw);
      if (s !== null) {
        distinct.add(s);
        // Email heuristics
        if (s.includes("@")) {
          emailLike++;
          if (!EMAIL_RE.test(s)) malformedEmail++;
        }
        // Date format tracking
        for (let i = 0; i < DATE_PATTERNS.length; i++) {
          if (DATE_PATTERNS[i]!.test(s)) {
            dateFormatsSeen.add(i);
            dateLike++;
            break;
          }
        }
      }
    }

    const nullRate = nullCount / total;
    if (nullRate > 0.5) {
      findings.push({
        column: col,
        issue: `High null rate: ${(nullRate * 100).toFixed(1)}%`,
        severity: nullRate > 0.9 ? "error" : "warn",
        affectedRows: nullCount,
        sampleValues: [],
      });
    }

    const nonNull = total - nullCount;
    if (nonNull > 0) {
      const cardinalityRatio = distinct.size / nonNull;
      if (cardinalityRatio < 0.001 && distinct.size <= 1) {
        findings.push({
          column: col,
          issue: "Constant column (single distinct non-null value)",
          severity: "info",
          affectedRows: nonNull,
          sampleValues: nonNullSamples,
        });
      }
    }

    if (emailLike > 0 && malformedEmail > 0) {
      findings.push({
        column: col,
        issue: `Malformed email values (${malformedEmail} of ${emailLike})`,
        severity: "warn",
        affectedRows: malformedEmail,
        sampleValues: nonNullSamples,
      });
    }

    if (dateLike > 0 && dateFormatsSeen.size > 1) {
      findings.push({
        column: col,
        issue: `Inconsistent date formats (${dateFormatsSeen.size} distinct patterns)`,
        severity: "warn",
        affectedRows: dateLike,
        sampleValues: nonNullSamples,
      });
    }

    // Numeric-looking string column
    if (nonNull > 0 && distinct.size > 0) {
      let digitCount = 0;
      for (const v of distinct) {
        if (DIGITS_RE.test(v)) digitCount++;
      }
      if (digitCount === distinct.size && distinct.size > 1) {
        // Entire column is numeric strings — informational.
        findings.push({
          column: col,
          issue: "Column contains only numeric strings (consider typing)",
          severity: "info",
          affectedRows: nonNull,
          sampleValues: nonNullSamples,
        });
      }
    }
  }

  return findings;
}

// ---------------------------------------------------------------------------
// runQualityCheck
// ---------------------------------------------------------------------------

/**
 * Pass-through runner: produce findings, echo rows unchanged.
 *
 * Mirrors `_scan_only` / `run_quality_check` from the Python module: no
 * GoldenCheck, no row rewrites, just reportable findings.
 */
export function runQualityCheck(
  rows: readonly Row[],
  config?: QualityConfig,
): QualityRunResult {
  const findings = scanQuality(rows, config);
  return { rows, findings };
}
