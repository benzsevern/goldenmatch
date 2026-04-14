/**
 * sensitivity.ts — Parameter sweep engine for GoldenMatch.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/sensitivity.py.
 */

import type { Row, GoldenMatchConfig } from "./types.js";
import { runDedupePipeline } from "./pipeline.js";
import { compareClusters } from "./compare-clusters.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SweepParam {
  /** Dot-path into the config, e.g. "threshold", "blocking.maxBlockSize". */
  readonly path: string;
  readonly values: readonly unknown[];
}

export interface SweepPoint {
  readonly params: Readonly<Record<string, unknown>>;
  readonly stats: Readonly<Record<string, number>>;
  readonly twi?: number;
  readonly error?: string;
}

export interface SensitivityResult {
  readonly baseline: SweepPoint;
  readonly points: readonly SweepPoint[];
  readonly stable: boolean;
}

// ---------------------------------------------------------------------------
// Dot-path config override
// ---------------------------------------------------------------------------

/** Set a nested property by dot-path, returning a new object (shallow-cloned chain). */
function setPath(
  root: Record<string, unknown>,
  path: string,
  value: unknown,
): Record<string, unknown> {
  // Simple dot path; array indices via [n] not supported in this edge-safe port
  const parts = path.split(".").filter((p) => p.length > 0);
  if (parts.length === 0) return root;
  const clone: Record<string, unknown> = { ...root };
  let cursor: Record<string, unknown> = clone;
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i]!;
    const child = cursor[key];
    const childObj =
      child !== null && typeof child === "object" && !Array.isArray(child)
        ? { ...(child as Record<string, unknown>) }
        : {};
    cursor[key] = childObj;
    cursor = childObj;
  }
  cursor[parts[parts.length - 1]!] = value;
  return clone;
}

// ---------------------------------------------------------------------------
// Stats extraction
// ---------------------------------------------------------------------------

function statsFrom(result: ReturnType<typeof runDedupePipeline>): Record<string, number> {
  return {
    totalRecords: result.stats.totalRecords,
    totalClusters: result.stats.totalClusters,
    matchedRecords: result.stats.matchedRecords,
    uniqueRecords: result.stats.uniqueRecords,
    matchRate: result.stats.matchRate,
    scoredPairs: result.scoredPairs.length,
  };
}

// ---------------------------------------------------------------------------
// Cartesian product of sweep values
// ---------------------------------------------------------------------------

function cartesianPoints(
  params: readonly SweepParam[],
): Readonly<Record<string, unknown>>[] {
  if (params.length === 0) return [];
  let acc: Record<string, unknown>[] = [{}];
  for (const p of params) {
    const next: Record<string, unknown>[] = [];
    for (const base of acc) {
      for (const v of p.values) {
        next.push({ ...base, [p.path]: v });
      }
    }
    acc = next;
  }
  return acc;
}

// ---------------------------------------------------------------------------
// runSensitivity
// ---------------------------------------------------------------------------

/**
 * Run a parameter sweep.
 *
 * Each point in the Cartesian product of `params` is applied to
 * `baselineConfig`, the dedupe pipeline runs, and the resulting clusters are
 * compared against the baseline via CCMS. A `stable` flag is set when every
 * point's TWI is within 0.05 of 1.0.
 *
 * Per-point errors are caught and stored on the point so that partial
 * results are preserved.
 */
export function runSensitivity(
  rows: readonly Row[],
  baselineConfig: GoldenMatchConfig,
  params: readonly SweepParam[],
): SensitivityResult {
  // Baseline run
  const baselineRun = runDedupePipeline(rows, baselineConfig);
  const baseline: SweepPoint = {
    params: {},
    stats: statsFrom(baselineRun),
    twi: 1.0,
  };

  const points: SweepPoint[] = [];
  const combos = cartesianPoints(params);

  let stable = true;
  for (const combo of combos) {
    let cfg: GoldenMatchConfig = baselineConfig;
    for (const [path, value] of Object.entries(combo)) {
      cfg = setPath(
        cfg as Record<string, unknown>,
        path,
        value,
      ) as GoldenMatchConfig;
    }

    try {
      const runResult = runDedupePipeline(rows, cfg);
      let twi: number | undefined;
      try {
        twi = compareClusters(baselineRun.clusters, runResult.clusters).twi;
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn(
          `TWI comparison failed for sweep point ${JSON.stringify(combo)}: ${
            err instanceof Error ? err.message : String(err)
          }`,
        );
        twi = undefined;
      }
      if (twi === undefined || Math.abs(1 - twi) > 0.05) stable = false;
      points.push({
        params: combo,
        stats: statsFrom(runResult),
        ...(twi !== undefined ? { twi } : {}),
      });
    } catch (err) {
      stable = false;
      points.push({
        params: combo,
        stats: {},
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return { baseline, points, stable };
}

// ---------------------------------------------------------------------------
// stabilityReport
// ---------------------------------------------------------------------------

/** Render a human-readable stability report for a sensitivity result. */
export function stabilityReport(result: SensitivityResult): string {
  const lines: string[] = [];
  lines.push("Sensitivity sweep:");
  lines.push(`  Baseline: ${JSON.stringify(result.baseline.stats)}`);
  lines.push(`  Points:   ${result.points.length}`);
  lines.push(`  Stable:   ${result.stable ? "yes" : "no"}`);
  for (const p of result.points) {
    const twiStr = p.twi !== undefined ? p.twi.toFixed(4) : "n/a";
    const errStr = p.error !== undefined ? ` error=${p.error}` : "";
    lines.push(
      `  - params=${JSON.stringify(p.params)} twi=${twiStr} clusters=${
        p.stats["totalClusters"] ?? "?"
      }${errStr}`,
    );
  }
  return lines.join("\n");
}
