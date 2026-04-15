/**
 * Auto-config verification layer — preflight + postflight.
 *
 * Ported from goldenmatch v1.5.0 (Python). Edge-safe: no node:* imports.
 *
 * Preflight runs at the end of autoConfigure(Rows); returns a PreflightReport
 * plus a possibly-repaired config. Raises ConfigValidationError on
 * unrepairable errors.
 *
 * Postflight runs inside runDedupePipeline / runMatchPipeline after scoring,
 * before clustering. Computes 4 signals; optionally adjusts threshold on
 * clear bimodality (unless config._strictAutoconfig is true).
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Snake_case chosen to match the Python PreflightCheckName string values;
 *  cross-repo parity fixtures compare these as literal strings. */
export type PreflightCheckName =
  | "missing_column"
  | "cardinality_high"
  | "cardinality_low"
  | "block_size"
  | "remote_asset"
  | "weight_confidence"
  | "no_matchkeys_remain"
  | "remote_asset_matchkey_empty";

export type Severity = "error" | "warning" | "info";

export interface PreflightFinding {
  readonly check: PreflightCheckName;
  readonly severity: Severity;
  readonly subject: string;
  readonly message: string;
  readonly repaired: boolean;
  readonly repairNote: string | null;
}

export interface PreflightReport {
  readonly findings: readonly PreflightFinding[];
  readonly configWasModified: boolean;
  /** True iff any finding has severity "error" and repaired === false. */
  readonly hasErrors: boolean;
}

export function makePreflightReport(
  findings: readonly PreflightFinding[],
  configWasModified: boolean,
): PreflightReport {
  const hasErrors = findings.some(
    (f) => f.severity === "error" && !f.repaired,
  );
  return Object.freeze({ findings, configWasModified, hasErrors });
}

export class ConfigValidationError extends Error {
  public readonly report: PreflightReport;
  constructor(report: PreflightReport) {
    const errors = report.findings.filter(
      (f) => f.severity === "error" && !f.repaired,
    );
    const msg =
      `auto-config produced ${errors.length} unrepairable error(s): ` +
      errors.map((e) => `${e.check}[${e.subject}]: ${e.message}`).join("; ");
    super(msg);
    this.name = "ConfigValidationError";
    this.report = report;
  }
}

export interface ScoreHistogram {
  readonly bins: readonly number[];
  readonly counts: readonly number[];
}

export interface BlockSizePercentiles {
  readonly p50: number;
  readonly p95: number;
  readonly p99: number;
  readonly max: number;
}

export interface ClusterSizePercentiles extends BlockSizePercentiles {
  readonly count: number;
}

export interface OversizedCluster {
  readonly clusterId: number;
  readonly size: number;
  readonly bottleneckPair: readonly [number, number];
}

export interface PostflightSignals {
  readonly scoreHistogram: ScoreHistogram;
  readonly blockingRecall: number | "deferred";
  readonly blockSizePercentiles: BlockSizePercentiles;
  readonly thresholdOverlapPct: number;
  readonly totalPairsScored: number;
  readonly currentThreshold: number;
  readonly preliminaryClusterSizes: ClusterSizePercentiles;
  readonly oversizedClusters: readonly OversizedCluster[];
}

export interface PostflightAdjustment {
  readonly field: string;
  readonly fromValue: unknown;
  readonly toValue: unknown;
  readonly reason: string;
  readonly signal: string;
}

export interface PostflightReport {
  readonly signals: PostflightSignals;
  readonly adjustments: readonly PostflightAdjustment[];
  readonly advisories: readonly string[];
}

// ---------------------------------------------------------------------------
// Utility: strip convention-private fields before YAML/JSON serialization.
// ---------------------------------------------------------------------------

export function stripConventionPrivate<T extends object>(cfg: T): T {
  const out = { ...cfg } as T & Record<string, unknown>;
  for (const k of Object.keys(out)) {
    if (k.startsWith("_")) {
      delete out[k];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// preflight / postflight — bodies land in Phase 2 / Phase 3.
// ---------------------------------------------------------------------------

export function preflight(): never {
  throw new Error("preflight not yet implemented — see Phase 2 of the plan");
}

export function postflight(): never {
  throw new Error("postflight not yet implemented — see Phase 3 of the plan");
}
