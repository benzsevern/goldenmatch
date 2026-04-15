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

// NOTE: cycle avoidance — types.ts imports PreflightReport/PostflightReport
// from this file, so this file must use `import type` for anything from
// types.ts. Value imports from ./domain.js are fine (domain.ts does not
// import from this module).
import type {
  GoldenMatchConfig,
  MatchkeyConfig,
  MatchkeyField,
  DomainConfig,
} from "./types.js";
import type { ColumnProfile } from "./profiler.js";
import { DOMAIN_EXTRACTED_COLS } from "./domain.js";

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
// preflight — six static checks, config-aware, returns possibly-repaired cfg.
// ---------------------------------------------------------------------------

export interface PreflightOptions {
  readonly profiles?: readonly ColumnProfile[];
  readonly allowRemoteAssets?: boolean;
}

function collectReferencedColumns(config: GoldenMatchConfig): Set<string> {
  const cols = new Set<string>();
  const b = config.blocking;
  if (b !== undefined) {
    for (const k of b.keys ?? []) for (const f of k.fields ?? []) cols.add(f);
    for (const k of b.passes ?? []) for (const f of k.fields ?? []) cols.add(f);
  }
  for (const mk of config.matchkeys ?? []) {
    for (const f of mk.fields ?? []) {
      if (f.field) cols.add(f.field);
      for (const c of f.columns ?? []) cols.add(c);
    }
  }
  return cols;
}

function checkColumns(
  rows: readonly Record<string, unknown>[],
  config: GoldenMatchConfig,
  findings: PreflightFinding[],
): GoldenMatchConfig {
  const present = new Set<string>(
    rows.length > 0 ? Object.keys(rows[0] as object) : [],
  );
  let repairedConfig = config;

  for (const col of collectReferencedColumns(config)) {
    if (present.has(col)) continue;
    // Matchkey-intermediate columns produced at runtime by the scorer.
    if (col.startsWith("__mk_")) continue;
    if (DOMAIN_EXTRACTED_COLS.has(col)) {
      const profile = repairedConfig._domainProfile;
      if (profile !== undefined) {
        const existing = repairedConfig.domain;
        const newDomain: DomainConfig = {
          enabled: true,
          mode: profile.name,
          confidenceThreshold: existing?.confidenceThreshold ?? 0.7,
          llmValidation: existing?.llmValidation ?? false,
          ...(existing?.budget !== undefined
            ? { budget: existing.budget }
            : {}),
        };
        repairedConfig = { ...repairedConfig, domain: newDomain };
        findings.push({
          check: "missing_column",
          severity: "warning",
          subject: col,
          message: `domain-extracted column ${col} auto-resolved via config.domain`,
          repaired: true,
          repairNote: `set config.domain.enabled=true, mode=${profile.name}`,
        });
        continue;
      }
    }
    findings.push({
      check: "missing_column",
      severity: "error",
      subject: col,
      message: `column ${col} referenced but not present and not producible`,
      repaired: false,
      repairNote: null,
    });
  }

  return repairedConfig;
}

function cardinalityRatio(
  rows: readonly Record<string, unknown>[],
  col: string,
): number {
  if (rows.length === 0) return 0;
  const distinct = new Set<unknown>();
  for (const r of rows) distinct.add(r[col]);
  return distinct.size / rows.length;
}

function checkCardinality(
  rows: readonly Record<string, unknown>[],
  config: GoldenMatchConfig,
  findings: PreflightFinding[],
): GoldenMatchConfig {
  const matchkeys = config.matchkeys;
  if (matchkeys === undefined || matchkeys.length === 0) return config;
  if (rows.length === 0) return config;

  const present = new Set<string>(Object.keys(rows[0] as object));
  const kept: MatchkeyConfig[] = [];
  let dropped = 0;

  for (const mk of matchkeys) {
    if (mk.type !== "exact") {
      kept.push(mk);
      continue;
    }
    let drop = false;
    for (const f of mk.fields) {
      if (!present.has(f.field)) continue;
      const ratio = cardinalityRatio(rows, f.field);
      if (ratio >= 0.99) {
        findings.push({
          check: "cardinality_high",
          severity: "warning",
          subject: mk.name,
          message: `exact matchkey ${mk.name} dropped: column ${f.field} has cardinality ${ratio.toFixed(3)} (near-unique)`,
          repaired: true,
          repairNote: `dropped matchkey (cardinality_ratio=${ratio.toFixed(3)} >= 0.99)`,
        });
        drop = true;
        break;
      }
      if (ratio <= 0.01) {
        findings.push({
          check: "cardinality_low",
          severity: "warning",
          subject: mk.name,
          message: `exact matchkey ${mk.name} dropped: column ${f.field} has cardinality ${ratio.toFixed(3)} (too few distinct values)`,
          repaired: true,
          repairNote: `dropped matchkey (cardinality_ratio=${ratio.toFixed(3)} <= 0.01)`,
        });
        drop = true;
        break;
      }
    }
    if (drop) {
      dropped += 1;
    } else {
      kept.push(mk);
    }
  }

  if (dropped === 0) return config;

  if (kept.length === 0) {
    findings.push({
      check: "no_matchkeys_remain",
      severity: "error",
      subject: "matchkeys",
      message:
        "all matchkeys dropped by cardinality guards — nothing left to score",
      repaired: false,
      repairNote: null,
    });
  }

  return { ...config, matchkeys: kept };
}

export function preflight(
  rows: readonly Record<string, unknown>[],
  config: GoldenMatchConfig,
  options?: PreflightOptions,
): { readonly report: PreflightReport; readonly config: GoldenMatchConfig } {
  void options;
  const findings: PreflightFinding[] = [];
  let current = config;

  current = checkColumns(rows, current, findings);
  current = checkCardinality(rows, current, findings);
  // Checks 4-6 added in subsequent tasks.

  const report = makePreflightReport(findings, current !== config);
  return { report, config: current };
}

export function postflight(): never {
  throw new Error("postflight not yet implemented — see Phase 3 of the plan");
}
