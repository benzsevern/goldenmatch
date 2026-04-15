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

function percentile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor(q * (sorted.length - 1))),
  );
  return sorted[idx] ?? 0;
}

function checkBlockSizes(
  rows: readonly Record<string, unknown>[],
  config: GoldenMatchConfig,
  findings: PreflightFinding[],
): GoldenMatchConfig {
  const blocking = config.blocking;
  if (blocking === undefined || blocking.keys === undefined) return config;
  if (rows.length === 0) return config;

  const sample = rows.length <= 10_000 ? rows : rows.slice(0, 10_000);

  for (const key of blocking.keys) {
    if (key.fields.length === 0) continue;
    const groups = new Map<string, number>();
    for (const r of sample) {
      const parts: string[] = [];
      for (const f of key.fields) {
        const v = r[f];
        parts.push(v === null || v === undefined ? "\u0000" : String(v));
      }
      const gkey = parts.join("\u0001");
      groups.set(gkey, (groups.get(gkey) ?? 0) + 1);
    }
    if (groups.size === 0) continue;
    const sizes = [...groups.values()].sort((a, b) => a - b);
    const p50 = percentile(sizes, 0.5);
    const p99 = percentile(sizes, 0.99);
    const subject = key.fields.join("+");
    if (p99 > 5000) {
      findings.push({
        check: "block_size",
        severity: "warning",
        subject,
        message: `blocking key [${subject}] yields p99 block size ${p99} (>5000) — fuzzy scoring will be slow`,
        repaired: false,
        repairNote: null,
      });
    } else if (p50 < 2 && sizes.length > 1) {
      findings.push({
        check: "block_size",
        severity: "warning",
        subject,
        message: `blocking key [${subject}] is too selective: p50 block size ${p50} — most records will be singletons`,
        repaired: false,
        repairNote: null,
      });
    }
  }

  return config;
}

function checkRemoteAssets(
  config: GoldenMatchConfig,
  findings: PreflightFinding[],
  allowRemoteAssets: boolean,
): GoldenMatchConfig {
  // Opt-in escape hatches: explicit caller permission or an LLM scorer
  // already committed to remote round-trips.
  if (allowRemoteAssets) return config;
  if (config.llmScorer?.enabled === true) return config;

  const matchkeys = config.matchkeys;
  if (matchkeys === undefined || matchkeys.length === 0) return config;

  const newMatchkeys: MatchkeyConfig[] = [];
  let changed = false;

  for (const mk of matchkeys) {
    const keptFields: MatchkeyField[] = [];
    let mkChanged = false;

    for (const f of mk.fields) {
      if (f.scorer === "embedding") {
        keptFields.push({ ...f, scorer: "ensemble" });
        mkChanged = true;
        findings.push({
          check: "remote_asset",
          severity: "warning",
          subject: `${mk.name}.${f.field}`,
          message: `scorer 'embedding' requires network + model download — demoted to 'ensemble'`,
          repaired: true,
          repairNote: "scorer embedding -> ensemble",
        });
      } else if (f.scorer === "record_embedding") {
        mkChanged = true;
        findings.push({
          check: "remote_asset",
          severity: "warning",
          subject: `${mk.name}.${f.field}`,
          message: `scorer 'record_embedding' requires network + model download — field dropped`,
          repaired: true,
          repairNote: "field removed (record_embedding)",
        });
        // drop field
      } else {
        keptFields.push(f);
      }
    }

    if (!mkChanged) {
      newMatchkeys.push(mk);
      continue;
    }
    changed = true;

    if (keptFields.length === 0) {
      findings.push({
        check: "remote_asset_matchkey_empty",
        severity: "info",
        subject: mk.name,
        message: `matchkey ${mk.name} dropped: all fields depended on remote-asset scorers`,
        repaired: true,
        repairNote: "matchkey removed",
      });
      continue;
    }

    if (mk.type === "weighted") {
      const next: MatchkeyConfig = {
        ...mk,
        fields: keptFields,
        // Rerank requires a cross-encoder model download too.
        ...(mk.rerank === true ? { rerank: false } : {}),
      };
      if (mk.rerank === true) {
        findings.push({
          check: "remote_asset",
          severity: "warning",
          subject: mk.name,
          message: `matchkey ${mk.name} rerank=true requires cross-encoder download — disabled`,
          repaired: true,
          repairNote: "rerank true -> false",
        });
      }
      newMatchkeys.push(next);
    } else {
      newMatchkeys.push({ ...mk, fields: keptFields });
    }
  }

  if (!changed) return config;
  return { ...config, matchkeys: newMatchkeys };
}

function checkWeightConfidence(
  config: GoldenMatchConfig,
  profiles: readonly ColumnProfile[] | undefined,
  findings: PreflightFinding[],
): GoldenMatchConfig {
  if (profiles === undefined) return config;
  const matchkeys = config.matchkeys;
  if (matchkeys === undefined || matchkeys.length === 0) return config;

  const byName = new Map<string, ColumnProfile>();
  for (const p of profiles) byName.set(p.name, p);

  const newMatchkeys: MatchkeyConfig[] = [];
  let changed = false;

  for (const mk of matchkeys) {
    if (mk.type !== "weighted") {
      newMatchkeys.push(mk);
      continue;
    }
    let mkChanged = false;
    const newFields: MatchkeyField[] = mk.fields.map((f) => {
      const profile = byName.get(f.field);
      if (
        profile !== undefined &&
        profile.confidence < 0.5 &&
        f.weight > 0.5
      ) {
        mkChanged = true;
        findings.push({
          check: "weight_confidence",
          severity: "warning",
          subject: `${mk.name}.${f.field}`,
          message: `field ${f.field} has classifier confidence ${profile.confidence.toFixed(2)} — weight capped from ${f.weight} to 0.5`,
          repaired: true,
          repairNote: `weight ${f.weight} -> 0.5`,
        });
        return { ...f, weight: 0.5 };
      }
      return f;
    });
    if (mkChanged) {
      changed = true;
      newMatchkeys.push({ ...mk, fields: newFields });
    } else {
      newMatchkeys.push(mk);
    }
  }

  if (!changed) return config;
  return { ...config, matchkeys: newMatchkeys };
}

export function preflight(
  rows: readonly Record<string, unknown>[],
  config: GoldenMatchConfig,
  options?: PreflightOptions,
): { readonly report: PreflightReport; readonly config: GoldenMatchConfig } {
  const findings: PreflightFinding[] = [];
  let current = config;

  current = checkColumns(rows, current, findings);
  current = checkCardinality(rows, current, findings);
  current = checkBlockSizes(rows, current, findings);
  current = checkRemoteAssets(
    current,
    findings,
    options?.allowRemoteAssets ?? false,
  );
  current = checkWeightConfidence(current, options?.profiles, findings);

  const report = makePreflightReport(findings, current !== config);
  return { report, config: current };
}

// ---------------------------------------------------------------------------
// postflight — score histogram + bimodality threshold nudge (Task 3.1)
// ---------------------------------------------------------------------------

interface HistResult {
  readonly histogram: ScoreHistogram;
  readonly valleyLocation: number | null;
  readonly isBimodal: boolean;
}

function signalScoreHistogram(
  pairScores: readonly { score: number }[],
): HistResult {
  const bins: number[] = [];
  for (let i = 0; i <= 100; i++) bins.push(i / 100);
  const counts = new Array<number>(100).fill(0);
  for (const p of pairScores) {
    const idx = Math.min(99, Math.max(0, Math.floor(p.score * 100)));
    counts[idx]! += 1;
  }
  // 5-bin smoothing
  const smoothed = counts.map((_, i) => {
    let sum = 0;
    let n = 0;
    for (let j = -2; j <= 2; j++) {
      const k = i + j;
      if (k >= 0 && k < counts.length) {
        sum += counts[k]!;
        n += 1;
      }
    }
    return sum / Math.max(1, n);
  });
  const max = Math.max(...smoothed, 0);
  const mean =
    smoothed.reduce((a, b) => a + b, 0) / Math.max(1, smoothed.length);
  const minHeight = Math.max(max * 0.3, mean * 2);
  const peaks: number[] = [];
  for (let i = 1; i < smoothed.length - 1; i++) {
    if (
      smoothed[i]! >= minHeight &&
      smoothed[i]! > smoothed[i - 1]! &&
      smoothed[i]! >= smoothed[i + 1]!
    ) {
      peaks.push(i);
    }
  }
  if (peaks.length < 2) {
    return {
      histogram: { bins, counts },
      valleyLocation: null,
      isBimodal: false,
    };
  }
  const first = peaks[0]!;
  const last = peaks[peaks.length - 1]!;
  if (last - first <= 10) {
    return {
      histogram: { bins, counts },
      valleyLocation: null,
      isBimodal: false,
    };
  }
  let valleyIdx = first;
  let valleyVal = smoothed[first]!;
  for (let i = first + 1; i < last; i++) {
    if (smoothed[i]! < valleyVal) {
      valleyVal = smoothed[i]!;
      valleyIdx = i;
    }
  }
  const depthRatio = valleyVal / Math.min(smoothed[first]!, smoothed[last]!);
  if (depthRatio >= 0.5) {
    return {
      histogram: { bins, counts },
      valleyLocation: null,
      isBimodal: false,
    };
  }
  return {
    histogram: { bins, counts },
    valleyLocation: valleyIdx / 100,
    isBimodal: true,
  };
}

function getFirstWeightedThreshold(
  config: GoldenMatchConfig,
): number | null {
  for (const mk of config.matchkeys ?? []) {
    if (mk.type === "weighted") return mk.threshold;
  }
  return null;
}

// TODO(autoconfig-iterative): full brute-force recall estimation lands with
// the iterative auto-config loop (separate future spec). For v0.3 we ship
// the sentinel so the schema contract is stable while the implementation is
// deferred.
function signalBlockingRecall(): "deferred" {
  return "deferred";
}

function signalClusterSizes(
  pairScores: readonly { idA: number; idB: number; score: number }[],
  threshold: number,
): {
  readonly percentiles: ClusterSizePercentiles;
  readonly oversized: readonly OversizedCluster[];
} {
  const above = pairScores.filter((p) => p.score >= threshold);
  const parent = new Map<number, number>();
  const find = (x: number): number => {
    let root = x;
    while ((parent.get(root) ?? root) !== root) root = parent.get(root)!;
    let y = x;
    while ((parent.get(y) ?? y) !== root) {
      const next = parent.get(y) ?? y;
      parent.set(y, root);
      y = next;
    }
    return root;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };
  for (const p of above) {
    if (!parent.has(p.idA)) parent.set(p.idA, p.idA);
    if (!parent.has(p.idB)) parent.set(p.idB, p.idB);
    union(p.idA, p.idB);
  }
  const sizeByRoot = new Map<number, number>();
  const membersByRoot = new Map<number, Set<number>>();
  for (const id of parent.keys()) {
    const r = find(id);
    sizeByRoot.set(r, (sizeByRoot.get(r) ?? 0) + 1);
    if (!membersByRoot.has(r)) membersByRoot.set(r, new Set());
    membersByRoot.get(r)!.add(id);
  }
  const sizes = Array.from(sizeByRoot.values()).sort((a, b) => a - b);
  const pct = (q: number) =>
    sizes.length === 0
      ? 0
      : sizes[Math.min(sizes.length - 1, Math.floor(sizes.length * q))] ?? 0;
  const percentiles: ClusterSizePercentiles = {
    p50: pct(0.5),
    p95: pct(0.95),
    p99: pct(0.99),
    max: sizes.length === 0 ? 0 : sizes[sizes.length - 1]!,
    count: sizes.length,
  };
  const oversized: OversizedCluster[] = [];
  let clusterId = 0;
  for (const [root, size] of sizeByRoot) {
    if (size <= 100) continue;
    const members = membersByRoot.get(root)!;
    let bottleneckPair: [number, number] = [-1, -1];
    let minScore = Infinity;
    for (const p of above) {
      if (members.has(p.idA) && members.has(p.idB) && p.score < minScore) {
        minScore = p.score;
        bottleneckPair = [
          Math.min(p.idA, p.idB),
          Math.max(p.idA, p.idB),
        ];
      }
    }
    oversized.push({ clusterId: clusterId++, size, bottleneckPair });
  }
  return { percentiles, oversized };
}

export function postflight(
  _rows: readonly Record<string, unknown>[],
  config: GoldenMatchConfig,
  options: {
    readonly pairScores: readonly { idA: number; idB: number; score: number }[];
    readonly currentThreshold?: number;
  },
): PostflightReport {
  const currentThreshold =
    options.currentThreshold ?? getFirstWeightedThreshold(config) ?? 0.7;
  const hist = signalScoreHistogram(options.pairScores);
  const adjustments: PostflightAdjustment[] = [];
  const advisories: string[] = [];

  if (hist.isBimodal && hist.valleyLocation !== null) {
    if (
      !config._strictAutoconfig &&
      Math.abs(hist.valleyLocation - currentThreshold) > 0.05
    ) {
      adjustments.push({
        field: "threshold",
        fromValue: currentThreshold,
        toValue: hist.valleyLocation,
        reason: "histogram valley location differs from current threshold",
        signal: "scoreHistogram",
      });
    }
  } else {
    advisories.push(
      "score distribution is unimodal; threshold cannot be auto-set",
    );
  }

  const clusterResult = signalClusterSizes(
    options.pairScores,
    currentThreshold,
  );

  // Placeholder values for signals added in Tasks 3.4-3.5.
  const signals: PostflightSignals = {
    scoreHistogram: hist.histogram,
    blockingRecall: signalBlockingRecall(),
    blockSizePercentiles: { p50: 0, p95: 0, p99: 0, max: 0 },
    thresholdOverlapPct: 0,
    totalPairsScored: options.pairScores.length,
    currentThreshold,
    preliminaryClusterSizes: clusterResult.percentiles,
    oversizedClusters: clusterResult.oversized,
  };

  return { signals, adjustments, advisories };
}
