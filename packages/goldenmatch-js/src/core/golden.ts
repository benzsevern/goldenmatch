/**
 * golden.ts — Golden record builder with per-field merge strategies.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 */

import type {
  ClusterInfo,
  ClusterProvenance,
  FieldProvenance,
  GoldenFieldRule,
  GoldenRulesConfig,
  Row,
} from "./types.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INTERNAL_PREFIXES = [
  "__row_id__",
  "__source__",
  "__block_key__",
  "__mk_",
  "__cluster_id__",
  "__golden_confidence__",
];

function isInternal(col: string): boolean {
  return (
    col === "__mk_" ||
    INTERNAL_PREFIXES.some((prefix) => col.startsWith(prefix))
  );
}

// ---------------------------------------------------------------------------
// MergeField result
// ---------------------------------------------------------------------------

export interface MergeFieldResult {
  readonly value: unknown;
  readonly confidence: number;
  readonly sourceIndex: number | null;
}

// ---------------------------------------------------------------------------
// MergeField options
// ---------------------------------------------------------------------------

export interface MergeFieldOptions {
  readonly sources?: readonly string[];
  readonly dates?: readonly unknown[];
  readonly qualityWeights?: readonly number[];
}

// ---------------------------------------------------------------------------
// mergeField
// ---------------------------------------------------------------------------

/**
 * Merge a list of values using the given strategy.
 *
 * Strategies:
 * - most_complete: pick longest string value; tie-break by quality weight
 * - majority_vote: pick most frequent value; weighted by quality if available
 * - source_priority: pick first non-null from priority list
 * - most_recent: pick value with most recent date
 * - first_non_null: pick first non-null; prefer highest quality weight
 */
export function mergeField(
  values: readonly unknown[],
  rule: GoldenFieldRule,
  options?: MergeFieldOptions,
): MergeFieldResult {
  const nonNull: [number, unknown][] = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i] != null) {
      nonNull.push([i, values[i]]);
    }
  }

  if (nonNull.length === 0) {
    return { value: null, confidence: 0.0, sourceIndex: null };
  }

  // All non-null values identical -> confidence 1.0 shortcut
  const uniqueVals = new Set(nonNull.map(([, v]) => v));
  if (uniqueVals.size === 1) {
    return { value: nonNull[0]![1], confidence: 1.0, sourceIndex: nonNull[0]![0] };
  }

  const strategy = rule.strategy;

  switch (strategy) {
    case "most_complete":
      return _mostComplete(nonNull, options?.qualityWeights);
    case "majority_vote":
      return _majorityVote(nonNull, options?.qualityWeights);
    case "source_priority":
      return _sourcePriority(values, rule, options?.sources);
    case "most_recent":
      return _mostRecent(values, options?.dates);
    case "first_non_null":
      return _firstNonNull(nonNull, options?.qualityWeights);
    default:
      throw new Error(`Unknown strategy: ${strategy}`);
  }
}

// ---------------------------------------------------------------------------
// Strategy implementations
// ---------------------------------------------------------------------------

function _mostComplete(
  nonNull: [number, unknown][],
  qualityWeights?: readonly number[],
): MergeFieldResult {
  const strVals = nonNull.map(
    ([i, v]) => [i, String(v), v] as [number, string, unknown],
  );
  const maxLen = Math.max(...strVals.map(([, s]) => s.length));
  const longest = strVals.filter(([, s]) => s.length === maxLen);

  if (longest.length === 1) {
    return { value: longest[0]![2], confidence: 1.0, sourceIndex: longest[0]![0] };
  }

  // Tie-break by quality weight
  if (qualityWeights) {
    const best = longest.reduce((a, b) => {
      const wa =
        a[0] < qualityWeights.length ? qualityWeights[a[0]]! : 1.0;
      const wb =
        b[0] < qualityWeights.length ? qualityWeights[b[0]]! : 1.0;
      return wa >= wb ? a : b;
    });
    const w =
      best[0] < qualityWeights.length ? qualityWeights[best[0]]! : 1.0;
    const conf = Math.min(1.0, 0.7 * w);
    return { value: best[2], confidence: conf, sourceIndex: best[0] };
  }

  return { value: longest[0]![2], confidence: 0.7, sourceIndex: longest[0]![0] };
}

function _majorityVote(
  nonNull: [number, unknown][],
  qualityWeights?: readonly number[],
): MergeFieldResult {
  if (qualityWeights) {
    // Weighted vote: sum quality weights per value
    const valueWeights = new Map<unknown, number>();
    const valueIdx = new Map<unknown, number>();

    for (const [i, v] of nonNull) {
      const w = i < qualityWeights.length ? qualityWeights[i]! : 1.0;
      valueWeights.set(v, (valueWeights.get(v) ?? 0) + w);
      if (!valueIdx.has(v)) valueIdx.set(v, i);
    }

    let winner: unknown = null;
    let bestWeight = -Infinity;
    for (const [v, w] of valueWeights) {
      if (w > bestWeight) {
        bestWeight = w;
        winner = v;
      }
    }

    let totalWeight = 0;
    for (const w of valueWeights.values()) totalWeight += w;
    const conf = totalWeight > 0 ? bestWeight / totalWeight : 0.0;
    return { value: winner, confidence: conf, sourceIndex: valueIdx.get(winner)! };
  }

  // Unweighted: count occurrences
  const counts = new Map<unknown, number>();
  for (const [, v] of nonNull) {
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }

  let winner: unknown = null;
  let bestCount = -1;
  for (const [v, c] of counts) {
    if (c > bestCount) {
      bestCount = c;
      winner = v;
    }
  }

  const winnerIdx = nonNull.find(([, v]) => v === winner)![0];
  return {
    value: winner,
    confidence: bestCount / nonNull.length,
    sourceIndex: winnerIdx,
  };
}

function _sourcePriority(
  values: readonly unknown[],
  rule: GoldenFieldRule,
  sources?: readonly string[],
): MergeFieldResult {
  if (!sources) {
    throw new Error("source_priority strategy requires sources list");
  }

  const sourceVal = new Map<string, unknown>();
  const sourceIdx = new Map<string, number>();

  for (let i = 0; i < sources.length; i++) {
    const src = sources[i]!;
    if (!sourceVal.has(src)) {
      sourceVal.set(src, values[i]);
      sourceIdx.set(src, i);
    }
  }

  const priority = rule.sourcePriority ?? [];
  for (let idx = 0; idx < priority.length; idx++) {
    const src = priority[idx]!;
    const val = sourceVal.get(src);
    if (val != null) {
      const conf = Math.max(0.1, 1.0 - idx * 0.1);
      return { value: val, confidence: conf, sourceIndex: sourceIdx.get(src)! };
    }
  }

  // Fallback: no match in priority list
  return { value: null, confidence: 0.0, sourceIndex: null };
}

function _mostRecent(
  values: readonly unknown[],
  dates?: readonly unknown[],
): MergeFieldResult {
  if (!dates) {
    throw new Error("most_recent strategy requires dates list");
  }

  const indexed: [number, unknown, unknown][] = [];
  for (let i = 0; i < values.length; i++) {
    if (values[i] != null && dates[i] != null) {
      indexed.push([i, dates[i], values[i]]);
    }
  }

  if (indexed.length === 0) {
    return { value: null, confidence: 0.0, sourceIndex: null };
  }

  // Sort by date descending (works for ISO strings and numbers)
  indexed.sort((a, b) => {
    if (a[1]! > b[1]!) return -1;
    if (a[1]! < b[1]!) return 1;
    return 0;
  });

  const topDate = indexed[0]![1];
  const tied = indexed.filter(([, d]) => d === topDate);
  const conf = tied.length === 1 ? 1.0 : 0.5;

  return { value: indexed[0]![2], confidence: conf, sourceIndex: indexed[0]![0] };
}

function _firstNonNull(
  nonNull: [number, unknown][],
  qualityWeights?: readonly number[],
): MergeFieldResult {
  if (qualityWeights) {
    // Pick the non-null value with the highest quality weight
    let bestIdx = nonNull[0]![0];
    let bestVal = nonNull[0]![1];
    let bestWeight =
      nonNull[0]![0] < qualityWeights.length
        ? qualityWeights[nonNull[0]![0]]!
        : 1.0;

    for (let i = 1; i < nonNull.length; i++) {
      const [idx, val] = nonNull[i]!;
      const w = idx < qualityWeights.length ? qualityWeights[idx]! : 1.0;
      if (w > bestWeight) {
        bestWeight = w;
        bestIdx = idx;
        bestVal = val;
      }
    }

    return { value: bestVal, confidence: 0.6, sourceIndex: bestIdx };
  }

  return { value: nonNull[0]![1], confidence: 0.6, sourceIndex: nonNull[0]![0] };
}

// ---------------------------------------------------------------------------
// GoldenRecord
// ---------------------------------------------------------------------------

export interface GoldenRecord {
  readonly fields: Readonly<Record<string, { value: unknown; confidence: number }>>;
  readonly goldenConfidence: number;
}

// ---------------------------------------------------------------------------
// buildGoldenRecord
// ---------------------------------------------------------------------------

/**
 * Build a golden record from cluster rows.
 *
 * @param clusterRows - Array of row objects belonging to one cluster.
 * @param rules - Golden rules config with default strategy and field rules.
 * @param qualityScores - Optional map of `"rowId:column"` -> quality score.
 */
export function buildGoldenRecord(
  clusterRows: readonly Row[],
  rules: GoldenRulesConfig,
  qualityScores?: ReadonlyMap<string, number>,
): GoldenRecord {
  if (clusterRows.length === 0) {
    return { fields: {}, goldenConfidence: 0.0 };
  }

  // Collect all column names
  const columns = new Set<string>();
  for (const row of clusterRows) {
    for (const col of Object.keys(row)) {
      columns.add(col);
    }
  }

  const rowIds: number[] = clusterRows.map(
    (r) => (r.__row_id__ as number) ?? 0,
  );

  const fields: Record<string, { value: unknown; confidence: number }> = {};
  const confidences: number[] = [];

  for (const col of columns) {
    if (isInternal(col)) continue;

    const values = clusterRows.map((r) => r[col] ?? null);

    // Look up field rule or use default
    const fieldRule: GoldenFieldRule =
      rules.fieldRules[col] ?? { strategy: rules.defaultStrategy as GoldenFieldRule["strategy"] };

    // Gather optional lists
    let sources: string[] | undefined;
    let dates: unknown[] | undefined;
    let weights: number[] | undefined;

    if (fieldRule.strategy === "source_priority") {
      sources = clusterRows.map((r) => String(r.__source__ ?? ""));
    }
    if (fieldRule.strategy === "most_recent" && fieldRule.dateColumn) {
      dates = clusterRows.map((r) => r[fieldRule.dateColumn!] ?? null);
    }
    if (qualityScores) {
      weights = rowIds.map((rid) => qualityScores.get(`${rid}:${col}`) ?? 1.0);
    }

    const mergeOpts: MergeFieldOptions = {
      ...(sources !== undefined && { sources }),
      ...(dates !== undefined && { dates }),
      ...(weights !== undefined && { qualityWeights: weights }),
    };
    const result = mergeField(values, fieldRule, mergeOpts);
    fields[col] = { value: result.value, confidence: result.confidence };
    confidences.push(result.confidence);
  }

  const goldenConfidence =
    confidences.length > 0
      ? confidences.reduce((a, b) => a + b, 0) / confidences.length
      : 0.0;

  return { fields, goldenConfidence };
}

// ---------------------------------------------------------------------------
// buildGoldenRecordWithProvenance
// ---------------------------------------------------------------------------

export interface GoldenRecordWithProvenanceResult {
  readonly goldenRecords: readonly (Row & {
    __cluster_id__: number;
    __golden_confidence__: number;
  })[];
  readonly provenance: readonly ClusterProvenance[];
}

/**
 * Build golden records with full field-level provenance tracking.
 *
 * @param allRows - All rows with `__cluster_id__` and `__row_id__` columns.
 * @param rules - Golden rules config.
 * @param clusters - Cluster map from buildClusters.
 * @param qualityScores - Optional `"rowId:column"` -> quality score map.
 */
export function buildGoldenRecordWithProvenance(
  allRows: readonly Row[],
  rules: GoldenRulesConfig,
  clusters: ReadonlyMap<number, ClusterInfo>,
  qualityScores?: ReadonlyMap<string, number>,
): GoldenRecordWithProvenanceResult {
  // Group rows by cluster ID
  const clusterDfs = new Map<number, Row[]>();
  for (const row of allRows) {
    const cid = (row.__cluster_id__ as number) ?? 1;
    let arr = clusterDfs.get(cid);
    if (!arr) {
      arr = [];
      clusterDfs.set(cid, arr);
    }
    arr.push(row);
  }

  const clusterIds = [...clusterDfs.keys()].sort((a, b) => a - b);
  const goldenRecords: (Row & {
    __cluster_id__: number;
    __golden_confidence__: number;
  })[] = [];
  const provenanceList: ClusterProvenance[] = [];

  for (const cid of clusterIds) {
    const clusterRows = clusterDfs.get(cid)!;
    const cinfo = clusters.get(cid);
    const rowIds = clusterRows.map(
      (r) => (r.__row_id__ as number) ?? 0,
    );

    // Collect columns
    const columns = new Set<string>();
    for (const row of clusterRows) {
      for (const col of Object.keys(row)) {
        columns.add(col);
      }
    }

    const fieldProvenance: Record<string, FieldProvenance> = {};
    const goldenRow: Record<string, unknown> = { __cluster_id__: cid };
    const confidences: number[] = [];

    for (const col of columns) {
      if (isInternal(col)) continue;

      const values = clusterRows.map((r) => r[col] ?? null);

      const fieldRule: GoldenFieldRule =
        rules.fieldRules[col] ?? { strategy: rules.defaultStrategy as GoldenFieldRule["strategy"] };

      let sources: string[] | undefined;
      let dates: unknown[] | undefined;
      let weights: number[] | undefined;

      if (fieldRule.strategy === "source_priority") {
        sources = clusterRows.map((r) => String(r.__source__ ?? ""));
      }
      if (fieldRule.strategy === "most_recent" && fieldRule.dateColumn) {
        dates = clusterRows.map((r) => r[fieldRule.dateColumn!] ?? null);
      }
      if (qualityScores) {
        weights = rowIds.map(
          (rid) => qualityScores.get(`${rid}:${col}`) ?? 1.0,
        );
      }

      const mergeOpts: MergeFieldOptions = {
        ...(sources !== undefined && { sources }),
        ...(dates !== undefined && { dates }),
        ...(weights !== undefined && { qualityWeights: weights }),
      };
      const result = mergeField(values, fieldRule, mergeOpts);
      confidences.push(result.confidence);

      const sourceRowId =
        result.sourceIndex != null && result.sourceIndex < rowIds.length
          ? rowIds[result.sourceIndex]!
          : rowIds[0]!;

      const candidates = rowIds.map((rid, idx) => {
        const q = qualityScores
          ? (qualityScores.get(`${rid}:${col}`) ?? 1.0)
          : 1.0;
        return { row_id: rid, value: values[idx], quality: q };
      });

      fieldProvenance[col] = {
        value: result.value,
        sourceRowId,
        strategy: fieldRule.strategy,
        confidence: result.confidence,
        candidates,
      };

      goldenRow[col] = result.value;
    }

    const goldenConfidence =
      confidences.length > 0
        ? confidences.reduce((a, b) => a + b, 0) / confidences.length
        : 0.0;

    goldenRow.__golden_confidence__ = goldenConfidence;

    goldenRecords.push(
      goldenRow as Row & {
        __cluster_id__: number;
        __golden_confidence__: number;
      },
    );

    provenanceList.push({
      clusterId: cid,
      clusterQuality: cinfo?.clusterQuality ?? "strong",
      clusterConfidence: cinfo?.confidence ?? 0.0,
      fields: fieldProvenance,
    });
  }

  return { goldenRecords, provenance: provenanceList };
}
