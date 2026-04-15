/**
 * types.ts — GoldenMatch config interfaces and result types.
 * Edge-safe: no Node.js imports, no `process`.
 */

import type {
  PreflightReport,
  PostflightReport,
} from "./autoconfigVerify.js";

// ---------------------------------------------------------------------------
// Primitive types
// ---------------------------------------------------------------------------

export type ColumnValue = string | number | boolean | null;
export type Row = Readonly<Record<string, unknown>>;

/** A canonical pair key in the form "minId:maxId". Only produced by pairKey(). */
export type PairKey = string & { readonly __brand: "PairKey" };

// ---------------------------------------------------------------------------
// Matchkey field config
// ---------------------------------------------------------------------------

export interface MatchkeyField {
  readonly field: string;
  readonly transforms: readonly string[];
  readonly scorer: string;
  readonly weight: number;
  readonly model?: string;
  readonly columns?: readonly string[];
  readonly columnWeights?: Readonly<Record<string, number>>;
  readonly levels?: number;
  readonly partialThreshold?: number;
}

export interface ExactMatchkey {
  readonly name: string;
  readonly type: "exact";
  readonly fields: readonly MatchkeyField[];
}

export interface WeightedMatchkey {
  readonly name: string;
  readonly type: "weighted";
  readonly fields: readonly MatchkeyField[];
  readonly threshold: number;
  readonly autoThreshold?: boolean;
  readonly rerank?: boolean;
  readonly rerankModel?: string;
  readonly rerankBand?: number;
}

export interface ProbabilisticMatchkey {
  readonly name: string;
  readonly type: "probabilistic";
  readonly fields: readonly MatchkeyField[];
  readonly threshold?: number;
  readonly emIterations?: number;
  readonly convergenceThreshold?: number;
  readonly linkThreshold?: number;
  readonly reviewThreshold?: number;
}

export type MatchkeyConfig =
  | ExactMatchkey
  | WeightedMatchkey
  | ProbabilisticMatchkey;

// ---------------------------------------------------------------------------
// Blocking config
// ---------------------------------------------------------------------------

export interface BlockingKeyConfig {
  readonly fields: readonly string[];
  readonly transforms: readonly string[];
}

export interface SortKeyField {
  readonly column: string;
  readonly transforms: readonly string[];
}

export interface CanopyConfig {
  readonly fields: readonly string[];
  readonly looseThreshold: number;
  readonly tightThreshold: number;
  readonly maxCanopySize: number;
}

export interface BlockingConfig {
  readonly strategy:
    | "static"
    | "adaptive"
    | "sorted_neighborhood"
    | "multi_pass"
    | "ann"
    | "canopy"
    | "ann_pairs"
    | "learned";
  readonly keys: readonly BlockingKeyConfig[];
  readonly maxBlockSize: number;
  readonly skipOversized: boolean;
  readonly autoSuggest?: boolean;
  readonly autoSelect?: boolean;
  readonly subBlockKeys?: readonly BlockingKeyConfig[];
  readonly windowSize?: number;
  readonly sortKey?: readonly SortKeyField[];
  readonly passes?: readonly BlockingKeyConfig[];
  readonly unionMode?: boolean;
  readonly maxTotalComparisons?: number;
  readonly annColumn?: string;
  readonly annModel?: string;
  readonly annTopK?: number;
  readonly canopy?: CanopyConfig;
  readonly learnedSampleSize?: number;
  readonly learnedMinRecall?: number;
  readonly learnedMinReduction?: number;
  readonly learnedPredicateDepth?: number;
  readonly learnedCachePath?: string;
}

// ---------------------------------------------------------------------------
// Golden rules config
// ---------------------------------------------------------------------------

export interface GoldenFieldRule {
  readonly strategy:
    | "most_complete"
    | "majority_vote"
    | "source_priority"
    | "most_recent"
    | "first_non_null";
  readonly dateColumn?: string;
  readonly sourcePriority?: readonly string[];
}

export interface GoldenRulesConfig {
  readonly defaultStrategy: string;
  readonly fieldRules: Readonly<Record<string, GoldenFieldRule>>;
  readonly maxClusterSize: number;
  readonly autoSplit: boolean;
  readonly qualityWeighting: boolean;
  readonly weakClusterThreshold: number;
}

// ---------------------------------------------------------------------------
// Standardization, validation, quality, transform
// ---------------------------------------------------------------------------

export interface StandardizationConfig {
  readonly rules: Readonly<Record<string, readonly string[]>>;
}

export interface ValidationRuleConfig {
  readonly column: string;
  readonly ruleType:
    | "regex"
    | "min_length"
    | "max_length"
    | "not_null"
    | "in_set"
    | "format";
  readonly params: Readonly<Record<string, unknown>>;
  readonly action: "null" | "quarantine" | "flag";
}

export interface ValidationConfig {
  readonly rules: readonly ValidationRuleConfig[];
  readonly autoFix: boolean;
}

export interface QualityConfig {
  readonly enabled: boolean;
  readonly mode: "silent" | "announced" | "disabled";
  readonly fixMode: "safe" | "moderate" | "none";
  readonly domain?: string;
}

export interface TransformConfig {
  readonly enabled: boolean;
  readonly mode: "silent" | "announced" | "disabled";
}

// ---------------------------------------------------------------------------
// LLM scorer & budget
// ---------------------------------------------------------------------------

export interface BudgetConfig {
  readonly maxCostUsd?: number;
  readonly maxCalls?: number;
  readonly escalationModel?: string;
  readonly escalationBand?: readonly number[];
  readonly escalationBudgetPct?: number;
  readonly warnAtPct?: number;
}

export interface LLMScorerConfig {
  readonly enabled: boolean;
  readonly provider?: string;
  readonly model?: string;
  readonly autoThreshold: number;
  readonly candidateLo: number;
  readonly candidateHi: number;
  readonly batchSize: number;
  readonly maxWorkers: number;
  readonly budget?: BudgetConfig;
  readonly mode: "pairwise" | "cluster";
  readonly clusterMaxSize?: number;
  readonly clusterMinSize?: number;
}

// ---------------------------------------------------------------------------
// Domain config
// ---------------------------------------------------------------------------

export interface DomainConfig {
  readonly enabled: boolean;
  readonly mode?: string;
  readonly confidenceThreshold: number;
  readonly llmValidation: boolean;
  readonly budget?: BudgetConfig;
}

// ---------------------------------------------------------------------------
// Memory & learning
// ---------------------------------------------------------------------------

export interface LearningConfig {
  readonly thresholdMinCorrections: number;
  readonly weightsMinCorrections: number;
}

export interface MemoryConfig {
  readonly enabled: boolean;
  readonly backend: "sqlite" | "postgres";
  readonly path?: string;
  readonly trust: number;
  readonly learning: LearningConfig;
}

// ---------------------------------------------------------------------------
// Input & output config
// ---------------------------------------------------------------------------

export interface InputFileConfig {
  readonly path: string;
  readonly idColumn?: string;
  readonly sourceLabel?: string;
  readonly sourceName?: string;
  readonly columnMap?: Readonly<Record<string, string>>;
  readonly delimiter?: string;
  readonly encoding?: string;
  readonly sheet?: string;
  readonly parseMode?: string;
  readonly headerRow?: number;
  readonly hasHeader?: boolean;
  readonly skipRows?: readonly number[];
}

export interface InputConfig {
  readonly files: readonly InputFileConfig[];
  readonly fileA?: InputFileConfig;
  readonly fileB?: InputFileConfig;
}

export interface OutputConfig {
  readonly path?: string;
  readonly format?: string;
  readonly directory?: string;
  readonly runName?: string;
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

export interface GoldenMatchConfig {
  readonly matchkeys?: readonly MatchkeyConfig[];
  readonly matchSettings?: readonly MatchkeyConfig[];
  readonly blocking?: BlockingConfig;
  readonly threshold?: number;
  readonly goldenRules?: GoldenRulesConfig;
  readonly standardization?: StandardizationConfig;
  readonly validation?: ValidationConfig;
  readonly quality?: QualityConfig;
  readonly transform?: TransformConfig;
  readonly llmScorer?: LLMScorerConfig;
  readonly domain?: DomainConfig;
  readonly memory?: MemoryConfig;
  readonly input?: InputConfig;
  readonly output?: OutputConfig;
  readonly backend?: string;
  readonly llmAuto?: boolean;
  readonly llmBoost?: boolean;

  /** Internal: auto-config hand-off. Do not read from outside the library.
   *  Non-readonly so preflight / postflight can populate. Stripped by
   *  stripConventionPrivate before YAML/JSON export.
   *
   *  This list is CLOSED. Future internal state should use a side-table
   *  pattern (WeakMap) instead — see spec §11 risks. Adding more underscore
   *  fields here weakens the readonly contract for every consumer. */
  _preflightReport?: PreflightReport;
  _strictAutoconfig?: boolean;
  _domainProfile?: import("./domain.js").DomainProfile;
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

export interface ScoredPair {
  readonly idA: number;
  readonly idB: number;
  readonly score: number;
}

export interface ClusterInfo {
  readonly members: readonly number[];
  readonly size: number;
  readonly oversized: boolean;
  readonly pairScores: ReadonlyMap<PairKey, number>;
  readonly confidence: number;
  readonly bottleneckPair: readonly [number, number] | null;
  readonly clusterQuality: "strong" | "weak" | "split";
}

export interface DedupeStats {
  readonly totalRecords: number;
  readonly totalClusters: number;
  readonly matchRate: number;
  readonly matchedRecords: number;
  readonly uniqueRecords: number;
}

export interface DedupeResult {
  readonly goldenRecords: readonly Row[];
  readonly clusters: ReadonlyMap<number, ClusterInfo>;
  readonly dupes: readonly Row[];
  readonly unique: readonly Row[];
  readonly stats: DedupeStats;
  readonly scoredPairs: readonly ScoredPair[];
  readonly config: GoldenMatchConfig;
  readonly postflightReport?: PostflightReport;
}

export interface MatchResult {
  readonly matched: readonly Row[];
  readonly unmatched: readonly Row[];
  readonly stats: Readonly<Record<string, unknown>>;
  readonly postflightReport?: PostflightReport;
}

export interface FieldProvenance {
  readonly value: unknown;
  readonly sourceRowId: number;
  readonly strategy: string;
  readonly confidence: number;
  readonly candidates: readonly Readonly<Record<string, unknown>>[];
}

export interface ClusterProvenance {
  readonly clusterId: number;
  readonly clusterQuality: string;
  readonly clusterConfidence: number;
  readonly fields: Readonly<Record<string, FieldProvenance>>;
}

export interface BlockResult {
  readonly blockKey: string;
  readonly rows: readonly Row[];
  readonly strategy: string;
  readonly depth: number;
  readonly parentKey?: string;
  readonly preScoredPairs?: readonly ScoredPair[];
}

// ---------------------------------------------------------------------------
// Valid enum sets
// ---------------------------------------------------------------------------

export const VALID_SCORERS = new Set([
  "exact",
  "jaro_winkler",
  "levenshtein",
  "token_sort",
  "soundex_match",
  "embedding",
  "record_embedding",
  "ensemble",
  "dice",
  "jaccard",
] as const);

export const VALID_TRANSFORMS = new Set([
  "lowercase",
  "uppercase",
  "strip",
  "strip_all",
  "soundex",
  "metaphone",
  "digits_only",
  "alpha_only",
  "normalize_whitespace",
  "token_sort",
  "first_token",
  "last_token",
] as const);

export const VALID_STRATEGIES = new Set([
  "most_recent",
  "source_priority",
  "most_complete",
  "majority_vote",
  "first_non_null",
] as const);

export const VALID_STANDARDIZERS = new Set([
  "email",
  "name_proper",
  "name_upper",
  "name_lower",
  "phone",
  "zip5",
  "address",
  "state",
  "strip",
  "trim_whitespace",
] as const);

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/**
 * Create a ScoredPair guaranteeing idA <= idB (canonical order).
 * Always use this instead of constructing `{ idA, idB, score }` directly.
 */
export function makeScoredPair(
  a: number,
  b: number,
  score: number,
): ScoredPair {
  const lo = a < b ? a : b;
  const hi = a < b ? b : a;
  return { idA: lo, idB: hi, score };
}

/** Create a MatchkeyField with sensible defaults. */
export function makeMatchkeyField(
  partial: Partial<MatchkeyField> & Pick<MatchkeyField, "field">,
): MatchkeyField {
  return {
    transforms: [],
    scorer: "jaro_winkler",
    weight: 1.0,
    ...partial,
  };
}

/**
 * Shape accepted by `makeMatchkeyConfig`. All variant-specific fields are
 * optional; the factory picks the right variant based on `type`.
 */
export interface MakeMatchkeyConfigInput {
  readonly name: string;
  readonly type?: "exact" | "weighted" | "probabilistic";
  readonly fields?: readonly MatchkeyField[];
  readonly threshold?: number;
  readonly autoThreshold?: boolean;
  readonly rerank?: boolean;
  readonly rerankModel?: string;
  readonly rerankBand?: number;
  readonly emIterations?: number;
  readonly convergenceThreshold?: number;
  readonly linkThreshold?: number;
  readonly reviewThreshold?: number;
}

/** Create a MatchkeyConfig with sensible defaults. Produces the correct variant. */
export function makeMatchkeyConfig(
  partial: MakeMatchkeyConfigInput,
): MatchkeyConfig {
  const type = partial.type ?? "weighted";
  const fields = partial.fields ?? [];
  if (type === "exact") {
    return { name: partial.name, type: "exact", fields };
  }
  if (type === "probabilistic") {
    const out: ProbabilisticMatchkey = {
      name: partial.name,
      type: "probabilistic",
      fields,
      ...(partial.threshold !== undefined
        ? { threshold: partial.threshold }
        : {}),
      ...(partial.emIterations !== undefined
        ? { emIterations: partial.emIterations }
        : {}),
      ...(partial.convergenceThreshold !== undefined
        ? { convergenceThreshold: partial.convergenceThreshold }
        : {}),
      ...(partial.linkThreshold !== undefined
        ? { linkThreshold: partial.linkThreshold }
        : {}),
      ...(partial.reviewThreshold !== undefined
        ? { reviewThreshold: partial.reviewThreshold }
        : {}),
    };
    return out;
  }
  // weighted (default)
  const out: WeightedMatchkey = {
    name: partial.name,
    type: "weighted",
    fields,
    threshold: partial.threshold ?? 0.85,
    ...(partial.autoThreshold !== undefined
      ? { autoThreshold: partial.autoThreshold }
      : {}),
    ...(partial.rerank !== undefined ? { rerank: partial.rerank } : {}),
    ...(partial.rerankModel !== undefined
      ? { rerankModel: partial.rerankModel }
      : {}),
    ...(partial.rerankBand !== undefined
      ? { rerankBand: partial.rerankBand }
      : {}),
  };
  return out;
}

/** Create a BlockingConfig with sensible defaults. */
export function makeBlockingConfig(
  partial?: Partial<BlockingConfig>,
): BlockingConfig {
  return {
    strategy: "static",
    keys: [],
    maxBlockSize: 5000,
    skipOversized: false,
    ...partial,
  };
}

/** Create a GoldenRulesConfig with sensible defaults. */
export function makeGoldenRulesConfig(
  partial?: Partial<GoldenRulesConfig>,
): GoldenRulesConfig {
  return {
    defaultStrategy: "most_complete",
    fieldRules: {},
    maxClusterSize: 10,
    autoSplit: true,
    qualityWeighting: true,
    weakClusterThreshold: 0.3,
    ...partial,
  };
}

/** Create a full GoldenMatchConfig with sensible defaults. */
export function makeConfig(
  partial?: Partial<GoldenMatchConfig>,
): GoldenMatchConfig {
  return {
    threshold: 0.85,
    blocking: makeBlockingConfig(partial?.blocking),
    goldenRules: makeGoldenRulesConfig(partial?.goldenRules),
    ...partial,
    // Re-apply blocking/goldenRules after spread so partial overrides win
    ...(partial?.blocking !== undefined
      ? { blocking: makeBlockingConfig(partial.blocking) }
      : {}),
    ...(partial?.goldenRules !== undefined
      ? { goldenRules: makeGoldenRulesConfig(partial.goldenRules) }
      : {}),
  };
}

/**
 * Return matchkeys from config, checking both `matchkeys` and `matchSettings`.
 * Mirrors Python's `GoldenMatchConfig.get_matchkeys()`.
 */
export function getMatchkeys(
  config: GoldenMatchConfig,
): readonly MatchkeyConfig[] {
  return config.matchkeys ?? config.matchSettings ?? [];
}
