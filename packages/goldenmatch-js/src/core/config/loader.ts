/**
 * config/loader.ts — Config loader that parses raw objects (from YAML/JSON)
 * into typed GoldenMatchConfig.
 *
 * Edge-safe: no `node:` imports, no `require()`.
 */

import type {
  GoldenMatchConfig,
  MatchkeyConfig,
  MatchkeyField,
  BlockingConfig,
  BlockingKeyConfig,
  GoldenRulesConfig,
  GoldenFieldRule,
  StandardizationConfig,
  LLMScorerConfig,
  BudgetConfig,
  ValidationConfig,
  ValidationRuleConfig,
  DomainConfig,
  QualityConfig,
  TransformConfig,
  MemoryConfig,
  LearningConfig,
  InputConfig,
  InputFileConfig,
  OutputConfig,
  SortKeyField,
  CanopyConfig,
} from "../types.js";
import {
  VALID_SCORERS,
  VALID_TRANSFORMS,
  VALID_STRATEGIES,
  VALID_STANDARDIZERS,
} from "../types.js";

// ---------------------------------------------------------------------------
// String-union validation
// ---------------------------------------------------------------------------

const VALID_MATCHKEY_TYPES = new Set([
  "exact",
  "weighted",
  "probabilistic",
] as const);

const VALID_BLOCKING_STRATEGIES = new Set([
  "static",
  "adaptive",
  "sorted_neighborhood",
  "multi_pass",
  "ann",
  "canopy",
  "ann_pairs",
  "learned",
] as const);

const VALID_MEMORY_BACKENDS = new Set(["sqlite", "postgres"] as const);

const VALID_QUALITY_MODES = new Set([
  "silent",
  "announced",
  "disabled",
] as const);

const VALID_QUALITY_FIX_MODES = new Set(["safe", "moderate", "none"] as const);

const VALID_LLM_MODES = new Set(["pairwise", "cluster"] as const);

const VALID_VALIDATION_RULE_TYPES = new Set([
  "regex",
  "min_length",
  "max_length",
  "not_null",
  "in_set",
  "format",
] as const);

const VALID_VALIDATION_ACTIONS = new Set([
  "null",
  "quarantine",
  "flag",
] as const);

/**
 * Validate that `value` is one of `allowed`. If `defaultValue` is provided,
 * return it when `value` is null/undefined. Throws a clear error otherwise.
 */
function requireIn<T extends string>(
  value: unknown,
  allowed: ReadonlySet<T>,
  fieldName: string,
  defaultValue?: T,
): T {
  if (value === undefined || value === null) {
    if (defaultValue !== undefined) return defaultValue;
    throw new Error(`Required field '${fieldName}' is missing`);
  }
  if (typeof value !== "string" || !(allowed as ReadonlySet<string>).has(value)) {
    const valid = [...allowed].sort().join(", ");
    throw new Error(
      `Invalid value '${String(value)}' for '${fieldName}'. Valid options: ${valid}`,
    );
  }
  return value as T;
}

/**
 * Accept known transforms plus parametric forms:
 *   - substring:<n>:<n>
 *   - qgram:<n>
 *   - bloom_filter, bloom_filter:<...>
 */
function isValidTransform(t: string): boolean {
  if ((VALID_TRANSFORMS as ReadonlySet<string>).has(t)) return true;
  if (/^substring:\d+:\d+$/.test(t)) return true;
  if (/^qgram:\d+$/.test(t)) return true;
  if (t === "bloom_filter" || /^bloom_filter:/.test(t)) return true;
  return false;
}

// ---------------------------------------------------------------------------
// Snake_case to camelCase conversion
// ---------------------------------------------------------------------------

/** Convert a snake_case key to camelCase. */
function snakeToCamel(s: string): string {
  return s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}

/** Recursively convert all keys of a plain object from snake_case to camelCase. */
function camelizeKeys(obj: unknown): unknown {
  if (obj === null || obj === undefined) return obj;
  if (Array.isArray(obj)) return obj.map(camelizeKeys);
  if (typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(obj as Record<string, unknown>)) {
      result[snakeToCamel(key)] = camelizeKeys(val);
    }
    return result;
  }
  return obj;
}

/** Recursively convert all keys from camelCase to snake_case. */
function camelToSnake(s: string): string {
  return s.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

function snakeifyKeys(obj: unknown): unknown {
  if (obj === null || obj === undefined) return obj;
  if (Array.isArray(obj)) return obj.map(snakeifyKeys);
  if (typeof obj === "object") {
    const result: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(obj as Record<string, unknown>)) {
      result[camelToSnake(key)] = snakeifyKeys(val);
    }
    return result;
  }
  return obj;
}

// ---------------------------------------------------------------------------
// Helpers: strip undefined values for exactOptionalPropertyTypes
// ---------------------------------------------------------------------------

/**
 * Remove keys whose value is `undefined` from a plain object.
 * Required because TypeScript's `exactOptionalPropertyTypes` disallows
 * assigning `undefined` to optional properties.
 */
function stripUndefined<T extends Record<string, unknown>>(obj: T): T {
  const result = {} as Record<string, unknown>;
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined) result[k] = v;
  }
  return result as T;
}

// ---------------------------------------------------------------------------
// Helpers: safe getters
// ---------------------------------------------------------------------------

type RawObj = Record<string, unknown>;

function asObj(v: unknown, ctx: string): RawObj {
  if (typeof v !== "object" || v === null || Array.isArray(v)) {
    throw new Error(`${ctx}: expected object, got ${typeof v}`);
  }
  return v as RawObj;
}

function asArr(v: unknown, ctx: string): unknown[] {
  if (!Array.isArray(v)) {
    throw new Error(`${ctx}: expected array, got ${typeof v}`);
  }
  return v;
}

function asStr(v: unknown, ctx: string): string {
  if (typeof v !== "string") {
    throw new Error(`${ctx}: expected string, got ${typeof v}`);
  }
  return v;
}

function asNum(v: unknown, ctx: string): number {
  if (typeof v !== "number") {
    throw new Error(`${ctx}: expected number, got ${typeof v}`);
  }
  return v;
}

function asBool(v: unknown, ctx: string): boolean {
  if (typeof v !== "boolean") {
    throw new Error(`${ctx}: expected boolean, got ${typeof v}`);
  }
  return v;
}

function optStr(v: unknown): string | undefined {
  return typeof v === "string" ? v : undefined;
}

function optNum(v: unknown): number | undefined {
  return typeof v === "number" ? v : undefined;
}

function optBool(v: unknown): boolean | undefined {
  return typeof v === "boolean" ? v : undefined;
}

// ---------------------------------------------------------------------------
// Parsers for nested config objects
// ---------------------------------------------------------------------------

function parseMatchkeyField(raw: unknown, ctx: string): MatchkeyField {
  const obj = asObj(raw, ctx);
  const fieldName = typeof obj.field === "string" ? obj.field : "<unknown>";

  // Validate transforms. Allow parametric forms like "substring:0:3", "qgram:3",
  // "bloom_filter:high".
  const transforms: string[] = Array.isArray(obj.transforms)
    ? (obj.transforms as unknown[]).map((t, i) => {
        if (typeof t !== "string") {
          throw new Error(
            `${ctx}.transforms[${i}]: expected string, got ${typeof t}`,
          );
        }
        return t;
      })
    : [];
  for (const t of transforms) {
    if (!isValidTransform(t)) {
      const valid = [...VALID_TRANSFORMS].sort().join(", ");
      throw new Error(
        `Invalid transform '${t}' on field '${fieldName}'. ` +
          `Valid: ${valid}, or 'substring:<n>:<n>', 'qgram:<n>', 'bloom_filter[:...]'.`,
      );
    }
  }

  // Scorer is optional for exact matchkeys. Allow plugin scorers — warn only
  // if the name is unknown (plugin registration may fill it in later).
  if (obj.scorer !== undefined && obj.scorer !== null) {
    if (
      typeof obj.scorer !== "string" ||
      !(VALID_SCORERS as ReadonlySet<string>).has(obj.scorer)
    ) {
      // eslint-disable-next-line no-console
      console.warn(
        `Unknown scorer '${String(obj.scorer)}' on field '${fieldName}' ` +
          `(will be rejected at score-time if no plugin is registered).`,
      );
    }
  }

  return stripUndefined({
    field: asStr(obj.field, `${ctx}.field`),
    transforms,
    scorer: typeof obj.scorer === "string" ? obj.scorer : "jaro_winkler",
    weight: typeof obj.weight === "number" ? obj.weight : 1.0,
    model: optStr(obj.model),
    columns: Array.isArray(obj.columns)
      ? (obj.columns as string[])
      : undefined,
    columnWeights:
      typeof obj.columnWeights === "object" && obj.columnWeights !== null
        ? (obj.columnWeights as Record<string, number>)
        : undefined,
    levels: optNum(obj.levels),
    partialThreshold: optNum(obj.partialThreshold),
  }) as MatchkeyField;
}

function parseMatchkeyConfig(raw: unknown, ctx: string): MatchkeyConfig {
  const obj = asObj(raw, ctx);
  const fields = Array.isArray(obj.fields)
    ? obj.fields.map((f: unknown, i: number) =>
        parseMatchkeyField(f, `${ctx}.fields[${i}]`),
      )
    : [];

  return stripUndefined({
    name: asStr(obj.name, `${ctx}.name`),
    type: requireIn(
      obj.type,
      VALID_MATCHKEY_TYPES,
      `${ctx}.type`,
      "weighted",
    ),
    fields,
    threshold: optNum(obj.threshold),
    autoThreshold: optBool(obj.autoThreshold),
    rerank: optBool(obj.rerank),
    rerankModel: optStr(obj.rerankModel),
    rerankBand: optNum(obj.rerankBand),
    emIterations: optNum(obj.emIterations),
    convergenceThreshold: optNum(obj.convergenceThreshold),
    linkThreshold: optNum(obj.linkThreshold),
    reviewThreshold: optNum(obj.reviewThreshold),
  }) as MatchkeyConfig;
}

function parseBlockingKeyConfig(
  raw: unknown,
  ctx: string,
): BlockingKeyConfig {
  const obj = asObj(raw, ctx);
  return {
    fields: Array.isArray(obj.fields) ? (obj.fields as string[]) : [],
    transforms: Array.isArray(obj.transforms)
      ? (obj.transforms as string[])
      : [],
  };
}

function parseSortKeyField(raw: unknown, ctx: string): SortKeyField {
  const obj = asObj(raw, ctx);
  return {
    column: asStr(obj.column, `${ctx}.column`),
    transforms: Array.isArray(obj.transforms)
      ? (obj.transforms as string[])
      : [],
  };
}

function parseCanopyConfig(raw: unknown, ctx: string): CanopyConfig {
  const obj = asObj(raw, ctx);
  return {
    fields: Array.isArray(obj.fields) ? (obj.fields as string[]) : [],
    looseThreshold: typeof obj.looseThreshold === "number" ? obj.looseThreshold : 0.7,
    tightThreshold: typeof obj.tightThreshold === "number" ? obj.tightThreshold : 0.9,
    maxCanopySize: typeof obj.maxCanopySize === "number" ? obj.maxCanopySize : 1000,
  };
}

function parseBlockingConfig(raw: unknown, ctx: string): BlockingConfig {
  const obj = asObj(raw, ctx);
  const keys = Array.isArray(obj.keys)
    ? obj.keys.map((k: unknown, i: number) =>
        parseBlockingKeyConfig(k, `${ctx}.keys[${i}]`),
      )
    : [];
  const passes = Array.isArray(obj.passes)
    ? obj.passes.map((p: unknown, i: number) =>
        parseBlockingKeyConfig(p, `${ctx}.passes[${i}]`),
      )
    : undefined;
  const subBlockKeys = Array.isArray(obj.subBlockKeys)
    ? obj.subBlockKeys.map((k: unknown, i: number) =>
        parseBlockingKeyConfig(k, `${ctx}.subBlockKeys[${i}]`),
      )
    : undefined;
  const sortKey = Array.isArray(obj.sortKey)
    ? obj.sortKey.map((s: unknown, i: number) =>
        parseSortKeyField(s, `${ctx}.sortKey[${i}]`),
      )
    : undefined;
  const canopy =
    typeof obj.canopy === "object" && obj.canopy !== null
      ? parseCanopyConfig(obj.canopy, `${ctx}.canopy`)
      : undefined;

  return stripUndefined({
    strategy: requireIn(
      obj.strategy,
      VALID_BLOCKING_STRATEGIES,
      `${ctx}.strategy`,
      "static",
    ),
    keys,
    maxBlockSize:
      typeof obj.maxBlockSize === "number" ? obj.maxBlockSize : 5000,
    skipOversized:
      typeof obj.skipOversized === "boolean" ? obj.skipOversized : false,
    autoSuggest: optBool(obj.autoSuggest),
    autoSelect: optBool(obj.autoSelect),
    subBlockKeys,
    windowSize: optNum(obj.windowSize),
    sortKey,
    passes,
    unionMode: optBool(obj.unionMode),
    maxTotalComparisons: optNum(obj.maxTotalComparisons),
    annColumn: optStr(obj.annColumn),
    annModel: optStr(obj.annModel),
    annTopK: optNum(obj.annTopK),
    canopy,
    learnedSampleSize: optNum(obj.learnedSampleSize),
    learnedMinRecall: optNum(obj.learnedMinRecall),
    learnedMinReduction: optNum(obj.learnedMinReduction),
    learnedPredicateDepth: optNum(obj.learnedPredicateDepth),
    learnedCachePath: optStr(obj.learnedCachePath),
  }) as BlockingConfig;
}

function parseGoldenFieldRule(raw: unknown, ctx: string): GoldenFieldRule {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    strategy: requireIn(
      obj.strategy,
      VALID_STRATEGIES,
      `${ctx}.strategy`,
    ) as GoldenFieldRule["strategy"],
    dateColumn: optStr(obj.dateColumn),
    sourcePriority: Array.isArray(obj.sourcePriority)
      ? (obj.sourcePriority as string[])
      : undefined,
  }) as GoldenFieldRule;
}

function parseGoldenRulesConfig(
  raw: unknown,
  ctx: string,
): GoldenRulesConfig {
  const obj = asObj(raw, ctx);

  // Normalize: YAML uses `default`, TS interface uses `defaultStrategy`
  const defaultStrategy =
    typeof obj.defaultStrategy === "string"
      ? obj.defaultStrategy
      : typeof obj.default === "string"
        ? obj.default
        : "most_complete";

  const fieldRules: Record<string, GoldenFieldRule> = {};
  if (
    typeof obj.fieldRules === "object" &&
    obj.fieldRules !== null &&
    !Array.isArray(obj.fieldRules)
  ) {
    for (const [key, val] of Object.entries(
      obj.fieldRules as Record<string, unknown>,
    )) {
      fieldRules[key] = parseGoldenFieldRule(val, `${ctx}.fieldRules.${key}`);
    }
  }

  return {
    defaultStrategy,
    fieldRules,
    maxClusterSize:
      typeof obj.maxClusterSize === "number" ? obj.maxClusterSize : 10,
    autoSplit:
      typeof obj.autoSplit === "boolean" ? obj.autoSplit : true,
    qualityWeighting:
      typeof obj.qualityWeighting === "boolean"
        ? obj.qualityWeighting
        : true,
    weakClusterThreshold:
      typeof obj.weakClusterThreshold === "number"
        ? obj.weakClusterThreshold
        : 0.3,
  };
}

function parseStandardizationConfig(
  raw: unknown,
  ctx: string,
): StandardizationConfig {
  const obj = asObj(raw, ctx);

  // Normalize: in YAML the rules may be at top level or nested under `rules`
  let rulesObj: Record<string, unknown>;
  if (
    typeof obj.rules === "object" &&
    obj.rules !== null &&
    !Array.isArray(obj.rules)
  ) {
    rulesObj = obj.rules as Record<string, unknown>;
  } else {
    // Flat form: each key is a column name mapping to standardizers
    rulesObj = obj;
  }

  const rules: Record<string, readonly string[]> = {};
  for (const [key, val] of Object.entries(rulesObj)) {
    if (Array.isArray(val)) {
      const arr = val as unknown[];
      for (const rule of arr) {
        if (typeof rule !== "string") {
          throw new Error(
            `${ctx}.${key}: expected array of strings, got ${typeof rule}`,
          );
        }
        if (!(VALID_STANDARDIZERS as ReadonlySet<string>).has(rule)) {
          const valid = [...VALID_STANDARDIZERS].sort().join(", ");
          throw new Error(
            `Invalid standardizer '${rule}' on column '${key}'. Valid: ${valid}`,
          );
        }
      }
      rules[key] = arr as string[];
    }
  }

  return { rules };
}

function parseBudgetConfig(raw: unknown, ctx: string): BudgetConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    maxCostUsd: optNum(obj.maxCostUsd),
    maxCalls: optNum(obj.maxCalls),
    escalationModel: optStr(obj.escalationModel),
    escalationBand: Array.isArray(obj.escalationBand)
      ? (obj.escalationBand as number[])
      : undefined,
    escalationBudgetPct: optNum(obj.escalationBudgetPct),
    warnAtPct: optNum(obj.warnAtPct),
  }) as BudgetConfig;
}

function parseLLMScorerConfig(
  raw: unknown,
  ctx: string,
): LLMScorerConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    enabled: typeof obj.enabled === "boolean" ? obj.enabled : false,
    provider: optStr(obj.provider),
    model: optStr(obj.model),
    autoThreshold:
      typeof obj.autoThreshold === "number" ? obj.autoThreshold : 0.9,
    candidateLo:
      typeof obj.candidateLo === "number" ? obj.candidateLo : 0.6,
    candidateHi:
      typeof obj.candidateHi === "number" ? obj.candidateHi : 0.9,
    batchSize:
      typeof obj.batchSize === "number" ? obj.batchSize : 10,
    maxWorkers:
      typeof obj.maxWorkers === "number" ? obj.maxWorkers : 4,
    budget:
      typeof obj.budget === "object" && obj.budget !== null
        ? parseBudgetConfig(obj.budget, `${ctx}.budget`)
        : undefined,
    mode: requireIn(obj.mode, VALID_LLM_MODES, `${ctx}.mode`, "pairwise"),
    clusterMaxSize: optNum(obj.clusterMaxSize),
    clusterMinSize: optNum(obj.clusterMinSize),
  }) as LLMScorerConfig;
}

function parseValidationRuleConfig(
  raw: unknown,
  ctx: string,
): ValidationRuleConfig {
  const obj = asObj(raw, ctx);
  return {
    column: asStr(obj.column, `${ctx}.column`),
    ruleType: requireIn(
      obj.ruleType,
      VALID_VALIDATION_RULE_TYPES,
      `${ctx}.ruleType`,
    ),
    params:
      typeof obj.params === "object" && obj.params !== null
        ? (obj.params as Record<string, unknown>)
        : {},
    action: requireIn(
      obj.action,
      VALID_VALIDATION_ACTIONS,
      `${ctx}.action`,
      "flag",
    ),
  };
}

function parseValidationConfig(
  raw: unknown,
  ctx: string,
): ValidationConfig {
  const obj = asObj(raw, ctx);
  return {
    rules: Array.isArray(obj.rules)
      ? obj.rules.map((r: unknown, i: number) =>
          parseValidationRuleConfig(r, `${ctx}.rules[${i}]`),
        )
      : [],
    autoFix: typeof obj.autoFix === "boolean" ? obj.autoFix : false,
  };
}

function parseDomainConfig(raw: unknown, ctx: string): DomainConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    enabled: typeof obj.enabled === "boolean" ? obj.enabled : false,
    mode: optStr(obj.mode),
    confidenceThreshold:
      typeof obj.confidenceThreshold === "number"
        ? obj.confidenceThreshold
        : 0.8,
    llmValidation:
      typeof obj.llmValidation === "boolean" ? obj.llmValidation : false,
    budget:
      typeof obj.budget === "object" && obj.budget !== null
        ? parseBudgetConfig(obj.budget, `${ctx}.budget`)
        : undefined,
  }) as DomainConfig;
}

function parseQualityConfig(raw: unknown, ctx: string): QualityConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    enabled: typeof obj.enabled === "boolean" ? obj.enabled : true,
    mode: requireIn(obj.mode, VALID_QUALITY_MODES, `${ctx}.mode`, "silent"),
    fixMode: requireIn(
      obj.fixMode,
      VALID_QUALITY_FIX_MODES,
      `${ctx}.fixMode`,
      "safe",
    ),
    domain: optStr(obj.domain),
  }) as QualityConfig;
}

function parseTransformConfig(raw: unknown, ctx: string): TransformConfig {
  const obj = asObj(raw, ctx);
  return {
    enabled: typeof obj.enabled === "boolean" ? obj.enabled : true,
    mode: requireIn(obj.mode, VALID_QUALITY_MODES, `${ctx}.mode`, "silent"),
  };
}

function parseLearningConfig(raw: unknown, ctx: string): LearningConfig {
  const obj = asObj(raw, ctx);
  return {
    thresholdMinCorrections:
      typeof obj.thresholdMinCorrections === "number"
        ? obj.thresholdMinCorrections
        : 10,
    weightsMinCorrections:
      typeof obj.weightsMinCorrections === "number"
        ? obj.weightsMinCorrections
        : 50,
  };
}

function parseMemoryConfig(raw: unknown, ctx: string): MemoryConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    enabled: typeof obj.enabled === "boolean" ? obj.enabled : false,
    backend: requireIn(
      obj.backend,
      VALID_MEMORY_BACKENDS,
      `${ctx}.backend`,
      "sqlite",
    ),
    path: optStr(obj.path),
    trust: typeof obj.trust === "number" ? obj.trust : 0.9,
    learning:
      typeof obj.learning === "object" && obj.learning !== null
        ? parseLearningConfig(obj.learning, `${ctx}.learning`)
        : { thresholdMinCorrections: 10, weightsMinCorrections: 50 },
  }) as MemoryConfig;
}

function parseInputFileConfig(
  raw: unknown,
  ctx: string,
): InputFileConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    path: asStr(obj.path, `${ctx}.path`),
    idColumn: optStr(obj.idColumn),
    sourceLabel: optStr(obj.sourceLabel),
    sourceName: optStr(obj.sourceName),
    columnMap:
      typeof obj.columnMap === "object" && obj.columnMap !== null
        ? (obj.columnMap as Record<string, string>)
        : undefined,
    delimiter: optStr(obj.delimiter),
    encoding: optStr(obj.encoding),
    sheet: optStr(obj.sheet),
    parseMode: optStr(obj.parseMode),
    headerRow: optNum(obj.headerRow),
    hasHeader: optBool(obj.hasHeader),
    skipRows: Array.isArray(obj.skipRows)
      ? (obj.skipRows as number[])
      : undefined,
  }) as InputFileConfig;
}

function parseInputConfig(raw: unknown, ctx: string): InputConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    files: Array.isArray(obj.files)
      ? obj.files.map((f: unknown, i: number) =>
          parseInputFileConfig(f, `${ctx}.files[${i}]`),
        )
      : [],
    fileA:
      typeof obj.fileA === "object" && obj.fileA !== null
        ? parseInputFileConfig(obj.fileA, `${ctx}.fileA`)
        : undefined,
    fileB:
      typeof obj.fileB === "object" && obj.fileB !== null
        ? parseInputFileConfig(obj.fileB, `${ctx}.fileB`)
        : undefined,
  }) as InputConfig;
}

function parseOutputConfig(raw: unknown, ctx: string): OutputConfig {
  const obj = asObj(raw, ctx);
  return stripUndefined({
    path: optStr(obj.path),
    format: optStr(obj.format),
    directory: optStr(obj.directory),
    runName: optStr(obj.runName),
  }) as OutputConfig;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Parse a raw JS object (already deserialized from YAML or JSON) into a
 * validated GoldenMatchConfig.
 *
 * Handles:
 * - Snake_case to camelCase key conversion
 * - Normalization of `matchkeys` / `match_settings`
 * - Parsing of all nested config objects
 * - `default` -> `defaultStrategy` normalization in golden_rules
 */
export function parseConfig(raw: unknown): GoldenMatchConfig {
  if (typeof raw !== "object" || raw === null) {
    throw new Error("Invalid config: expected a non-null object");
  }

  // Camelize all keys recursively
  const obj = camelizeKeys(raw) as RawObj;

  // Normalize matchkeys: accept either `matchkeys` or `matchSettings`
  const rawMatchkeys = obj.matchkeys ?? obj.matchSettings;
  const matchkeys = Array.isArray(rawMatchkeys)
    ? rawMatchkeys.map((mk: unknown, i: number) =>
        parseMatchkeyConfig(mk, `matchkeys[${i}]`),
      )
    : undefined;

  const config = stripUndefined({
    matchkeys,
    blocking:
      typeof obj.blocking === "object" && obj.blocking !== null
        ? parseBlockingConfig(obj.blocking, "blocking")
        : undefined,
    threshold: optNum(obj.threshold),
    goldenRules:
      typeof obj.goldenRules === "object" && obj.goldenRules !== null
        ? parseGoldenRulesConfig(obj.goldenRules, "goldenRules")
        : undefined,
    standardization:
      typeof obj.standardization === "object" && obj.standardization !== null
        ? parseStandardizationConfig(obj.standardization, "standardization")
        : undefined,
    validation:
      typeof obj.validation === "object" && obj.validation !== null
        ? parseValidationConfig(obj.validation, "validation")
        : undefined,
    quality:
      typeof obj.quality === "object" && obj.quality !== null
        ? parseQualityConfig(obj.quality, "quality")
        : undefined,
    transform:
      typeof obj.transform === "object" && obj.transform !== null
        ? parseTransformConfig(obj.transform, "transform")
        : undefined,
    llmScorer:
      typeof obj.llmScorer === "object" && obj.llmScorer !== null
        ? parseLLMScorerConfig(obj.llmScorer, "llmScorer")
        : undefined,
    domain:
      typeof obj.domain === "object" && obj.domain !== null
        ? parseDomainConfig(obj.domain, "domain")
        : undefined,
    memory:
      typeof obj.memory === "object" && obj.memory !== null
        ? parseMemoryConfig(obj.memory, "memory")
        : undefined,
    input:
      typeof obj.input === "object" && obj.input !== null
        ? parseInputConfig(obj.input, "input")
        : undefined,
    output:
      typeof obj.output === "object" && obj.output !== null
        ? parseOutputConfig(obj.output, "output")
        : undefined,
    backend: optStr(obj.backend),
    llmAuto: optBool(obj.llmAuto),
    llmBoost: optBool(obj.llmBoost),
  }) as GoldenMatchConfig;

  return config;
}

/**
 * Parse a YAML string into a GoldenMatchConfig.
 *
 * Requires the caller to provide a YAML parse function (e.g. from the `yaml`
 * npm package) to keep this module edge-safe with no dynamic imports.
 *
 * @param yamlStr - The YAML configuration string.
 * @param yamlParseFn - A function that parses a YAML string into a JS object.
 */
export function parseConfigYaml(
  yamlStr: string,
  yamlParseFn: (s: string) => unknown,
): GoldenMatchConfig {
  const raw = yamlParseFn(yamlStr);
  if (typeof raw !== "object" || raw === null) {
    throw new Error("Invalid YAML config: expected a non-null object at root");
  }
  return parseConfig(raw);
}

/**
 * Convert a GoldenMatchConfig back to a plain JS object suitable for
 * YAML or JSON serialization (snake_case keys).
 *
 * @param config - The typed config object.
 * @param yamlStringifyFn - A function that serializes a JS object to YAML.
 */
export function configToYaml(
  config: GoldenMatchConfig,
  yamlStringifyFn: (obj: unknown) => string,
): string {
  // Strip undefined values then convert keys to snake_case
  const plain = JSON.parse(JSON.stringify(config));
  const snaked = snakeifyKeys(plain);
  return yamlStringifyFn(snaked);
}
