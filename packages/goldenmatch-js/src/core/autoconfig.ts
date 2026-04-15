/**
 * autoconfig.ts — Auto-generate a GoldenMatch config from sample data.
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/core/autoconfig.py. Profiles the rows, classifies
 * columns, and builds exact/weighted matchkeys + blocking config.
 */

import type {
  Row,
  GoldenMatchConfig,
  MatchkeyConfig,
  MatchkeyField,
  BlockingKeyConfig,
  BlockingConfig,
} from "./types.js";
import {
  makeConfig,
  makeMatchkeyConfig,
  makeMatchkeyField,
  makeBlockingConfig,
  makeGoldenRulesConfig,
} from "./types.js";
import { profileRows, type ColumnProfile, type DatasetProfile } from "./profiler.js";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface AutoconfigOptions {
  readonly llmProvider?: string;
  readonly llmAuto?: boolean;
}

// ---------------------------------------------------------------------------
// Name-based classification patterns (authoritative over data profiling for
// some signals — matches Python's _DATE_PATTERNS / _GEO_PATTERNS behavior).
// ---------------------------------------------------------------------------

const EMAIL_NAME_PATTERNS = [/email/i, /e_mail/i, /e-mail/i];
const PHONE_NAME_PATTERNS = [/phone/i, /tel(?!e)/i, /mobile/i, /cell/i];
const NAME_NAME_PATTERNS = [/name/i, /first/i, /last/i, /full_name/i, /surname/i];
const ZIP_NAME_PATTERNS = [/zip/i, /postal/i, /postcode/i];
const GEO_NAME_PATTERNS = [
  /^city/i,
  /city_desc/i,
  /^state/i,
  /state_cd/i,
  /county/i,
  /country/i,
  /^region/i,
  /province/i,
];
const DATE_NAME_PATTERNS = [
  /date/i,
  /created/i,
  /modified/i,
  /updated/i,
  /_at$/i,
  /birth(?!_year)/i, // "birth" but not "birth_year" — year takes precedence
  /dob/i,
];
const YEAR_NAME_PATTERNS = [/(^|_)(year|yr)(_|$)/i];
const ID_NAME_PATTERNS = [
  /^id$/i,
  /_id$/i,
  /uuid/i,
  /guid/i,
  // v0.3 additions — targeted suffixes + whole-name anchors. Deliberately
  // NOT adding /_(no|num)$/ alone — would false-positive on yes_no, num_kids.
  /_(ref|ref_num|reg_num|account_no|account_num|account)$/i,
  /^(account_no|account_num)$/i,
  /^guid_/i,
  /^uuid_/i,
];

// Re-exported for consumers that wanted the spec-level constants.
export const EMAIL_PATTERNS = EMAIL_NAME_PATTERNS;
export const PHONE_PATTERNS = PHONE_NAME_PATTERNS;
export const NAME_PATTERNS = NAME_NAME_PATTERNS;
export const ZIP_PATTERNS = ZIP_NAME_PATTERNS;
export const GEO_PATTERNS = GEO_NAME_PATTERNS;
export const DATE_PATTERNS = DATE_NAME_PATTERNS;
export const ID_PATTERNS = ID_NAME_PATTERNS;

function nameMatches(name: string, patterns: readonly RegExp[]): boolean {
  return patterns.some((re) => re.test(name));
}

// ---------------------------------------------------------------------------
// Column classification (authoritative: date > geo > name heuristics)
// ---------------------------------------------------------------------------

type ClassifiedKind =
  | "email"
  | "phone"
  | "zip"
  | "geo"
  | "date"
  | "year"
  | "name"
  | "multi_name"
  | "id"
  | "numeric"
  | "text";

function classifyColumn(profile: ColumnProfile): ClassifiedKind {
  const name = profile.name;

  // Cardinality guard (spec §5.2) — a column where virtually every value is
  // unique cannot be a phone, zip, or numeric feature UNLESS its name
  // explicitly asserts it (e.g. "phone" or "zip"). Scoped to samples >= 10
  // to avoid false positives on tiny fixtures. Only overrides data-heuristic
  // classifications; explicit name patterns still win below.
  if (
    profile.totalCount >= 10 &&
    profile.cardinalityRatio >= 0.95 &&
    !nameMatches(name, EMAIL_NAME_PATTERNS) &&
    !nameMatches(name, PHONE_NAME_PATTERNS) &&
    !nameMatches(name, ZIP_NAME_PATTERNS) &&
    !nameMatches(name, NAME_NAME_PATTERNS) &&
    !nameMatches(name, GEO_NAME_PATTERNS) &&
    !nameMatches(name, DATE_NAME_PATTERNS) &&
    !nameMatches(name, YEAR_NAME_PATTERNS) &&
    profile.inferredType !== "year" &&
    (profile.inferredType === "phone" ||
      profile.inferredType === "zip" ||
      profile.inferredType === "numeric")
  ) {
    return "id";
  }

  // Year checked before date so "birth_year" routes to year, not date.
  if (nameMatches(name, YEAR_NAME_PATTERNS)) return "year";
  if (profile.inferredType === "year") return "year";

  // Date is checked first so that date-like columns never get misclassified
  // as phones by the profiler's value heuristic.
  if (nameMatches(name, DATE_NAME_PATTERNS)) return "date";
  if (profile.inferredType === "date") return "date";

  if (nameMatches(name, GEO_NAME_PATTERNS)) return "geo";
  if (profile.inferredType === "geo") return "geo";

  if (nameMatches(name, EMAIL_NAME_PATTERNS) || profile.inferredType === "email") {
    return "email";
  }
  if (nameMatches(name, PHONE_NAME_PATTERNS) || profile.inferredType === "phone") {
    return "phone";
  }
  if (nameMatches(name, ZIP_NAME_PATTERNS) || profile.inferredType === "zip") {
    return "zip";
  }
  // Multi-name (delimited author/entity list) checked before plain name so
  // "authors" column with comma-separated values routes to token_sort, not
  // jaro_winkler.
  if (profile.inferredType === "multi_name") return "multi_name";
  if (nameMatches(name, NAME_NAME_PATTERNS) || profile.inferredType === "name") {
    return "name";
  }
  if (nameMatches(name, ID_NAME_PATTERNS) || profile.inferredType === "id") {
    return "id";
  }
  if (profile.inferredType === "numeric") return "numeric";
  return "text";
}

// ---------------------------------------------------------------------------
// Heuristic builders
// ---------------------------------------------------------------------------

function buildExactMatchkeys(
  profiles: readonly ColumnProfile[],
): MatchkeyConfig[] {
  const out: MatchkeyConfig[] = [];
  for (const p of profiles) {
    const kind = classifyColumn(p);
    // zip/geo/year are blocking signals, NOT identity claims.
    // id/numeric/date are skipped from scoring (Python parity: id is a
    // primary-key column, not a match signal).
    if (
      kind === "zip" ||
      kind === "geo" ||
      kind === "date" ||
      kind === "year" ||
      kind === "text" ||
      kind === "id" ||
      kind === "numeric"
    ) {
      continue;
    }

    // Skip sparse & near-constant columns
    if (p.nullRate > 0.4) continue;
    if (p.cardinalityRatio < 0.01) continue;

    // Only identifier-like columns (email, phone) get exact matchkeys with >=0.5 cardinality.
    const isIdentifier = kind === "email" || kind === "phone";
    if (!isIdentifier) continue;
    if (p.cardinalityRatio < 0.5) continue;

    const transforms: string[] =
      kind === "email"
        ? ["lowercase", "strip"]
        : kind === "phone"
          ? ["digits_only"]
          : ["strip"];

    out.push(
      makeMatchkeyConfig({
        name: `exact_${p.name}`,
        type: "exact",
        fields: [
          makeMatchkeyField({
            field: p.name,
            transforms,
            scorer: "exact",
            weight: 1.0,
          }),
        ],
        threshold: 1.0,
      }),
    );
  }
  return out;
}

function buildWeightedMatchkey(
  profiles: readonly ColumnProfile[],
): MatchkeyConfig | null {
  const fields: MatchkeyField[] = [];

  for (const p of profiles) {
    const kind = classifyColumn(p);
    if (p.nullRate > 0.5) continue;

    if (kind === "multi_name") {
      fields.push(
        makeMatchkeyField({
          field: p.name,
          transforms: ["lowercase", "strip", "normalize_whitespace"],
          scorer: "token_sort",
          weight: 1.0,
        }),
      );
    } else if (kind === "name") {
      fields.push(
        makeMatchkeyField({
          field: p.name,
          transforms: ["lowercase", "strip", "normalize_whitespace"],
          scorer: "jaro_winkler",
          weight: 0.6,
        }),
      );
    } else if (kind === "email") {
      fields.push(
        makeMatchkeyField({
          field: p.name,
          transforms: ["lowercase", "strip"],
          scorer: "jaro_winkler",
          weight: 0.3,
        }),
      );
    } else if (kind === "phone") {
      fields.push(
        makeMatchkeyField({
          field: p.name,
          transforms: ["digits_only"],
          scorer: "exact",
          weight: 0.25,
        }),
      );
    } else if (kind === "zip") {
      fields.push(
        makeMatchkeyField({
          field: p.name,
          transforms: ["digits_only"],
          scorer: "exact",
          weight: 0.15,
        }),
      );
    } else if (kind === "geo") {
      fields.push(
        makeMatchkeyField({
          field: p.name,
          transforms: ["lowercase", "strip"],
          scorer: "exact",
          weight: 0.1,
        }),
      );
    } else if (kind === "text" && p.avgLength >= 10) {
      // Long free-text columns: token_sort to catch reordering
      fields.push(
        makeMatchkeyField({
          field: p.name,
          transforms: ["lowercase", "strip", "token_sort"],
          scorer: "token_sort",
          weight: 0.2,
        }),
      );
    }
  }

  if (fields.length === 0) return null;

  return makeMatchkeyConfig({
    name: "weighted_identity",
    type: "weighted",
    fields,
    threshold: 0.85,
    rerank: false,
  });
}

function buildBlocking(profiles: readonly ColumnProfile[]): BlockingConfig {
  const keys: BlockingKeyConfig[] = [];

  // Prefer zip > geo > first-letter of name
  for (const p of profiles) {
    const kind = classifyColumn(p);
    if (kind !== "zip") continue;
    if (p.nullRate > 0.2) continue;
    if (p.cardinalityRatio >= 0.95) continue;
    keys.push({
      fields: [p.name],
      transforms: ["digits_only", "substring:0:5"],
    });
    break;
  }

  if (keys.length === 0) {
    for (const p of profiles) {
      const kind = classifyColumn(p);
      if (kind !== "year") continue;
      if (p.nullRate > 0.2) continue;
      if (p.cardinalityRatio >= 0.95) continue;
      keys.push({
        fields: [p.name],
        transforms: ["strip"],
      });
      break;
    }
  }

  if (keys.length === 0) {
    for (const p of profiles) {
      const kind = classifyColumn(p);
      if (kind !== "geo") continue;
      if (p.nullRate > 0.2) continue;
      if (p.cardinalityRatio >= 0.95) continue;
      keys.push({
        fields: [p.name],
        transforms: ["lowercase", "strip"],
      });
      break;
    }
  }

  if (keys.length === 0) {
    for (const p of profiles) {
      const kind = classifyColumn(p);
      if (kind !== "name") continue;
      if (p.nullRate > 0.2) continue;
      if (p.cardinalityRatio >= 0.95) continue;
      keys.push({
        fields: [p.name],
        transforms: ["lowercase", "strip", "substring:0:1"],
      });
      break;
    }
  }

  // Last resort: first non-null column that isn't near-unique or sparse
  if (keys.length === 0) {
    for (const p of profiles) {
      if (p.nullRate > 0.2) continue;
      if (p.cardinalityRatio >= 0.95) continue;
      if (p.cardinalityRatio < 0.01) continue;
      keys.push({
        fields: [p.name],
        transforms: ["lowercase", "strip"],
      });
      break;
    }
  }

  return makeBlockingConfig({
    strategy: "static",
    keys,
    maxBlockSize: 1000,
    skipOversized: true,
  });
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/**
 * Build a GoldenMatchConfig by profiling the provided rows.
 *
 * Mirrors goldenmatch.core.autoconfig.auto_configure_df. Does not apply
 * standardization rules directly — callers can merge them onto the result.
 */
export function autoConfigureRows(
  rows: readonly Row[],
  options?: AutoconfigOptions,
): GoldenMatchConfig {
  const profile: DatasetProfile = profileRows(rows);
  const profiles = profile.columns;

  const exactKeys = buildExactMatchkeys(profiles);
  const weighted = buildWeightedMatchkey(profiles);
  const matchkeys: MatchkeyConfig[] = [...exactKeys];
  if (weighted) matchkeys.push(weighted);

  const blocking = buildBlocking(profiles);
  const goldenRules = makeGoldenRulesConfig({ defaultStrategy: "most_complete" });

  const config = makeConfig({
    matchkeys,
    blocking,
    goldenRules,
    threshold: 0.85,
    ...(options?.llmAuto !== undefined ? { llmAuto: options.llmAuto } : {}),
  });

  return config;
}

/**
 * Convenience alias for API parity with the Python function that starts
 * from "files" (which, in edge-safe land, means pre-loaded row arrays).
 */
export function autoConfigure(
  rows: readonly Row[],
  options?: AutoconfigOptions,
): GoldenMatchConfig {
  return autoConfigureRows(rows, options);
}
