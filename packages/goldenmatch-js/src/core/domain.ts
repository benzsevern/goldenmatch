/**
 * domain.ts — Domain detection & lightweight feature extraction.
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/core/domain.py. Detects the subject area (product,
 * person, bibliographic, company, generic) from column names and extracts
 * per-row features (brand, model, version, etc.) as extra columns.
 */

import type { Row } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DomainProfile {
  readonly name: string;
  readonly confidence: number;
  readonly textColumns: readonly string[];
  readonly featureColumns: readonly string[];
}

// ---------------------------------------------------------------------------
// Domain signature tables
// ---------------------------------------------------------------------------

type Signature = { readonly pattern: RegExp; readonly weight: number };

const PRODUCT_SIGNATURES: readonly Signature[] = [
  { pattern: /brand|manufacturer|mfr/i, weight: 2 },
  { pattern: /model/i, weight: 2 },
  { pattern: /sku|upc|ean|asin|mpn/i, weight: 3 },
  { pattern: /price|msrp|cost/i, weight: 1 },
  { pattern: /category|dept|department/i, weight: 1 },
  { pattern: /product|item/i, weight: 1 },
];

const PERSON_SIGNATURES: readonly Signature[] = [
  { pattern: /^first|first_name|fname/i, weight: 2 },
  { pattern: /^last|last_name|lname|surname/i, weight: 2 },
  { pattern: /full_name|person_name/i, weight: 2 },
  { pattern: /email/i, weight: 2 },
  { pattern: /phone|mobile|cell/i, weight: 1 },
  { pattern: /dob|birth|birthday/i, weight: 2 },
  { pattern: /ssn|nin/i, weight: 3 },
];

const BIBLIOGRAPHIC_SIGNATURES: readonly Signature[] = [
  { pattern: /^title$|article_title/i, weight: 2 },
  { pattern: /authors?|by_line/i, weight: 3 },
  { pattern: /year|pub_year|published/i, weight: 1 },
  { pattern: /venue|journal|conference/i, weight: 2 },
  { pattern: /doi|issn|isbn/i, weight: 3 },
  { pattern: /abstract/i, weight: 1 },
];

const COMPANY_SIGNATURES: readonly Signature[] = [
  { pattern: /company|employer|org(?!anization_id)/i, weight: 2 },
  { pattern: /industry|sector/i, weight: 2 },
  { pattern: /website|domain|url/i, weight: 1 },
  { pattern: /ein|duns|cik|lei/i, weight: 3 },
  { pattern: /hq|headquarters/i, weight: 1 },
];

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

function scoreDomain(
  columns: readonly string[],
  signatures: readonly Signature[],
): number {
  let score = 0;
  for (const col of columns) {
    for (const sig of signatures) {
      if (sig.pattern.test(col)) {
        score += sig.weight;
        break;
      }
    }
  }
  return score;
}

function findMatchingColumns(
  columns: readonly string[],
  signatures: readonly Signature[],
): string[] {
  const hits: string[] = [];
  for (const col of columns) {
    if (signatures.some((s) => s.pattern.test(col))) {
      hits.push(col);
    }
  }
  return hits;
}

const TEXT_NAME_RE = /name|title|description|notes|text|body/i;

/**
 * Detect the domain of a dataset based on its column names.
 */
export function detectDomain(columns: readonly string[]): DomainProfile {
  const candidates: ReadonlyArray<{
    name: string;
    score: number;
    features: string[];
  }> = [
    {
      name: "product",
      score: scoreDomain(columns, PRODUCT_SIGNATURES),
      features: findMatchingColumns(columns, PRODUCT_SIGNATURES),
    },
    {
      name: "person",
      score: scoreDomain(columns, PERSON_SIGNATURES),
      features: findMatchingColumns(columns, PERSON_SIGNATURES),
    },
    {
      name: "bibliographic",
      score: scoreDomain(columns, BIBLIOGRAPHIC_SIGNATURES),
      features: findMatchingColumns(columns, BIBLIOGRAPHIC_SIGNATURES),
    },
    {
      name: "company",
      score: scoreDomain(columns, COMPANY_SIGNATURES),
      features: findMatchingColumns(columns, COMPANY_SIGNATURES),
    },
  ];

  let winner = candidates[0]!;
  for (const c of candidates) if (c.score > winner.score) winner = c;

  const MAX_SCORE = 10;
  const confidence =
    winner.score <= 0 ? 0 : Math.min(1, winner.score / MAX_SCORE);

  const textColumns = columns.filter((c) => TEXT_NAME_RE.test(c));

  if (winner.score === 0) {
    return {
      name: "generic",
      confidence: 0,
      textColumns,
      featureColumns: [],
    };
  }

  return {
    name: winner.name,
    confidence,
    textColumns,
    featureColumns: winner.features,
  };
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

function asString(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const s = typeof value === "string" ? value : String(value);
  const trimmed = s.trim();
  return trimmed.length === 0 ? null : trimmed;
}

const KNOWN_BRANDS = new Set(
  [
    "apple",
    "samsung",
    "sony",
    "lg",
    "dell",
    "hp",
    "lenovo",
    "asus",
    "acer",
    "microsoft",
    "google",
    "amazon",
    "bose",
    "canon",
    "nikon",
    "panasonic",
    "philips",
    "toshiba",
  ].map((s) => s.toLowerCase()),
);

const MODEL_RE = /\b([A-Z0-9]{2,}[\-_]?[A-Z0-9]{2,}|[A-Z][A-Z0-9]{3,})\b/;
const SEMVER_RE = /\b(\d+\.\d+(?:\.\d+)?(?:[\-+][A-Za-z0-9.]+)?)\b/;

function extractBrand(row: Row, profile: DomainProfile): string | null {
  const manufacturer =
    asString(row["manufacturer"]) ??
    asString(row["brand"]) ??
    asString(row["mfr"]);
  if (manufacturer) return manufacturer.toLowerCase();

  for (const col of profile.textColumns) {
    const val = asString(row[col]);
    if (!val) continue;
    const first = val.split(/\s+/)[0];
    if (first && KNOWN_BRANDS.has(first.toLowerCase())) {
      return first.toLowerCase();
    }
  }
  return null;
}

function extractModel(row: Row, profile: DomainProfile): string | null {
  const explicit = asString(row["model"]) ?? asString(row["mpn"]);
  if (explicit) {
    return explicit.replace(/[\-_\s]/g, "").toUpperCase();
  }
  for (const col of profile.textColumns) {
    const val = asString(row[col]);
    if (!val) continue;
    const m = MODEL_RE.exec(val);
    if (m && m[1]) return m[1].replace(/[\-_]/g, "").toUpperCase();
  }
  return null;
}

function extractVersion(row: Row, profile: DomainProfile): string | null {
  const explicit = asString(row["version"]) ?? asString(row["ver"]);
  if (explicit) return explicit;
  for (const col of profile.textColumns) {
    const val = asString(row[col]);
    if (!val) continue;
    const m = SEMVER_RE.exec(val);
    if (m && m[1]) return m[1];
  }
  return null;
}

/**
 * Annotate rows with domain-specific extracted columns.
 * Returns enriched rows plus indices with low extraction confidence.
 */
export function extractFeatures(
  rows: readonly Row[],
  profile: DomainProfile,
  confidenceThreshold: number = 0.3,
): { rows: Row[]; lowConfidenceIds: readonly number[] } {
  if (profile.name === "generic" || profile.confidence === 0) {
    return { rows: rows.map((r) => ({ ...r })), lowConfidenceIds: [] };
  }

  const lowConfidenceIds: number[] = [];
  const out: Row[] = [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    const enriched: Record<string, unknown> = { ...row };

    if (profile.name === "product") {
      const brand = extractBrand(row, profile);
      const model = extractModel(row, profile);
      const version = extractVersion(row, profile);

      if (brand !== null) enriched["__brand__"] = brand;
      if (model !== null) enriched["__model__"] = model;
      if (version !== null) enriched["__version__"] = version;

      const expected = 3;
      const got = [brand, model, version].filter((v) => v !== null).length;
      const conf = got / expected;
      if (conf < confidenceThreshold) lowConfidenceIds.push(i);
    }

    out.push(enriched as Row);
  }

  return { rows: out, lowConfidenceIds };
}
