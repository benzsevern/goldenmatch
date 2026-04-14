/**
 * standardize.ts — Data standardization for GoldenMatch-JS.
 * Edge-safe: no `node:` imports, pure TypeScript only.
 *
 * Ports standardization from goldenmatch/core/standardize.py.
 * These are data cleaning transforms applied to columns before matching.
 */

import type { Row } from "./types.js";

// ---------------------------------------------------------------------------
// Address abbreviations (USPS standard)
// ---------------------------------------------------------------------------

/** Map of full word (lowercase) to USPS abbreviation. */
const ADDRESS_ABBREVIATIONS: Readonly<Record<string, string>> = {
  street: "St",
  avenue: "Ave",
  boulevard: "Blvd",
  drive: "Dr",
  lane: "Ln",
  road: "Rd",
  court: "Ct",
  place: "Pl",
  circle: "Cir",
  terrace: "Ter",
  highway: "Hwy",
  parkway: "Pkwy",
  expressway: "Expy",
  freeway: "Fwy",
  trail: "Trl",
  way: "Way",
  north: "N",
  south: "S",
  east: "E",
  west: "W",
  northeast: "NE",
  northwest: "NW",
  southeast: "SE",
  southwest: "SW",
  apartment: "Apt",
  suite: "Ste",
  building: "Bldg",
  floor: "Fl",
  room: "Rm",
  unit: "Unit",
  department: "Dept",
  "post office box": "PO Box",
  "p.o. box": "PO Box",
  "po box": "PO Box",
};

// ---------------------------------------------------------------------------
// Individual standardizer functions
// ---------------------------------------------------------------------------

/**
 * Standardize email: lowercase, strip, validate basic structure.
 * Returns null for invalid emails.
 */
function stdEmail(value: string): string | null {
  const v = value.trim().toLowerCase();
  if (!v || !v.includes("@")) return null;
  const domain = v.split("@").pop();
  if (!domain || !domain.includes(".")) return null;
  return v;
}

/**
 * Standardize name to proper case (Title Case).
 * Handles hyphenated names: mary-jane -> Mary-Jane.
 */
function stdNameProper(value: string): string | null {
  const v = value.trim();
  if (!v) return null;
  // Collapse whitespace
  const collapsed = v.replace(/\s+/g, " ");
  // Title-case each whitespace-separated word; within a word handle hyphens
  const titleWord = (word: string): string => {
    if (!word) return "";
    const hyphenParts = word.split("-");
    return hyphenParts
      .map((p) => {
        if (!p) return "";
        return p.charAt(0).toUpperCase() + p.slice(1).toLowerCase();
      })
      .join("-");
  };
  return collapsed.split(" ").map(titleWord).join(" ");
}

/**
 * Standardize name to UPPER CASE.
 */
function stdNameUpper(value: string): string | null {
  const v = value.trim().replace(/\s+/g, " ").toUpperCase();
  return v || null;
}

/**
 * Standardize name to lower case.
 */
function stdNameLower(value: string): string | null {
  const v = value.trim().replace(/\s+/g, " ");
  return v ? v.toLowerCase() : null;
}

/**
 * Standardize phone: digits only, strip US country code if 11 digits starting with 1.
 * Returns null if fewer than 7 digits.
 */
function stdPhone(value: string): string | null {
  let digits = value.replace(/\D/g, "");
  if (!digits) return null;
  // Strip US country code
  if (digits.length === 11 && digits.startsWith("1")) {
    digits = digits.slice(1);
  }
  // Must be at least 7 digits
  if (digits.length < 7) return null;
  return digits;
}

/**
 * Standardize ZIP code to first 5 digits, zero-padded.
 */
function stdZip5(value: string): string | null {
  // Take part before hyphen or space
  const first = value.split("-")[0]!.split(" ")[0]!;
  const digits = first.replace(/\D/g, "");
  if (!digits) return null;
  return digits.slice(0, 5).padStart(5, "0");
}

/**
 * Title-case a single word.
 */
function titleCase(word: string): string {
  if (!word) return word;
  return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
}

/**
 * Standardize address: title case, USPS abbreviations, normalize whitespace.
 */
function stdAddress(value: string): string | null {
  let v = value.trim();
  if (!v) return null;
  // Normalize whitespace
  v = v.replace(/\s+/g, " ");
  const words = v.split(" ");
  const result: string[] = [];
  let i = 0;
  while (i < words.length) {
    // Check two-word phrases first (e.g. "post office")
    if (i + 1 < words.length) {
      const twoWord = `${words[i]} ${words[i + 1]}`.toLowerCase();
      if (twoWord in ADDRESS_ABBREVIATIONS) {
        result.push(ADDRESS_ABBREVIATIONS[twoWord]!);
        i += 2;
        continue;
      }
    }
    // Strip trailing punctuation for lookup
    const wordLower = words[i]!.toLowerCase().replace(/[.,]+$/, "");
    if (wordLower in ADDRESS_ABBREVIATIONS) {
      result.push(ADDRESS_ABBREVIATIONS[wordLower]!);
    } else {
      result.push(titleCase(words[i]!));
    }
    i += 1;
  }
  return result.join(" ");
}

/**
 * Standardize state to uppercase, strip.
 */
function stdState(value: string): string | null {
  const v = value.trim().toUpperCase();
  return v || null;
}

/**
 * Strip whitespace, normalize to null if empty.
 */
function stdStrip(value: string): string | null {
  const v = value.trim();
  return v || null;
}

/**
 * Collapse multiple spaces to one, strip.
 */
function stdTrimWhitespace(value: string): string | null {
  const v = value.replace(/\s+/g, " ").trim();
  return v || null;
}

// ---------------------------------------------------------------------------
// Standardizer registry
// ---------------------------------------------------------------------------

/** Map of standardizer name to function. */
const STANDARDIZERS: Readonly<Record<string, (value: string) => string | null>> = {
  email: stdEmail,
  name_proper: stdNameProper,
  name_upper: stdNameUpper,
  name_lower: stdNameLower,
  phone: stdPhone,
  zip5: stdZip5,
  address: stdAddress,
  state: stdState,
  strip: stdStrip,
  trim_whitespace: stdTrimWhitespace,
};

// ---------------------------------------------------------------------------
// applyStandardizer — dispatch to the correct standardizer
// ---------------------------------------------------------------------------

/**
 * Apply a named standardizer to a string value.
 *
 * @throws Error if the standardizer name is not recognized.
 */
export function applyStandardizer(value: string, name: string): string {
  const fn = STANDARDIZERS[name];
  if (!fn) {
    const available = Object.keys(STANDARDIZERS).sort().join(", ");
    throw new Error(
      `Unknown standardizer: "${name}". Available: ${available}`,
    );
  }
  const result = fn(value);
  // Standardizers may return null for invalid data; treat as empty string
  // so downstream pipeline can decide how to handle it.
  return result ?? "";
}

// ---------------------------------------------------------------------------
// applyStandardization — apply rules to all rows
// ---------------------------------------------------------------------------

/**
 * Apply standardization rules to rows.
 *
 * `rules` maps column names to arrays of standardizer names that are
 * applied in sequence. For example:
 *
 * ```ts
 * applyStandardization(rows, {
 *   email: ["email"],
 *   first_name: ["strip", "name_proper"],
 *   phone: ["phone"],
 * });
 * ```
 *
 * Returns new row objects (does not mutate originals).
 * Null/undefined column values are skipped (left as-is).
 */
export function applyStandardization(
  rows: readonly Row[],
  rules: Readonly<Record<string, readonly string[]>>,
): Row[] {
  return rows.map((row) => {
    const newRow: Record<string, unknown> = { ...row };
    for (const [column, standardizers] of Object.entries(rules)) {
      const val = row[column];
      if (val === null || val === undefined) continue;
      let str = String(val);
      for (const stdName of standardizers) {
        str = applyStandardizer(str, stdName);
      }
      newRow[column] = str;
    }
    return newRow as Row;
  });
}
