/**
 * validate.ts — Column validation rules with quarantine/flag actions.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/validate.py.
 */

import type { Row } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ValidationAction = "null" | "quarantine" | "flag";

export type ValidationRuleType =
  | "regex"
  | "min_length"
  | "max_length"
  | "not_null"
  | "in_set"
  | "format";

export interface ValidationRule {
  readonly column: string;
  readonly ruleType: ValidationRuleType;
  readonly params: Readonly<Record<string, unknown>>;
  readonly action: ValidationAction;
}

export interface ValidationReport {
  readonly totalRows: number;
  readonly quarantined: number;
  readonly flagged: number;
  readonly ruleViolations: Readonly<Record<string, number>>;
}

export interface ValidationResult {
  readonly valid: Row[];
  readonly quarantine: Row[];
  readonly report: ValidationReport;
}

// ---------------------------------------------------------------------------
// Built-in format matchers
// ---------------------------------------------------------------------------

const FORMAT_MATCHERS: Readonly<Record<string, RegExp>> = {
  email: /^[^@\s]+@[^@\s]+\.[^@\s]+$/,
  phone: /^\+?[\d\s().-]{7,}$/,
  zip: /^\d{5}(-\d{4})?$/,
  date: /^\d{4}-\d{2}-\d{2}$/,
};

// ---------------------------------------------------------------------------
// Rule checker
// ---------------------------------------------------------------------------

function valueToStr(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return null;
}

/** Returns true if the rule passes for this value, false otherwise. */
export function checkRule(value: unknown, rule: ValidationRule): boolean {
  const params = rule.params;

  if (rule.ruleType === "not_null") {
    return value !== null && value !== undefined && value !== "";
  }

  // Other rules short-circuit on null to pass (use not_null to force)
  if (value === null || value === undefined) return true;

  switch (rule.ruleType) {
    case "regex": {
      const pat = params["pattern"];
      if (typeof pat !== "string") return true;
      const str = valueToStr(value);
      if (str === null) return false;
      try {
        return new RegExp(pat).test(str);
      } catch {
        return false;
      }
    }
    case "min_length": {
      const min = typeof params["value"] === "number" ? params["value"] : 0;
      const str = valueToStr(value) ?? "";
      return str.length >= min;
    }
    case "max_length": {
      const max =
        typeof params["value"] === "number" ? params["value"] : Infinity;
      const str = valueToStr(value) ?? "";
      return str.length <= max;
    }
    case "in_set": {
      const allowed = params["values"];
      if (!Array.isArray(allowed)) return true;
      return allowed.includes(value);
    }
    case "format": {
      const name = params["name"];
      if (typeof name !== "string") return true;
      const matcher = FORMAT_MATCHERS[name];
      if (matcher === undefined) return true;
      const str = valueToStr(value);
      if (str === null) return false;
      return matcher.test(str);
    }
    default:
      return true;
  }
}

function ruleKey(rule: ValidationRule): string {
  return `${rule.column}:${rule.ruleType}`;
}

// ---------------------------------------------------------------------------
// validateRows
// ---------------------------------------------------------------------------

/**
 * Validate rows against a list of rules.
 *
 * Actions:
 * - "null":       replace the failing value with null, row stays valid
 * - "quarantine": move row to quarantine bucket
 * - "flag":       add __flags__ entry, row stays valid
 */
export function validateRows(
  rows: readonly Row[],
  rules: readonly ValidationRule[],
): ValidationResult {
  const valid: Row[] = [];
  const quarantine: Row[] = [];
  const violations = new Map<string, number>();
  let flagged = 0;

  for (const row of rows) {
    let current: Record<string, unknown> = { ...row };
    let shouldQuarantine = false;
    let wasFlagged = false;
    const flags: string[] = Array.isArray(current["__flags__"])
      ? [...(current["__flags__"] as unknown[])].filter(
          (f): f is string => typeof f === "string",
        )
      : [];

    for (const rule of rules) {
      const value = current[rule.column];
      if (checkRule(value, rule)) continue;

      const key = ruleKey(rule);
      violations.set(key, (violations.get(key) ?? 0) + 1);

      switch (rule.action) {
        case "null":
          current[rule.column] = null;
          break;
        case "quarantine":
          shouldQuarantine = true;
          break;
        case "flag":
          flags.push(key);
          wasFlagged = true;
          break;
      }
      if (shouldQuarantine) break;
    }

    if (shouldQuarantine) {
      quarantine.push(current as Row);
    } else {
      if (wasFlagged) {
        current["__flags__"] = flags;
        flagged++;
      }
      valid.push(current as Row);
    }
  }

  const ruleViolations: Record<string, number> = {};
  for (const [k, v] of violations) ruleViolations[k] = v;

  return {
    valid,
    quarantine,
    report: {
      totalRows: rows.length,
      quarantined: quarantine.length,
      flagged,
      ruleViolations,
    },
  };
}
