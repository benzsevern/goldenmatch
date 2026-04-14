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

/**
 * Compile a rule into a checker function. Expensive work (regex compilation)
 * happens here, not on every row. If a regex is invalid, log once and return
 * a checker that matches no rows.
 */
function compileRule(rule: ValidationRule): (value: unknown) => boolean {
  if (rule.ruleType === "not_null") {
    return (v) => v !== null && v !== undefined && v !== "";
  }

  if (rule.ruleType === "regex") {
    const pat = rule.params["pattern"];
    if (typeof pat !== "string") {
      return (v) => v === null || v === undefined;
    }
    let re: RegExp;
    try {
      re = new RegExp(pat);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        `Invalid regex pattern for rule on column '${rule.column}': ${pat}. ` +
          `Error: ${err instanceof Error ? err.message : String(err)}. ` +
          `Rule will match no rows.`,
      );
      return (v) => {
        if (v === null || v === undefined) return true;
        return false;
      };
    }
    return (v) => {
      if (v === null || v === undefined) return true;
      const str = valueToStr(v);
      if (str === null) return false;
      return re.test(str);
    };
  }

  if (rule.ruleType === "min_length") {
    const min =
      typeof rule.params["value"] === "number" ? rule.params["value"] : 0;
    return (v) => {
      if (v === null || v === undefined) return true;
      const str = valueToStr(v) ?? "";
      return str.length >= min;
    };
  }

  if (rule.ruleType === "max_length") {
    const max =
      typeof rule.params["value"] === "number"
        ? rule.params["value"]
        : Infinity;
    return (v) => {
      if (v === null || v === undefined) return true;
      const str = valueToStr(v) ?? "";
      return str.length <= max;
    };
  }

  if (rule.ruleType === "in_set") {
    const allowed = rule.params["values"];
    if (!Array.isArray(allowed)) return () => true;
    return (v) => {
      if (v === null || v === undefined) return true;
      return allowed.includes(v);
    };
  }

  if (rule.ruleType === "format") {
    const name = rule.params["name"];
    if (typeof name !== "string") return () => true;
    const matcher = FORMAT_MATCHERS[name];
    if (matcher === undefined) return () => true;
    return (v) => {
      if (v === null || v === undefined) return true;
      const str = valueToStr(v);
      if (str === null) return false;
      return matcher.test(str);
    };
  }

  return () => true;
}

/** Returns true if the rule passes for this value, false otherwise. */
export function checkRule(value: unknown, rule: ValidationRule): boolean {
  return compileRule(rule)(value);
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

  // Pre-compile all rule checkers once. Logs any regex errors exactly once.
  const compiled = rules.map((rule) => ({
    rule,
    check: compileRule(rule),
  }));

  for (const row of rows) {
    let current: Record<string, unknown> = { ...row };
    let shouldQuarantine = false;
    let wasFlagged = false;
    const flags: string[] = Array.isArray(current["__flags__"])
      ? [...(current["__flags__"] as unknown[])].filter(
          (f): f is string => typeof f === "string",
        )
      : [];

    for (const { rule, check } of compiled) {
      const value = current[rule.column];
      if (check(value)) continue;

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
