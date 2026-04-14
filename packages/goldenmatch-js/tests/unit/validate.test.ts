import { describe, it, expect, vi } from "vitest";
import { validateRows, checkRule } from "../../src/core/validate.js";
import type { ValidationRule } from "../../src/core/validate.js";
import type { Row } from "../../src/core/types.js";

describe("validateRows — not_null rule", () => {
  it("quarantines rows that fail not_null when action is quarantine", () => {
    const rows: Row[] = [
      { email: "a@x.com" },
      { email: null },
      { email: "" },
    ];
    const rules: ValidationRule[] = [
      {
        column: "email",
        ruleType: "not_null",
        params: {},
        action: "quarantine",
      },
    ];
    const res = validateRows(rows, rules);
    expect(res.valid.length).toBe(1);
    expect(res.quarantine.length).toBe(2);
    expect(res.report.quarantined).toBe(2);
    expect(res.report.ruleViolations["email:not_null"]).toBe(2);
  });

  it("action='null' sets the failing cell to null but keeps the row", () => {
    const rows: Row[] = [{ email: "" }];
    const rules: ValidationRule[] = [
      { column: "email", ruleType: "not_null", params: {}, action: "null" },
    ];
    const res = validateRows(rows, rules);
    expect(res.valid.length).toBe(1);
    expect(res.valid[0]!["email"]).toBeNull();
    expect(res.quarantine.length).toBe(0);
  });

  it("action='flag' keeps the row and adds to __flags__ without quarantine", () => {
    const rows: Row[] = [{ email: null }];
    const rules: ValidationRule[] = [
      { column: "email", ruleType: "not_null", params: {}, action: "flag" },
    ];
    const res = validateRows(rows, rules);
    expect(res.valid.length).toBe(1);
    const flags = res.valid[0]!["__flags__"] as string[];
    expect(flags).toContain("email:not_null");
    expect(res.report.flagged).toBe(1);
  });
});

describe("validateRows — regex rule", () => {
  it("valid values pass, invalid values trigger the action", () => {
    const rows: Row[] = [
      { email: "alice@example.com" },
      { email: "not-an-email" },
    ];
    const rules: ValidationRule[] = [
      {
        column: "email",
        ruleType: "regex",
        params: { pattern: "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$" },
        action: "quarantine",
      },
    ];
    const res = validateRows(rows, rules);
    expect(res.valid.length).toBe(1);
    expect(res.quarantine.length).toBe(1);
  });

  it("invalid regex pattern does not crash; value is treated as failing", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const rows: Row[] = [{ email: "alice@example.com" }];
    const rules: ValidationRule[] = [
      {
        column: "email",
        ruleType: "regex",
        params: { pattern: "([unclosed" },
        action: "flag",
      },
    ];
    // Must not throw.
    expect(() => validateRows(rows, rules)).not.toThrow();
    // Direct checkRule returns false for broken regex (via catch).
    expect(
      checkRule("a", {
        column: "x",
        ruleType: "regex",
        params: { pattern: "([unclosed" },
        action: "flag",
      }),
    ).toBe(false);
    warn.mockRestore();
  });
});

describe("checkRule — misc", () => {
  it("min_length / max_length respect the `value` parameter", () => {
    const minOk = checkRule("hello", {
      column: "x",
      ruleType: "min_length",
      params: { value: 3 },
      action: "flag",
    });
    const minFail = checkRule("hi", {
      column: "x",
      ruleType: "min_length",
      params: { value: 3 },
      action: "flag",
    });
    const maxOk = checkRule("hi", {
      column: "x",
      ruleType: "max_length",
      params: { value: 3 },
      action: "flag",
    });
    const maxFail = checkRule("toolong", {
      column: "x",
      ruleType: "max_length",
      params: { value: 3 },
      action: "flag",
    });
    expect(minOk).toBe(true);
    expect(minFail).toBe(false);
    expect(maxOk).toBe(true);
    expect(maxFail).toBe(false);
  });

  it("in_set accepts allowed and rejects disallowed values", () => {
    const allowed = checkRule("a", {
      column: "x",
      ruleType: "in_set",
      params: { values: ["a", "b"] },
      action: "flag",
    });
    const rejected = checkRule("z", {
      column: "x",
      ruleType: "in_set",
      params: { values: ["a", "b"] },
      action: "flag",
    });
    expect(allowed).toBe(true);
    expect(rejected).toBe(false);
  });
});
