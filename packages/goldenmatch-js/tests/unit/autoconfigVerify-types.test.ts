import { describe, it, expect } from "vitest";
import {
  makePreflightReport,
  stripConventionPrivate,
  ConfigValidationError,
  type PreflightFinding,
} from "../../src/core/autoconfigVerify.js";

describe("PreflightReport shape", () => {
  it("makePreflightReport computes hasErrors from findings", () => {
    const warn: PreflightFinding = {
      check: "cardinality_high",
      severity: "warning",
      subject: "email",
      message: "dropped useless matchkey",
      repaired: true,
      repairNote: "cardinality 1.0",
    };
    const error: PreflightFinding = {
      check: "missing_column",
      severity: "error",
      subject: "nonexistent",
      message: "unresolved",
      repaired: false,
      repairNote: null,
    };

    expect(makePreflightReport([warn], false).hasErrors).toBe(false);
    expect(makePreflightReport([warn, error], false).hasErrors).toBe(true);
    expect(makePreflightReport([], false).hasErrors).toBe(false);
    // Error + repaired=true: NOT an error
    expect(
      makePreflightReport([{ ...error, repaired: true }], false).hasErrors,
    ).toBe(false);
  });

  it("ConfigValidationError carries the report", () => {
    const err = makePreflightReport(
      [
        {
          check: "missing_column",
          severity: "error",
          subject: "nonexistent",
          message: "unresolved",
          repaired: false,
          repairNote: null,
        },
      ],
      false,
    );
    const exception = new ConfigValidationError(err);
    expect(exception.report).toBe(err);
    expect(exception.message).toContain("missing_column");
  });
});

describe("stripConventionPrivate", () => {
  it("removes underscore-prefixed keys", () => {
    const cfg = {
      matchkeys: [],
      _preflightReport: {},
      _strictAutoconfig: true,
    };
    const out = stripConventionPrivate(cfg);
    expect(Object.keys(out).sort()).toEqual(["matchkeys"]);
  });

  it("leaves non-underscore keys untouched", () => {
    const cfg = { a: 1, b: "two", _hidden: 99 };
    const out = stripConventionPrivate(cfg) as Record<string, unknown>;
    expect(out["a"]).toBe(1);
    expect(out["b"]).toBe("two");
    expect(out).not.toHaveProperty("_hidden");
  });
});
