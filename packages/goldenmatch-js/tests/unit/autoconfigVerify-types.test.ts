import { describe, it, expect } from "vitest";
import {
  makePreflightReport,
  stripConventionPrivate,
  ConfigValidationError,
  type PreflightFinding,
  type PostflightReport,
} from "../../src/core/autoconfigVerify.js";
import type { GoldenMatchConfig } from "../../src/core/types.js";

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

describe("GoldenMatchConfig + result types", () => {
  it("GoldenMatchConfig allows _preflightReport assignment", () => {
    const cfg: GoldenMatchConfig = { matchkeys: [] };
    const report = makePreflightReport([], false);
    cfg._preflightReport = report;
    expect(cfg._preflightReport).toBe(report);
  });
});

// Type-level compile check — no runtime assertion. If these fail to compile,
// DedupeResult/MatchResult is missing `postflightReport`.
const _postflightReportTypeCheck = (r: PostflightReport) => {
  type DedupeResultShape = import("../../src/core/types.js").DedupeResult;
  type MatchResultShape = import("../../src/core/types.js").MatchResult;
  const _d: DedupeResultShape = {
    config: {} as GoldenMatchConfig,
    duplicates: [],
    unique: [],
    golden: [],
    stats: {
      totalRecords: 0,
      duplicateGroups: 0,
      duplicates: 0,
      unique: 0,
      matchRate: 0,
      runtime: 0,
    },
    clusters: {} as Record<number, unknown>,
    postflightReport: r,
  } as unknown as DedupeResultShape;
  const _m: MatchResultShape = {
    matched: [],
    unmatched: [],
    stats: {} as unknown,
    postflightReport: r,
  } as unknown as MatchResultShape;
  return [_d, _m];
};
void _postflightReportTypeCheck;

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
