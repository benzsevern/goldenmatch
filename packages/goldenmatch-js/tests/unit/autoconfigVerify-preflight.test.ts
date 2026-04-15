import { describe, it, expect } from "vitest";
import {
  preflight,
  ConfigValidationError,
} from "../../src/core/autoconfigVerify.js";
import type { GoldenMatchConfig } from "../../src/core/types.js";

describe("preflight Check 1: missing column", () => {
  it("raises findings when referenced column missing", () => {
    const rows = [{ name: "a" }];
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk",
          type: "exact",
          fields: [
            { field: "nonexistent", transforms: [], scorer: "exact", weight: 1 },
          ],
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["nonexistent"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
    };
    const { report } = preflight(rows, cfg);
    expect(report.hasErrors).toBe(true);
    const missing = report.findings.find((f) => f.check === "missing_column");
    expect(missing).toBeDefined();
    expect(missing!.message).toContain("nonexistent");
  });

  it("auto-repairs config.domain when _domainProfile stashed and __<col>__ referenced", () => {
    const rows = [{ brand_raw: "acme", model_raw: "x" }];
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk",
          type: "exact",
          fields: [
            { field: "__brand__", transforms: [], scorer: "exact", weight: 1 },
          ],
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["__brand__"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
      _domainProfile: {
        name: "electronics",
        confidence: 0.9,
        textColumns: [],
        featureColumns: [],
      },
    };
    const { report, config: repaired } = preflight(rows, cfg);
    expect(report.hasErrors).toBe(false);
    expect(repaired.domain?.enabled).toBe(true);
    expect(repaired.domain?.mode).toBe("electronics");
    const repair = report.findings.find(
      (f) => f.check === "missing_column" && f.repaired,
    );
    expect(repair).toBeDefined();
  });

  it("ConfigValidationError can be thrown from report", () => {
    const rows = [{ a: 1 }];
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk",
          type: "exact",
          fields: [
            { field: "missing", transforms: [], scorer: "exact", weight: 1 },
          ],
        },
      ],
    };
    const { report } = preflight(rows, cfg);
    expect(() => {
      throw new ConfigValidationError(report);
    }).toThrow(/missing_column/);
  });
});
