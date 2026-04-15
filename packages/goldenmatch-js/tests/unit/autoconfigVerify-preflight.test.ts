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

describe("preflight Check 2: cardinality_high", () => {
  it("drops exact matchkey on near-unique column", () => {
    const rows = Array.from({ length: 100 }, (_, i) => ({
      id: String(i),
      name: "alice",
    }));
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk_id",
          type: "exact",
          fields: [{ field: "id", transforms: [], scorer: "exact", weight: 1 }],
        },
        {
          name: "mk_name",
          type: "weighted",
          fields: [{ field: "name", transforms: [], scorer: "exact", weight: 1 }],
          threshold: 0.7,
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["name"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
    };
    const { report, config: repaired } = preflight(rows, cfg);
    const names = (repaired.matchkeys ?? []).map((mk) => mk.name);
    expect(names).not.toContain("mk_id");
    expect(names).toContain("mk_name");
    const warning = report.findings.find((f) => f.check === "cardinality_high");
    expect(warning?.repaired).toBe(true);
  });
});

describe("preflight Check 3: cardinality_low", () => {
  it("drops exact matchkey on single-value column", () => {
    const rows = Array.from({ length: 100 }, (_, i) => ({
      state: "NC",
      last_name: `name${i}`,
    }));
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk_state",
          type: "exact",
          fields: [{ field: "state", transforms: [], scorer: "exact", weight: 1 }],
        },
        {
          name: "mk_name",
          type: "weighted",
          fields: [
            { field: "last_name", transforms: [], scorer: "token_sort", weight: 1 },
          ],
          threshold: 0.7,
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["last_name"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
    };
    const { report, config: repaired } = preflight(rows, cfg);
    const names = (repaired.matchkeys ?? []).map((mk) => mk.name);
    expect(names).not.toContain("mk_state");
    const warning = report.findings.find((f) => f.check === "cardinality_low");
    expect(warning?.repaired).toBe(true);
  });

  it("skips cardinality checks when no matchkeys remain before check", () => {
    // Just sanity — empty matchkeys shouldn't crash.
    const rows = [{ a: 1 }];
    const cfg: GoldenMatchConfig = { matchkeys: [] };
    const { report } = preflight(rows, cfg);
    expect(
      report.findings.some((f) => f.check === "no_matchkeys_remain"),
    ).toBe(false);
  });

  it("emits no_matchkeys_remain if drops empty the list", () => {
    const rows = Array.from({ length: 100 }, (_, i) => ({ id: String(i) }));
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk_id",
          type: "exact",
          fields: [{ field: "id", transforms: [], scorer: "exact", weight: 1 }],
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["id"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
    };
    const { report } = preflight(rows, cfg);
    expect(report.hasErrors).toBe(true);
    expect(
      report.findings.some((f) => f.check === "no_matchkeys_remain"),
    ).toBe(true);
  });
});

describe("preflight Check 4: block_size", () => {
  it("warns when blocking produces a mega-block", () => {
    const rows = Array.from({ length: 10_000 }, (_, i) => ({
      state: "NC",
      last_name: `name${i}`,
    }));
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk",
          type: "weighted",
          fields: [
            { field: "last_name", transforms: [], scorer: "token_sort", weight: 1 },
          ],
          threshold: 0.7,
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["state"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
    };
    const { report } = preflight(rows, cfg);
    const warning = report.findings.find((f) => f.check === "block_size");
    expect(warning).toBeDefined();
    expect(warning?.repaired).toBe(false);
  });

  it("does not warn when blocks look healthy", () => {
    const rows = Array.from({ length: 200 }, (_, i) => ({
      state: ["NC", "SC", "VA", "NY"][i % 4]!,
      last_name: `name${i}`,
    }));
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk",
          type: "weighted",
          fields: [
            { field: "last_name", transforms: [], scorer: "token_sort", weight: 1 },
          ],
          threshold: 0.7,
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["state"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
    };
    const { report } = preflight(rows, cfg);
    expect(report.findings.some((f) => f.check === "block_size")).toBe(false);
  });
});
