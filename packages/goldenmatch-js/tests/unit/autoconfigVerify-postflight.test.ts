import { describe, it, expect } from "vitest";
import { postflight } from "../../src/core/autoconfigVerify.js";
import type { GoldenMatchConfig } from "../../src/core/types.js";

function makeCfg(overrides: Partial<GoldenMatchConfig> = {}): GoldenMatchConfig {
  return {
    matchkeys: [
      {
        name: "mk",
        type: "weighted",
        fields: [
          { field: "name", transforms: [], scorer: "token_sort", weight: 1 },
        ],
        threshold: 0.7,
      },
    ],
    blocking: {
      strategy: "static",
      keys: [{ fields: ["name"], transforms: [] }],
      maxBlockSize: 1000,
      skipOversized: true,
    },
    ...overrides,
  };
}

// Seeded LCG + Box-Muller for deterministic gaussian-ish distributions.
function seeded(s: number): () => number {
  let x = s;
  return () => {
    x = (x * 1103515245 + 12345) & 0x7fffffff;
    return x / 0x7fffffff;
  };
}
function gauss(rng: () => number, mu: number, sigma: number): number {
  const u = Math.max(1e-9, rng());
  const v = rng();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

describe("postflight: score histogram", () => {
  it("bimodal distribution triggers threshold adjustment", () => {
    const r = seeded(42);
    const pairScores: { idA: number; idB: number; score: number }[] = [];
    for (let i = 0; i < 500; i++) {
      pairScores.push({
        idA: i, idB: i + 1000,
        score: Math.min(1, Math.max(0, gauss(r, 0.2, 0.05))),
      });
    }
    for (let i = 0; i < 500; i++) {
      pairScores.push({
        idA: i + 2000, idB: i + 3000,
        score: Math.min(1, Math.max(0, gauss(r, 0.9, 0.05))),
      });
    }
    const rows = Array.from({ length: 50 }, (_, i) => ({ name: `x${i}` }));
    const report = postflight(rows, makeCfg(), { pairScores });
    expect(report.adjustments.some((a) => a.field === "threshold")).toBe(true);
    const hist = report.signals.scoreHistogram;
    expect(hist.bins.length).toBe(101);
    expect(hist.counts.length).toBe(100);
  });

  it("unimodal distribution emits no adjustment", () => {
    const r = seeded(42);
    const pairScores = Array.from({ length: 1000 }, (_, i) => ({
      idA: i, idB: i + 1, score: r(),
    }));
    const rows = Array.from({ length: 20 }, (_, i) => ({ name: `x${i}` }));
    const report = postflight(rows, makeCfg(), { pairScores });
    expect(report.adjustments.some((a) => a.field === "threshold")).toBe(false);
  });

  it("returns 'deferred' sentinel for blockingRecall", () => {
    const rows = Array.from({ length: 500 }, (_, i) => ({ name: `x${i}` }));
    const pairScores = [{ idA: 0, idB: 1, score: 0.9 }];
    const report = postflight(rows, makeCfg(), { pairScores });
    expect(report.signals.blockingRecall).toBe("deferred");
  });

  it("detects oversized cluster of 151 from a pair chain", () => {
    // Chain: 0-1, 1-2, ..., 149-150 → one component of 151
    const pairScores = Array.from({ length: 150 }, (_, i) => ({
      idA: i, idB: i + 1, score: 0.9,
    }));
    const rows = Array.from({ length: 200 }, (_, i) => ({ name: `n${i}` }));
    const report = postflight(rows, makeCfg(), { pairScores });
    const oversized = report.signals.oversizedClusters;
    expect(oversized.length).toBe(1);
    expect(oversized[0]!.size).toBe(151);
    expect(oversized[0]!.bottleneckPair.length).toBe(2);
    const bp = oversized[0]!.bottleneckPair;
    // bottleneckPair must be a real edge in the input (canonicalized)
    const canonKey = (a: number, b: number) =>
      a < b ? `${a}-${b}` : `${b}-${a}`;
    const inputKeys = new Set(
      pairScores.map((p) => canonKey(p.idA, p.idB)),
    );
    expect(inputKeys.has(canonKey(bp[0]!, bp[1]!))).toBe(true);
  });

  it("emits llm advisory when >20% of pairs in threshold band and llm disabled", () => {
    const inBand = Array.from({ length: 300 }, (_, i) => ({
      idA: i, idB: i + 10000, score: 0.69,
    }));
    const outBand = Array.from({ length: 700 }, (_, i) => ({
      idA: i + 20000, idB: i + 30000, score: 0.2,
    }));
    const rows = Array.from({ length: 50 }, (_, i) => ({ name: `x${i}` }));
    const report = postflight(rows, makeCfg(), {
      pairScores: [...inBand, ...outBand],
    });
    expect(report.signals.thresholdOverlapPct).toBeGreaterThan(0.2);
    expect(
      report.advisories.some((a) => a.toLowerCase().includes("llm")),
    ).toBe(true);
  });

  it("signals has exactly the 8 documented keys", () => {
    const rows = Array.from({ length: 50 }, (_, i) => ({ a: String(i) }));
    const cfg: GoldenMatchConfig = {
      matchkeys: [
        {
          name: "mk",
          type: "weighted",
          fields: [
            { field: "a", transforms: [], scorer: "exact", weight: 1 },
          ],
          threshold: 0.7,
        },
      ],
      blocking: {
        strategy: "static",
        keys: [{ fields: ["a"], transforms: [] }],
        maxBlockSize: 1000,
        skipOversized: true,
      },
    };
    const pairScores = Array.from({ length: 48 }, (_, i) => ({
      idA: i, idB: i + 1, score: 0.8,
    }));
    const report = postflight(rows, cfg, { pairScores });
    const keys = new Set(Object.keys(report.signals));
    expect(keys).toEqual(new Set([
      "scoreHistogram", "blockingRecall", "blockSizePercentiles",
      "thresholdOverlapPct", "totalPairsScored", "currentThreshold",
      "preliminaryClusterSizes", "oversizedClusters",
    ]));
  });

  it("strict mode: signal computed, adjustments empty", () => {
    const r = seeded(42);
    const pairScores: { idA: number; idB: number; score: number }[] = [];
    for (let i = 0; i < 500; i++) {
      pairScores.push({
        idA: i, idB: i + 1000,
        score: Math.min(1, Math.max(0, gauss(r, 0.2, 0.05))),
      });
    }
    for (let i = 0; i < 500; i++) {
      pairScores.push({
        idA: i + 2000, idB: i + 3000,
        score: Math.min(1, Math.max(0, gauss(r, 0.9, 0.05))),
      });
    }
    const rows = Array.from({ length: 50 }, (_, i) => ({ name: `x${i}` }));
    const cfg = { ...makeCfg(), _strictAutoconfig: true };
    const report = postflight(rows, cfg, { pairScores });
    expect(report.adjustments).toHaveLength(0);
    expect(report.signals.scoreHistogram.bins.length).toBe(101);
  });
});

describe("postflight pipeline integration: dedupe", () => {
  it("auto-configured dedupe attaches postflightReport", async () => {
    const { dedupe } = await import("../../src/core/api.js");
    const { autoConfigureRows } = await import("../../src/core/autoconfig.js");
    const rows = Array.from({ length: 50 }, (_, i) => ({
      name: `alice ${i}`,
      zip: "90210",
    }));
    const cfg = autoConfigureRows(rows);
    const result = dedupe(rows, { config: cfg });
    expect(result.postflightReport).toBeDefined();
    expect(result.postflightReport!.signals.scoreHistogram.bins.length).toBe(101);
  });

  it("hand-written config (non-autoconfig) has undefined postflightReport", async () => {
    const { dedupe } = await import("../../src/core/api.js");
    const rows = Array.from({ length: 10 }, (_, i) => ({ name: `x${i}` }));
    // Hand-written shorthand: the shorthand path does NOT call
    // autoConfigureRows, so config._preflightReport is undefined and the
    // pipeline guard short-circuits. This documents the contract: a
    // PostflightReport is only attached when the config went through the
    // auto-config preflight layer.
    const result = dedupe(rows, { fuzzy: { name: 0.7 } });
    expect(result.postflightReport).toBeUndefined();
  });
});

describe("postflight pipeline integration: strict mode", () => {
  it("strict mode: postflightReport present, adjustments empty", async () => {
    const { autoConfigureRows } = await import("../../src/core/autoconfig.js");
    const { dedupe } = await import("../../src/core/api.js");
    const rows = Array.from({ length: 100 }, (_, i) => ({
      name: i < 50 ? `bob ${i}` : `bob ${i}x`,
    }));
    const cfg = autoConfigureRows(rows, { strict: true });
    const result = dedupe(rows, { config: cfg });
    expect(result.postflightReport).toBeDefined();
    expect(result.postflightReport!.adjustments).toHaveLength(0);
  });
});

describe("postflight pipeline integration: match", () => {
  it("auto-configured match attaches postflightReport", async () => {
    const { match } = await import("../../src/core/api.js");
    const { autoConfigureRows } = await import("../../src/core/autoconfig.js");
    const target = [
      { name: "alice smith" },
      { name: "bob jones" },
      { name: "carol white" },
    ];
    const reference = [
      { name: "alice smyth" },
      { name: "bob jonesy" },
      { name: "carol white" },
      { name: "dan brown" },
    ];
    const cfg = autoConfigureRows([...target, ...reference]);
    const result = match(target, reference, { config: cfg });
    expect(result.postflightReport).toBeDefined();
    const sig = result.postflightReport!.signals;
    expect("scoreHistogram" in sig).toBe(true);
    expect(sig.totalPairsScored).toBeGreaterThanOrEqual(0);
  });
});
