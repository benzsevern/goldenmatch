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
