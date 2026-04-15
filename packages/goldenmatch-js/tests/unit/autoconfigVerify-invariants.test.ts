/**
 * Property-based invariants for the autoconfig verification layer.
 *
 * Covers structural invariants that must hold across many random inputs:
 *   - histogram shape (bins/counts sizes, sum of counts)
 *   - percentile ordering (p50 <= p95 <= p99 <= max) for blocks & clusters
 *   - thresholdOverlapPct is a probability in [0, 1]
 *   - strict mode disables all threshold adjustments
 *   - oversized cluster bottleneckPair is always a real input edge
 *
 * Deterministic across runs via seeded LCG; 10 seeds per invariant.
 */
import { describe, it, expect } from "vitest";
import { postflight } from "../../src/core/autoconfigVerify.js";
import type { GoldenMatchConfig } from "../../src/core/types.js";

// ---------------------------------------------------------------------------
// Helpers (mirror patterns in autoconfigVerify-postflight.test.ts)
// ---------------------------------------------------------------------------

function makeCfg(
  overrides: Partial<GoldenMatchConfig> = {},
): GoldenMatchConfig {
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

function seeded(s: number): () => number {
  let x = s || 1;
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

/**
 * Bimodal mixture of pair scores — one cluster around 0.2, one around 0.9.
 * Useful for exercising percentile + histogram invariants under realistic
 * distributions (not just uniform noise).
 */
function bimodalPairs(
  rng: () => number,
  n: number,
): { idA: number; idB: number; score: number }[] {
  const out: { idA: number; idB: number; score: number }[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    out.push({
      idA: i,
      idB: i + 10_000,
      score: Math.min(1, Math.max(0, gauss(rng, 0.2, 0.06))),
    });
  }
  for (let i = 0; i < n - half; i++) {
    out.push({
      idA: i + 20_000,
      idB: i + 30_000,
      score: Math.min(1, Math.max(0, gauss(rng, 0.9, 0.05))),
    });
  }
  return out;
}

const SEEDS = [1, 7, 13, 42, 101, 256, 1337, 4096, 9001, 65_535];

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

describe("postflight invariants: histogram shape", () => {
  for (const seed of SEEDS) {
    it(`seed ${seed}: bins.length===101, counts.length===100, sum(counts)===totalPairsScored`, () => {
      const r = seeded(seed);
      const pairs = bimodalPairs(r, 400 + Math.floor(r() * 400));
      const rows = Array.from({ length: 20 }, (_, i) => ({ name: `n${i}` }));
      const report = postflight(rows, makeCfg(), { pairScores: pairs });
      const hist = report.signals.scoreHistogram;
      expect(hist.bins.length).toBe(101);
      expect(hist.counts.length).toBe(100);
      const sum = hist.counts.reduce((a, b) => a + b, 0);
      expect(sum).toBe(report.signals.totalPairsScored);
    });
  }
});

describe("postflight invariants: percentile ordering", () => {
  for (const seed of SEEDS) {
    it(`seed ${seed}: blockSizePercentiles p50 <= p95 <= p99 <= max`, () => {
      const r = seeded(seed);
      const pairs = bimodalPairs(r, 200);
      const rows = Array.from({ length: 50 }, (_, i) => ({ name: `n${i}` }));
      const report = postflight(rows, makeCfg(), { pairScores: pairs });
      const p = report.signals.blockSizePercentiles;
      expect(p.p50).toBeLessThanOrEqual(p.p95);
      expect(p.p95).toBeLessThanOrEqual(p.p99);
      expect(p.p99).toBeLessThanOrEqual(p.max);
    });

    it(`seed ${seed}: preliminaryClusterSizes p50 <= p95 <= p99 <= max`, () => {
      const r = seeded(seed);
      // Chain graph so clustering produces a non-trivial distribution.
      const chainLen = 20 + Math.floor(r() * 30);
      const pairs = Array.from({ length: chainLen }, (_, i) => ({
        idA: i,
        idB: i + 1,
        score: 0.9,
      }));
      const rows = Array.from({ length: chainLen + 5 }, (_, i) => ({
        name: `n${i}`,
      }));
      const report = postflight(rows, makeCfg(), { pairScores: pairs });
      const c = report.signals.preliminaryClusterSizes;
      expect(c.p50).toBeLessThanOrEqual(c.p95);
      expect(c.p95).toBeLessThanOrEqual(c.p99);
      expect(c.p99).toBeLessThanOrEqual(c.max);
      expect(c.count).toBeGreaterThan(0);
    });
  }
});

describe("postflight invariants: thresholdOverlapPct in [0, 1]", () => {
  for (const seed of SEEDS) {
    it(`seed ${seed}: thresholdOverlapPct bounded`, () => {
      const r = seeded(seed);
      const pairs = bimodalPairs(r, 300);
      const rows = Array.from({ length: 30 }, (_, i) => ({ name: `n${i}` }));
      const report = postflight(rows, makeCfg(), { pairScores: pairs });
      const t = report.signals.thresholdOverlapPct;
      expect(t).toBeGreaterThanOrEqual(0);
      expect(t).toBeLessThanOrEqual(1);
    });
  }
});

describe("postflight invariants: strict mode disables adjustments", () => {
  for (const seed of SEEDS) {
    it(`seed ${seed}: adjustments.length === 0 under _strictAutoconfig`, () => {
      const r = seeded(seed);
      // Bimodal input would normally trigger a threshold adjustment.
      const pairs = bimodalPairs(r, 1000);
      const rows = Array.from({ length: 30 }, (_, i) => ({ name: `n${i}` }));
      const cfg = makeCfg({ _strictAutoconfig: true });
      const report = postflight(rows, cfg, { pairScores: pairs });
      expect(report.adjustments.length).toBe(0);
    });
  }
});

describe("postflight invariants: oversized bottleneckPair is a real edge", () => {
  const canonKey = (a: number, b: number): string =>
    a < b ? `${a}-${b}` : `${b}-${a}`;

  for (const seed of SEEDS) {
    it(`seed ${seed}: every oversizedCluster.bottleneckPair is in input`, () => {
      const r = seeded(seed);
      // Build a chain longer than 150 so at least one oversized cluster
      // emerges (default threshold is typically 150).
      const chainLen = 160 + Math.floor(r() * 20);
      const pairs = Array.from({ length: chainLen }, (_, i) => ({
        idA: i,
        idB: i + 1,
        score: 0.9,
      }));
      const rows = Array.from({ length: chainLen + 5 }, (_, i) => ({
        name: `n${i}`,
      }));
      const report = postflight(rows, makeCfg(), { pairScores: pairs });
      const inputKeys = new Set(pairs.map((p) => canonKey(p.idA, p.idB)));
      for (const ov of report.signals.oversizedClusters) {
        const bp = ov.bottleneckPair;
        expect(bp.length).toBe(2);
        expect(
          inputKeys.has(canonKey(bp[0]!, bp[1]!)),
          `bottleneck ${bp[0]}-${bp[1]} must be a real input edge`,
        ).toBe(true);
      }
    });
  }
});
