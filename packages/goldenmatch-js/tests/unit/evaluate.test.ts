import { describe, it, expect } from "vitest";
import { evaluatePairs, evaluateClusters, buildClusters } from "../../src/core/index.js";
import type { ScoredPair } from "../../src/core/index.js";

describe("evaluatePairs", () => {
  it("perfect predictions -> precision=recall=f1=1.0", () => {
    const predicted: ScoredPair[] = [
      { idA: 1, idB: 2, score: 0.9 },
      { idA: 3, idB: 4, score: 0.9 },
    ];
    const truth: [number, number][] = [
      [1, 2],
      [3, 4],
    ];
    const result = evaluatePairs(predicted, truth);
    expect(result.precision).toBe(1);
    expect(result.recall).toBe(1);
    expect(result.f1).toBe(1);
    expect(result.truePositives).toBe(2);
    expect(result.falsePositives).toBe(0);
    expect(result.falseNegatives).toBe(0);
  });

  it("mix of TP, FP, FN", () => {
    const predicted: ScoredPair[] = [
      { idA: 1, idB: 2, score: 0.9 }, // TP
      { idA: 5, idB: 6, score: 0.9 }, // FP
    ];
    const truth: [number, number][] = [
      [1, 2], // in predicted -> TP
      [3, 4], // not predicted -> FN
    ];
    const result = evaluatePairs(predicted, truth);
    expect(result.truePositives).toBe(1);
    expect(result.falsePositives).toBe(1);
    expect(result.falseNegatives).toBe(1);
    expect(result.precision).toBe(0.5);
    expect(result.recall).toBe(0.5);
    expect(result.f1).toBe(0.5);
  });

  it("no predictions and no truth -> all zeros", () => {
    const result = evaluatePairs([], []);
    expect(result.precision).toBe(0);
    expect(result.recall).toBe(0);
    expect(result.f1).toBe(0);
  });

  it("canonicalizes pair ordering", () => {
    // predicted (2,1) should match truth (1,2)
    const predicted: ScoredPair[] = [{ idA: 2, idB: 1, score: 0.9 }];
    const truth: [number, number][] = [[1, 2]];
    const result = evaluatePairs(predicted, truth);
    expect(result.truePositives).toBe(1);
  });
});

describe("evaluateClusters", () => {
  it("expands clusters to pairs then evaluates", () => {
    // cluster {1,2,3} -> pairs (1,2),(1,3),(2,3)
    const pairs: [number, number, number][] = [
      [1, 2, 0.9],
      [2, 3, 0.9],
      [1, 3, 0.9],
    ];
    const clusters = buildClusters(pairs, [1, 2, 3]);
    const truth: [number, number][] = [
      [1, 2],
      [1, 3],
      [2, 3],
    ];
    const result = evaluateClusters(clusters, truth, [1, 2, 3]);
    expect(result.precision).toBe(1);
    expect(result.recall).toBe(1);
  });

  it("singleton clusters produce no pairs", () => {
    const clusters = buildClusters([], [1, 2, 3]);
    // No predicted pairs, but truth has (1,2) -> all false negatives
    const truth: [number, number][] = [[1, 2]];
    const result = evaluateClusters(clusters, truth, [1, 2, 3]);
    expect(result.truePositives).toBe(0);
    expect(result.falseNegatives).toBe(1);
  });
});
