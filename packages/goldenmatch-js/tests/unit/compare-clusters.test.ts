import { describe, it, expect } from "vitest";
import { compareClusters, buildClusters } from "../../src/core/index.js";

describe("compareClusters (CCMS)", () => {
  it("identical clustering -> all unchanged, TWI = 1", () => {
    const pairs: [number, number, number][] = [[1, 2, 0.9], [3, 4, 0.9]];
    const a = buildClusters(pairs, [1, 2, 3, 4]);
    const b = buildClusters(pairs, [1, 2, 3, 4]);
    const result = compareClusters(a, b);
    expect(result.unchanged).toBe(a.size);
    expect(result.merged).toBe(0);
    expect(result.partitioned).toBe(0);
    expect(result.overlapping).toBe(0);
    expect(result.twi).toBeCloseTo(1.0, 5);
  });

  it("different clustering produces non-zero classifications", () => {
    // A has {1,2,3}, B has {1,2},{3}
    const pairsA: [number, number, number][] = [[1, 2, 0.9], [2, 3, 0.9]];
    const pairsB: [number, number, number][] = [[1, 2, 0.9]];
    const a = buildClusters(pairsA, [1, 2, 3]);
    const b = buildClusters(pairsB, [1, 2, 3]);
    const result = compareClusters(a, b);
    // A cluster {1,2,3} is partitioned in B
    expect(result.partitioned).toBeGreaterThanOrEqual(1);
  });

  it("throws if row id coverage differs", () => {
    const a = buildClusters([], [1, 2]);
    const b = buildClusters([], [1, 2, 3]);
    expect(() => compareClusters(a, b)).toThrow();
  });

  it("returns cc1, cc2, rc metadata", () => {
    const a = buildClusters([[1, 2, 0.9]], [1, 2, 3]);
    const b = buildClusters([[1, 2, 0.9]], [1, 2, 3]);
    const result = compareClusters(a, b);
    expect(result.cc1).toBe(a.size);
    expect(result.cc2).toBe(b.size);
    expect(result.rc).toBe(3);
  });
});
