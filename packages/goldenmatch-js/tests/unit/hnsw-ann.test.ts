import { describe, it, expect, vi } from "vitest";
import {
  ANNBlocker,
  HNSWANNBlocker,
  createANNBlocker,
  type HNSWModule,
  type HNSWIndexLike,
} from "../../src/core/index.js";

// ---------------------------------------------------------------------------
// Mock hnswlib-node
// ---------------------------------------------------------------------------

/**
 * Minimal in-memory stand-in for `hnswlib.HierarchicalNSW`. It records every
 * init/add/search call so assertions can verify the fast-path is being used,
 * and returns deterministic neighbours so pair assertions are stable.
 */
class MockIndex implements HNSWIndexLike {
  public readonly points: Array<{ label: number; vec: number[] }> = [];
  public initArgs: number[] | null = null;
  public ef: number | null = null;
  public searchCalls = 0;

  constructor(public metric: string, public dim: number) {}

  initIndex(
    maxElements: number,
    M?: number,
    efConstruction?: number,
    randomSeed?: number,
  ): void {
    this.initArgs = [maxElements, M ?? 16, efConstruction ?? 200, randomSeed ?? 100];
  }

  setEf(ef: number): void {
    this.ef = ef;
  }

  addPoint(vector: number[] | Float32Array, labelId: number): void {
    this.points.push({ label: labelId, vec: Array.from(vector) });
  }

  searchKnn(
    _query: number[] | Float32Array,
    k: number,
  ): { distances: number[]; neighbors: number[] } {
    this.searchCalls++;
    const n = Math.min(k, this.points.length);
    // Return the first n labels with small increasing distances. This makes
    // the test deterministic without having to actually compute distances.
    const neighbors: number[] = [];
    const distances: number[] = [];
    for (let i = 0; i < n; i++) {
      neighbors.push(this.points[i]!.label);
      distances.push(i * 0.1);
    }
    return { neighbors, distances };
  }
}

function makeMockModule(): {
  module: HNSWModule;
  instances: MockIndex[];
} {
  const instances: MockIndex[] = [];
  const module: HNSWModule = {
    HierarchicalNSW: class extends MockIndex {
      constructor(metric: string, dim: number) {
        super(metric, dim);
        instances.push(this);
      }
    } as unknown as HNSWModule["HierarchicalNSW"],
  };
  return { module, instances };
}

function vec(...nums: number[]): Float32Array {
  return new Float32Array(nums);
}

// ---------------------------------------------------------------------------
// HNSWANNBlocker — direct unit tests
// ---------------------------------------------------------------------------

describe("HNSWANNBlocker", () => {
  it("buildIndex initialises the native index and adds every point", () => {
    const { module, instances } = makeMockModule();
    const blocker = new HNSWANNBlocker({
      hnswModule: module,
      topK: 3,
      metric: "cosine",
      M: 8,
      efConstruction: 100,
      efSearch: 42,
    });
    blocker.buildIndex([vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)]);
    expect(instances).toHaveLength(1);
    const idx = instances[0]!;
    expect(idx.metric).toBe("cosine");
    expect(idx.dim).toBe(3);
    expect(idx.initArgs?.[1]).toBe(8); // M
    expect(idx.initArgs?.[2]).toBe(100); // efConstruction
    expect(idx.ef).toBe(42);
    expect(idx.points.map((p) => p.label)).toEqual([0, 1, 2]);
    expect(blocker.indexSize).toBe(3);
  });

  it("buildIndex with empty input clears the index", () => {
    const { module } = makeMockModule();
    const blocker = new HNSWANNBlocker({ hnswModule: module, topK: 3 });
    blocker.buildIndex([]);
    expect(blocker.indexSize).toBe(0);
    expect(blocker.query([vec(1, 2, 3)])).toEqual([]);
  });

  it("maps 'euclidean' option to the 'l2' metric string", () => {
    const { module, instances } = makeMockModule();
    const blocker = new HNSWANNBlocker({
      hnswModule: module,
      metric: "euclidean",
    });
    blocker.buildIndex([vec(1, 0), vec(0, 1)]);
    expect(instances[0]!.metric).toBe("l2");
  });

  it("query canonicalises pairs and drops self-matches", () => {
    const { module } = makeMockModule();
    const blocker = new HNSWANNBlocker({
      hnswModule: module,
      topK: 3,
    });
    blocker.buildIndex([vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)]);
    const pairs = blocker.query([vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)]);
    // MockIndex returns labels [0,1,2] for every query — after self-filter +
    // canonicalisation we should get {0,1}, {0,2}, {1,2}.
    const sorted = pairs.map(([a, b]) => `${a}-${b}`).sort();
    expect(sorted).toEqual(["0-1", "0-2", "1-2"]);
  });

  it("queryWithScores converts cosine distance to similarity = 1 - d", () => {
    const { module } = makeMockModule();
    const blocker = new HNSWANNBlocker({
      hnswModule: module,
      topK: 3,
      metric: "cosine",
    });
    blocker.buildIndex([vec(1, 0), vec(0, 1), vec(1, 1)]);
    const scored = blocker.queryWithScores([vec(1, 0), vec(0, 1), vec(1, 1)]);
    // Distances produced by MockIndex: 0, 0.1, 0.2 → similarities 1.0, 0.9, 0.8
    // After dedup (best score kept per pair) all three unique pairs survive.
    expect(scored).toHaveLength(3);
    for (const [, , score] of scored) {
      expect(score).toBeGreaterThanOrEqual(0.8 - 1e-6);
      expect(score).toBeLessThanOrEqual(1.0 + 1e-6);
    }
  });

  it("queryWithScores converts l2 distance to 1 / (1 + d)", () => {
    const { module } = makeMockModule();
    const blocker = new HNSWANNBlocker({
      hnswModule: module,
      topK: 2,
      metric: "euclidean",
    });
    blocker.buildIndex([vec(0, 0), vec(1, 1)]);
    const scored = blocker.queryWithScores([vec(0, 0), vec(1, 1)]);
    // Best distance across the two queries for pair (0,1) is 0 → score 1.0.
    expect(scored).toHaveLength(1);
    const [, , score] = scored[0]!;
    expect(score).toBeCloseTo(1.0, 6);
  });

  it("queryOne returns (neighbour, score) pairs", () => {
    const { module } = makeMockModule();
    const blocker = new HNSWANNBlocker({
      hnswModule: module,
      topK: 2,
      metric: "cosine",
    });
    blocker.buildIndex([vec(1, 0), vec(0, 1), vec(1, 1)]);
    const out = blocker.queryOne(vec(0.5, 0.5));
    expect(out).toHaveLength(2);
    // First mock neighbour has distance 0 → similarity 1.0.
    expect(out[0]![1]).toBeCloseTo(1.0, 6);
  });

  it("addToIndex grows the index and returns sequential ids", () => {
    const { module, instances } = makeMockModule();
    const blocker = new HNSWANNBlocker({ hnswModule: module });
    blocker.buildIndex([vec(1, 0), vec(0, 1)]);
    const id1 = blocker.addToIndex(vec(1, 1));
    const id2 = blocker.addToIndex(vec(0.5, 0.5));
    expect(id1).toBe(2);
    expect(id2).toBe(3);
    expect(blocker.indexSize).toBe(4);
    expect(instances[0]!.points).toHaveLength(4);
  });

  it("addToIndex before buildIndex throws", () => {
    const { module } = makeMockModule();
    const blocker = new HNSWANNBlocker({ hnswModule: module });
    expect(() => blocker.addToIndex(vec(1, 2))).toThrow(/buildIndex/);
  });
});

// ---------------------------------------------------------------------------
// createANNBlocker factory
// ---------------------------------------------------------------------------

describe("createANNBlocker", () => {
  it("returns a brute-force ANNBlocker when useHNSW=false", async () => {
    const blocker = await createANNBlocker({ useHNSW: false, topK: 5 });
    expect(blocker).toBeInstanceOf(ANNBlocker);
  });

  it("returns a brute-force ANNBlocker with no options", async () => {
    const blocker = await createANNBlocker();
    expect(blocker).toBeInstanceOf(ANNBlocker);
  });

  it("uses the provided hnswModule when useHNSW=true", async () => {
    const { module, instances } = makeMockModule();
    const blocker = await createANNBlocker({
      useHNSW: true,
      hnswModule: module,
      topK: 4,
      metric: "cosine",
    });
    expect(blocker).toBeInstanceOf(HNSWANNBlocker);
    blocker.buildIndex([vec(1, 0), vec(0, 1)]);
    expect(instances).toHaveLength(1);
  });

  it("falls back to brute-force when hnswlib-node is not installed", async () => {
    const warnings: string[] = [];
    const blocker = await createANNBlocker({
      useHNSW: true,
      onFallbackWarning: (m) => warnings.push(m),
    });
    // hnswlib-node is not in devDependencies, so the dynamic import fails.
    expect(blocker).toBeInstanceOf(ANNBlocker);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/hnswlib-node/);
  });

  it("forwards tuning knobs to HNSWANNBlocker", async () => {
    const { module, instances } = makeMockModule();
    const blocker = await createANNBlocker({
      useHNSW: true,
      hnswModule: module,
      topK: 7,
      metric: "euclidean",
      M: 24,
      efConstruction: 300,
      efSearch: 77,
      maxElements: 500,
    });
    expect(blocker).toBeInstanceOf(HNSWANNBlocker);
    blocker.buildIndex([vec(1, 2), vec(3, 4)]);
    const idx = instances[0]!;
    expect(idx.metric).toBe("l2");
    expect(idx.initArgs?.[0]).toBe(500); // maxElements
    expect(idx.initArgs?.[1]).toBe(24); // M
    expect(idx.initArgs?.[2]).toBe(300); // efConstruction
    expect(idx.ef).toBe(77);
  });

  it("silences the warn sink via onFallbackWarning", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      await createANNBlocker({
        useHNSW: true,
        onFallbackWarning: () => {
          // swallow
        },
      });
      expect(warnSpy).not.toHaveBeenCalled();
    } finally {
      warnSpy.mockRestore();
    }
  });
});
