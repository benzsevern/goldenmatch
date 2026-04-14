import { describe, it, expect } from "vitest";
import {
  ANNBlocker,
  cosineSim,
  euclideanDist,
} from "../../src/core/index.js";

function vec(...nums: number[]): Float32Array {
  return new Float32Array(nums);
}

describe("cosineSim", () => {
  it("orthogonal vectors -> 0", () => {
    expect(cosineSim(vec(1, 0, 0), vec(0, 1, 0))).toBe(0);
  });

  it("identical vectors -> 1", () => {
    expect(cosineSim(vec(1, 0, 0), vec(1, 0, 0))).toBeCloseTo(1, 6);
  });

  it("opposite vectors -> -1", () => {
    expect(cosineSim(vec(1, 0, 0), vec(-1, 0, 0))).toBeCloseTo(-1, 6);
  });

  it("zero vector -> 0 (no NaN)", () => {
    expect(cosineSim(vec(0, 0, 0), vec(1, 2, 3))).toBe(0);
  });
});

describe("euclideanDist", () => {
  it("3-4-5 Pythagorean triple", () => {
    expect(euclideanDist(vec(0, 0), vec(3, 4))).toBeCloseTo(5, 6);
  });

  it("identical -> 0", () => {
    expect(euclideanDist(vec(1, 2, 3), vec(1, 2, 3))).toBe(0);
  });

  it("5-12-13 triple", () => {
    expect(euclideanDist(vec(0, 0), vec(5, 12))).toBeCloseTo(13, 6);
  });
});

describe("ANNBlocker", () => {
  it("buildIndex then query returns top-K", () => {
    const blocker = new ANNBlocker({ topK: 2 });
    const embeddings = [
      vec(1, 0, 0),
      vec(0.99, 0.01, 0), // close to 0
      vec(0, 1, 0),
      vec(0, 0.98, 0.02), // close to 2
    ];
    blocker.buildIndex(embeddings);
    const pairs = blocker.query(embeddings);
    // Should pair (0,1) and (2,3) at minimum
    const pairKeys = new Set(pairs.map((p) => `${p[0]}-${p[1]}`));
    expect(pairKeys.has("0-1")).toBe(true);
    expect(pairKeys.has("2-3")).toBe(true);
  });

  it("indexSize reflects buildIndex", () => {
    const blocker = new ANNBlocker();
    expect(blocker.indexSize).toBe(0);
    blocker.buildIndex([vec(1, 0), vec(0, 1)]);
    expect(blocker.indexSize).toBe(2);
  });

  it("addToIndex grows index and returns position", () => {
    const blocker = new ANNBlocker();
    blocker.buildIndex([vec(1, 0)]);
    expect(blocker.indexSize).toBe(1);
    const pos = blocker.addToIndex(vec(0, 1));
    expect(pos).toBe(1);
    expect(blocker.indexSize).toBe(2);
  });

  it("queryWithScores returns [a, b, score] tuples", () => {
    const blocker = new ANNBlocker({ topK: 2 });
    const embeddings = [vec(1, 0), vec(0.99, 0.01), vec(0, 1)];
    blocker.buildIndex(embeddings);
    // Use the same array reference to enable self-pair filtering.
    const scored = blocker.queryWithScores(
      embeddings as unknown as readonly Float32Array[],
    );
    expect(scored.length).toBeGreaterThan(0);
    for (const [a, b, score] of scored) {
      expect(typeof a).toBe("number");
      expect(typeof b).toBe("number");
      expect(typeof score).toBe("number");
      // Pairs are canonicalized, so a <= b. Self-pairs are excluded when
      // the same array reference is used as both index and queries.
      expect(a).toBeLessThanOrEqual(b);
    }
  });

  it("queryOne returns top-K neighbors with scores", () => {
    const blocker = new ANNBlocker({ topK: 2 });
    blocker.buildIndex([vec(1, 0), vec(0, 1), vec(0.9, 0.1)]);
    const top = blocker.queryOne(vec(1, 0));
    expect(top.length).toBe(2);
    // Best match should be index 0 (cosine=1)
    expect(top[0]![0]).toBe(0);
  });

  it("query on empty index -> empty", () => {
    const blocker = new ANNBlocker();
    blocker.buildIndex([]);
    expect(blocker.query([vec(1, 0)])).toEqual([]);
  });

  it("euclidean metric still ranks closest first", () => {
    const blocker = new ANNBlocker({ topK: 1, metric: "euclidean" });
    blocker.buildIndex([vec(0, 0), vec(10, 10), vec(0.1, 0.1)]);
    const top = blocker.queryOne(vec(0, 0));
    expect(top[0]![0]).toBe(0); // self is closest
  });
});
