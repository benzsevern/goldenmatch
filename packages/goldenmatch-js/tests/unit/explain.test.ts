import { describe, it, expect } from "vitest";
import { explainPair, explainCluster, buildClusters, pairKey } from "../../src/core/index.js";
import type { ClusterInfo, MatchkeyConfig, Row } from "../../src/core/index.js";

const MK: MatchkeyConfig = {
  name: "name_mk",
  type: "weighted",
  threshold: 0.8,
  fields: [
    { field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 },
    { field: "city", transforms: [], scorer: "exact", weight: 1.0 },
  ],
};

describe("explainPair", () => {
  it("produces reasoning strings and overall score", () => {
    const rowA: Row = { name: "John Smith", city: "NYC" };
    const rowB: Row = { name: "John Smith", city: "NYC" };
    const exp = explainPair(rowA, rowB, MK);
    expect(exp.score).toBe(1.0);
    expect(exp.reasoning.length).toBeGreaterThan(0);
    expect(exp.confidence).toBe("high");
    expect(typeof exp.explanation).toBe("string");
    expect(exp.explanation.length).toBeGreaterThan(0);
  });

  it("includes fieldScores for each field", () => {
    const rowA: Row = { name: "John", city: "NYC" };
    const rowB: Row = { name: "Jon", city: "NYC" };
    const exp = explainPair(rowA, rowB, MK);
    expect(exp.fieldScores.name).not.toBe(null);
    expect(exp.fieldScores.city).toBe(1.0);
  });

  it("low-confidence when overall score is low", () => {
    const rowA: Row = { name: "Alice", city: "NYC" };
    const rowB: Row = { name: "Zeke", city: "LA" };
    const exp = explainPair(rowA, rowB, MK);
    expect(exp.confidence).toBe("low");
  });
});

describe("explainCluster", () => {
  it("summarizes multi-member cluster", () => {
    const pairs: [number, number, number][] = [[1, 2, 0.9], [2, 3, 0.85], [1, 3, 0.88]];
    const clusters = buildClusters(pairs, [1, 2, 3]);
    const cid = [...clusters.keys()][0]!;
    const cinfo = clusters.get(cid)!;

    const rows: Row[] = [
      { __row_id__: 1, name: "A", city: "X" },
      { __row_id__: 2, name: "A", city: "X" },
      { __row_id__: 3, name: "A", city: "X" },
    ];
    const exp = explainCluster(cid, cinfo, rows, MK);
    expect(exp.size).toBe(3);
    expect(exp.summary).toContain("Cluster of 3");
  });

  it("singleton cluster has specialized summary", () => {
    const cinfo: ClusterInfo = {
      members: [7],
      size: 1,
      oversized: false,
      pairScores: new Map(),
      confidence: 1.0,
      bottleneckPair: null,
      clusterQuality: "strong",
    };
    const exp = explainCluster(7, cinfo, [{ __row_id__: 7 }], MK);
    expect(exp.summary).toContain("Singleton");
  });
});
