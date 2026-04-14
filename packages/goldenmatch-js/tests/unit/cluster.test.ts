import { describe, it, expect } from "vitest";
import {
  UnionFind,
  buildClusters,
  computeClusterConfidence,
  unmergeRecord,
  unmergeCluster,
  addToCluster,
  pairKey,
} from "../../src/core/index.js";
import type { ClusterInfo } from "../../src/core/index.js";

describe("UnionFind", () => {
  it("add + find returns self", () => {
    const uf = new UnionFind();
    uf.add(1);
    expect(uf.find(1)).toBe(1);
  });

  it("union joins two elements", () => {
    const uf = new UnionFind();
    uf.add(1);
    uf.add(2);
    uf.union(1, 2);
    expect(uf.find(1)).toBe(uf.find(2));
  });

  it("getClusters returns grouping", () => {
    const uf = new UnionFind();
    uf.addMany([1, 2, 3, 4]);
    uf.union(1, 2);
    uf.union(3, 4);
    const clusters = uf.getClusters();
    expect(clusters.length).toBe(2);
    const sizes = clusters.map((c) => c.size).sort();
    expect(sizes).toEqual([2, 2]);
  });

  it("transitive closure", () => {
    const uf = new UnionFind();
    uf.addMany([1, 2, 3]);
    uf.union(1, 2);
    uf.union(2, 3);
    expect(uf.find(1)).toBe(uf.find(3));
  });
});

describe("buildClusters", () => {
  it("simple pairs produce expected clusters", () => {
    const pairs: [number, number, number][] = [
      [1, 2, 0.95],
      [3, 4, 0.9],
    ];
    const allIds = [1, 2, 3, 4, 5];
    const clusters = buildClusters(pairs, allIds);
    expect(clusters.size).toBe(3); // {1,2}, {3,4}, {5}
    const sizes = [...clusters.values()].map((c) => c.size).sort();
    expect(sizes).toEqual([1, 2, 2]);
  });

  it("cluster with only singletons", () => {
    const clusters = buildClusters([], [1, 2, 3]);
    expect(clusters.size).toBe(3);
    for (const c of clusters.values()) {
      expect(c.size).toBe(1);
    }
  });

  it("weak cluster detection downgrades confidence", () => {
    // Chain with weak link: edges (1,2)=0.95, (2,3)=0.95, (1,3)=0.3
    // avg - min = (0.95+0.95+0.3)/3 - 0.3 ~= 0.433, > 0.3 threshold
    const pairs: [number, number, number][] = [
      [1, 2, 0.95],
      [2, 3, 0.95],
      [1, 3, 0.3],
    ];
    const clusters = buildClusters(pairs, [1, 2, 3]);
    const single = [...clusters.values()][0]!;
    expect(single.clusterQuality).toBe("weak");
  });

  it("oversized cluster auto-splits", () => {
    // With maxClusterSize=2, 3 fully-connected nodes should split
    const pairs: [number, number, number][] = [
      [1, 2, 0.9],
      [2, 3, 0.5], // weakest
      [1, 3, 0.9],
    ];
    const clusters = buildClusters(pairs, [1, 2, 3], { maxClusterSize: 2 });
    // Should be split into 2+ clusters
    expect(clusters.size).toBeGreaterThan(1);
  });

  it("auto-split disabled leaves oversized", () => {
    const pairs: [number, number, number][] = [
      [1, 2, 0.9],
      [2, 3, 0.9],
      [1, 3, 0.9],
    ];
    const clusters = buildClusters(pairs, [1, 2, 3], {
      maxClusterSize: 2,
      autoSplit: false,
    });
    expect(clusters.size).toBe(1);
    const single = [...clusters.values()][0]!;
    expect(single.oversized).toBe(true);
  });
});

describe("computeClusterConfidence", () => {
  it("singleton confidence 1.0", () => {
    const conf = computeClusterConfidence(new Map(), 1);
    expect(conf.confidence).toBe(1.0);
    expect(conf.minEdge).toBe(null);
  });

  it("confidence formula: 0.4*min + 0.3*avg + 0.3*connectivity", () => {
    // One pair, size=2 — fully connected so connectivity=1.0
    const pairs = new Map<string, number>([[pairKey(1, 2), 0.8]]);
    const conf = computeClusterConfidence(pairs, 2);
    expect(conf.minEdge).toBe(0.8);
    expect(conf.avgEdge).toBe(0.8);
    expect(conf.connectivity).toBe(1.0);
    // 0.4*0.8 + 0.3*0.8 + 0.3*1.0 = 0.32 + 0.24 + 0.30 = 0.86
    expect(conf.confidence).toBeCloseTo(0.86, 5);
  });

  it("bottleneck pair is weakest edge", () => {
    const pairs = new Map<string, number>([
      [pairKey(1, 2), 0.9],
      [pairKey(2, 3), 0.5],
    ]);
    const conf = computeClusterConfidence(pairs, 3);
    expect(conf.bottleneckPair).toEqual([2, 3]);
  });
});

describe("unmergeRecord", () => {
  it("removes a record and makes it singleton", () => {
    const pairs: [number, number, number][] = [
      [1, 2, 0.95],
      [2, 3, 0.95],
      [1, 3, 0.95],
    ];
    const clusters = buildClusters(pairs, [1, 2, 3]);
    // Cluster has {1,2,3}
    expect(clusters.size).toBe(1);

    const updated = unmergeRecord(1, clusters);
    // Now record 1 is a singleton; 2,3 may still be together
    const allMembers: number[] = [];
    for (const c of updated.values()) {
      allMembers.push(...c.members);
    }
    expect(allMembers.sort()).toEqual([1, 2, 3]);

    // Find cluster containing record 1
    let foundSingleton = false;
    for (const c of updated.values()) {
      if (c.members.length === 1 && c.members[0] === 1) {
        foundSingleton = true;
      }
    }
    expect(foundSingleton).toBe(true);
  });
});

describe("unmergeCluster", () => {
  it("shatters cluster into singletons", () => {
    const pairs: [number, number, number][] = [[1, 2, 0.95], [2, 3, 0.95]];
    const clusters = buildClusters(pairs, [1, 2, 3]);
    const cid = [...clusters.keys()][0]!;
    const updated = unmergeCluster(cid, clusters);
    expect(updated.size).toBe(3);
    for (const c of updated.values()) {
      expect(c.size).toBe(1);
    }
  });
});

describe("addToCluster", () => {
  it("no match -> singleton", () => {
    const clusters = new Map<number, ClusterInfo>();
    addToCluster(5, [], clusters);
    expect(clusters.size).toBe(1);
    const c = [...clusters.values()][0]!;
    expect(c.members).toEqual([5]);
    expect(c.size).toBe(1);
  });

  it("1 cluster match -> joins that cluster", () => {
    // Start with cluster {1,2}
    const pairs: [number, number, number][] = [[1, 2, 0.9]];
    const clusters = buildClusters(pairs, [1, 2]);
    addToCluster(3, [[1, 0.9]], clusters);
    // Should have one cluster of size 3
    const sizes = [...clusters.values()].map((c) => c.size);
    expect(sizes).toContain(3);
  });

  it("2+ cluster matches -> merges clusters", () => {
    const pairs: [number, number, number][] = [
      [1, 2, 0.9],
      [3, 4, 0.9],
    ];
    const clusters = buildClusters(pairs, [1, 2, 3, 4]);
    expect(clusters.size).toBe(2);
    addToCluster(5, [[1, 0.9], [3, 0.9]], clusters);
    // Should have merged into one cluster of size 5
    const sizes = [...clusters.values()].map((c) => c.size);
    expect(sizes).toContain(5);
  });
});
