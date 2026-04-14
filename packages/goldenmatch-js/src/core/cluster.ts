/**
 * cluster.ts — Union-Find clustering with MST splitting.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 */

import type { ClusterInfo } from "./types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Canonicalize a pair key: always min:max. */
export function pairKey(a: number, b: number): string {
  return `${Math.min(a, b)}:${Math.max(a, b)}`;
}

/** Parse a pair key back into [idA, idB]. */
function parsePairKey(key: string): [number, number] {
  const idx = key.indexOf(":");
  return [Number(key.slice(0, idx)), Number(key.slice(idx + 1))];
}

// ---------------------------------------------------------------------------
// UnionFind
// ---------------------------------------------------------------------------

export class UnionFind {
  private parent = new Map<number, number>();
  private rank = new Map<number, number>();

  /** Add element as its own root. */
  add(x: number): void {
    if (!this.parent.has(x)) {
      this.parent.set(x, x);
      this.rank.set(x, 0);
    }
  }

  /** Batch add multiple elements. */
  addMany(ids: readonly number[]): void {
    for (const x of ids) {
      if (!this.parent.has(x)) {
        this.parent.set(x, x);
        this.rank.set(x, 0);
      }
    }
  }

  /** Find root with iterative path compression. */
  find(x: number): number {
    let root = x;
    while (this.parent.get(root) !== root) {
      root = this.parent.get(root)!;
    }
    // Path compression
    let current = x;
    while (this.parent.get(current) !== root) {
      const next = this.parent.get(current)!;
      this.parent.set(current, root);
      current = next;
    }
    return root;
  }

  /** Union by rank. */
  union(a: number, b: number): void {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return;
    const rankA = this.rank.get(ra)!;
    const rankB = this.rank.get(rb)!;
    if (rankA < rankB) {
      [ra, rb] = [rb, ra];
    }
    this.parent.set(rb, ra);
    if (rankA === rankB) {
      this.rank.set(ra, rankA + 1);
    }
  }

  /** Return all clusters as arrays of sets. */
  getClusters(): Set<number>[] {
    const groups = new Map<number, Set<number>>();
    for (const x of this.parent.keys()) {
      const root = this.find(x);
      let group = groups.get(root);
      if (!group) {
        group = new Set<number>();
        groups.set(root, group);
      }
      group.add(x);
    }
    return Array.from(groups.values());
  }
}

// ---------------------------------------------------------------------------
// MST (max-weight spanning tree via Kruskal)
// ---------------------------------------------------------------------------

/**
 * Build a max-weight spanning tree using Kruskal's algorithm.
 * Returns edges as [idA, idB, score] sorted by descending weight.
 */
export function buildMst(
  members: readonly number[],
  pairScores: ReadonlyMap<string, number>,
): [number, number, number][] {
  // Collect and sort edges by score descending
  const edges: [number, number, number][] = [];
  for (const [key, score] of pairScores) {
    const [a, b] = parsePairKey(key);
    edges.push([a, b, score]);
  }
  edges.sort((x, y) => y[2] - x[2]);

  const uf = new UnionFind();
  uf.addMany(members);

  const mst: [number, number, number][] = [];
  const target = members.length - 1;
  for (const [a, b, s] of edges) {
    if (uf.find(a) !== uf.find(b)) {
      uf.union(a, b);
      mst.push([a, b, s]);
      if (mst.length === target) break;
    }
  }
  return mst;
}

// ---------------------------------------------------------------------------
// Cluster confidence
// ---------------------------------------------------------------------------

export interface ClusterConfidence {
  readonly minEdge: number | null;
  readonly avgEdge: number | null;
  readonly connectivity: number;
  readonly bottleneckPair: readonly [number, number] | null;
  readonly confidence: number;
}

/**
 * Compute confidence metrics for a cluster.
 * confidence = 0.4 * minEdge + 0.3 * avgEdge + 0.3 * connectivity
 */
export function computeClusterConfidence(
  pairScores: ReadonlyMap<string, number>,
  size: number,
): ClusterConfidence {
  if (size <= 1 || pairScores.size === 0) {
    return {
      minEdge: null,
      avgEdge: null,
      connectivity: size <= 1 ? 1.0 : 0.0,
      bottleneckPair: null,
      confidence: size <= 1 ? 1.0 : 0.0,
    };
  }

  let minEdge = Infinity;
  let sum = 0;
  let bottleneckKey = "";

  for (const [key, score] of pairScores) {
    sum += score;
    if (score < minEdge) {
      minEdge = score;
      bottleneckKey = key;
    }
  }

  const avgEdge = sum / pairScores.size;
  const maxPossibleEdges = (size * (size - 1)) / 2;
  const connectivity =
    maxPossibleEdges > 0 ? pairScores.size / maxPossibleEdges : 0.0;

  const bottleneckPair: readonly [number, number] | null = bottleneckKey
    ? parsePairKey(bottleneckKey)
    : null;

  const confidence = 0.4 * minEdge + 0.3 * avgEdge + 0.3 * connectivity;

  return { minEdge, avgEdge, connectivity, bottleneckPair, confidence };
}

// ---------------------------------------------------------------------------
// Split oversized cluster
// ---------------------------------------------------------------------------

/** Internal mutable cluster info used during building. */
interface MutableClusterInfo {
  members: number[];
  size: number;
  oversized: boolean;
  pairScores: Map<string, number>;
  confidence: number;
  bottleneckPair: readonly [number, number] | null;
  clusterQuality: "strong" | "weak" | "split";
  _wasSplit?: boolean;
}

/**
 * Split a cluster by removing the weakest MST edge.
 * Returns sub-cluster infos.
 */
export function splitOversizedCluster(
  members: readonly number[],
  pairScores: ReadonlyMap<string, number>,
): MutableClusterInfo[] {
  if (members.length <= 1 || pairScores.size === 0) {
    return [
      {
        members: [...members].sort((a, b) => a - b),
        size: members.length,
        oversized: false,
        pairScores: new Map(pairScores),
        confidence: 1.0,
        bottleneckPair: null,
        clusterQuality: "strong",
      },
    ];
  }

  const mst = buildMst(members, pairScores);
  if (mst.length === 0) {
    return [
      {
        members: [...members].sort((a, b) => a - b),
        size: members.length,
        oversized: false,
        pairScores: new Map(pairScores),
        confidence: 1.0,
        bottleneckPair: null,
        clusterQuality: "strong",
      },
    ];
  }

  // Find weakest edge
  let weakestIdx = 0;
  let weakestScore = mst[0]![2];
  for (let i = 1; i < mst.length; i++) {
    if (mst[i]![2] < weakestScore) {
      weakestScore = mst[i]![2];
      weakestIdx = i;
    }
  }

  // Rebuild without weakest edge
  const uf = new UnionFind();
  uf.addMany(members as number[]);
  for (let i = 0; i < mst.length; i++) {
    if (i !== weakestIdx) {
      uf.union(mst[i]![0], mst[i]![1]);
    }
  }

  const result: MutableClusterInfo[] = [];
  for (const subMembers of uf.getClusters()) {
    const subList = [...subMembers].sort((a, b) => a - b);
    const subPairs = new Map<string, number>();
    for (const [key, score] of pairScores) {
      const [a, b] = parsePairKey(key);
      if (subMembers.has(a) && subMembers.has(b)) {
        subPairs.set(key, score);
      }
    }
    const conf = computeClusterConfidence(subPairs, subList.length);
    result.push({
      members: subList,
      size: subList.length,
      oversized: false,
      pairScores: subPairs,
      confidence: conf.confidence,
      bottleneckPair: conf.bottleneckPair,
      clusterQuality: "strong",
    });
  }
  return result;
}

// ---------------------------------------------------------------------------
// buildClusters options
// ---------------------------------------------------------------------------

export interface BuildClustersOptions {
  readonly maxClusterSize?: number;
  readonly weakClusterThreshold?: number;
  readonly autoSplit?: boolean;
}

// ---------------------------------------------------------------------------
// buildClusters
// ---------------------------------------------------------------------------

/**
 * Build clusters from scored pairs using Union-Find.
 *
 * Auto-splits oversized clusters via MST (iterative, not recursive).
 * Assigns cluster_quality: "strong", "weak" (avg-min > weakThreshold), or "split".
 * Downgrades confidence by 0.7 for weak clusters.
 */
export function buildClusters(
  pairs: readonly (readonly [number, number, number])[],
  allIds: readonly number[],
  options?: BuildClustersOptions,
): Map<number, ClusterInfo> {
  const maxClusterSize = options?.maxClusterSize ?? 100;
  const weakClusterThreshold = options?.weakClusterThreshold ?? 0.3;
  const autoSplit = options?.autoSplit ?? true;

  // Build Union-Find from pairs
  const uf = new UnionFind();
  uf.addMany(allIds);
  for (const [idA, idB] of pairs) {
    uf.union(idA, idB);
  }

  const clusters = uf.getClusters();

  // Sort clusters by minimum member for deterministic IDs.
  // Use for-loop min — Math.min(...set) crashes on Sets with >65K elements.
  const minOf = (s: Set<number>): number => {
    let m = Infinity;
    for (const v of s) if (v < m) m = v;
    return m;
  };
  clusters.sort((a, b) => minOf(a) - minOf(b));

  // Map members to cluster IDs
  const memberToCid = new Map<number, number>();
  for (let i = 0; i < clusters.length; i++) {
    const cid = i + 1;
    for (const m of clusters[i]!) {
      memberToCid.set(m, cid);
    }
  }

  // Build mutable result
  const result = new Map<number, MutableClusterInfo>();
  for (let i = 0; i < clusters.length; i++) {
    const cid = i + 1;
    const memberArr = [...clusters[i]!].sort((a, b) => a - b);
    result.set(cid, {
      members: memberArr,
      size: memberArr.length,
      oversized: memberArr.length > maxClusterSize,
      pairScores: new Map(),
      confidence: 0,
      bottleneckPair: null,
      clusterQuality: "strong",
    });
  }

  // Assign pair scores to clusters (canonicalized keys)
  for (const [idA, idB, score] of pairs) {
    const cid = memberToCid.get(idA)!;
    const info = result.get(cid)!;
    info.pairScores.set(pairKey(idA, idB), score);
  }

  // Compute initial confidence
  for (const [, cinfo] of result) {
    const conf = computeClusterConfidence(cinfo.pairScores, cinfo.size);
    cinfo.confidence = conf.confidence;
    cinfo.bottleneckPair = conf.bottleneckPair;
  }

  // Auto-split oversized clusters (iterative)
  if (autoSplit) {
    const toSplit: number[] = [];
    for (const [cid, c] of result) {
      if (c.oversized) toSplit.push(cid);
    }

    while (toSplit.length > 0) {
      const cid = toSplit.pop()!;
      const cinfo = result.get(cid)!;
      result.delete(cid);

      const subClusters = splitOversizedCluster(
        cinfo.members,
        cinfo.pairScores,
      );
      let nextCid = 0;
      for (const [k] of result) {
        if (k > nextCid) nextCid = k;
      }
      nextCid += 1;

      for (const sc of subClusters) {
        sc.oversized = sc.size > maxClusterSize;
        sc._wasSplit = true;
        result.set(nextCid, sc);
        if (sc.oversized) {
          toSplit.push(nextCid);
        }
        nextCid++;
      }
    }
  }

  // Assign cluster_quality and apply confidence downgrade
  for (const [, cinfo] of result) {
    if (cinfo._wasSplit) {
      cinfo.clusterQuality = "split";
    } else if (cinfo.size > 1 && cinfo.pairScores.size > 0) {
      const scores = [...cinfo.pairScores.values()];
      let minE = Infinity;
      let sumE = 0;
      for (const s of scores) {
        if (s < minE) minE = s;
        sumE += s;
      }
      const avgE = sumE / scores.length;
      if (avgE - minE > weakClusterThreshold) {
        cinfo.clusterQuality = "weak";
        cinfo.confidence *= 0.7;
      } else {
        cinfo.clusterQuality = "strong";
      }
    } else {
      cinfo.clusterQuality = "strong";
    }
    delete cinfo._wasSplit;
  }

  // Freeze into ClusterInfo
  const frozen = new Map<number, ClusterInfo>();
  for (const [cid, c] of result) {
    frozen.set(cid, {
      members: c.members,
      size: c.size,
      oversized: c.oversized,
      pairScores: c.pairScores,
      confidence: c.confidence,
      bottleneckPair: c.bottleneckPair,
      clusterQuality: c.clusterQuality,
    });
  }
  return frozen;
}

// ---------------------------------------------------------------------------
// addToCluster
// ---------------------------------------------------------------------------

/**
 * Add a new record to existing clusters based on matches.
 *
 * - No matches: new singleton cluster
 * - Single cluster match: join that cluster
 * - Multiple cluster match: merge all matched clusters
 *
 * Flags oversized but does NOT auto-split. Caller should call
 * splitOversizedCluster() if desired.
 */
export function addToCluster(
  recordId: number,
  matches: readonly (readonly [number, number])[],
  clusters: Map<number, ClusterInfo>,
  maxClusterSize = 100,
): Map<number, ClusterInfo> {
  const makeSingleton = (): ClusterInfo => ({
    members: [recordId],
    size: 1,
    oversized: false,
    pairScores: new Map(),
    confidence: 1.0,
    bottleneckPair: null,
    clusterQuality: "strong",
  });

  if (matches.length === 0) {
    const nextCid = _nextCid(clusters);
    clusters.set(nextCid, makeSingleton());
    return clusters;
  }

  // Map members to cluster IDs
  const memberToCid = new Map<number, number>();
  for (const [cid, cinfo] of clusters) {
    for (const m of cinfo.members) {
      memberToCid.set(m, cid);
    }
  }

  const matchedCids = new Set<number>();
  for (const [matchedId] of matches) {
    const cid = memberToCid.get(matchedId);
    if (cid !== undefined) matchedCids.add(cid);
  }

  if (matchedCids.size === 0) {
    const nextCid = _nextCid(clusters);
    clusters.set(nextCid, makeSingleton());
    return clusters;
  }

  if (matchedCids.size === 1) {
    const cid = matchedCids.values().next().value!;
    const old = clusters.get(cid)!;
    const newPairs = new Map(old.pairScores);

    for (const [matchedId, score] of matches) {
      if (memberToCid.get(matchedId) === cid) {
        newPairs.set(pairKey(recordId, matchedId), score);
      }
    }

    const newMembers = [...old.members, recordId].sort((a, b) => a - b);
    const newSize = newMembers.length;
    const conf = computeClusterConfidence(newPairs, newSize);

    clusters.set(cid, {
      members: newMembers,
      size: newSize,
      oversized: newSize > maxClusterSize,
      pairScores: newPairs,
      confidence: conf.confidence,
      bottleneckPair: conf.bottleneckPair,
      clusterQuality: old.clusterQuality,
    });
    return clusters;
  }

  // Multiple clusters: merge all
  const mergedMembers: number[] = [recordId];
  const mergedPairs = new Map<string, number>();

  for (const cid of matchedCids) {
    const cinfo = clusters.get(cid)!;
    mergedMembers.push(...cinfo.members);
    for (const [k, v] of cinfo.pairScores) {
      mergedPairs.set(k, v);
    }
    clusters.delete(cid);
  }

  for (const [matchedId, score] of matches) {
    mergedPairs.set(pairKey(recordId, matchedId), score);
  }

  const sortedMembers = mergedMembers.sort((a, b) => a - b);
  const size = sortedMembers.length;
  const conf = computeClusterConfidence(mergedPairs, size);
  const nextCid = _nextCid(clusters);

  clusters.set(nextCid, {
    members: sortedMembers,
    size,
    oversized: size > maxClusterSize,
    pairScores: mergedPairs,
    confidence: conf.confidence,
    bottleneckPair: conf.bottleneckPair,
    clusterQuality: "strong",
  });

  return clusters;
}

// ---------------------------------------------------------------------------
// unmergeRecord
// ---------------------------------------------------------------------------

/**
 * Remove a record from its cluster and re-cluster remaining members.
 * The removed record becomes a singleton.
 */
export function unmergeRecord(
  recordId: number,
  clusters: Map<number, ClusterInfo>,
  threshold = 0.0,
): Map<number, ClusterInfo> {
  // Find which cluster contains this record
  let sourceCid: number | null = null;
  for (const [cid, cinfo] of clusters) {
    if (cinfo.members.includes(recordId)) {
      sourceCid = cid;
      break;
    }
  }

  if (sourceCid === null) return clusters; // Not found
  const cinfo = clusters.get(sourceCid)!;
  if (cinfo.size <= 1) return clusters; // Already singleton

  // Extract pairs excluding the removed record, applying threshold
  const remainingMembers = cinfo.members.filter((m) => m !== recordId);
  const remainingPairs: [number, number, number][] = [];
  for (const [key, score] of cinfo.pairScores) {
    const [a, b] = parsePairKey(key);
    if (a !== recordId && b !== recordId && score >= threshold) {
      remainingPairs.push([a, b, score]);
    }
  }

  // Re-cluster remaining members
  const subClusters = buildClusters(remainingPairs, remainingMembers);

  // Remove the original cluster
  clusters.delete(sourceCid);

  // Assign new cluster IDs
  let nextCid = _nextCid(clusters);

  // Add the removed record as a singleton
  clusters.set(nextCid, {
    members: [recordId],
    size: 1,
    oversized: false,
    pairScores: new Map(),
    confidence: 1.0,
    bottleneckPair: null,
    clusterQuality: "strong",
  });
  nextCid++;

  // Add re-clustered groups
  for (const [, subInfo] of subClusters) {
    clusters.set(nextCid, subInfo);
    nextCid++;
  }

  return clusters;
}

// ---------------------------------------------------------------------------
// unmergeCluster
// ---------------------------------------------------------------------------

/**
 * Shatter a cluster into individual singletons.
 * All members become their own cluster. Pair scores are discarded.
 */
export function unmergeCluster(
  clusterId: number,
  clusters: Map<number, ClusterInfo>,
): Map<number, ClusterInfo> {
  const cinfo = clusters.get(clusterId);
  if (!cinfo) return clusters;

  const members = cinfo.members;
  clusters.delete(clusterId);

  let nextCid = _nextCid(clusters);
  for (const memberId of members) {
    clusters.set(nextCid, {
      members: [memberId],
      size: 1,
      oversized: false,
      pairScores: new Map(),
      confidence: 1.0,
      bottleneckPair: null,
      clusterQuality: "strong",
    });
    nextCid++;
  }

  return clusters;
}

// ---------------------------------------------------------------------------
// getClusterPairScores
// ---------------------------------------------------------------------------

/**
 * Get pair scores for a specific set of cluster members from all pairs.
 * Call on-demand, not in hot path.
 */
export function getClusterPairScores(
  members: readonly number[],
  allPairs: readonly (readonly [number, number, number])[],
): Map<string, number> {
  const memberSet = new Set(members);
  const result = new Map<string, number>();
  for (const [a, b, s] of allPairs) {
    if (memberSet.has(a) && memberSet.has(b)) {
      result.set(pairKey(a, b), s);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function _nextCid(clusters: ReadonlyMap<number, unknown>): number {
  let max = 0;
  for (const k of clusters.keys()) {
    if (k > max) max = k;
  }
  return max + 1;
}
