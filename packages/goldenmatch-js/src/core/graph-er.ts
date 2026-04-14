/**
 * graph-er.ts — Multi-table entity resolution with evidence propagation.
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/core/graph_er.py. Each table is deduped independently
 * first, then cluster assignments propagate across foreign-key edges:
 * if row A.fk points into B's cluster, rows of A whose FK shares a cluster
 * get a similarity boost before re-clustering.
 */

import type { ClusterInfo, Row, ScoredPair } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TableSchema {
  readonly name: string;
  readonly rows: readonly Row[];
  readonly idColumn: string;
}

export interface Relationship {
  readonly tableA: string;
  readonly tableB: string;
  readonly fkColumn: string; // column in tableA referencing tableB
}

export interface GraphERResult {
  readonly clustersByTable: ReadonlyMap<string, ReadonlyMap<number, ClusterInfo>>;
  readonly converged: boolean;
  readonly iterations: number;
}

export interface GraphERScorer {
  (rows: readonly Row[]): readonly ScoredPair[];
}

export interface RunGraphEROptions {
  readonly maxIterations?: number;
  readonly convergenceThreshold?: number;
  readonly similarityBoost?: number;
  /** Per-table scorer: takes rows, returns scored pairs. Required. */
  readonly scorerByTable: ReadonlyMap<string, GraphERScorer>;
  /** Match threshold for building clusters. Default 0.85. */
  readonly threshold?: number;
}

// ---------------------------------------------------------------------------
// Minimal Union-Find
// ---------------------------------------------------------------------------

class UnionFind {
  private parent: number[] = [];
  private size: number[] = [];

  add(id: number): void {
    while (this.parent.length <= id) {
      this.parent.push(this.parent.length);
      this.size.push(1);
    }
  }

  find(id: number): number {
    this.add(id);
    let cur = id;
    while (this.parent[cur] !== cur) {
      const parent = this.parent[cur]!;
      this.parent[cur] = this.parent[parent]!; // path compression
      cur = this.parent[cur]!;
    }
    return cur;
  }

  union(a: number, b: number): void {
    const rootA = this.find(a);
    const rootB = this.find(b);
    if (rootA === rootB) return;
    if (this.size[rootA]! < this.size[rootB]!) {
      this.parent[rootA] = rootB;
      this.size[rootB]! += this.size[rootA]!;
    } else {
      this.parent[rootB] = rootA;
      this.size[rootA]! += this.size[rootB]!;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toRowIndex(rows: readonly Row[], idColumn: string): Map<unknown, number> {
  const map = new Map<unknown, number>();
  for (let i = 0; i < rows.length; i++) {
    const v = rows[i]![idColumn];
    if (v !== null && v !== undefined) map.set(v, i);
  }
  return map;
}

function clustersFromPairs(
  rowCount: number,
  pairs: readonly ScoredPair[],
  threshold: number,
): Map<number, ClusterInfo> {
  const uf = new UnionFind();
  for (let i = 0; i < rowCount; i++) uf.add(i);

  const scoreMap = new Map<string, number>();
  for (const p of pairs) {
    if (p.score < threshold) continue;
    uf.union(p.idA, p.idB);
    const k = p.idA < p.idB ? `${p.idA}|${p.idB}` : `${p.idB}|${p.idA}`;
    scoreMap.set(k, p.score);
  }

  const rootMembers = new Map<number, number[]>();
  for (let i = 0; i < rowCount; i++) {
    const root = uf.find(i);
    const list = rootMembers.get(root);
    if (list) list.push(i);
    else rootMembers.set(root, [i]);
  }

  const clusters = new Map<number, ClusterInfo>();
  let clusterId = 0;
  for (const members of rootMembers.values()) {
    const pairScores = new Map<string, number>();
    let minEdge = 1;
    let edgeSum = 0;
    let edgeCount = 0;
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        const a = members[i]!;
        const b = members[j]!;
        const k = a < b ? `${a}|${b}` : `${b}|${a}`;
        const s = scoreMap.get(k);
        if (s !== undefined) {
          pairScores.set(k, s);
          if (s < minEdge) minEdge = s;
          edgeSum += s;
          edgeCount++;
        }
      }
    }
    const avgEdge = edgeCount > 0 ? edgeSum / edgeCount : 0;
    const connectivity = members.length <= 1 ? 0 : Math.min(1, edgeCount / (members.length - 1));
    const confidence = members.length <= 1 ? 1 : 0.4 * minEdge + 0.3 * avgEdge + 0.3 * connectivity;

    clusters.set(clusterId++, {
      members,
      size: members.length,
      oversized: false,
      pairScores,
      confidence,
      bottleneckPair: null,
      clusterQuality: "strong",
    });
  }

  return clusters;
}

function rowIdToCluster(clusters: ReadonlyMap<number, ClusterInfo>): Map<number, number> {
  const map = new Map<number, number>();
  for (const [cid, c] of clusters) {
    for (const m of c.members) map.set(m, cid);
  }
  return map;
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/**
 * Run multi-table entity resolution with iterative evidence propagation.
 *
 * For each table, the caller provides a scorer that produces pair scores
 * from a row array. The algorithm:
 *   1. Score & cluster each table independently.
 *   2. For every relationship A->B: find pairs in A whose fk resolves to
 *      the same cluster in B. Boost those pair scores by `similarityBoost`.
 *   3. Re-cluster every table. Repeat until clusters stabilize or
 *      `maxIterations` is reached.
 */
export function runGraphER(
  tables: readonly TableSchema[],
  relationships: readonly Relationship[],
  options: RunGraphEROptions,
): GraphERResult {
  const maxIterations = options.maxIterations ?? 5;
  const convergenceThreshold = options.convergenceThreshold ?? 0.01;
  const similarityBoost = options.similarityBoost ?? 0.1;
  const threshold = options.threshold ?? 0.85;

  // Per-table state.
  const tableByName = new Map<string, TableSchema>();
  for (const t of tables) tableByName.set(t.name, t);

  const idIndexByTable = new Map<string, Map<unknown, number>>();
  for (const t of tables) {
    idIndexByTable.set(t.name, toRowIndex(t.rows, t.idColumn));
  }

  // Initial pair scores per table (without boost).
  const basePairsByTable = new Map<string, ScoredPair[]>();
  for (const t of tables) {
    const scorer = options.scorerByTable.get(t.name);
    if (!scorer) {
      throw new Error(`Missing scorer for table "${t.name}"`);
    }
    basePairsByTable.set(t.name, [...scorer(t.rows)]);
  }

  let clustersByTable = new Map<string, Map<number, ClusterInfo>>();
  for (const t of tables) {
    clustersByTable.set(
      t.name,
      clustersFromPairs(t.rows.length, basePairsByTable.get(t.name) ?? [], threshold),
    );
  }

  let converged = false;
  let iter = 0;

  for (; iter < maxIterations; iter++) {
    const rowToCluster = new Map<string, Map<number, number>>();
    for (const [name, clusters] of clustersByTable) {
      rowToCluster.set(name, rowIdToCluster(clusters));
    }

    const nextClusters = new Map<string, Map<number, ClusterInfo>>();
    let maxDelta = 0;

    for (const t of tables) {
      const basePairs = basePairsByTable.get(t.name) ?? [];
      const boosted = basePairs.map((p) => ({ ...p }));

      // For each relationship where this table is the source, boost pairs
      // whose FK targets land in the same cluster in the referenced table.
      for (const rel of relationships) {
        if (rel.tableA !== t.name) continue;
        const bClusters = rowToCluster.get(rel.tableB);
        if (!bClusters) continue;
        const bIndex = idIndexByTable.get(rel.tableB);
        if (!bIndex) continue;

        // Build: rowId in A -> cluster id in B (or null)
        const fkClusterById = new Map<number, number>();
        for (let i = 0; i < t.rows.length; i++) {
          const fkVal = t.rows[i]![rel.fkColumn];
          if (fkVal === null || fkVal === undefined) continue;
          const bRowIdx = bIndex.get(fkVal);
          if (bRowIdx === undefined) continue;
          const bCid = bClusters.get(bRowIdx);
          if (bCid !== undefined) fkClusterById.set(i, bCid);
        }

        for (const pair of boosted) {
          const ca = fkClusterById.get(pair.idA);
          const cb = fkClusterById.get(pair.idB);
          if (ca !== undefined && cb !== undefined && ca === cb) {
            const newScore = Math.min(1, pair.score + similarityBoost);
            (pair as { score: number }).score = newScore;
          }
        }
      }

      const newClusters = clustersFromPairs(t.rows.length, boosted, threshold);
      const prevClusters = clustersByTable.get(t.name);
      if (prevClusters) {
        const delta = clusterSetDelta(prevClusters, newClusters);
        if (delta > maxDelta) maxDelta = delta;
      }
      nextClusters.set(t.name, newClusters);
    }

    clustersByTable = nextClusters;
    if (maxDelta < convergenceThreshold) {
      converged = true;
      break;
    }
  }

  const finalMap = new Map<string, ReadonlyMap<number, ClusterInfo>>();
  for (const [k, v] of clustersByTable) finalMap.set(k, v);

  return {
    clustersByTable: finalMap,
    converged,
    iterations: iter + (converged ? 1 : 0),
  };
}

/**
 * Compare two cluster assignments over the same row set. Returns fraction of
 * rows whose cluster signature changed — a rough "delta" proxy. Two rows
 * have the same signature if they are in the same cluster in both sets.
 */
function clusterSetDelta(
  a: ReadonlyMap<number, ClusterInfo>,
  b: ReadonlyMap<number, ClusterInfo>,
): number {
  const mapA = rowIdToCluster(a);
  const mapB = rowIdToCluster(b);

  // Align cluster IDs between a and b by finding the most common b-id for
  // each a-id. Anything mismatched counts as a change.
  const aToB = new Map<number, Map<number, number>>();
  for (const [rowId, aCid] of mapA) {
    const bCid = mapB.get(rowId);
    if (bCid === undefined) continue;
    let sub = aToB.get(aCid);
    if (!sub) {
      sub = new Map();
      aToB.set(aCid, sub);
    }
    sub.set(bCid, (sub.get(bCid) ?? 0) + 1);
  }

  const majority = new Map<number, number>();
  for (const [aCid, counts] of aToB) {
    let best: [number, number] | null = null;
    for (const [bCid, count] of counts) {
      if (best === null || count > best[1]) best = [bCid, count];
    }
    if (best) majority.set(aCid, best[0]);
  }

  let changed = 0;
  let total = 0;
  for (const [rowId, aCid] of mapA) {
    const bCid = mapB.get(rowId);
    if (bCid === undefined) {
      changed++;
      total++;
      continue;
    }
    total++;
    if (majority.get(aCid) !== bCid) changed++;
  }

  return total === 0 ? 0 : changed / total;
}
