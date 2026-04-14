/**
 * ann-blocker.ts — Approximate nearest neighbour blocking.
 *
 * Edge-safe: no `node:` imports, no FAISS. Implements a brute-force kNN
 * (O(n^2)) which is appropriate for <= ~10K records. Embeddings are
 * fetched via `getEmbedder()` which uses HTTP `fetch()`.
 *
 * Ports `goldenmatch/core/ann_blocker.py`.
 */

import type { BlockResult, Row, ScoredPair } from "./types.js";
import { makeScoredPair } from "./types.js";
import { getEmbedder, type EmbedderOptions } from "./embedder.js";

// ---------------------------------------------------------------------------
// Public option types
// ---------------------------------------------------------------------------

export interface ANNBlockerOptions {
  readonly topK?: number;
  readonly metric?: "cosine" | "euclidean";
}

export interface BuildANNOptions {
  readonly topK?: number;
  readonly model?: string;
  readonly apiKey?: string;
  readonly provider?: EmbedderOptions["provider"];
  /** Row identifier column (default `__row_id__`). */
  readonly idColumn?: string;
  /** Maximum block size produced by Union-Find grouping. */
  readonly maxBlockSize?: number;
  /** Use hnswlib-node fast-path when available (falls back to brute-force). */
  readonly useHNSW?: boolean;
}

/**
 * Minimal shape of the `hnswlib-node` module that we rely on. The caller
 * passes in the loaded module; we deliberately keep the surface tiny so we
 * don't hard-depend on its types.
 */
export interface HNSWModule {
  readonly HierarchicalNSW: new (
    metric: string,
    dim: number,
  ) => HNSWIndexLike;
}

export interface HNSWIndexLike {
  initIndex(
    maxElements: number,
    M?: number,
    efConstruction?: number,
    randomSeed?: number,
  ): void;
  setEf(ef: number): void;
  addPoint(vector: number[] | Float32Array, labelId: number): void;
  searchKnn(
    query: number[] | Float32Array,
    k: number,
  ): { distances: number[]; neighbors: number[] };
}

export interface HNSWOptions {
  readonly hnswModule: HNSWModule;
  readonly topK?: number;
  readonly metric?: "cosine" | "euclidean";
  readonly maxElements?: number;
  readonly M?: number;
  readonly efConstruction?: number;
  readonly efSearch?: number;
}

/** Shared interface so `ANNBlocker` and `HNSWANNBlocker` are interchangeable. */
export interface ANNBlockerBase {
  buildIndex(embeddings: readonly Float32Array[]): void;
  addToIndex(embedding: Float32Array): number;
  query(queryEmbeddings: readonly Float32Array[]): Array<[number, number]>;
  queryWithScores(
    queryEmbeddings: readonly Float32Array[],
  ): Array<[number, number, number]>;
  queryOne(queryEmbedding: Float32Array): Array<[number, number]>;
  readonly indexSize: number;
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

export function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const av = a[i]!;
    const bv = b[i]!;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export function euclideanDist(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!;
    s += d * d;
  }
  return Math.sqrt(s);
}

// ---------------------------------------------------------------------------
// ANNBlocker
// ---------------------------------------------------------------------------

export class ANNBlocker implements ANNBlockerBase {
  private embeddings: Float32Array[] = [];
  private readonly topK: number;
  private readonly metric: "cosine" | "euclidean";

  constructor(options: ANNBlockerOptions = {}) {
    this.topK = options.topK ?? 20;
    this.metric = options.metric ?? "cosine";
  }

  /** Replace the index with a fresh set of embeddings. */
  buildIndex(embeddings: readonly Float32Array[]): void {
    this.embeddings = embeddings.map((e) => e);
  }

  /** Number of vectors currently in the index. */
  get indexSize(): number {
    return this.embeddings.length;
  }

  /** Append a single embedding; returns its position. */
  addToIndex(embedding: Float32Array): number {
    this.embeddings.push(embedding);
    return this.embeddings.length - 1;
  }

  // ──────────────────────────────────────────────────────────
  // Querying
  // ──────────────────────────────────────────────────────────

  /**
   * For each query embedding, return up to topK (queryIdx, indexIdx) pairs.
   * Self-matches (same index when queries == embeddings) are excluded only
   * when the underlying object identity matches; otherwise the caller is
   * responsible for filtering self-pairs.
   *
   * Pairs are canonicalised so the lower index is always first when querying
   * against the same index population (queryIdx === indexIdx case removed).
   */
  query(queryEmbeddings: readonly Float32Array[]): Array<[number, number]> {
    const seen = new Set<number>();
    const out: Array<[number, number]> = [];
    const sameIndex = queryEmbeddings === (this.embeddings as readonly Float32Array[]);
    for (let i = 0; i < queryEmbeddings.length; i++) {
      const top = this.topKFor(queryEmbeddings[i]!);
      for (const [neighbour] of top) {
        if (sameIndex && neighbour === i) continue;
        if (neighbour < 0) continue;
        const a = Math.min(i, neighbour);
        const b = Math.max(i, neighbour);
        const key = a * 100000003 + b; // rough Cantor-like dedup key
        if (seen.has(key)) continue;
        seen.add(key);
        out.push([a, b]);
      }
    }
    return out;
  }

  /** Same as `query` but also returns the similarity score for each pair. */
  queryWithScores(
    queryEmbeddings: readonly Float32Array[],
  ): Array<[number, number, number]> {
    const best = new Map<number, [number, number, number]>();
    const sameIndex = queryEmbeddings === (this.embeddings as readonly Float32Array[]);
    for (let i = 0; i < queryEmbeddings.length; i++) {
      const top = this.topKFor(queryEmbeddings[i]!);
      for (const [neighbour, score] of top) {
        if (sameIndex && neighbour === i) continue;
        if (neighbour < 0) continue;
        const a = Math.min(i, neighbour);
        const b = Math.max(i, neighbour);
        const key = a * 100000003 + b;
        const prev = best.get(key);
        if (!prev || score > prev[2]) {
          best.set(key, [a, b, score]);
        }
      }
    }
    return [...best.values()];
  }

  /** Top-K matches for a single query. Returns (neighborIdx, score). */
  queryOne(queryEmbedding: Float32Array): Array<[number, number]> {
    return this.topKFor(queryEmbedding);
  }

  // ──────────────────────────────────────────────────────────
  // Internals
  // ──────────────────────────────────────────────────────────

  private topKFor(query: Float32Array): Array<[number, number]> {
    const n = this.embeddings.length;
    if (n === 0) return [];
    // Score every vector. For 10k×10k that's 100M ops — acceptable for an
    // edge-safe brute-force fallback, but callers should pre-filter for very
    // large datasets.
    const scores = new Array<{ idx: number; score: number }>(n);
    for (let i = 0; i < n; i++) {
      const s =
        this.metric === "cosine"
          ? cosineSim(query, this.embeddings[i]!)
          : -euclideanDist(query, this.embeddings[i]!);
      scores[i] = { idx: i, score: s };
    }
    // Partial sort: top-K only.
    const k = Math.min(this.topK, n);
    scores.sort((a, b) => b.score - a.score);
    const out: Array<[number, number]> = new Array(k);
    for (let i = 0; i < k; i++) out[i] = [scores[i]!.idx, scores[i]!.score];
    return out;
  }
}

// ---------------------------------------------------------------------------
// HNSWANNBlocker — optional fast-path backed by `hnswlib-node`.
//
// The caller provides the loaded `hnswlib-node` module via `opts.hnswModule`,
// keeping this file edge-safe (we never import the native module here).
// ---------------------------------------------------------------------------

export class HNSWANNBlocker implements ANNBlockerBase {
  private index: HNSWIndexLike | null = null;
  private count = 0;
  private readonly opts: HNSWOptions;
  private readonly topK: number;
  private readonly metric: "cosine" | "euclidean";

  constructor(opts: HNSWOptions) {
    this.opts = opts;
    this.topK = opts.topK ?? 20;
    this.metric = opts.metric ?? "cosine";
  }

  get indexSize(): number {
    return this.count;
  }

  buildIndex(embeddings: readonly Float32Array[]): void {
    if (embeddings.length === 0) {
      this.index = null;
      this.count = 0;
      return;
    }
    const dim = embeddings[0]!.length;
    const metricStr = this.metric === "euclidean" ? "l2" : "cosine";
    const HierarchicalNSW = this.opts.hnswModule.HierarchicalNSW;
    const index = new HierarchicalNSW(metricStr, dim);
    const maxElements = this.opts.maxElements ?? Math.max(embeddings.length * 2, 16);
    const M = this.opts.M ?? 16;
    const efConstruction = this.opts.efConstruction ?? 200;
    index.initIndex(maxElements, M, efConstruction, 100);
    const efSearch = this.opts.efSearch ?? Math.max(this.topK, 50);
    index.setEf(efSearch);
    for (let i = 0; i < embeddings.length; i++) {
      index.addPoint(Array.from(embeddings[i]!), i);
    }
    this.index = index;
    this.count = embeddings.length;
  }

  addToIndex(embedding: Float32Array): number {
    if (!this.index) {
      throw new Error("HNSWANNBlocker.addToIndex called before buildIndex");
    }
    const id = this.count;
    this.index.addPoint(Array.from(embedding), id);
    this.count++;
    return id;
  }

  query(queryEmbeddings: readonly Float32Array[]): Array<[number, number]> {
    const pairs: Array<[number, number]> = [];
    if (!this.index || this.count === 0) return pairs;
    const k = Math.min(this.topK, this.count);
    const seen = new Set<number>();
    for (let i = 0; i < queryEmbeddings.length; i++) {
      const q = Array.from(queryEmbeddings[i]!);
      const result = this.index.searchKnn(q, k);
      for (const neighbour of result.neighbors) {
        if (neighbour === i) continue;
        if (neighbour < 0) continue;
        const a = Math.min(i, neighbour);
        const b = Math.max(i, neighbour);
        const key = a * 100000003 + b;
        if (seen.has(key)) continue;
        seen.add(key);
        pairs.push([a, b]);
      }
    }
    return pairs;
  }

  queryWithScores(
    queryEmbeddings: readonly Float32Array[],
  ): Array<[number, number, number]> {
    const best = new Map<number, [number, number, number]>();
    if (!this.index || this.count === 0) return [];
    const k = Math.min(this.topK, this.count);
    for (let i = 0; i < queryEmbeddings.length; i++) {
      const q = Array.from(queryEmbeddings[i]!);
      const result = this.index.searchKnn(q, k);
      for (let idx = 0; idx < result.neighbors.length; idx++) {
        const neighbour = result.neighbors[idx]!;
        const d = result.distances[idx]!;
        if (neighbour === i) continue;
        if (neighbour < 0) continue;
        // For "cosine" metric hnswlib returns (1 - cos_sim); for "l2" it
        // returns squared Euclidean distance. Convert to a similarity score
        // bounded in (roughly) [0, 1].
        const score = this.metric === "euclidean" ? 1 / (1 + d) : 1 - d;
        const a = Math.min(i, neighbour);
        const b = Math.max(i, neighbour);
        const key = a * 100000003 + b;
        const prev = best.get(key);
        if (!prev || score > prev[2]) {
          best.set(key, [a, b, score]);
        }
      }
    }
    return [...best.values()];
  }

  queryOne(queryEmbedding: Float32Array): Array<[number, number]> {
    if (!this.index || this.count === 0) return [];
    const k = Math.min(this.topK, this.count);
    const result = this.index.searchKnn(Array.from(queryEmbedding), k);
    const out: Array<[number, number]> = [];
    for (let idx = 0; idx < result.neighbors.length; idx++) {
      const neighbour = result.neighbors[idx]!;
      const d = result.distances[idx]!;
      const score = this.metric === "euclidean" ? 1 / (1 + d) : 1 - d;
      out.push([neighbour, score]);
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Factory — auto-loads `hnswlib-node` when requested, falls back to brute-force.
// ---------------------------------------------------------------------------

export interface CreateANNBlockerOptions extends ANNBlockerOptions {
  /** Attempt to use the hnswlib-node fast-path. */
  readonly useHNSW?: boolean;
  /** Pre-loaded hnswlib-node module (skips dynamic import). */
  readonly hnswModule?: HNSWModule;
  /** Additional HNSW tuning knobs. Ignored when falling back to brute-force. */
  readonly maxElements?: number;
  readonly M?: number;
  readonly efConstruction?: number;
  readonly efSearch?: number;
  /**
   * Override the warning sink when the fast-path is unavailable. Defaults to
   * `console.warn`. Tests pass a spy here.
   */
  readonly onFallbackWarning?: (message: string) => void;
}

/**
 * Build an ANN blocker, preferring the `hnswlib-node` fast-path when
 * `useHNSW` is `true` and the module can be loaded. Falls back to the
 * brute-force `ANNBlocker` when the module is missing (e.g. edge runtime,
 * peer dep not installed) and emits a single warning.
 */
export async function createANNBlocker(
  options: CreateANNBlockerOptions = {},
): Promise<ANNBlockerBase> {
  const bruteOptions: ANNBlockerOptions = {
    ...(options.topK !== undefined ? { topK: options.topK } : {}),
    ...(options.metric !== undefined ? { metric: options.metric } : {}),
  };

  if (!options.useHNSW) {
    return new ANNBlocker(bruteOptions);
  }

  let hnsw: HNSWModule | null = options.hnswModule ?? null;
  if (!hnsw) {
    try {
      // `as string` prevents tsup / bundlers from trying to resolve this
      // optional native module at build time; it stays a runtime dynamic
      // import that we catch below if it fails.
      const mod = (await import("hnswlib-node" as string)) as unknown as {
        HierarchicalNSW: HNSWModule["HierarchicalNSW"];
        default?: { HierarchicalNSW: HNSWModule["HierarchicalNSW"] };
      };
      const ctor =
        mod.HierarchicalNSW ?? mod.default?.HierarchicalNSW ?? null;
      if (ctor) {
        hnsw = { HierarchicalNSW: ctor };
      }
    } catch {
      hnsw = null;
    }
  }

  if (!hnsw) {
    const warn = options.onFallbackWarning ?? ((m: string) => console.warn(m));
    warn("hnswlib-node not installed; falling back to brute-force ANN");
    return new ANNBlocker(bruteOptions);
  }

  return new HNSWANNBlocker({
    hnswModule: hnsw,
    ...(options.topK !== undefined ? { topK: options.topK } : {}),
    ...(options.metric !== undefined ? { metric: options.metric } : {}),
    ...(options.maxElements !== undefined
      ? { maxElements: options.maxElements }
      : {}),
    ...(options.M !== undefined ? { M: options.M } : {}),
    ...(options.efConstruction !== undefined
      ? { efConstruction: options.efConstruction }
      : {}),
    ...(options.efSearch !== undefined ? { efSearch: options.efSearch } : {}),
  });
}

// ---------------------------------------------------------------------------
// Block builders
// ---------------------------------------------------------------------------

/** Pull an ANN-relevant text from a row. Coerces non-strings, drops null. */
function getText(row: Row, col: string): string | null {
  const v = row[col];
  if (v === null || v === undefined) return null;
  const s = String(v).trim();
  return s === "" ? null : s;
}

/** Trivial Union-Find. */
class UnionFind {
  private parent: number[];
  constructor(n: number) {
    this.parent = new Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }
  find(x: number): number {
    let r = x;
    while (this.parent[r]! !== r) r = this.parent[r]!;
    // Path compression.
    let cur = x;
    while (this.parent[cur]! !== r) {
      const next = this.parent[cur]!;
      this.parent[cur] = r;
      cur = next;
    }
    return r;
  }
  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent[ra] = rb;
  }
}

/**
 * Embed one column, query top-K neighbours, and group connected pairs
 * into micro-blocks via Union-Find.
 */
export async function buildANNBlocks(
  rows: readonly Row[],
  annColumn: string,
  options: BuildANNOptions = {},
): Promise<BlockResult[]> {
  if (rows.length < 2) return [];

  const topK = options.topK ?? 20;
  const maxBlockSize = options.maxBlockSize ?? 1000;

  const embedder = getEmbedder({
    ...(options.model !== undefined ? { model: options.model } : {}),
    ...(options.apiKey !== undefined ? { apiKey: options.apiKey } : {}),
    ...(options.provider !== undefined ? { provider: options.provider } : {}),
  });

  // Extract texts.
  const texts: (string | null)[] = rows.map((r) => getText(r, annColumn));
  const embeddings = await embedder.embedColumn(texts);

  // Build index across all rows.
  const blocker = await createANNBlocker({
    topK,
    ...(options.useHNSW !== undefined ? { useHNSW: options.useHNSW } : {}),
  });
  blocker.buildIndex(embeddings);
  const pairs = blocker.query(embeddings);
  if (pairs.length === 0) return [];

  // Union-Find on the connected pairs.
  const uf = new UnionFind(rows.length);
  for (const [a, b] of pairs) uf.union(a, b);

  // Group by root.
  const groups = new Map<number, number[]>();
  for (let i = 0; i < rows.length; i++) {
    // Only include rows that participated in at least one pair (avoid singletons).
    // To detect that, we just include any row whose root has more than itself.
    const root = uf.find(i);
    let arr = groups.get(root);
    if (!arr) {
      arr = [];
      groups.set(root, arr);
    }
    arr.push(i);
  }

  const results: BlockResult[] = [];
  let blockNum = 0;
  for (const [, members] of groups) {
    if (members.length < 2) continue;
    if (members.length > maxBlockSize) continue; // skip oversized
    results.push({
      blockKey: `ann_${blockNum++}`,
      rows: members.map((idx) => rows[idx]!),
      strategy: "ann",
      depth: 0,
    });
  }
  return results;
}

/**
 * Variant that returns one BlockResult containing every row plus
 * pre-scored pairs derived from ANN cosine similarity. Useful when the
 * scorer should reuse the embedding-based scores instead of recomputing.
 */
export async function buildANNPairBlocks(
  rows: readonly Row[],
  annColumn: string,
  options: BuildANNOptions = {},
): Promise<BlockResult[]> {
  if (rows.length < 2) return [];

  const topK = options.topK ?? 20;
  const idColumn = options.idColumn ?? "__row_id__";

  const embedder = getEmbedder({
    ...(options.model !== undefined ? { model: options.model } : {}),
    ...(options.apiKey !== undefined ? { apiKey: options.apiKey } : {}),
    ...(options.provider !== undefined ? { provider: options.provider } : {}),
  });

  const texts: (string | null)[] = rows.map((r) => getText(r, annColumn));
  const embeddings = await embedder.embedColumn(texts);

  const blocker = await createANNBlocker({
    topK,
    ...(options.useHNSW !== undefined ? { useHNSW: options.useHNSW } : {}),
  });
  blocker.buildIndex(embeddings);
  const scored = blocker.queryWithScores(embeddings);
  if (scored.length === 0) return [];

  // Map index positions -> __row_id__ values (fall back to index).
  const rowIdAt = (idx: number): number => {
    const v = rows[idx]?.[idColumn];
    return typeof v === "number" ? v : idx;
  };

  const preScoredPairs: ScoredPair[] = scored.map(([a, b, score]) =>
    makeScoredPair(rowIdAt(a), rowIdAt(b), score),
  );

  return [
    {
      blockKey: "ann_pairs_0",
      rows: rows.slice(),
      strategy: "ann_pairs",
      depth: 0,
      preScoredPairs,
    },
  ];
}
