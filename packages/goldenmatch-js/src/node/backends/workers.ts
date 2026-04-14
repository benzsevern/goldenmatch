/**
 * workers.ts -- Concurrent and parallel block scoring for Node.
 *
 * Python uses ThreadPoolExecutor (rapidfuzz releases the GIL). JS is
 * single-threaded by default; true parallelism requires `worker_threads`
 * with serialization overhead that's only worth it for large blocks.
 *
 * This module ships two schedulers:
 *   - `scoreBlocksConcurrent` -- Promise.all batching on the main thread.
 *     No real parallelism, but zero setup cost and good for small/medium
 *     block counts.
 *   - `scoreBlocksParallel`   -- piscina-backed worker pool for true CPU
 *     parallelism. Optional peer dep; falls back to `scoreBlocksConcurrent`
 *     when piscina isn't installed.
 *
 * Mirrors the shape of `goldenmatch.backends.ray_backend.score_blocks_ray`
 * from the Python source, but stays inside one Node process.
 */

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import type {
  BlockResult,
  MatchkeyConfig,
  ScoredPair,
} from "../../core/types.js";

export interface WorkerPoolOptions {
  /** Max blocks scored concurrently per batch. Defaults to 4. */
  readonly batchSize?: number;
}

export interface ParallelWorkerOptions {
  /** Max worker threads. Defaults to min(8, max(2, blocks.length)). */
  readonly maxThreads?: number;
  /** Min worker threads kept warm. Defaults to 1. */
  readonly minThreads?: number;
  /** Idle timeout in ms before workers exit. Defaults to 1000. */
  readonly idleTimeout?: number;
}

/**
 * Score blocks with cooperative concurrency.
 *
 * - For 0 blocks: returns empty.
 * - For <= 2 blocks: skips batching overhead and runs sequentially via
 *   `scoreBlocksSequential` (mirrors Python's small-block fast path).
 * - Otherwise: schedules blocks in batches of `batchSize`, awaiting each
 *   batch with `Promise.all` so the event loop can interleave I/O.
 *
 * Note: `matchedPairs` is mutated as new pairs are discovered (consistent
 * with `scoreBlocksSequential`). A frozen snapshot is used per block so
 * concurrent batches see a stable exclusion set, matching Python's
 * `score_blocks_parallel` contract.
 */
export async function scoreBlocksConcurrent(
  blocks: readonly BlockResult[],
  mk: MatchkeyConfig,
  matchedPairs: Set<string>,
  options: WorkerPoolOptions = {},
): Promise<readonly ScoredPair[]> {
  if (blocks.length === 0) return [];

  // Small-block fast path -- sequential is cheaper than batching.
  if (blocks.length <= 2) {
    const { scoreBlocksSequential } = await import("../../core/scorer.js");
    return scoreBlocksSequential(blocks, mk, matchedPairs);
  }

  const { findFuzzyMatches } = await import("../../core/scorer.js");
  const batchSize = options.batchSize ?? 4;
  const results: ScoredPair[] = [];

  for (let i = 0; i < blocks.length; i += batchSize) {
    const batch = blocks.slice(i, i + batchSize);

    // Snapshot exclude set per batch so concurrent block scoring is stable.
    const excludeSnapshot: ReadonlySet<string> = new Set(matchedPairs);

    const batchResults = await Promise.all(
      batch.map((block) =>
        Promise.resolve().then(() =>
          findFuzzyMatches(
            block.rows,
            mk,
            excludeSnapshot,
            block.preScoredPairs,
          ),
        ),
      ),
    );

    for (const pairs of batchResults) {
      for (const p of pairs) {
        const key = `${p.idA}:${p.idB}`;
        if (matchedPairs.has(key)) continue;
        matchedPairs.add(key);
        results.push(p);
      }
    }
  }

  return results;
}

/**
 * Score blocks in true parallel via piscina worker_threads.
 *
 * - For 0 blocks: returns empty.
 * - For <= 2 blocks: runs sequentially (spinning up workers isn't worth it).
 * - Otherwise: dispatches each block to a piscina worker that runs
 *   `findFuzzyMatches` in its own V8 isolate, giving true CPU parallelism.
 *
 * Falls back to `scoreBlocksConcurrent` with a console warning if piscina
 * isn't installed (it's an optional peer dep).
 *
 * `matchedPairs` is mutated in place with newly discovered pairs, matching
 * the contract of `scoreBlocksSequential` / `scoreBlocksConcurrent`.
 */
export async function scoreBlocksParallel(
  blocks: readonly BlockResult[],
  mk: MatchkeyConfig,
  matchedPairs: Set<string>,
  options: ParallelWorkerOptions = {},
): Promise<readonly ScoredPair[]> {
  if (blocks.length === 0) return [];

  // Small-block fast path -- worker startup isn't worth it.
  if (blocks.length <= 2) {
    const { scoreBlocksSequential } = await import("../../core/scorer.js");
    return scoreBlocksSequential(blocks, mk, matchedPairs);
  }

  // Dynamically load piscina so it stays an optional peer dep.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let PiscinaCtor: any;
  try {
    const mod = await import("piscina" as string);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const m = mod as any;
    PiscinaCtor = m.Piscina ?? m.default ?? m;
  } catch {
    console.warn(
      "piscina not installed; falling back to Promise.all concurrency. " +
        "Install `piscina` as a peer dep for true worker-thread parallelism.",
    );
    return scoreBlocksConcurrent(blocks, mk, matchedPairs);
  }

  const workerScript = resolveWorkerScript();

  const pool = new PiscinaCtor({
    filename: workerScript,
    maxThreads: options.maxThreads ?? Math.max(2, Math.min(8, blocks.length)),
    minThreads: options.minThreads ?? 1,
    idleTimeout: options.idleTimeout ?? 1000,
  });

  try {
    // Snapshot the exclude set as an array so it serializes across threads.
    const snapshot = Array.from(matchedPairs);

    const results = (await Promise.all(
      blocks.map(
        (block) =>
          pool.run({ block, mk, matchedPairs: snapshot }) as Promise<{
            pairs: readonly ScoredPair[];
          }>,
      ),
    )) as { pairs: readonly ScoredPair[] }[];

    const all: ScoredPair[] = [];
    for (const r of results) {
      for (const p of r.pairs) {
        const key =
          p.idA < p.idB ? `${p.idA}:${p.idB}` : `${p.idB}:${p.idA}`;
        if (matchedPairs.has(key)) continue;
        matchedPairs.add(key);
        all.push(p);
      }
    }
    return all;
  } finally {
    await pool.destroy();
  }
}

/**
 * Resolve the on-disk path of the compiled worker script.
 *
 * tsup is configured to emit `score-worker.js` / `.cjs` alongside this
 * module in the `dist/node/backends/` directory. In dev (pre-build) the
 * caller can set `GOLDENMATCH_WORKER_SCRIPT` to point at a custom path
 * (e.g. a ts-node loader).
 *
 * Picks `.js` (ESM) first, falling back to `.cjs`. piscina resolves the
 * file itself -- we just hand it a path string.
 */
function resolveWorkerScript(): string {
  const override = process.env["GOLDENMATCH_WORKER_SCRIPT"];
  if (override !== undefined && override.length > 0) return override;

  const here = fileURLToPath(import.meta.url);
  const dir = dirname(here);

  // Preferred: sibling in the same dist/node/backends/ directory.
  // tsup emits both .js (ESM) and .cjs depending on the package format;
  // piscina accepts either.
  return join(dir, "score-worker.js");
}
