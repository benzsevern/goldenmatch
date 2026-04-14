/**
 * score-worker.ts -- piscina worker that scores a single block.
 *
 * Invoked by piscina with input { block, mk, matchedPairs }.
 * Returns the scored pairs for that block.
 *
 * Built as a separate tsup entry so it can be loaded by piscina from disk
 * at runtime. The worker runs in its own V8 isolate (worker_thread) so CPU
 * work here is truly parallel with the main thread and other workers.
 */
import { findFuzzyMatches } from "../../core/scorer.js";
import type {
  BlockResult,
  MatchkeyConfig,
  ScoredPair,
} from "../../core/types.js";

export interface ScoreWorkerInput {
  readonly block: BlockResult;
  readonly mk: MatchkeyConfig;
  /** Serialized Set<string> contents -- piscina can't transfer Sets. */
  readonly matchedPairs: readonly string[];
}

export interface ScoreWorkerOutput {
  readonly pairs: readonly ScoredPair[];
}

export default function scoreWorker(
  input: ScoreWorkerInput,
): ScoreWorkerOutput {
  const excludeSet = new Set(input.matchedPairs);
  const pairs = findFuzzyMatches(
    input.block.rows,
    input.mk,
    excludeSet,
    input.block.preScoredPairs,
  );
  return { pairs };
}
