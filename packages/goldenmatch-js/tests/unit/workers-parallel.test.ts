/**
 * workers-parallel.test.ts -- Smoke tests for `scoreBlocksParallel`.
 *
 * piscina is an optional peer dep; these tests assert that the function
 * degrades gracefully (fast-path / fallback) even when piscina isn't
 * installed. When piscina IS installed, the worker path exercises true
 * worker_threads, but we use block shapes that also work in the fallback.
 */
import { describe, it, expect } from "vitest";
import { scoreBlocksParallel } from "../../src/node/backends/workers.js";
import type {
  BlockResult,
  MatchkeyConfig,
  Row,
} from "../../src/core/index.js";

const mk: MatchkeyConfig = {
  name: "name_match",
  type: "weighted",
  threshold: 0.5,
  fields: [
    { field: "name", transforms: [], scorer: "jaro_winkler", weight: 1 },
  ],
};

function makeBlock(blockKey: string, rows: Row[]): BlockResult {
  return { blockKey, rows, strategy: "static", depth: 0 };
}

describe("scoreBlocksParallel", () => {
  it("returns empty for 0 blocks", async () => {
    const result = await scoreBlocksParallel([], mk, new Set());
    expect(result).toEqual([]);
  });

  it("uses sequential fast-path for <= 2 blocks", async () => {
    const blocks = [
      makeBlock("b0", [
        { __row_id__: 1, name: "John" },
        { __row_id__: 2, name: "Jon" },
      ]),
    ];
    const result = await scoreBlocksParallel(blocks, mk, new Set());
    expect(Array.isArray(result)).toBe(true);
    // "John" vs "Jon" with jaro_winkler clears 0.5 threshold.
    expect(result.length).toBeGreaterThanOrEqual(1);
  });

  it("falls back to concurrent path when piscina not installed", async () => {
    // piscina is not installed in this dev env -- the dynamic import
    // inside scoreBlocksParallel fails and we should land on the
    // scoreBlocksConcurrent fallback, not throw.
    const blocks: BlockResult[] = Array.from({ length: 5 }, (_, i) =>
      makeBlock(`b${i}`, [
        { __row_id__: i * 10 + 1, name: "John" },
        { __row_id__: i * 10 + 2, name: "Jon" },
      ]),
    );
    const result = await scoreBlocksParallel(blocks, mk, new Set());
    expect(Array.isArray(result)).toBe(true);
    // 5 blocks × at least 1 pair each clearing 0.5 threshold.
    expect(result.length).toBeGreaterThanOrEqual(5);
  });

  it("mutates matchedPairs with newly discovered pairs", async () => {
    const blocks: BlockResult[] = Array.from({ length: 4 }, (_, i) =>
      makeBlock(`b${i}`, [
        { __row_id__: i * 10 + 1, name: "Alice" },
        { __row_id__: i * 10 + 2, name: "Alyce" },
      ]),
    );
    const matched = new Set<string>();
    const result = await scoreBlocksParallel(blocks, mk, matched);
    // Every newly returned pair must be in matchedPairs.
    for (const p of result) {
      const key =
        p.idA < p.idB ? `${p.idA}:${p.idB}` : `${p.idB}:${p.idA}`;
      expect(matched.has(key)).toBe(true);
    }
  });

  it("respects the exclude set (no duplicate pairs)", async () => {
    const blocks: BlockResult[] = Array.from({ length: 3 }, (_, i) =>
      makeBlock(`b${i}`, [
        { __row_id__: i * 10 + 1, name: "Alice" },
        { __row_id__: i * 10 + 2, name: "Alyce" },
      ]),
    );
    const result = await scoreBlocksParallel(blocks, mk, new Set());
    const keys = new Set<string>();
    for (const p of result) {
      const key =
        p.idA < p.idB ? `${p.idA}:${p.idB}` : `${p.idB}:${p.idA}`;
      expect(keys.has(key)).toBe(false);
      keys.add(key);
    }
  });
});
