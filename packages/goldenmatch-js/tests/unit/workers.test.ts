import { describe, it, expect } from "vitest";
import { scoreBlocksConcurrent } from "../../src/node/backends/workers.js";
import { scoreBlocksSequential } from "../../src/core/index.js";
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

describe("scoreBlocksConcurrent", () => {
  it("0 blocks -> empty", async () => {
    const out = await scoreBlocksConcurrent([], mk, new Set());
    expect(out).toEqual([]);
  });

  it("small N (<=2 blocks) returns same pairs as sequential", async () => {
    const block: Row[] = [
      { __row_id__: 0, name: "Alice" },
      { __row_id__: 1, name: "Alyce" },
    ];
    const blocks = [makeBlock("b0", block)];
    const concurrent = await scoreBlocksConcurrent(blocks, mk, new Set());
    const sequential = scoreBlocksSequential(blocks, mk, new Set());
    expect(concurrent.length).toBe(sequential.length);
    if (sequential.length > 0) {
      expect(concurrent[0]!.idA).toBe(sequential[0]!.idA);
      expect(concurrent[0]!.idB).toBe(sequential[0]!.idB);
    }
  });

  it("many blocks (>2) batched concurrently returns same set as sequential", async () => {
    const blocks: BlockResult[] = [];
    for (let i = 0; i < 6; i++) {
      const base = i * 10;
      blocks.push(
        makeBlock(`b${i}`, [
          { __row_id__: base, name: "Alice" },
          { __row_id__: base + 1, name: "Alyce" },
        ]),
      );
    }
    const concurrent = await scoreBlocksConcurrent(blocks, mk, new Set());
    const sequential = scoreBlocksSequential(blocks, mk, new Set());
    expect(concurrent.length).toBe(sequential.length);

    const cKeys = new Set(concurrent.map((p) => `${p.idA}:${p.idB}`));
    const sKeys = new Set(sequential.map((p) => `${p.idA}:${p.idB}`));
    expect(cKeys).toEqual(sKeys);
  });

  it("handles empty/singleton blocks gracefully", async () => {
    const blocks: BlockResult[] = [
      makeBlock("empty", []),
      makeBlock("singleton", [{ __row_id__: 0, name: "Alice" }]),
      makeBlock("empty2", []),
      makeBlock("singleton2", [{ __row_id__: 1, name: "Bob" }]),
    ];
    const out = await scoreBlocksConcurrent(blocks, mk, new Set());
    expect(out).toEqual([]);
  });
});
