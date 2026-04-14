import { describe, it, expect } from "vitest";
import {
  buildBlocks,
  buildBlocksAsync,
  makeBlockingConfig,
} from "../../src/core/index.js";
import type { Row, BlockingConfig } from "../../src/core/index.js";

describe("buildBlocksAsync", () => {
  it("strategy='static' delegates to buildBlocks (same result)", async () => {
    const rows: Row[] = [
      { __row_id__: 0, zip: "12345" },
      { __row_id__: 1, zip: "12345" },
      { __row_id__: 2, zip: "67890" },
      { __row_id__: 3, zip: "67890" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
    });
    const sync = buildBlocks(rows, config);
    const async_ = await buildBlocksAsync(rows, config);
    expect(async_.length).toBe(sync.length);
    for (let i = 0; i < sync.length; i++) {
      expect(async_[i]!.blockKey).toBe(sync[i]!.blockKey);
      expect(async_[i]!.rows.length).toBe(sync[i]!.rows.length);
    }
  });

  it("strategy='ann' without apiKey/env throws clear error", async () => {
    const rows: Row[] = [
      { __row_id__: 0, name: "Alice" },
      { __row_id__: 1, name: "Alyce" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "ann",
      keys: [],
      annColumn: "name",
    });
    // Stash any real API key so the test is deterministic.
    const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
    const saved = env?.OPENAI_API_KEY;
    if (env) delete env.OPENAI_API_KEY;
    try {
      await expect(buildBlocksAsync(rows, config)).rejects.toThrow(/OpenAI|API key|apiKey/);
    } finally {
      if (env && saved !== undefined) env.OPENAI_API_KEY = saved;
    }
  });

  it("strategy='ann' missing annColumn throws", async () => {
    const rows: Row[] = [
      { __row_id__: 0, name: "Alice" },
      { __row_id__: 1, name: "Alyce" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "ann",
      keys: [],
      // annColumn omitted
    });
    await expect(buildBlocksAsync(rows, config)).rejects.toThrow(/annColumn/);
  });
});
