import { describe, it, expect } from "vitest";
import {
  buildStaticBlocks,
  buildMultiPassBlocks,
  buildAdaptiveBlocks,
  buildBlocks,
  makeBlockingConfig,
} from "../../src/core/index.js";
import type { Row, BlockingConfig } from "../../src/core/index.js";

describe("buildStaticBlocks", () => {
  it("3 rows with same zip -> 1 block of 3", () => {
    const rows: Row[] = [
      { __row_id__: 0, zip: "12345" },
      { __row_id__: 1, zip: "12345" },
      { __row_id__: 2, zip: "12345" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
    });
    const blocks = buildStaticBlocks(rows, config);
    expect(blocks.length).toBe(1);
    expect(blocks[0]!.rows.length).toBe(3);
  });

  it("different zips -> no blocks (singletons skipped)", () => {
    const rows: Row[] = [
      { __row_id__: 0, zip: "11111" },
      { __row_id__: 1, zip: "22222" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
    });
    const blocks = buildStaticBlocks(rows, config);
    expect(blocks.length).toBe(0);
  });

  it("applies transforms to block key (lowercase)", () => {
    const rows: Row[] = [
      { __row_id__: 0, city: "NYC" },
      { __row_id__: 1, city: "nyc" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["city"], transforms: ["lowercase"] }],
    });
    const blocks = buildStaticBlocks(rows, config);
    expect(blocks.length).toBe(1);
    expect(blocks[0]!.rows.length).toBe(2);
  });

  it("missing field produces null block key and row is skipped", () => {
    const rows: Row[] = [
      { __row_id__: 0, zip: "12345" },
      { __row_id__: 1, zip: null },
      { __row_id__: 2, zip: "12345" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
    });
    const blocks = buildStaticBlocks(rows, config);
    expect(blocks.length).toBe(1);
    expect(blocks[0]!.rows.length).toBe(2);
  });

  it("oversized block with skipOversized=true is dropped", () => {
    const rows: Row[] = Array.from({ length: 10 }, (_, i) => ({
      __row_id__: i,
      zip: "12345",
    }));
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
      maxBlockSize: 5,
      skipOversized: true,
    });
    const blocks = buildStaticBlocks(rows, config);
    expect(blocks.length).toBe(0);
  });

  it("oversized block with skipOversized=false is kept", () => {
    const rows: Row[] = Array.from({ length: 10 }, (_, i) => ({
      __row_id__: i,
      zip: "12345",
    }));
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
      maxBlockSize: 5,
      skipOversized: false,
    });
    const blocks = buildStaticBlocks(rows, config);
    expect(blocks.length).toBe(1);
    expect(blocks[0]!.rows.length).toBe(10);
  });
});

describe("buildMultiPassBlocks", () => {
  it("runs multiple passes with different keys", () => {
    const rows: Row[] = [
      { __row_id__: 0, zip: "111", last: "Smith" },
      { __row_id__: 1, zip: "111", last: "Jones" },
      { __row_id__: 2, zip: "222", last: "Smith" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "multi_pass",
      keys: [{ fields: ["zip"], transforms: [] }],
      passes: [
        { fields: ["zip"], transforms: [] },
        { fields: ["last"], transforms: [] },
      ],
    });
    const blocks = buildMultiPassBlocks(rows, config);
    // Pass 1: zip 111 has 2 rows -> 1 block
    // Pass 2: last Smith has 2 rows -> 1 block
    expect(blocks.length).toBe(2);
  });
});

describe("buildAdaptiveBlocks", () => {
  it("auto-split oversized block", () => {
    const rows: Row[] = [
      { __row_id__: 0, zip: "111", city: "A" },
      { __row_id__: 1, zip: "111", city: "A" },
      { __row_id__: 2, zip: "111", city: "B" },
      { __row_id__: 3, zip: "111", city: "B" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "adaptive",
      keys: [{ fields: ["zip"], transforms: [] }],
      maxBlockSize: 3,
      skipOversized: false,
    });
    const blocks = buildAdaptiveBlocks(rows, config);
    // Should split by city
    expect(blocks.length).toBeGreaterThanOrEqual(2);
  });
});

describe("buildBlocks dispatch", () => {
  it("static strategy routes correctly", () => {
    const rows: Row[] = [
      { __row_id__: 0, zip: "111" },
      { __row_id__: 1, zip: "111" },
    ];
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
    });
    const blocks = buildBlocks(rows, config);
    expect(blocks.length).toBe(1);
  });

  it("fewer than 2 rows returns empty", () => {
    const config: BlockingConfig = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
    });
    expect(buildBlocks([], config)).toEqual([]);
  });
});
