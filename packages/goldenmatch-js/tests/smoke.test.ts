import { describe, it, expect } from "vitest";
import {
  dedupe,
  scoreField,
  scorePair,
  applyTransform,
  applyTransforms,
  buildClusters,
  UnionFind,
  scoreStrings,
} from "../src/core/index.js";

describe("smoke", () => {
  it("imports work", () => {
    expect(typeof dedupe).toBe("function");
    expect(typeof scoreField).toBe("function");
    expect(typeof scorePair).toBe("function");
    expect(typeof applyTransform).toBe("function");
    expect(typeof applyTransforms).toBe("function");
    expect(typeof buildClusters).toBe("function");
    expect(typeof UnionFind).toBe("function");
    expect(typeof scoreStrings).toBe("function");
  });

  it("basic dedupe works end-to-end", () => {
    const rows = [
      { id: 1, name: "John Smith", email: "john@example.com", zip: "12345" },
      { id: 2, name: "Jon Smith", email: "jon@example.com", zip: "12345" },
      { id: 3, name: "Jane Doe", email: "jane@example.com", zip: "54321" },
    ];
    const result = dedupe(rows, {
      fuzzy: { name: 0.7 },
      blocking: ["zip"],
      threshold: 0.7,
    });
    expect(result.stats.totalRecords).toBe(3);
  });

  it("scoreStrings returns a number between 0 and 1", () => {
    const s = scoreStrings("hello", "hello");
    expect(s).toBe(1.0);
    const s2 = scoreStrings("hello", "world");
    expect(s2).toBeGreaterThanOrEqual(0);
    expect(s2).toBeLessThanOrEqual(1);
  });

  it("autoconfigVerify symbols re-exported from goldenmatch", async () => {
    const gm = await import("../src/index.js");
    expect(typeof gm.makePreflightReport).toBe("function");
    expect(typeof gm.ConfigValidationError).toBe("function");
    expect(typeof gm.stripConventionPrivate).toBe("function");
  });
});
