import { describe, it, expect } from "vitest";
import { dedupe, match, scoreStrings, scorePairRecord } from "../../src/core/index.js";
import type { MatchkeyField, Row } from "../../src/core/index.js";

describe("dedupe() — shorthand API", () => {
  it("with exact catches identical emails", () => {
    const rows: Row[] = [
      { email: "a@x.com", name: "Alice" },
      { email: "a@x.com", name: "A." },
      { email: "b@x.com", name: "Bob" },
    ];
    const result = dedupe(rows, { exact: ["email"] });
    expect(result.stats.totalRecords).toBe(3);
    expect(result.dupes.length).toBeGreaterThanOrEqual(2);
  });

  it("with fuzzy catches similar names", () => {
    const rows: Row[] = [
      { name: "John Smith", zip: "111" },
      { name: "Jon Smith", zip: "111" },
      { name: "Zeke Xavier", zip: "222" },
    ];
    const result = dedupe(rows, {
      fuzzy: { name: 1.0 },
      blocking: ["zip"],
      threshold: 0.7,
    });
    expect(result.stats.totalRecords).toBe(3);
    expect(result.scoredPairs.length).toBeGreaterThanOrEqual(1);
  });
});

describe("match() — cross-dataset", () => {
  it("finds matches across datasets", () => {
    const target: Row[] = [{ email: "a@x.com" }];
    const reference: Row[] = [{ email: "a@x.com" }, { email: "b@x.com" }];
    const result = match(target, reference, { exact: ["email"] });
    expect(result.matched.length).toBe(1);
  });
});

describe("scoreStrings()", () => {
  it("exact identical", () => {
    expect(scoreStrings("hello", "hello", "exact")).toBe(1.0);
  });

  it("jaro_winkler default", () => {
    const s = scoreStrings("John", "John");
    expect(s).toBe(1.0);
  });

  it("returns 0-1 range", () => {
    const s = scoreStrings("foo", "bar");
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });

  it("levenshtein", () => {
    expect(scoreStrings("abc", "abc", "levenshtein")).toBe(1.0);
  });

  it("token_sort reorders", () => {
    expect(scoreStrings("a b", "b a", "token_sort")).toBe(1.0);
  });
});

describe("scorePairRecord()", () => {
  it("scores two row objects across fields", () => {
    const rowA: Row = { name: "John", city: "NYC" };
    const rowB: Row = { name: "John", city: "NYC" };
    const fields: MatchkeyField[] = [
      { field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 },
      { field: "city", transforms: [], scorer: "exact", weight: 1.0 },
    ];
    expect(scorePairRecord(rowA, rowB, fields)).toBe(1.0);
  });
});
