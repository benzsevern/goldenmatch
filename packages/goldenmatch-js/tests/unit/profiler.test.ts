import { describe, it, expect } from "vitest";
import { profileRows } from "../../src/core/profiler.js";
import type { Row } from "../../src/core/types.js";

describe("profileRows", () => {
  it("empty input returns rowCount=0 and empty columns", () => {
    const p = profileRows([]);
    expect(p.rowCount).toBe(0);
    expect(p.columns).toEqual([]);
    expect(p.byName).toEqual({});
  });

  it("infers 'email' for a column of email-shaped values", () => {
    const rows: Row[] = [];
    for (let i = 0; i < 10; i++) {
      rows.push({ email: `user${i}@example.com`, name: `Name${i}` });
    }
    const p = profileRows(rows);
    expect(p.byName["email"]!.inferredType).toBe("email");
  });

  it("computes accurate null counts and rates", () => {
    const rows: Row[] = [
      { a: "x" },
      { a: null },
      { a: "" }, // treated as null
      { a: "y" },
    ];
    const p = profileRows(rows);
    const a = p.byName["a"]!;
    expect(a.totalCount).toBe(4);
    expect(a.nullCount).toBe(2);
    expect(a.nullRate).toBeCloseTo(0.5, 5);
  });

  it("cardinality ratio reflects distinct-per-total for mostly-unique vs repeating", () => {
    // Unique column
    const unique: Row[] = [];
    for (let i = 0; i < 10; i++) unique.push({ id: `v${i}` });
    const pu = profileRows(unique);
    expect(pu.byName["id"]!.cardinalityRatio).toBeCloseTo(1.0, 5);

    // Repeating column (2 distinct values)
    const repeating: Row[] = [];
    for (let i = 0; i < 10; i++) {
      repeating.push({ status: i % 2 === 0 ? "active" : "inactive" });
    }
    const pr = profileRows(repeating);
    expect(pr.byName["status"]!.cardinalityRatio).toBeCloseTo(0.2, 5);
    expect(pr.byName["status"]!.distinctCount).toBe(2);
  });

  it("computes accurate avgLength and maxLength for string columns", () => {
    const rows: Row[] = [{ s: "a" }, { s: "abc" }, { s: "ab" }];
    const p = profileRows(rows);
    const s = p.byName["s"]!;
    // avg = (1 + 3 + 2) / 3 = 2.0, max = 3
    expect(s.avgLength).toBeCloseTo(2.0, 5);
    expect(s.maxLength).toBe(3);
  });

  it("ignores internal __ columns", () => {
    const rows: Row[] = [{ __row_id__: 0, name: "Alice" }];
    const p = profileRows(rows);
    expect(p.byName["__row_id__"]).toBeUndefined();
    expect(p.byName["name"]).toBeDefined();
  });
});
