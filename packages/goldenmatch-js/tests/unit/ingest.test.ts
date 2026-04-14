import { describe, it, expect } from "vitest";
import {
  applyColumnMap,
  validateColumns,
  concatRows,
  tagSource,
  assignRowIds,
} from "../../src/core/ingest.js";
import type { Row } from "../../src/core/types.js";

describe("applyColumnMap", () => {
  it("renames columns per the map", () => {
    const rows: Row[] = [
      { first: "Alice", last: "Brown", email: "a@x.com" },
      { first: "Bob", last: "Smith", email: "b@x.com" },
    ];
    const out = applyColumnMap(rows, { first: "first_name", last: "last_name" });
    expect(Object.keys(out[0]!).sort()).toEqual(
      ["email", "first_name", "last_name"].sort(),
    );
    expect(out[0]!["first_name"]).toBe("Alice");
    expect(out[0]!["last_name"]).toBe("Brown");
  });

  it("passes unmapped keys through untouched", () => {
    const rows: Row[] = [{ a: 1, b: 2 }];
    const out = applyColumnMap(rows, { a: "aa" });
    expect(out[0]!["aa"]).toBe(1);
    expect(out[0]!["b"]).toBe(2);
  });

  it("empty map clones rows unchanged", () => {
    const rows: Row[] = [{ a: 1 }];
    const out = applyColumnMap(rows, {});
    expect(out).toEqual(rows);
    // Different reference (shallow clone).
    expect(out[0]).not.toBe(rows[0]);
  });
});

describe("validateColumns", () => {
  it("passes when all required columns are present", () => {
    const rows: Row[] = [{ a: 1, b: 2, c: 3 }];
    expect(() => validateColumns(rows, ["a", "b"])).not.toThrow();
  });

  it("throws with the missing columns listed", () => {
    const rows: Row[] = [{ a: 1 }];
    expect(() => validateColumns(rows, ["a", "b", "c"])).toThrow(
      /Required columns missing:.*b.*c/,
    );
  });

  it("no-ops on empty input", () => {
    expect(() => validateColumns([], ["a"])).not.toThrow();
  });
});

describe("concatRows", () => {
  it("unions schemas and fills missing fields with null", () => {
    const a: Row[] = [{ x: 1, y: 2 }];
    const b: Row[] = [{ y: 3, z: 4 }];
    const out = concatRows([a, b]);
    expect(out.length).toBe(2);
    // Both rows have all keys from union.
    expect(Object.keys(out[0]!).sort()).toEqual(["x", "y", "z"]);
    expect(Object.keys(out[1]!).sort()).toEqual(["x", "y", "z"]);
    expect(out[0]!["z"]).toBeNull(); // missing in a
    expect(out[1]!["x"]).toBeNull(); // missing in b
    expect(out[0]!["x"]).toBe(1);
    expect(out[1]!["z"]).toBe(4);
  });

  it("handles empty arrays gracefully", () => {
    const out = concatRows([]);
    expect(out).toEqual([]);
    const out2 = concatRows([[], []]);
    expect(out2).toEqual([]);
  });
});

describe("tagSource / assignRowIds", () => {
  it("tagSource adds __source__ to each row", () => {
    const rows: Row[] = [{ a: 1 }, { a: 2 }];
    const out = tagSource(rows, "csv_a");
    expect(out[0]!["__source__"]).toBe("csv_a");
    expect(out[1]!["__source__"]).toBe("csv_a");
  });

  it("assignRowIds fills missing __row_id__ with sequential ids", () => {
    const rows: Row[] = [{ a: 1 }, { a: 2, __row_id__: 99 }, { a: 3 }];
    const out = assignRowIds(rows, 10);
    expect(out[0]!["__row_id__"]).toBe(10);
    expect(out[1]!["__row_id__"]).toBe(99); // existing preserved
    expect(out[2]!["__row_id__"]).toBe(12);
  });
});
