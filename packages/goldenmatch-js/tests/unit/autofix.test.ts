import { describe, it, expect } from "vitest";
import { autoFixRows } from "../../src/core/autofix.js";
import type { Row } from "../../src/core/types.js";

describe("autoFixRows", () => {
  it("trims whitespace from string values", () => {
    const rows: Row[] = [{ name: "  Alice  ", email: " a@x.com" }];
    const { rows: out, log } = autoFixRows(rows);
    expect(out[0]!["name"]).toBe("Alice");
    expect(out[0]!["email"]).toBe("a@x.com");
    const trimLog = log.find((l) => l.fixType === "trim_whitespace");
    expect(trimLog).toBeDefined();
  });

  it("converts empty strings to null", () => {
    const rows: Row[] = [{ a: "", b: "   " }];
    const { rows: out } = autoFixRows(rows);
    expect(out[0]!["a"]).toBeNull();
    expect(out[0]!["b"]).toBeNull();
  });

  it("converts common null tokens to null (case-insensitive)", () => {
    const rows: Row[] = [
      { a: "N/A", b: "NULL", c: "Unknown", d: "-", e: "n/a" },
    ];
    const { rows: out } = autoFixRows(rows);
    expect(out[0]!["a"]).toBeNull();
    expect(out[0]!["b"]).toBeNull();
    expect(out[0]!["c"]).toBeNull();
    expect(out[0]!["d"]).toBeNull();
    expect(out[0]!["e"]).toBeNull();
  });

  it("passes non-string values through unchanged", () => {
    const rows: Row[] = [{ n: 42, b: true, x: null }];
    const { rows: out } = autoFixRows(rows);
    expect(out[0]!["n"]).toBe(42);
    expect(out[0]!["b"]).toBe(true);
    expect(out[0]!["x"]).toBeNull();
  });

  it("leaves internal __ columns untouched", () => {
    const rows: Row[] = [
      { __row_id__: 0, __source__: "  sensitive  ", name: "  X  " },
    ];
    const { rows: out } = autoFixRows(rows);
    expect(out[0]!["__row_id__"]).toBe(0);
    // Internal columns preserved as-is (not trimmed, not nulled).
    expect(out[0]!["__source__"]).toBe("  sensitive  ");
    expect(out[0]!["name"]).toBe("X");
  });

  it("returns a log that aggregates affected rows per column/fix-type", () => {
    const rows: Row[] = [
      { a: "  x  ", b: "N/A" },
      { a: "  y  ", b: "" },
      { a: "ok", b: "hello" },
    ];
    const { log } = autoFixRows(rows);
    const trimA = log.find(
      (l) => l.column === "a" && l.fixType === "trim_whitespace",
    );
    const nullB = log.find(
      (l) => l.column === "b" && l.fixType === "null_empty_or_token",
    );
    expect(trimA).toBeDefined();
    expect(trimA!.affectedRows).toBe(2);
    expect(nullB).toBeDefined();
    expect(nullB!.affectedRows).toBe(2);
  });
});
