import { describe, it, expect } from "vitest";
import { applyStandardizer, applyStandardization } from "../../src/core/index.js";
import type { Row } from "../../src/core/index.js";

describe("applyStandardizer", () => {
  it("email: lowercase and strip", () => {
    expect(applyStandardizer("  USER@Example.COM  ", "email")).toBe("user@example.com");
  });

  it("email: invalid returns empty string (null->empty)", () => {
    expect(applyStandardizer("not-an-email", "email")).toBe("");
  });

  it("name_proper: JOHN smith -> John Smith", () => {
    expect(applyStandardizer("JOHN smith", "name_proper")).toBe("John Smith");
  });

  it("name_proper: hyphenated mary-jane -> Mary-Jane", () => {
    expect(applyStandardizer("mary-jane", "name_proper")).toBe("Mary-Jane");
  });

  it("phone: digits only, strips US country code", () => {
    expect(applyStandardizer("1-800-555-1234", "phone")).toBe("8005551234");
  });

  it("phone: pure digits retained", () => {
    expect(applyStandardizer("(415) 555-1234", "phone")).toBe("4155551234");
  });

  it("zip5: 12345-6789 -> 12345", () => {
    expect(applyStandardizer("12345-6789", "zip5")).toBe("12345");
  });

  it("zip5: short padded", () => {
    expect(applyStandardizer("123", "zip5")).toBe("00123");
  });

  it("address: MAIN ST -> Main St", () => {
    expect(applyStandardizer("MAIN ST", "address")).toBe("Main St");
  });

  it("address: MAIN STREET -> Main St (abbreviated)", () => {
    expect(applyStandardizer("MAIN STREET", "address")).toBe("Main St");
  });

  it("state uppercases", () => {
    expect(applyStandardizer(" ca ", "state")).toBe("CA");
  });

  it("strip removes whitespace", () => {
    expect(applyStandardizer("  hello  ", "strip")).toBe("hello");
  });

  it("unknown standardizer throws", () => {
    expect(() => applyStandardizer("x", "not-a-thing")).toThrow();
  });
});

describe("applyStandardization", () => {
  it("applies rules dict to rows", () => {
    const rows: Row[] = [
      { email: "USER@X.COM", first: "JOHN" },
    ];
    const out = applyStandardization(rows, {
      email: ["email"],
      first: ["name_proper"],
    });
    expect(out[0]!.email).toBe("user@x.com");
    expect(out[0]!.first).toBe("John");
  });

  it("leaves nulls as-is", () => {
    const rows: Row[] = [{ email: null, first: "A" }];
    const out = applyStandardization(rows, { email: ["email"], first: ["name_proper"] });
    expect(out[0]!.email).toBe(null);
    expect(out[0]!.first).toBe("A");
  });

  it("chains multiple standardizers", () => {
    const rows: Row[] = [{ first: "  JOHN  " }];
    const out = applyStandardization(rows, { first: ["strip", "name_proper"] });
    expect(out[0]!.first).toBe("John");
  });
});
