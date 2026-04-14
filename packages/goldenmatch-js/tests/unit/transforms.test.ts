import { describe, it, expect } from "vitest";
import { applyTransform, applyTransforms, soundex, metaphone } from "../../src/core/index.js";

describe("applyTransform - basic transforms", () => {
  it("lowercase", () => {
    expect(applyTransform("HELLO", "lowercase")).toBe("hello");
  });

  it("uppercase", () => {
    expect(applyTransform("hello", "uppercase")).toBe("HELLO");
  });

  it("strip", () => {
    expect(applyTransform("  hello  ", "strip")).toBe("hello");
  });

  it("strip_all", () => {
    expect(applyTransform("a b\tc\nd", "strip_all")).toBe("abcd");
  });

  it("digits_only", () => {
    expect(applyTransform("abc123def", "digits_only")).toBe("123");
  });

  it("alpha_only", () => {
    expect(applyTransform("abc123def!", "alpha_only")).toBe("abcdef");
  });

  it("normalize_whitespace", () => {
    expect(applyTransform("  a   b\tc  ", "normalize_whitespace")).toBe("a b c");
  });

  it("token_sort", () => {
    expect(applyTransform("Smith John", "token_sort")).toBe("John Smith");
  });

  it("first_token", () => {
    expect(applyTransform("John Smith Doe", "first_token")).toBe("John");
  });

  it("last_token", () => {
    expect(applyTransform("John Smith Doe", "last_token")).toBe("Doe");
  });

  it("returns null for null input", () => {
    expect(applyTransform(null, "lowercase")).toBe(null);
  });

  it("unknown transform returns value unchanged", () => {
    expect(applyTransform("hello", "nonexistent")).toBe("hello");
  });
});

describe("applyTransform - parameterized", () => {
  it("substring:0:3", () => {
    expect(applyTransform("abcdef", "substring:0:3")).toBe("abc");
  });

  it("substring:2:5", () => {
    expect(applyTransform("abcdef", "substring:2:5")).toBe("cde");
  });

  it("qgram:3 splits to 3-grams", () => {
    const result = applyTransform("abcde", "qgram:3");
    // 3-grams of "abcde": abc, bcd, cde (sorted)
    expect(result).toBe("abc bcd cde");
  });

  it("qgram:2 splits to bigrams", () => {
    const result = applyTransform("abc", "qgram:2");
    // bigrams: ab, bc
    expect(result).toBe("ab bc");
  });
});

describe("soundex", () => {
  it("Robert -> R163", () => {
    expect(soundex("Robert")).toBe("R163");
  });

  it("Smith and Smyth have same code", () => {
    expect(soundex("Smith")).toBe(soundex("Smyth"));
  });

  it("Rupert -> R163 (same as Robert)", () => {
    expect(soundex("Rupert")).toBe("R163");
  });

  it("empty string -> 0000", () => {
    expect(soundex("")).toBe("0000");
  });

  it("returns 4-character code", () => {
    expect(soundex("Washington").length).toBe(4);
  });
});

describe("metaphone", () => {
  it("returns a string", () => {
    expect(typeof metaphone("Thompson")).toBe("string");
  });

  it("empty string returns empty", () => {
    expect(metaphone("")).toBe("");
  });

  it("code has at most 4 characters", () => {
    expect(metaphone("Washington").length).toBeLessThanOrEqual(4);
  });
});

describe("applyTransforms - chain", () => {
  it("applies multiple in order", () => {
    // lowercase then strip
    expect(applyTransforms("  HELLO  ", ["lowercase", "strip"])).toBe("hello");
  });

  it("strip then digits_only", () => {
    expect(applyTransforms("  abc123  ", ["strip", "digits_only"])).toBe("123");
  });

  it("empty chain returns value unchanged", () => {
    expect(applyTransforms("hello", [])).toBe("hello");
  });

  it("propagates null through chain", () => {
    expect(applyTransforms(null, ["lowercase", "strip"])).toBe(null);
  });
});
