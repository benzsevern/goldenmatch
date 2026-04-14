import { describe, it, expect } from "vitest";
import { applyTransform, applyTransforms, soundex, metaphone } from "../../src/core/index.js";
import { sha256Hex, hmacSha256Hex } from "../../src/core/transforms.js";

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

describe("sha256Hex / hmacSha256Hex - Python parity", () => {
  it("sha256 of empty string", () => {
    expect(sha256Hex("")).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });

  it("sha256 of 'abc' (FIPS 180-2 reference vector)", () => {
    expect(sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });

  it("sha256 handles UTF-8 multibyte input", () => {
    // hashlib.sha256("héllo".encode()).hexdigest()
    expect(sha256Hex("héllo")).toBe(
      "3c48591d8d098a4538f5e013dfcf406e948eac4d3277b10bf614e295d6068179",
    );
  });

  it("hmac-sha256 matches Python reference (empty key, empty msg)", () => {
    // hmac.new(b"", b"", hashlib.sha256).hexdigest()
    expect(hmacSha256Hex("", "")).toBe(
      "b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad",
    );
  });

  it("hmac-sha256 matches Python reference (RFC 4231 test case)", () => {
    // hmac.new(b"key", b"The quick brown fox jumps over the lazy dog", hashlib.sha256).hexdigest()
    expect(hmacSha256Hex("key", "The quick brown fox jumps over the lazy dog")).toBe(
      "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8",
    );
  });
});

describe("bloom_filter - default size + hex length", () => {
  it("default bloom_filter produces 256 hex chars (1024 bits)", () => {
    const hex = applyTransform("hello", "bloom_filter");
    expect(hex).not.toBe(null);
    expect(hex!).toMatch(/^[0-9a-f]+$/);
    expect(hex!.length).toBe(256); // 1024 bits / 4 bits per hex char
  });

  it("bloom_filter:standard produces 128 hex chars (512 bits)", () => {
    const hex = applyTransform("hello", "bloom_filter:standard");
    expect(hex!.length).toBe(128);
  });

  it("bloom_filter:high produces 256 hex chars (1024 bits)", () => {
    const hex = applyTransform("hello", "bloom_filter:high");
    expect(hex!.length).toBe(256);
  });

  it("bloom_filter:paranoid produces 512 hex chars (2048 bits)", () => {
    const hex = applyTransform("hello", "bloom_filter:paranoid");
    expect(hex!.length).toBe(512);
  });

  it("same input produces same output (deterministic)", () => {
    expect(applyTransform("hello", "bloom_filter")).toBe(
      applyTransform("hello", "bloom_filter"),
    );
  });

  it("different inputs produce different outputs", () => {
    const a = applyTransform("hello", "bloom_filter");
    const b = applyTransform("world", "bloom_filter");
    expect(a).not.toBe(b);
  });

  it("byte-for-byte Python parity for 'hello' default bloom_filter", () => {
    // Reference generated by goldenmatch.utils.transforms.apply_transform("hello", "bloom_filter")
    const expected =
      "a008a1041000204000000400a000140000100048810004000008102010004000008400000000000080010100800000011014000000200000008002101000010002002100010000022000010020800000c00060000040000010010000000002400080000000800004008900200090000080800001000009000001000000100c20";
    expect(applyTransform("hello", "bloom_filter")).toBe(expected);
  });
});
