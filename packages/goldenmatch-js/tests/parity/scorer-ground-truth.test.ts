/**
 * Python parity ground-truth for every scorer.
 *
 * Each row: (scorer, input_a, input_b, python_score).
 * Tolerance: 1e-4 (four decimal places) — tight enough to catch a real
 * bug but loose enough to survive last-digit floating-point drift.
 *
 * The canonical Jaro-Winkler reference values (MARTHA/MARHTA = 0.9611,
 * DIXON/DICKSONX = 0.8133, etc.) come from Winkler's original paper and
 * are reproduced by rapidfuzz, jellyfish, and every other mainstream
 * implementation. Other values (token_sort, levenshtein) are computed
 * from their well-defined formulas:
 *   - Levenshtein similarity: 1 - dist / max(|a|, |b|)
 *   - Indel similarity (rapidfuzz token_sort base): 1 - d_indel / (|a| + |b|)
 *   - Soundex match: 1.0 iff soundex codes match
 *
 * To regenerate / extend: run the equivalent Python via
 *   from rapidfuzz.fuzz import token_sort_ratio
 *   from rapidfuzz.distance import JaroWinkler, Levenshtein, Indel
 * and paste the results below.
 */
import { describe, it, expect } from "vitest";
import {
  scoreField,
  jaro,
  diceCoefficient,
  jaccardSimilarity,
} from "../../src/core/index.js";

type Case = readonly [scorer: string, a: string, b: string, expected: number];

const CASES: readonly Case[] = [
  // Jaro-Winkler — canonical reference values from Winkler's paper
  ["jaro_winkler", "MARTHA", "MARHTA", 0.9611],
  ["jaro_winkler", "DIXON", "DICKSONX", 0.8133],
  ["jaro_winkler", "JELLYFISH", "SMELLYFISH", 0.8963],
  ["jaro_winkler", "DWAYNE", "DUANE", 0.84],
  ["jaro_winkler", "abc", "abc", 1.0],
  ["jaro_winkler", "", "", 1.0],
  ["jaro_winkler", "abc", "", 0.0],
  // John/Jon: jaro = (3/4 + 3/3 + 1)/3 = 11/12 = 0.9167; prefix=3, jw = 0.9167 + 3*0.1*(1-0.9167) = 0.9333
  ["jaro_winkler", "John", "Jon", 0.9333],

  // Levenshtein similarity (1 - dist/max_len)
  ["levenshtein", "kitten", "sitting", 0.5714], // 1 - 3/7
  ["levenshtein", "saturday", "sunday", 0.625], // 1 - 3/8
  ["levenshtein", "abc", "abc", 1.0],
  ["levenshtein", "", "", 1.0],
  ["levenshtein", "abc", "xyz", 0.0],

  // token_sort via rapidfuzz Indel ratio, with lowercase + strip-nonalnum preprocessing
  ["token_sort", "New York Mets", "Mets New York", 1.0],
  // "john smith" (10) vs "johnson smith" (13): indel dist = 3 (insert 's','o','n')
  // similarity = 1 - 3/23 = 20/23 ≈ 0.8696
  ["token_sort", "John Smith", "Smith Johnson", 0.8696],
  ["token_sort", "the quick brown fox", "fox quick the brown", 1.0],
  ["token_sort", "a b c", "c b a", 1.0],
  ["token_sort", "John, Smith!", "smith john.", 1.0], // strips punctuation
  ["token_sort", "John SMITH", "smith john", 1.0], // lowercases

  // Exact
  ["exact", "abc", "abc", 1.0],
  ["exact", "abc", "xyz", 0.0],

  // Soundex
  ["soundex_match", "Robert", "Rupert", 1.0], // both R163
  ["soundex_match", "Robert", "Smith", 0.0],
  ["soundex_match", "Smith", "Smyth", 1.0], // both S530
];

describe("scorer Python parity (4-decimal tolerance)", () => {
  for (const [scorer, a, b, expected] of CASES) {
    it(`${scorer}(${JSON.stringify(a)}, ${JSON.stringify(b)}) ≈ ${expected}`, () => {
      const actual = scoreField(a, b, scorer);
      expect(actual).not.toBeNull();
      expect(actual as number).toBeCloseTo(expected, 4);
    });
  }
});

describe("jaro parity (not exposed via scoreField)", () => {
  it("jaro(MARTHA, MARHTA) ≈ 0.9444", () => {
    expect(jaro("MARTHA", "MARHTA")).toBeCloseTo(0.9444, 4);
  });
});

// Bloom-filter scorers — dice / jaccard. Not Python-sourced since Python's
// implementation is trivial (bitwise ops on CLK bitvectors); we lock behavior
// on hand-computed edge cases.
describe("bloom-filter similarity sanity", () => {
  it("identical bloom filters score 1.0 (dice)", () => {
    const hex = "ff00ff00";
    expect(diceCoefficient(hex, hex)).toBe(1.0);
  });

  it("identical bloom filters score 1.0 (jaccard)", () => {
    const hex = "ff00ff00";
    expect(jaccardSimilarity(hex, hex)).toBe(1.0);
  });

  it("non-overlapping bloom filters score 0.0 (dice)", () => {
    expect(diceCoefficient("ff00", "00ff")).toBe(0.0);
  });

  it("non-overlapping bloom filters score 0.0 (jaccard)", () => {
    expect(jaccardSimilarity("ff00", "00ff")).toBe(0.0);
  });

  it("half-overlap: ff00 vs ffff -> dice = 2*8/(8+16) = 0.6667", () => {
    // ff00 has 8 bits set; ffff has 16 bits; intersection = 8
    // dice = 2*8 / (8+16) = 16/24 = 0.6667
    expect(diceCoefficient("ff00", "ffff")).toBeCloseTo(0.6667, 4);
  });

  it("half-overlap: ff00 vs ffff -> jaccard = 8/16 = 0.5", () => {
    expect(jaccardSimilarity("ff00", "ffff")).toBeCloseTo(0.5, 4);
  });
});
