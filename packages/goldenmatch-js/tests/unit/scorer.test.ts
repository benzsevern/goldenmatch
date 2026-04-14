import { describe, it, expect } from "vitest";
import {
  scoreField,
  scorePair,
  findExactMatches,
  findFuzzyMatches,
  jaro,
  jaroWinkler,
  levenshteinDistance,
  levenshteinSimilarity,
  tokenSortRatio,
  soundexMatch,
  diceCoefficient,
  jaccardSimilarity,
  ensembleScore,
  scoreMatrix,
  applyTransform,
} from "../../src/core/index.js";
import type { MatchkeyConfig, MatchkeyField, Row } from "../../src/core/index.js";

describe("jaro / jaroWinkler", () => {
  it("jaro MARTHA ~= MARHTA matches Python (0.9444)", () => {
    expect(jaro("MARTHA", "MARHTA")).toBeCloseTo(0.9444, 4);
  });

  it("jaroWinkler MARTHA ~= MARHTA matches Python (0.9611)", () => {
    expect(jaroWinkler("MARTHA", "MARHTA")).toBeCloseTo(0.9611, 4);
  });

  it("jaroWinkler DIXON / DICKSONX matches Python (0.8133)", () => {
    expect(jaroWinkler("DIXON", "DICKSONX")).toBeCloseTo(0.8133, 4);
  });

  it("jaroWinkler JELLYFISH / SMELLYFISH matches Python (0.8963)", () => {
    expect(jaroWinkler("JELLYFISH", "SMELLYFISH")).toBeCloseTo(0.8963, 4);
  });

  it("jaro identical -> 1.0", () => {
    expect(jaro("hello", "hello")).toBe(1.0);
  });

  it("jaro empty -> 0", () => {
    expect(jaro("", "hello")).toBe(0.0);
  });
});

describe("levenshtein", () => {
  it("kitten -> sitting is distance 3", () => {
    expect(levenshteinDistance("kitten", "sitting")).toBe(3);
  });

  it("identical distance 0", () => {
    expect(levenshteinDistance("abc", "abc")).toBe(0);
  });

  it("empty -> len", () => {
    expect(levenshteinDistance("", "abc")).toBe(3);
  });

  it("similarity 1.0 for identical", () => {
    expect(levenshteinSimilarity("abc", "abc")).toBe(1.0);
  });

  it("kitten/sitting similarity matches Python (1 - 3/7 = 0.5714)", () => {
    expect(levenshteinSimilarity("kitten", "sitting")).toBeCloseTo(0.5714, 4);
  });

  it("saturday/sunday similarity matches Python (1 - 3/8 = 0.6250)", () => {
    expect(levenshteinSimilarity("saturday", "sunday")).toBeCloseTo(0.625, 4);
  });
});

describe("tokenSortRatio (rapidfuzz-compatible)", () => {
  it("John Smith / Smith John -> 1.0 (same token set)", () => {
    expect(tokenSortRatio("John Smith", "Smith John")).toBe(1.0);
  });

  it("New York Mets / Mets New York -> 1.0", () => {
    expect(tokenSortRatio("New York Mets", "Mets New York")).toBe(1.0);
  });

  it("John Smith / Smith Johnson matches Indel ratio (0.8696)", () => {
    // sorted: "john smith" (10) vs "johnson smith" (13)
    // indel distance = 3 (insert s, o, n), 1 - 3/23 = 20/23 ≈ 0.8696
    expect(tokenSortRatio("John Smith", "Smith Johnson")).toBeCloseTo(0.8696, 4);
  });

  it("lowercases before sorting (case-insensitive)", () => {
    expect(tokenSortRatio("John SMITH", "smith JOHN")).toBe(1.0);
  });

  it("strips punctuation (rapidfuzz preprocessing)", () => {
    expect(tokenSortRatio("John, Smith!", "smith john.")).toBe(1.0);
  });

  it("different tokens return < 1", () => {
    expect(tokenSortRatio("John", "Jane")).toBeLessThan(1.0);
  });
});

describe("soundexMatch", () => {
  it("Robert/Rupert same code -> 1.0 (both R163)", () => {
    expect(soundexMatch("Robert", "Rupert")).toBe(1.0);
  });

  it("Smith/Smyth same code -> 1.0 (both S530)", () => {
    expect(soundexMatch("Smith", "Smyth")).toBe(1.0);
  });

  it("Smith/Doe -> 0", () => {
    expect(soundexMatch("Smith", "Doe")).toBe(0.0);
  });
});

describe("ensembleScore", () => {
  it("identical strings -> 1", () => {
    expect(ensembleScore("hello", "hello")).toBe(1.0);
  });

  it("returns at least jaro_winkler", () => {
    const jw = jaroWinkler("Smith", "Smyth");
    const en = ensembleScore("Smith", "Smyth");
    expect(en).toBeGreaterThanOrEqual(jw);
  });
});

describe("dice / jaccard (bloom filter hex)", () => {
  it("dice of identical bloom filters -> 1.0", () => {
    const bloom = applyTransform("hello", "bloom_filter");
    expect(bloom).not.toBe(null);
    expect(diceCoefficient(bloom!, bloom!)).toBe(1.0);
  });

  it("jaccard of identical -> 1.0", () => {
    const bloom = applyTransform("hello", "bloom_filter");
    expect(jaccardSimilarity(bloom!, bloom!)).toBe(1.0);
  });

  it("all-zero filters -> 0", () => {
    // 256 bits / 8 = 32 bytes, so 64 hex chars
    const zero = "0".repeat(64);
    expect(diceCoefficient(zero, zero)).toBe(0.0);
    expect(jaccardSimilarity(zero, zero)).toBe(0.0);
  });
});

describe("scoreField", () => {
  it("exact a==a -> 1.0", () => {
    expect(scoreField("a", "a", "exact")).toBe(1.0);
  });

  it("exact a!=b -> 0.0", () => {
    expect(scoreField("a", "b", "exact")).toBe(0.0);
  });

  it("returns null if either input is null", () => {
    expect(scoreField(null, "a", "exact")).toBe(null);
    expect(scoreField("a", null, "jaro_winkler")).toBe(null);
    expect(scoreField(null, null, "exact")).toBe(null);
  });

  it("unknown scorer throws", () => {
    expect(() => scoreField("a", "b", "fake_scorer")).toThrow();
  });

  it("jaro_winkler returns similarity", () => {
    const s = scoreField("abc", "abc", "jaro_winkler");
    expect(s).toBe(1.0);
  });

  it("levenshtein", () => {
    const s = scoreField("abc", "abc", "levenshtein");
    expect(s).toBe(1.0);
  });

  it("token_sort", () => {
    const s = scoreField("a b", "b a", "token_sort");
    expect(s).toBe(1.0);
  });

  it("token_sort strips punctuation and lowercases (rapidfuzz parity)", () => {
    // "John, Smith!" vs "smith john." → both sort to "john smith" → 1.0
    expect(scoreField("John, Smith!", "smith john.", "token_sort")).toBe(1.0);
  });
});

describe("scorePair - weighted fields", () => {
  it("weighted aggregation of fields", () => {
    const rowA: Row = { name: "John", city: "NYC" };
    const rowB: Row = { name: "John", city: "NYC" };
    const fields: MatchkeyField[] = [
      { field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 },
      { field: "city", transforms: [], scorer: "exact", weight: 1.0 },
    ];
    expect(scorePair(rowA, rowB, fields)).toBe(1.0);
  });

  it("returns 0 when weightSum=0 (all fields null)", () => {
    const rowA: Row = { name: null };
    const rowB: Row = { name: null };
    const fields: MatchkeyField[] = [
      { field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 },
    ];
    expect(scorePair(rowA, rowB, fields)).toBe(0);
  });

  it("weighted average of partial matches", () => {
    const rowA: Row = { name: "John", city: "NYC" };
    const rowB: Row = { name: "John", city: "LA" };
    const fields: MatchkeyField[] = [
      { field: "name", transforms: [], scorer: "exact", weight: 1.0 },
      { field: "city", transforms: [], scorer: "exact", weight: 1.0 },
    ];
    // (1.0 * 1 + 0.0 * 1) / 2 = 0.5
    expect(scorePair(rowA, rowB, fields)).toBe(0.5);
  });
});

describe("findExactMatches", () => {
  it("groups by matchkey column", () => {
    const rows: Row[] = [
      { __row_id__: 0, email: "a@x.com" },
      { __row_id__: 1, email: "a@x.com" },
      { __row_id__: 2, email: "b@x.com" },
    ];
    const mk: MatchkeyConfig = {
      name: "email",
      type: "exact",
      fields: [{ field: "email", transforms: [], scorer: "exact", weight: 1.0 }],
    };
    const pairs = findExactMatches(rows, mk);
    expect(pairs.length).toBe(1);
    expect(pairs[0]!.idA).toBe(0);
    expect(pairs[0]!.idB).toBe(1);
    expect(pairs[0]!.score).toBe(1.0);
  });

  it("returns empty for 0 or 1 rows", () => {
    const mk: MatchkeyConfig = {
      name: "email",
      type: "exact",
      fields: [{ field: "email", transforms: [], scorer: "exact", weight: 1.0 }],
    };
    expect(findExactMatches([], mk)).toEqual([]);
    expect(findExactMatches([{ __row_id__: 0, email: "a" }], mk)).toEqual([]);
  });

  it("skips rows where matchkey field is null", () => {
    const rows: Row[] = [
      { __row_id__: 0, email: null },
      { __row_id__: 1, email: null },
      { __row_id__: 2, email: "x@x.com" },
    ];
    const mk: MatchkeyConfig = {
      name: "email",
      type: "exact",
      fields: [{ field: "email", transforms: [], scorer: "exact", weight: 1.0 }],
    };
    const pairs = findExactMatches(rows, mk);
    expect(pairs.length).toBe(0);
  });
});

describe("findFuzzyMatches", () => {
  it("NxN scoring within block", () => {
    const rows: Row[] = [
      { __row_id__: 0, name: "Jon Smith" },
      { __row_id__: 1, name: "John Smith" },
      { __row_id__: 2, name: "Zeke Xavier" },
    ];
    const mk: MatchkeyConfig = {
      name: "name_fuzzy",
      type: "weighted",
      threshold: 0.7,
      fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 }],
    };
    const pairs = findFuzzyMatches(rows, mk);
    // Jon/John should match; Zeke should not match either
    const hasPair01 = pairs.some((p) => p.idA === 0 && p.idB === 1);
    expect(hasPair01).toBe(true);
  });

  it("empty if fewer than 2 rows", () => {
    const mk: MatchkeyConfig = {
      name: "f",
      type: "weighted",
      threshold: 0.85,
      fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 }],
    };
    expect(findFuzzyMatches([], mk)).toEqual([]);
  });
});

describe("scoreMatrix", () => {
  it("symmetric with 0 diagonal", () => {
    const m = scoreMatrix(["abc", "abd", "xyz"], "jaro_winkler");
    expect(m.length).toBe(3);
    expect(m[0]![0]).toBe(0); // diagonal
    expect(m[0]![1]).toBe(m[1]![0]); // symmetric
  });
});
