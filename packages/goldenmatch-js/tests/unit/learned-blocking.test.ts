import { describe, it, expect } from "vitest";
import {
  learnBlockingRules,
  applyLearnedBlocks,
} from "../../src/core/learned-blocking.js";
import type { Row, ScoredPair } from "../../src/core/types.js";

// Build a small dataset where known duplicate pairs share last-name/email prefixes
// so predicates like equal-on-last-name or soundex actually discriminate.
function makeDataset(): { rows: Row[]; pairs: ScoredPair[] } {
  const dupeGroups: Array<[string, string][]> = [
    [["John", "Smith"], ["Jon", "Smith"]],
    [["Mary", "Jones"], ["Marie", "Jones"]],
    [["Alice", "Brown"], ["Alicia", "Brown"]],
    [["Bob", "Miller"], ["Robert", "Miller"]],
    [["Carol", "Davis"], ["Caroline", "Davis"]],
  ];
  // Plus distractor rows to give the predicate some reduction work to do.
  const distractorLasts = [
    "Zygmunt", "Xiong", "Petrov", "Kowalski", "Nakamura",
    "Rasmussen", "Tanaka", "Vasquez", "Wojcik", "Yamamoto",
  ];
  const rows: Row[] = [];
  let id = 0;
  const pairs: ScoredPair[] = [];
  for (const pair of dupeGroups) {
    const idA = id++;
    rows.push({ __row_id__: idA, first_name: pair[0]![0], last_name: pair[0]![1] });
    const idB = id++;
    rows.push({ __row_id__: idB, first_name: pair[1]![0], last_name: pair[1]![1] });
    pairs.push({ idA, idB, score: 1.0 });
  }
  for (const last of distractorLasts) {
    rows.push({ __row_id__: id++, first_name: `First${id}`, last_name: last });
  }
  return { rows, pairs };
}

describe("learnBlockingRules", () => {
  it("produces predicates for a dataset with known matching pairs", () => {
    const { rows, pairs } = makeDataset();
    const rules = learnBlockingRules(rows, pairs, ["first_name", "last_name"], {
      minRecall: 0.8,
      minReduction: 0.5,
      predicateDepth: 3,
    });
    expect(rules.predicates.length).toBeGreaterThan(0);
    expect(rules.predicates.length).toBeLessThanOrEqual(3);
    // Every selected predicate should have positive recall.
    for (const p of rules.predicates) expect(p.recall).toBeGreaterThan(0);
    expect(rules.minRecall).toBe(0.8);
    expect(rules.minReduction).toBe(0.5);
    expect(typeof rules.learnedAt).toBe("string");
  });

  it("learned predicates cover at least one of the known pairs (useful recall)", () => {
    const { rows, pairs } = makeDataset();
    const rules = learnBlockingRules(rows, pairs, ["last_name"], {
      minRecall: 0.95,
      minReduction: 0.5,
      predicateDepth: 3,
    });
    // At least one predicate should retain some useful recall.
    const bestRecall = Math.max(...rules.predicates.map((p) => p.recall));
    expect(bestRecall).toBeGreaterThan(0);
  });

  it("empty known-pair input returns rules with no predicates (graceful fallback)", () => {
    const { rows } = makeDataset();
    const rules = learnBlockingRules(rows, [], ["first_name", "last_name"]);
    expect(rules.predicates).toEqual([]);
  });

  it("is deterministic given the same inputs", () => {
    const { rows, pairs } = makeDataset();
    const r1 = learnBlockingRules(rows, pairs, ["first_name", "last_name"]);
    const r2 = learnBlockingRules(rows, pairs, ["first_name", "last_name"]);
    expect(r1.predicates.length).toBe(r2.predicates.length);
    for (let i = 0; i < r1.predicates.length; i++) {
      expect(r1.predicates[i]!.type).toBe(r2.predicates[i]!.type);
      expect(r1.predicates[i]!.field).toBe(r2.predicates[i]!.field);
      expect(r1.predicates[i]!.recall).toBeCloseTo(r2.predicates[i]!.recall, 10);
      expect(r1.predicates[i]!.reduction).toBeCloseTo(
        r2.predicates[i]!.reduction,
        10,
      );
    }
  });
});

describe("applyLearnedBlocks", () => {
  it("produces BlockResult[] from learned rules with rows >= 2", () => {
    const { rows, pairs } = makeDataset();
    const rules = learnBlockingRules(rows, pairs, ["last_name"]);
    const blocks = applyLearnedBlocks(rows, rules, 100);
    for (const b of blocks) {
      expect(b.rows.length).toBeGreaterThanOrEqual(2);
      expect(b.strategy).toBe("learned");
      expect(b.blockKey.startsWith("learned:")).toBe(true);
    }
    // Each duplicate pair should share at least one block.
    if (blocks.length > 0) {
      const groupsById = new Map<number, Set<string>>();
      for (const b of blocks) {
        for (const r of b.rows) {
          const id = r["__row_id__"] as number;
          let set = groupsById.get(id);
          if (!set) {
            set = new Set();
            groupsById.set(id, set);
          }
          set.add(b.blockKey);
        }
      }
      // Sanity: if a useful predicate was chosen, the first dupe pair shares a key.
      const a = groupsById.get(0);
      const b = groupsById.get(1);
      if (a && b) {
        const shared = [...a].some((k) => b.has(k));
        expect(shared).toBe(true);
      }
    }
  });

  it("returns empty array when rules contain no predicates", () => {
    const { rows } = makeDataset();
    const blocks = applyLearnedBlocks(
      rows,
      { predicates: [], minRecall: 0.9, minReduction: 0.9, learnedAt: "" },
      100,
    );
    expect(blocks).toEqual([]);
  });
});
