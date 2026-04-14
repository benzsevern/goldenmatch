import { describe, it, expect } from "vitest";
import { MemoryStore, MemoryLearner } from "../../src/core/index.js";
import type { Correction, MatchkeyConfig } from "../../src/core/index.js";

function makeCorrection(
  rowIdA: number,
  rowIdB: number,
  verdict: "match" | "no_match",
  score: number,
): Correction {
  return {
    rowIdA,
    rowIdB,
    verdict,
    feature: "overall",
    score,
    timestamp: Date.now(),
    trust: 0.9,
    source: "test",
  };
}

describe("MemoryStore", () => {
  it("add + list + count", () => {
    const store = new MemoryStore();
    expect(store.count()).toBe(0);
    store.add(makeCorrection(1, 2, "match", 0.9));
    store.add(makeCorrection(3, 4, "no_match", 0.3));
    expect(store.count()).toBe(2);
    expect(store.list().length).toBe(2);
  });

  it("listMatches and listNonMatches", () => {
    const store = new MemoryStore();
    store.add(makeCorrection(1, 2, "match", 0.9));
    store.add(makeCorrection(3, 4, "no_match", 0.3));
    expect(store.listMatches().length).toBe(1);
    expect(store.listNonMatches().length).toBe(1);
  });

  it("clear resets the store", () => {
    const store = new MemoryStore();
    store.add(makeCorrection(1, 2, "match", 0.9));
    store.clear();
    expect(store.count()).toBe(0);
  });

  it("upsert with higher trust replaces existing", () => {
    const store = new MemoryStore();
    const c1: Correction = {
      rowIdA: 1,
      rowIdB: 2,
      verdict: "match",
      feature: "name",
      score: 0.9,
      timestamp: 1000,
      trust: 0.5,
      source: "a",
    };
    const c2: Correction = { ...c1, trust: 0.9, source: "b", timestamp: 2000 };
    store.upsert(c1);
    store.upsert(c2);
    expect(store.count()).toBe(1);
    expect(store.list()[0]!.source).toBe("b");
  });
});

describe("MemoryLearner", () => {
  it("tunes threshold when given >= 10 corrections with mixed verdicts", () => {
    const baseline: MatchkeyConfig = {
      name: "m",
      type: "weighted",
      threshold: 0.85,
      fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 }],
    };
    const corrections: Correction[] = [];
    // 10 positives with score > 0.8, 10 negatives with score < 0.7
    for (let i = 0; i < 10; i++) {
      corrections.push(makeCorrection(i, i + 100, "match", 0.85));
      corrections.push(makeCorrection(i + 200, i + 300, "no_match", 0.4));
    }
    const learner = new MemoryLearner();
    const params = learner.learn(corrections, baseline);
    expect(params.correctionCount).toBe(20);
    expect(params.threshold).not.toBeUndefined();
    // Optimal threshold should be somewhere between 0.4 and 0.85
    expect(params.threshold!).toBeGreaterThanOrEqual(0.5);
    expect(params.threshold!).toBeLessThanOrEqual(0.95);
  });

  it("returns no threshold when fewer than 10 corrections", () => {
    const baseline: MatchkeyConfig = {
      name: "m",
      type: "weighted",
      threshold: 0.85,
      fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 }],
    };
    const corrections = [makeCorrection(1, 2, "match", 0.9)];
    const learner = new MemoryLearner();
    const params = learner.learn(corrections, baseline);
    expect(params.threshold).toBeUndefined();
  });

  it("returns no threshold when all verdicts are the same", () => {
    const baseline: MatchkeyConfig = {
      name: "m",
      type: "weighted",
      threshold: 0.85,
      fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 }],
    };
    const corrections: Correction[] = [];
    for (let i = 0; i < 15; i++) {
      corrections.push(makeCorrection(i, i + 100, "match", 0.9));
    }
    const learner = new MemoryLearner();
    const params = learner.learn(corrections, baseline);
    expect(params.threshold).toBeUndefined();
  });
});
