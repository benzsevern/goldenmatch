import { describe, it, expect } from "vitest";
import { gatePairs, ReviewQueue } from "../../src/core/index.js";
import type { ScoredPair } from "../../src/core/index.js";

describe("gatePairs", () => {
  it("default thresholds split pairs into 3 buckets", () => {
    const pairs: ScoredPair[] = [
      { idA: 1, idB: 2, score: 0.99 }, // approve
      { idA: 3, idB: 4, score: 0.80 }, // review
      { idA: 5, idB: 6, score: 0.50 }, // reject
    ];
    const result = gatePairs(pairs);
    expect(result.autoApproved.length).toBe(1);
    expect(result.needsReview.length).toBe(1);
    expect(result.rejected.length).toBe(1);
  });

  it("custom thresholds", () => {
    const pairs: ScoredPair[] = [{ idA: 1, idB: 2, score: 0.85 }];
    const result = gatePairs(pairs, { approveAbove: 0.8, rejectBelow: 0.5 });
    expect(result.autoApproved.length).toBe(1);
  });

  it("review items have canonical pair ids", () => {
    const pairs: ScoredPair[] = [{ idA: 5, idB: 3, score: 0.8 }];
    const result = gatePairs(pairs);
    expect(result.needsReview[0]!.pairId).toBe("3:5");
    expect(result.needsReview[0]!.idA).toBe(3);
    expect(result.needsReview[0]!.idB).toBe(5);
  });
});

describe("ReviewQueue lifecycle", () => {
  it("add -> approve", () => {
    const q = new ReviewQueue();
    q.add({ idA: 1, idB: 2, score: 0.8 });
    expect(q.size()).toBe(1);
    expect(q.pending().length).toBe(1);
    q.approve("1:2");
    expect(q.approved().length).toBe(1);
    expect(q.pending().length).toBe(0);
  });

  it("add -> reject", () => {
    const q = new ReviewQueue();
    q.add({ idA: 1, idB: 2, score: 0.8 });
    q.reject("1:2");
    expect(q.rejected().length).toBe(1);
  });

  it("canonicalizes pair id on add (2,1 same as 1,2)", () => {
    const q = new ReviewQueue();
    q.add({ idA: 2, idB: 1, score: 0.8 });
    q.add({ idA: 1, idB: 2, score: 0.9 }); // same pair -> idempotent
    expect(q.size()).toBe(1);
  });

  it("approve on unknown pair is a no-op", () => {
    const q = new ReviewQueue();
    q.approve("99:100"); // no throw
    expect(q.approved().length).toBe(0);
  });

  it("static pairIdFor helper", () => {
    expect(ReviewQueue.pairIdFor(5, 3)).toBe("3:5");
    expect(ReviewQueue.pairIdFor(1, 2)).toBe("1:2");
  });
});
