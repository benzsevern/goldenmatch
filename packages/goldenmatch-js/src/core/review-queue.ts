/**
 * review-queue.ts — Human-in-the-loop pair gating.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/review_queue.py. Default gates: >=0.95 auto-approve,
 * <0.75 auto-reject, everything in between needs review.
 */

import type { ScoredPair } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ReviewStatus = "pending" | "approved" | "rejected";

export interface ReviewItem {
  readonly pairId: string;
  readonly idA: number;
  readonly idB: number;
  readonly score: number;
  readonly status: ReviewStatus;
  readonly createdAt: number;
}

export interface GatedResult {
  readonly autoApproved: readonly ScoredPair[];
  readonly needsReview: readonly ReviewItem[];
  readonly rejected: readonly ScoredPair[];
}

export interface GateOptions {
  readonly approveAbove?: number;
  readonly rejectBelow?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function canonicalIds(a: number, b: number): [number, number] {
  return a < b ? [a, b] : [b, a];
}

function pairIdFor(a: number, b: number): string {
  const [lo, hi] = canonicalIds(a, b);
  return `${lo}:${hi}`;
}

function now(): number {
  // Date.now is edge-safe (no node imports).
  return Date.now();
}

// ---------------------------------------------------------------------------
// gatePairs
// ---------------------------------------------------------------------------

/**
 * Split pairs into auto-approved, needs-review, and rejected buckets.
 *
 * Defaults: approveAbove=0.95, rejectBelow=0.75.
 */
export function gatePairs(
  pairs: readonly ScoredPair[],
  options?: GateOptions,
): GatedResult {
  const approveAbove = options?.approveAbove ?? 0.95;
  const rejectBelow = options?.rejectBelow ?? 0.75;

  const autoApproved: ScoredPair[] = [];
  const needsReview: ReviewItem[] = [];
  const rejected: ScoredPair[] = [];
  const t = now();

  for (const p of pairs) {
    if (p.score >= approveAbove) {
      autoApproved.push(p);
    } else if (p.score < rejectBelow) {
      rejected.push(p);
    } else {
      const [lo, hi] = canonicalIds(p.idA, p.idB);
      needsReview.push({
        pairId: `${lo}:${hi}`,
        idA: lo,
        idB: hi,
        score: p.score,
        status: "pending",
        createdAt: t,
      });
    }
  }

  return { autoApproved, needsReview, rejected };
}

// ---------------------------------------------------------------------------
// ReviewQueue
// ---------------------------------------------------------------------------

/**
 * In-memory review queue for human adjudication of borderline pairs.
 */
export class ReviewQueue {
  private readonly items = new Map<string, ReviewItem>();

  /** Add a pair as a pending review item (idempotent by canonical pair id). */
  add(pair: ScoredPair): void {
    const [lo, hi] = canonicalIds(pair.idA, pair.idB);
    const pairId = `${lo}:${hi}`;
    if (this.items.has(pairId)) return;
    this.items.set(pairId, {
      pairId,
      idA: lo,
      idB: hi,
      score: pair.score,
      status: "pending",
      createdAt: now(),
    });
  }

  /** Get an item by canonical pair id ("minId:maxId"). */
  get(pairId: string): ReviewItem | undefined {
    return this.items.get(pairId);
  }

  /** Mark a pair approved. No-op if unknown. */
  approve(pairId: string): void {
    const item = this.items.get(pairId);
    if (item === undefined) return;
    this.items.set(pairId, { ...item, status: "approved" });
  }

  /** Mark a pair rejected. No-op if unknown. */
  reject(pairId: string): void {
    const item = this.items.get(pairId);
    if (item === undefined) return;
    this.items.set(pairId, { ...item, status: "rejected" });
  }

  /** All pending items. */
  pending(): ReviewItem[] {
    const out: ReviewItem[] = [];
    for (const item of this.items.values()) {
      if (item.status === "pending") out.push(item);
    }
    return out;
  }

  /** All approved items. */
  approved(): ReviewItem[] {
    const out: ReviewItem[] = [];
    for (const item of this.items.values()) {
      if (item.status === "approved") out.push(item);
    }
    return out;
  }

  /** All rejected items. */
  rejected(): ReviewItem[] {
    const out: ReviewItem[] = [];
    for (const item of this.items.values()) {
      if (item.status === "rejected") out.push(item);
    }
    return out;
  }

  /** Current queue size. */
  size(): number {
    return this.items.size;
  }

  /** Canonical pair id helper ("minId:maxId"). */
  static pairIdFor(a: number, b: number): string {
    return pairIdFor(a, b);
  }
}
