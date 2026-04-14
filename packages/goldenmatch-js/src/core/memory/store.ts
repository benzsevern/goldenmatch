/**
 * memory/store.ts — Learning Memory store (in-memory backend).
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/core/memory/store.py. SQLite / Postgres backends are
 * deferred (they require host-specific drivers); the in-memory backend
 * keeps all corrections in a plain array with trust-based upsert.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Correction {
  readonly rowIdA: number;
  readonly rowIdB: number;
  readonly verdict: "match" | "no_match";
  readonly feature: string;
  readonly score: number;
  readonly timestamp: number;
  readonly trust: number;
  readonly source: string;
}

export interface MemoryStoreConfig {
  readonly backend: "memory" | "sqlite" | "postgres";
  readonly path?: string;
  readonly trustDefault?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pairFeatureKey(c: Correction): string {
  const [a, b] = c.rowIdA < c.rowIdB ? [c.rowIdA, c.rowIdB] : [c.rowIdB, c.rowIdA];
  return `${a}|${b}|${c.feature}`;
}

// ---------------------------------------------------------------------------
// MemoryStore
// ---------------------------------------------------------------------------

export class MemoryStore {
  private corrections: Correction[] = [];

  constructor(private readonly config: MemoryStoreConfig = { backend: "memory" }) {
    if (config.backend !== "memory") {
      // SQLite/Postgres backends intentionally unsupported in edge-safe code.
      // Callers that need persistence should swap in a host-specific wrapper.
      // We don't throw here to keep the class usable for tests.
    }
  }

  /** Append a correction unconditionally. */
  add(correction: Correction): void {
    this.corrections.push(correction);
  }

  /** Append many corrections unconditionally. */
  addBatch(corrections: readonly Correction[]): void {
    for (const c of corrections) this.corrections.push(c);
  }

  /** All corrections, in insertion order. */
  list(): readonly Correction[] {
    return this.corrections;
  }

  /** Corrections whose verdict is "match". */
  listMatches(): readonly Correction[] {
    return this.corrections.filter((c) => c.verdict === "match");
  }

  /** Corrections whose verdict is "no_match". */
  listNonMatches(): readonly Correction[] {
    return this.corrections.filter((c) => c.verdict === "no_match");
  }

  count(): number {
    return this.corrections.length;
  }

  clear(): void {
    this.corrections = [];
  }

  /**
   * Trust-based upsert: if a correction for the same (pair, feature) already
   * exists, keep whichever has higher trust. Ties break toward the more recent
   * correction.
   */
  upsert(correction: Correction): void {
    const key = pairFeatureKey(correction);
    for (let i = 0; i < this.corrections.length; i++) {
      const existing = this.corrections[i]!;
      if (pairFeatureKey(existing) !== key) continue;

      const newer = correction.timestamp >= existing.timestamp;
      const higherTrust = correction.trust > existing.trust;
      const equalTrustButNewer = correction.trust === existing.trust && newer;
      if (higherTrust || equalTrustButNewer) {
        this.corrections[i] = correction;
      }
      return;
    }
    this.corrections.push(correction);
  }

  /** Return the effective config (for debugging). */
  getConfig(): MemoryStoreConfig {
    return this.config;
  }
}
