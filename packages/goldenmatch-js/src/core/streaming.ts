/**
 * streaming.ts — Incremental single-record match + cluster updates.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/streaming.py.
 */

import type { Row, MatchkeyConfig, ClusterInfo } from "./types.js";
import { addToCluster } from "./cluster.js";
import { matchOne } from "./match-one.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StreamAddResult {
  readonly rowId: number;
  readonly matchedIds: readonly number[];
  readonly clusterId: number;
}

export interface StreamProcessorConfig {
  readonly matchkey: MatchkeyConfig;
  readonly threshold: number;
  readonly maxClusterSize?: number;
}

export interface StreamSnapshot {
  readonly clusters: ReadonlyMap<number, ClusterInfo>;
  readonly rows: readonly Row[];
}

// ---------------------------------------------------------------------------
// StreamProcessor
// ---------------------------------------------------------------------------

/**
 * Incremental record processor.
 *
 * On each `add()` the new row is matched against all previously seen rows
 * using `matchOne`, then folded into the cluster map via `addToCluster`.
 */
export class StreamProcessor {
  private readonly clusters = new Map<number, ClusterInfo>();
  private readonly rowsById = new Map<number, Row>();
  private readonly order: number[] = [];
  private nextId = 0;

  constructor(private readonly config: StreamProcessorConfig) {}

  /** Add a new record and return match + cluster info. */
  add(row: Row): StreamAddResult {
    const rowId = (row["__row_id__"] as number | undefined) ?? this.nextId;
    if (typeof row["__row_id__"] !== "number") {
      // Attach row_id if missing
      row = { ...row, __row_id__: rowId };
    }
    if (rowId >= this.nextId) {
      this.nextId = rowId + 1;
    }

    // Matchkey with threshold override (exact variant has no threshold).
    const base = this.config.matchkey;
    const mk: MatchkeyConfig =
      base.type === "exact"
        ? base
        : { ...base, threshold: this.config.threshold };

    // Build snapshot of existing rows (exclude self if duplicate id)
    const existing: Row[] = [];
    for (const id of this.order) {
      if (id === rowId) continue;
      const r = this.rowsById.get(id);
      if (r !== undefined) existing.push(r);
    }

    const hits = matchOne(row, existing, mk);
    const matchPairs: [number, number][] = hits.map((h) => [h.rowId, h.score]);

    addToCluster(
      rowId,
      matchPairs,
      this.clusters,
      this.config.maxClusterSize ?? 100,
    );

    // Register the row
    if (!this.rowsById.has(rowId)) {
      this.order.push(rowId);
    }
    this.rowsById.set(rowId, row);

    // Find the cluster id the record landed in
    let landedCid = -1;
    for (const [cid, info] of this.clusters) {
      if (info.members.includes(rowId)) {
        landedCid = cid;
        break;
      }
    }

    return {
      rowId,
      matchedIds: hits.map((h) => h.rowId),
      clusterId: landedCid,
    };
  }

  /** Number of records ingested. */
  get size(): number {
    return this.rowsById.size;
  }

  /** Snapshot of current cluster state + rows. */
  snapshot(): StreamSnapshot {
    const rows: Row[] = [];
    for (const id of this.order) {
      const r = this.rowsById.get(id);
      if (r !== undefined) rows.push(r);
    }
    // Clone clusters to decouple callers from internal map
    const frozen = new Map<number, ClusterInfo>();
    for (const [cid, info] of this.clusters) {
      frozen.set(cid, info);
    }
    return { clusters: frozen, rows };
  }
}
