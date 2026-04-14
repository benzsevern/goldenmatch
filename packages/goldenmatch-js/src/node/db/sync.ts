/**
 * sync.ts -- Postgres-backed dedupe sync + watch helpers.
 *
 * Mirrors `goldenmatch.db.sync` from Python: read source table, run
 * dedupe, write golden + cluster tables back to Postgres.
 */

import { dedupe } from "../../core/api.js";
import type { DedupeResult, GoldenMatchConfig } from "../../core/types.js";
import {
  createPostgresConnector,
  type PostgresConfig,
} from "./postgres.js";

export interface SyncOptions {
  readonly pg: PostgresConfig;
  readonly sourceTable: string;
  readonly goldenTable: string;
  /** Optional table to write cluster summaries (cluster_id, members, size, ...). */
  readonly clustersTable?: string;
  readonly config: GoldenMatchConfig;
}

/**
 * Run a single dedupe pass against Postgres.
 *
 * 1. Read all rows from `sourceTable`.
 * 2. Dedupe via the core pipeline.
 * 3. Write golden records to `goldenTable`.
 * 4. Optionally write cluster summaries to `clustersTable`.
 *
 * Always closes the connection on the way out.
 */
export async function syncDedupe(options: SyncOptions): Promise<DedupeResult> {
  const conn = createPostgresConnector(options.pg);
  try {
    await conn.connect();
    const rows = await conn.readTable(options.sourceTable);
    const result = dedupe(rows, { config: options.config });

    await conn.writeTable(options.goldenTable, result.goldenRecords);

    if (options.clustersTable !== undefined && result.clusters.size > 0) {
      const clusterRows = Array.from(result.clusters.entries()).map(
        ([id, c]) => ({
          cluster_id: id,
          members: JSON.stringify(c.members),
          size: c.size,
          confidence: c.confidence,
          quality: c.clusterQuality,
        }),
      );
      await conn.writeTable(options.clustersTable, clusterRows);
    }

    return result;
  } finally {
    await conn.close();
  }
}

export interface WatchSyncOptions extends SyncOptions {
  /** Polling interval in ms. Defaults to 60_000 (1 minute). */
  readonly intervalMs?: number;
}

/**
 * Run `syncDedupe` on a recurring interval.
 *
 * Returns a `stop` function. Errors in any iteration are logged via
 * `console.warn` so the loop keeps running; callers should monitor
 * `onResult` to confirm forward progress.
 */
export async function watchSync(
  options: WatchSyncOptions,
  onResult?: (result: DedupeResult) => void,
): Promise<() => void> {
  const intervalMs = options.intervalMs ?? 60_000;
  let stopped = false;

  const loop = async (): Promise<void> => {
    while (!stopped) {
      try {
        const result = await syncDedupe(options);
        onResult?.(result);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn("syncDedupe failed:", msg);
      }
      if (stopped) break;
      await new Promise<void>((resolve) => setTimeout(resolve, intervalMs));
    }
  };

  loop().catch((err: unknown) => {
    const msg = err instanceof Error ? err.message : String(err);
    console.warn("watchSync loop error:", msg);
  });

  return (): void => {
    stopped = true;
  };
}
