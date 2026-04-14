/**
 * duckdb.ts -- Optional DuckDB connector for Node.
 *
 * Mirrors `goldenmatch.backends.duckdb_backend.DuckDBBackend` from Python.
 *
 * Peer dependency (NOT in package.json -- install on demand):
 *   npm install @duckdb/node-api
 *
 * The dep is loaded via `createRequire` so the package stays importable
 * on edge runtimes and in environments without DuckDB.
 */

import { createRequire } from "node:module";
import type { Row } from "../../core/types.js";

export interface DuckDBConfig {
  /** Database path. Defaults to `:memory:`. */
  readonly path?: string;
}

export interface DuckDBConnector {
  readTable(table: string): Promise<Row[]>;
  readQuery(sql: string): Promise<Row[]>;
  writeTable(
    table: string,
    rows: readonly Row[],
    schema?: Readonly<Record<string, string>>,
  ): Promise<void>;
  listTables(): Promise<string[]>;
  close(): void;
}

/**
 * Create a DuckDB connector. Throws if `@duckdb/node-api` isn't installed.
 *
 * Async because the underlying DuckDB API is async-only (instance + connection
 * setup both return Promises).
 */
export async function createDuckDBConnector(
  config: DuckDBConfig = {},
): Promise<DuckDBConnector> {
  const require = createRequire(import.meta.url);
  let duckdb: {
    DuckDBInstance: {
      create: (path: string) => Promise<{
        connect: () => Promise<DuckDBConnection>;
        closeSync?: () => void;
      }>;
    };
  };
  try {
    duckdb = require("@duckdb/node-api") as typeof duckdb;
  } catch {
    throw new Error(
      "'@duckdb/node-api' is required for DuckDB support. Install: npm install @duckdb/node-api",
    );
  }

  const path = config.path ?? ":memory:";
  const instance = await duckdb.DuckDBInstance.create(path);
  const conn = await instance.connect();

  const escapeIdent = (s: string): string => `"${s.replace(/"/g, '""')}"`;

  return {
    async readTable(table: string): Promise<Row[]> {
      const reader = await conn.runAndReadAll(`SELECT * FROM ${escapeIdent(table)}`);
      return reader.getRowObjects() as Row[];
    },

    async readQuery(sql: string): Promise<Row[]> {
      const reader = await conn.runAndReadAll(sql);
      return reader.getRowObjects() as Row[];
    },

    async writeTable(
      table: string,
      rows: readonly Row[],
      schema?: Readonly<Record<string, string>>,
    ): Promise<void> {
      const tableIdent = escapeIdent(table);

      if (rows.length === 0) {
        // Empty -- create stub table from schema if provided, else no-op.
        if (schema !== undefined) {
          const colDefs = Object.entries(schema)
            .map(([c, t]) => `${escapeIdent(c)} ${t}`)
            .join(", ");
          await conn.run(`CREATE TABLE IF NOT EXISTS ${tableIdent} (${colDefs})`);
        }
        return;
      }

      const first = rows[0]!;
      const cols = Object.keys(first);
      const colDefs = cols
        .map((c) => `${escapeIdent(c)} ${schema?.[c] ?? "VARCHAR"}`)
        .join(", ");
      await conn.run(`CREATE TABLE IF NOT EXISTS ${tableIdent} (${colDefs})`);

      const placeholders = cols.map(() => "?").join(", ");
      const colList = cols.map(escapeIdent).join(", ");
      const prepared = await conn.prepare(
        `INSERT INTO ${tableIdent} (${colList}) VALUES (${placeholders})`,
      );

      for (const row of rows) {
        const values = cols.map((c) => (row as Record<string, unknown>)[c]);
        await prepared.run(...values);
      }
    },

    async listTables(): Promise<string[]> {
      const reader = await conn.runAndReadAll("SHOW TABLES");
      const out = reader.getRowObjects() as Array<Record<string, unknown>>;
      return out.map((r) => String(r["name"] ?? ""));
    },

    close(): void {
      instance.closeSync?.();
    },
  };
}

/** Internal: minimal shape of the @duckdb/node-api connection we touch. */
interface DuckDBConnection {
  runAndReadAll(sql: string): Promise<{ getRowObjects(): unknown[] }>;
  run(sql: string): Promise<unknown>;
  prepare(sql: string): Promise<{ run(...values: unknown[]): Promise<unknown> }>;
}
