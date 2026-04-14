/**
 * postgres.ts -- Optional Postgres connector for Node.
 *
 * Mirrors `goldenmatch.db.connector.PostgresConnector` from Python.
 *
 * Peer dependency (NOT in package.json -- install on demand):
 *   npm install pg
 *
 * The dep is loaded via `createRequire` so the package stays importable
 * on edge runtimes and in environments without Postgres.
 */

import { createRequire } from "node:module";
import type { Row } from "../../core/types.js";

export interface PostgresConfig {
  readonly connectionString?: string;
  readonly host?: string;
  readonly port?: number;
  readonly database?: string;
  readonly user?: string;
  readonly password?: string;
  readonly ssl?: boolean;
}

export interface PostgresWriteOptions {
  readonly upsert?: boolean;
  readonly primaryKey?: string;
}

export interface PostgresConnector {
  connect(): Promise<void>;
  query<T = Row>(sql: string, params?: readonly unknown[]): Promise<T[]>;
  readTable(table: string): Promise<Row[]>;
  writeTable(
    table: string,
    rows: readonly Row[],
    options?: PostgresWriteOptions,
  ): Promise<void>;
  listTables(schema?: string): Promise<string[]>;
  close(): Promise<void>;
}

/** Minimal shape of `pg.Client` we use. */
interface PgClient {
  connect(): Promise<void>;
  query(sql: string, params?: readonly unknown[]): Promise<{ rows: unknown[] }>;
  end(): Promise<void>;
}

interface PgModule {
  Client: new (config: unknown) => PgClient;
}

/**
 * Create a Postgres connector. Throws if `pg` isn't installed.
 *
 * The returned connector requires `connect()` before any query. Inserts
 * are batched in chunks of 1000 rows. When `options.upsert` is set, the
 * write uses `INSERT ... ON CONFLICT (primaryKey) DO UPDATE`.
 */
export function createPostgresConnector(
  config: PostgresConfig,
): PostgresConnector {
  const require = createRequire(import.meta.url);
  let pg: PgModule;
  try {
    pg = require("pg") as PgModule;
  } catch {
    throw new Error(
      "'pg' is required for Postgres support. Install: npm install pg",
    );
  }

  const clientConfig: Record<string, unknown> = {};
  if (config.connectionString !== undefined) {
    clientConfig["connectionString"] = config.connectionString;
  }
  if (config.host !== undefined) clientConfig["host"] = config.host;
  if (config.port !== undefined) clientConfig["port"] = config.port;
  if (config.database !== undefined) clientConfig["database"] = config.database;
  if (config.user !== undefined) clientConfig["user"] = config.user;
  if (config.password !== undefined) clientConfig["password"] = config.password;
  if (config.ssl !== undefined) clientConfig["ssl"] = config.ssl;

  const client = new pg.Client(clientConfig);

  const escapeIdent = (s: string): string => `"${s.replace(/"/g, '""')}"`;

  const writeTable = async (
    table: string,
    rows: readonly Row[],
    options: PostgresWriteOptions = {},
  ): Promise<void> => {
    if (rows.length === 0) return;

    const first = rows[0]!;
    const cols = Object.keys(first);
    const tableIdent = escapeIdent(table);
    const colNames = cols.map(escapeIdent).join(", ");

    const BATCH_SIZE = 1000;
    for (let i = 0; i < rows.length; i += BATCH_SIZE) {
      const batch = rows.slice(i, i + BATCH_SIZE);

      const valueClauses = batch
        .map((_row, idx) => {
          const offset = idx * cols.length;
          return `(${cols.map((_, j) => `$${offset + j + 1}`).join(", ")})`;
        })
        .join(", ");

      const params: unknown[] = [];
      for (const row of batch) {
        for (const c of cols) {
          params.push((row as Record<string, unknown>)[c]);
        }
      }

      let sql = `INSERT INTO ${tableIdent} (${colNames}) VALUES ${valueClauses}`;

      if (options.upsert === true && options.primaryKey !== undefined) {
        const pk = options.primaryKey;
        const updates = cols
          .filter((c) => c !== pk)
          .map((c) => `${escapeIdent(c)} = EXCLUDED.${escapeIdent(c)}`)
          .join(", ");
        if (updates.length > 0) {
          sql += ` ON CONFLICT (${escapeIdent(pk)}) DO UPDATE SET ${updates}`;
        } else {
          sql += ` ON CONFLICT (${escapeIdent(pk)}) DO NOTHING`;
        }
      }

      await client.query(sql, params);
    }
  };

  return {
    async connect(): Promise<void> {
      await client.connect();
    },

    async query<T = Row>(
      sql: string,
      params: readonly unknown[] = [],
    ): Promise<T[]> {
      const result = await client.query(sql, params);
      return result.rows as T[];
    },

    async readTable(table: string): Promise<Row[]> {
      const result = await client.query(`SELECT * FROM ${escapeIdent(table)}`);
      return result.rows as Row[];
    },

    writeTable,

    async listTables(schema: string = "public"): Promise<string[]> {
      const result = await client.query(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
        [schema],
      );
      return (result.rows as Array<Record<string, unknown>>).map((r) =>
        String(r["table_name"] ?? ""),
      );
    },

    async close(): Promise<void> {
      await client.end();
    },
  };
}
