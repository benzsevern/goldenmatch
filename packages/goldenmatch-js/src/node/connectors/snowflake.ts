/**
 * snowflake.ts -- Snowflake connector via the optional `snowflake-sdk` peer dep.
 */

import { createRequire } from "node:module";
import type { Row } from "../../core/types.js";
import type { BaseConnector, ConnectorQuery } from "./base.js";

export interface SnowflakeConfig {
  readonly account: string;
  readonly username: string;
  readonly password?: string;
  readonly privateKey?: string;
  readonly warehouse?: string;
  readonly database?: string;
  readonly schema?: string;
  readonly role?: string;
}

export function createSnowflakeConnector(config: SnowflakeConfig): BaseConnector {
  const require = createRequire(import.meta.url);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let snowflake: any;
  try {
    snowflake = require("snowflake-sdk");
  } catch {
    throw new Error(
      "'snowflake-sdk' is required for the Snowflake connector. Install: npm install snowflake-sdk",
    );
  }

  const connection = snowflake.createConnection({
    account: config.account,
    username: config.username,
    password: config.password,
    privateKey: config.privateKey,
    warehouse: config.warehouse,
    database: config.database,
    schema: config.schema,
    role: config.role,
  });

  return {
    name: "snowflake",
    connect() {
      return new Promise<void>((resolve, reject) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        connection.connect((err: any) => (err ? reject(err) : resolve()));
      });
    },
    async read(query) {
      const sql = typeof query === "string" ? query : buildSelect(query);
      return new Promise<Row[]>((resolve, reject) => {
        connection.execute({
          sqlText: sql,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          complete: (err: any, _stmt: any, rows: Row[]) =>
            err ? reject(err) : resolve(rows ?? []),
        });
      });
    },
    async close() {
      await new Promise<void>((resolve) => connection.destroy(() => resolve()));
    },
  };
}

function buildSelect(q: ConnectorQuery): string {
  const cols = q.columns?.length ? q.columns.map((c) => `"${c}"`).join(",") : "*";
  let sql = `SELECT ${cols} FROM ${q.table}`;
  if (q.limit) sql += ` LIMIT ${q.limit}`;
  return sql;
}
