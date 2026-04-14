/**
 * bigquery.ts -- Google BigQuery connector via the optional `@google-cloud/bigquery` peer dep.
 */

import { createRequire } from "node:module";
import type { Row } from "../../core/types.js";
import type { BaseConnector, ConnectorQuery } from "./base.js";

export interface BigQueryConfig {
  readonly projectId: string;
  readonly keyFilename?: string;
  readonly credentials?: Readonly<Record<string, unknown>>;
  readonly dataset?: string;
  readonly location?: string;
}

export function createBigQueryConnector(config: BigQueryConfig): BaseConnector {
  const require = createRequire(import.meta.url);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let bq: any;
  try {
    bq = require("@google-cloud/bigquery");
  } catch {
    throw new Error(
      "'@google-cloud/bigquery' is required for the BigQuery connector. Install: npm install @google-cloud/bigquery",
    );
  }

  const client = new bq.BigQuery({
    projectId: config.projectId,
    keyFilename: config.keyFilename,
    credentials: config.credentials,
    location: config.location,
  });

  return {
    name: "bigquery",
    async connect() {
      // No-op: BigQuery client uses REST per-query.
    },
    async read(query) {
      const sql =
        typeof query === "string"
          ? query
          : buildSelect(query, config.dataset);
      const [rows] = await client.query({ query: sql, location: config.location });
      return rows as Row[];
    },
    async close() {
      // No persistent connection to tear down.
    },
  };
}

function buildSelect(q: ConnectorQuery, dataset?: string): string {
  const cols = q.columns?.length ? q.columns.join(",") : "*";
  const tableRef = `${dataset ? `${dataset}.` : ""}${q.table}`;
  let sql = `SELECT ${cols} FROM \`${tableRef}\``;
  if (q.limit) sql += ` LIMIT ${q.limit}`;
  return sql;
}
