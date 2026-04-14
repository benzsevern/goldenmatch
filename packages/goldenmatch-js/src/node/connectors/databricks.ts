/**
 * databricks.ts -- Databricks SQL warehouse connector via the optional `@databricks/sql` peer dep.
 */

import { createRequire } from "node:module";
import type { Row } from "../../core/types.js";
import type { BaseConnector, ConnectorQuery } from "./base.js";

export interface DatabricksConfig {
  readonly serverHostname: string;
  readonly httpPath: string;
  readonly token: string;
  readonly catalog?: string;
  readonly schema?: string;
}

export function createDatabricksConnector(config: DatabricksConfig): BaseConnector {
  const require = createRequire(import.meta.url);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let db: any;
  try {
    db = require("@databricks/sql");
  } catch {
    throw new Error(
      "'@databricks/sql' is required for the Databricks connector. Install: npm install @databricks/sql",
    );
  }

  const client = new db.DBSQLClient();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let session: any = null;

  return {
    name: "databricks",
    async connect() {
      await client.connect({
        host: config.serverHostname,
        path: config.httpPath,
        token: config.token,
      });
      session = await client.openSession({
        initialCatalog: config.catalog,
        initialSchema: config.schema,
      });
    },
    async read(query) {
      if (!session) throw new Error("Databricks connector not connected. Call connect() first.");
      const sql = typeof query === "string" ? query : buildSelect(query);
      const operation = await session.executeStatement(sql);
      const rows = await operation.fetchAll();
      await operation.close();
      return rows as Row[];
    },
    async close() {
      if (session) {
        await session.close();
        session = null;
      }
      await client.close();
    },
  };
}

function buildSelect(q: ConnectorQuery): string {
  const cols = q.columns?.length ? q.columns.map((c) => `\`${c}\``).join(",") : "*";
  let sql = `SELECT ${cols} FROM ${q.table}`;
  if (q.limit) sql += ` LIMIT ${q.limit}`;
  return sql;
}
