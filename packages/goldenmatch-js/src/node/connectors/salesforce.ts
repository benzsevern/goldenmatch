/**
 * salesforce.ts -- Salesforce connector using the REST API via fetch().
 *
 * No SDK dependency: stays edge-adjacent. Supports either a pre-issued
 * accessToken or the OAuth 2.0 password grant flow.
 */

import type { Row } from "../../core/types.js";
import type { BaseConnector, ConnectorQuery } from "./base.js";

export interface SalesforceConfig {
  readonly instanceUrl: string; // e.g. "https://xxx.my.salesforce.com"
  readonly accessToken?: string;
  readonly clientId?: string;
  readonly clientSecret?: string;
  readonly username?: string;
  readonly password?: string;
  readonly securityToken?: string;
  readonly apiVersion?: string; // default "v61.0"
}

interface OAuthResponse {
  access_token: string;
  instance_url: string;
}

interface QueryResponse {
  records: Row[];
  done: boolean;
  nextRecordsUrl?: string;
}

export function createSalesforceConnector(config: SalesforceConfig): BaseConnector {
  const apiVersion = config.apiVersion ?? "v61.0";
  let accessToken = config.accessToken;
  let instanceUrl = config.instanceUrl;

  return {
    name: "salesforce",
    async connect() {
      if (accessToken) return;
      const body = new URLSearchParams({
        grant_type: "password",
        client_id: config.clientId ?? "",
        client_secret: config.clientSecret ?? "",
        username: config.username ?? "",
        password: (config.password ?? "") + (config.securityToken ?? ""),
      });
      const resp = await fetch(`${instanceUrl}/services/oauth2/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString(),
      });
      if (!resp.ok) {
        throw new Error(`Salesforce auth failed: ${resp.status} ${await resp.text()}`);
      }
      const j = (await resp.json()) as OAuthResponse;
      accessToken = j.access_token;
      instanceUrl = j.instance_url;
    },
    async read(query) {
      if (!accessToken) {
        throw new Error("Salesforce connector not connected. Call connect() first.");
      }
      const soql = typeof query === "string" ? query : buildSOQL(query);
      const url = `${instanceUrl}/services/data/${apiVersion}/query?q=${encodeURIComponent(soql)}`;
      const rows: Row[] = [];
      let next: string | undefined = url;
      while (next) {
        const resp = await fetch(next, {
          headers: { Authorization: `Bearer ${accessToken}` },
        });
        if (!resp.ok) {
          throw new Error(`Salesforce query failed: ${resp.status} ${await resp.text()}`);
        }
        const j = (await resp.json()) as QueryResponse;
        rows.push(...j.records);
        next = j.nextRecordsUrl ? `${instanceUrl}${j.nextRecordsUrl}` : undefined;
      }
      return rows;
    },
    async close() {
      // Stateless: nothing to tear down.
    },
  };
}

function buildSOQL(q: ConnectorQuery): string {
  const cols = q.columns?.length ? q.columns.join(",") : "Id";
  let sql = `SELECT ${cols} FROM ${q.table}`;
  if (q.limit) sql += ` LIMIT ${q.limit}`;
  return sql;
}
