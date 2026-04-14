/**
 * hubspot.ts -- HubSpot CRM connector via REST API + fetch().
 *
 * `query.table` selects the HubSpot object: "contacts", "companies", "deals", etc.
 * SQL strings are not supported -- use object queries.
 */

import type { Row } from "../../core/types.js";
import type { BaseConnector } from "./base.js";

export interface HubSpotConfig {
  readonly apiKey: string; // HubSpot private app access token
  readonly apiBase?: string; // default "https://api.hubapi.com"
}

interface HubSpotResult {
  id: string;
  properties: Record<string, unknown>;
}

interface HubSpotPage {
  results: HubSpotResult[];
  paging?: { next?: { link: string } };
}

export function createHubSpotConnector(config: HubSpotConfig): BaseConnector {
  const base = config.apiBase ?? "https://api.hubapi.com";

  return {
    name: "hubspot",
    async connect() {
      // No-op: stateless REST.
    },
    async read(query) {
      if (typeof query === "string") {
        throw new Error("HubSpot connector requires an object query (table/columns/limit), not SQL.");
      }
      const limit = query.limit ?? 100;
      const propsParam = query.columns?.length ? `&properties=${query.columns.join(",")}` : "";
      const endpoint = `${base}/crm/v3/objects/${query.table}?limit=${limit}${propsParam}`;
      const rows: Row[] = [];
      let next: string | undefined = endpoint;
      while (next) {
        const resp = await fetch(next, {
          headers: { Authorization: `Bearer ${config.apiKey}` },
        });
        if (!resp.ok) {
          throw new Error(`HubSpot query failed: ${resp.status} ${await resp.text()}`);
        }
        const j = (await resp.json()) as HubSpotPage;
        for (const r of j.results) {
          rows.push({ id: r.id, ...r.properties });
        }
        next = j.paging?.next?.link;
      }
      return rows;
    },
    async close() {
      // Stateless.
    },
  };
}
