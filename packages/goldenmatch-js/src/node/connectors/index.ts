/**
 * index.ts -- Connector registry bootstrap and re-exports.
 */

import { registerConnector, loadConnector, listConnectors } from "./base.js";
import { createSnowflakeConnector, type SnowflakeConfig } from "./snowflake.js";
import { createBigQueryConnector, type BigQueryConfig } from "./bigquery.js";
import { createDatabricksConnector, type DatabricksConfig } from "./databricks.js";
import { createSalesforceConnector, type SalesforceConfig } from "./salesforce.js";
import { createHubSpotConnector, type HubSpotConfig } from "./hubspot.js";

registerConnector("snowflake", (c) => createSnowflakeConnector(c as unknown as SnowflakeConfig));
registerConnector("bigquery", (c) => createBigQueryConnector(c as unknown as BigQueryConfig));
registerConnector("databricks", (c) => createDatabricksConnector(c as unknown as DatabricksConfig));
registerConnector("salesforce", (c) => createSalesforceConnector(c as unknown as SalesforceConfig));
registerConnector("hubspot", (c) => createHubSpotConnector(c as unknown as HubSpotConfig));

export {
  createSnowflakeConnector,
  createBigQueryConnector,
  createDatabricksConnector,
  createSalesforceConnector,
  createHubSpotConnector,
  registerConnector,
  loadConnector,
  listConnectors,
};

export type {
  BaseConnector,
  ConnectorConfig,
  ConnectorFactory,
  ConnectorQuery,
} from "./base.js";
export type { SnowflakeConfig } from "./snowflake.js";
export type { BigQueryConfig } from "./bigquery.js";
export type { DatabricksConfig } from "./databricks.js";
export type { SalesforceConfig } from "./salesforce.js";
export type { HubSpotConfig } from "./hubspot.js";

// Re-export the existing local-file connector for convenience.
export { readFile, readCsv, readJson, writeCsv, writeJson } from "./file.js";
export type { ReadCsvOptions, WriteCsvOptions } from "./file.js";
