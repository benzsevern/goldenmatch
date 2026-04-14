/**
 * index.ts -- Node.js entry point for GoldenMatch.
 *
 * Re-exports the edge-safe core API plus Node-only helpers that use
 * `node:fs`, `node:path`, and the optional `yaml` peer dependency.
 */

// Re-export everything from the edge-safe core
export * from "../core/index.js";

// Node-only file I/O
export {
  readCsv,
  readJson,
  readFile,
  writeCsv,
  writeJson,
} from "./connectors/file.js";
export type {
  ReadCsvOptions,
  WriteCsvOptions,
} from "./connectors/file.js";

// File-based dedupe / match
export { dedupeFile, matchFiles } from "./dedupe-file.js";
export type { FileDedupeOptions, FileSpec } from "./dedupe-file.js";

// YAML config file I/O
export { loadConfigFile, writeConfigFile } from "./config-file.js";

// Cloud connectors (registers built-in connectors as a side-effect)
export {
  createSnowflakeConnector,
  createBigQueryConnector,
  createDatabricksConnector,
  createSalesforceConnector,
  createHubSpotConnector,
  registerConnector,
  loadConnector,
  listConnectors,
} from "./connectors/index.js";
export type {
  BaseConnector,
  ConnectorConfig,
  ConnectorFactory,
  ConnectorQuery,
  SnowflakeConfig,
  BigQueryConfig,
  DatabricksConfig,
  SalesforceConfig,
  HubSpotConfig,
} from "./connectors/index.js";

// Servers (MCP / REST / A2A)
export { startMcpServer, handleTool, TOOLS } from "./mcp/server.js";
export { startApiServer, ReviewQueue } from "./api/server.js";
export type { StartApiOptions } from "./api/server.js";
export { startA2aServer, AGENT_CARD } from "./a2a/server.js";
export type { StartA2aOptions, AgentSkill } from "./a2a/server.js";

// Concurrent / parallel block scoring (Node-only; uses dynamic import of core scorer).
// `scoreBlocksParallel` uses piscina for true worker-thread parallelism when the
// optional `piscina` peer dep is installed; falls back to `scoreBlocksConcurrent`.
export {
  scoreBlocksConcurrent,
  scoreBlocksParallel,
} from "./backends/workers.js";
export type {
  WorkerPoolOptions,
  ParallelWorkerOptions,
} from "./backends/workers.js";

// Optional DuckDB connector (peer dep: @duckdb/node-api -- install on demand)
export { createDuckDBConnector } from "./backends/duckdb.js";
export type { DuckDBConfig, DuckDBConnector } from "./backends/duckdb.js";

// Optional Postgres connector + sync helpers (peer dep: pg -- install on demand)
export { createPostgresConnector } from "./db/postgres.js";
export type {
  PostgresConfig,
  PostgresConnector,
  PostgresWriteOptions,
} from "./db/postgres.js";
export { syncDedupe, watchSync } from "./db/sync.js";
export type { SyncOptions, WatchSyncOptions } from "./db/sync.js";

// Interactive TUI (optional peer deps: ink + react)
export { startTui } from "./tui/app.js";
export type { TuiOptions } from "./tui/app.js";
