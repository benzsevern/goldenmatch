/**
 * base.ts -- Base connector interface and registry.
 *
 * Mirrors goldenmatch.connectors.base from the Python package: a small
 * abstraction over external data sources (Snowflake, BigQuery, etc.) that
 * exposes connect/read/close lifecycle and a name-based registry.
 */

import type { Row } from "../../core/types.js";

export interface ConnectorConfig {
  readonly [key: string]: unknown;
}

export interface ConnectorQuery {
  readonly table: string;
  readonly columns?: readonly string[];
  readonly limit?: number;
}

export interface BaseConnector {
  readonly name: string;
  connect(): Promise<void>;
  read(query: string | ConnectorQuery): Promise<Row[]>;
  close(): Promise<void>;
}

export interface ConnectorFactory<C extends ConnectorConfig = ConnectorConfig> {
  (config: C): BaseConnector;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

const registry = new Map<string, ConnectorFactory>();

export function registerConnector(name: string, factory: ConnectorFactory): void {
  registry.set(name, factory);
}

export function loadConnector<C extends ConnectorConfig = ConnectorConfig>(
  name: string,
  config: C,
): BaseConnector {
  const factory = registry.get(name);
  if (!factory) {
    throw new Error(
      `Unknown connector: ${name}. Registered: ${[...registry.keys()].join(", ")}`,
    );
  }
  return factory(config);
}

export function listConnectors(): readonly string[] {
  return [...registry.keys()];
}
