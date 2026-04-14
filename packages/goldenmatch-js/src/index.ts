/**
 * index.ts -- Main package entry point.
 *
 * Re-exports the edge-safe core API. For Node-only helpers (file I/O,
 * config loading, CLI), import from `goldenmatch/node`.
 */

export * from "./core/index.js";
