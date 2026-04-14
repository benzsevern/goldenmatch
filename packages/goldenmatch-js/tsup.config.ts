import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "core/index": "src/core/index.ts",
    "node/index": "src/node/index.ts",
    cli: "src/cli.ts",
    // Separate entry so piscina can load it at runtime from disk.
    "node/backends/score-worker": "src/node/backends/score-worker.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "node20",
  splitting: false,
  treeshake: true,
  external: [
    "hnswlib-node",
    "@huggingface/transformers",
    "piscina",
    "ink",
    "ink-table",
    "ink-select-input",
    "ink-text-input",
    "ink-spinner",
    "ink-gradient",
    "react",
    "pg",
    "@duckdb/node-api",
    "snowflake-sdk",
    "@google-cloud/bigquery",
    "@databricks/sql",
    "yaml",
  ],
});
