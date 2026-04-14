/**
 * Build a full GoldenMatchConfig manually (matchkeys, standardization,
 * blocking, golden rules). Save to YAML, reload, use it.
 *
 * Run: npx tsx examples/05-custom-config.ts
 * Requires: `npm install yaml` (optional peer dep for YAML I/O).
 */
import { writeFileSync, unlinkSync } from "node:fs";
import {
  dedupe,
  makeConfig,
  makeMatchkeyConfig,
  makeMatchkeyField,
  makeBlockingConfig,
  makeGoldenRulesConfig,
} from "goldenmatch";
import { loadConfigFile, writeConfigFile } from "goldenmatch/node";

// Build config programmatically
const config = makeConfig({
  matchkeys: [
    makeMatchkeyConfig({
      name: "email_exact",
      type: "exact",
      fields: [makeMatchkeyField({ field: "email", transforms: ["lowercase", "strip"], scorer: "exact" })],
    }),
    makeMatchkeyConfig({
      name: "identity",
      type: "weighted",
      threshold: 0.85,
      fields: [
        makeMatchkeyField({ field: "first_name", transforms: ["lowercase"], scorer: "jaro_winkler", weight: 0.3 }),
        makeMatchkeyField({ field: "last_name",  transforms: ["lowercase"], scorer: "jaro_winkler", weight: 0.4 }),
        makeMatchkeyField({ field: "phone",      transforms: ["digits_only"], scorer: "exact",     weight: 0.3 }),
      ],
    }),
  ],
  blocking: makeBlockingConfig({
    strategy: "multi_pass",
    keys: [{ fields: ["zip"], transforms: ["lowercase", "strip"] }],
    passes: [
      { fields: ["zip"],        transforms: ["lowercase", "strip"] },
      { fields: ["last_name"],  transforms: ["soundex"] },
    ],
  }),
  goldenRules: makeGoldenRulesConfig({
    defaultStrategy: "most_complete",
    fieldRules: {
      email: { strategy: "first_non_null" },
      phone: { strategy: "most_complete" },
    },
  }),
  standardization: {
    rules: {
      email:      ["email"],
      phone:      ["phone"],
      first_name: ["strip", "name_proper"],
      last_name:  ["strip", "name_proper"],
    },
  },
  threshold: 0.85,
});

// Save to YAML (requires `yaml` peer dep)
const YAML_PATH = "custom_config_demo.yml";
try {
  writeConfigFile(YAML_PATH, config);
  console.log(`Wrote config to ${YAML_PATH}`);

  const reloaded = loadConfigFile(YAML_PATH);
  console.log(`Reloaded config has ${reloaded.matchkeys?.length ?? 0} matchkeys`);

  unlinkSync(YAML_PATH);
} catch (err) {
  console.warn(`YAML save/load skipped: ${(err as Error).message}`);
  console.warn("(install the `yaml` peer dep to enable: npm install yaml)");
}

// Use the config
const rows = [
  { id: 1, first_name: "john", last_name: "smith", email: "j@x.com",  phone: "555-123-4567", zip: "12345" },
  { id: 2, first_name: "John", last_name: "Smyth", email: "J@X.COM",  phone: "(555) 123-4567", zip: "12345" },
  { id: 3, first_name: "Jane", last_name: "Doe",   email: "jd@x.com", phone: "555-000-0000", zip: "54321" },
];

const result = dedupe(rows, { config });
console.log(`\nDeduped: ${result.stats.totalRecords} -> ${result.stats.totalClusters} clusters`);

/**
 * Python -> TS differences:
 *   - Python: `StandardizationConfig(rules={...})`; TS: `{ rules: {...} }` plain object.
 *   - Python: pydantic validation; TS: `makeConfig` normalizes defaults, `parseConfig`
 *     (used internally by loadConfigFile) validates YAML input.
 */
