/**
 * config-file.ts -- YAML config loading/saving from disk.
 *
 * Node-only. Uses `createRequire` so the optional `yaml` peer dependency
 * is resolved lazily without breaking edge-safe ESM builds.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { createRequire } from "node:module";
import { parseConfigYaml, configToYaml } from "../core/config/loader.js";
import type { GoldenMatchConfig } from "../core/types.js";

const require = createRequire(import.meta.url);

interface YamlModule {
  parse: (s: string) => unknown;
  stringify: (v: unknown) => string;
}

function loadYamlModule(): YamlModule {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const mod = require("yaml") as YamlModule;
    if (typeof mod.parse !== "function" || typeof mod.stringify !== "function") {
      throw new Error("'yaml' module missing parse/stringify exports");
    }
    return mod;
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new Error(
      `'yaml' package is required for config file I/O. Install: npm install yaml (${detail})`,
    );
  }
}

/**
 * Load and parse a YAML config file into a typed GoldenMatchConfig.
 *
 * @throws if the file cannot be read, `yaml` is not installed, or the
 *   document does not describe a valid config.
 */
export function loadConfigFile(path: string): GoldenMatchConfig {
  const resolved = resolve(path);
  if (!existsSync(resolved)) {
    throw new Error(`Config file not found: ${resolved}`);
  }
  const content = readFileSync(resolved, "utf8");
  const yamlMod = loadYamlModule();
  return parseConfigYaml(content, yamlMod.parse);
}

/**
 * Serialize a GoldenMatchConfig to YAML and write it to disk.
 * Creates parent directories as needed.
 */
export function writeConfigFile(path: string, config: GoldenMatchConfig): void {
  const resolved = resolve(path);
  const dir = dirname(resolved);
  if (dir && dir !== "." && !existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  const yamlMod = loadYamlModule();
  const yamlStr = configToYaml(config, yamlMod.stringify);
  writeFileSync(resolved, yamlStr, "utf8");
}
