#!/usr/bin/env node
/**
 * cli.ts -- GoldenMatch command-line interface.
 *
 * Built on commander. Exposes `dedupe`, `match`, `score`, `profile`,
 * `info`, and `demo` subcommands.
 */

import { Command } from "commander";
import { extname, basename } from "node:path";
import {
  readFile,
  writeCsv,
  writeJson,
} from "./node/connectors/file.js";
import { dedupe, match, scoreStrings } from "./core/api.js";
import { loadConfigFile } from "./node/config-file.js";
import type { Row } from "./core/types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseFuzzyArg(raw: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const pair of raw.split(",")) {
    const trimmed = pair.trim();
    if (trimmed === "") continue;
    const idx = trimmed.indexOf(":");
    let field: string;
    let threshold = 0.85;
    if (idx === -1) {
      field = trimmed;
    } else {
      field = trimmed.slice(0, idx).trim();
      const rawThreshold = trimmed.slice(idx + 1).trim();
      const parsed = parseFloat(rawThreshold);
      if (Number.isFinite(parsed)) threshold = parsed;
    }
    if (field !== "") out[field] = threshold;
  }
  return out;
}

function parseCsvList(raw: string): string[] {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function loadFilesWithSource(paths: readonly string[]): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < paths.length; i++) {
    const p = paths[i]!;
    const source = basename(p, extname(p)) || `file_${i}`;
    const fileRows = readFile(p);
    for (const r of fileRows) {
      rows.push({ ...r, __source__: source });
    }
  }
  return rows;
}

interface SharedMatchOpts {
  config?: string;
  exact?: string;
  fuzzy?: string;
  blocking?: string;
  threshold?: number;
  output?: string;
  format?: string;
}

function buildOptionsFromFlags(opts: SharedMatchOpts) {
  const out: {
    config?: ReturnType<typeof loadConfigFile>;
    exact?: string[];
    fuzzy?: Record<string, number>;
    blocking?: string[];
    threshold?: number;
  } = {};
  if (opts.config) out.config = loadConfigFile(opts.config);
  if (opts.exact) out.exact = parseCsvList(opts.exact);
  if (opts.fuzzy) out.fuzzy = parseFuzzyArg(opts.fuzzy);
  if (opts.blocking) out.blocking = parseCsvList(opts.blocking);
  if (opts.threshold !== undefined) out.threshold = opts.threshold;
  return out;
}

function writeOutputRows(
  path: string,
  rows: readonly Row[],
  format: string,
): void {
  const ext = extname(path).toLowerCase();
  const useJson =
    format === "json" ||
    ext === ".json" ||
    ext === ".jsonl" ||
    ext === ".ndjson";
  if (useJson) {
    writeJson(path, rows);
  } else {
    const delimiter = ext === ".tsv" ? "\t" : ",";
    writeCsv(path, rows, { delimiter });
  }
}

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

const program = new Command();

program
  .name("goldenmatch-js")
  .description("Entity resolution toolkit -- dedupe, match, build golden records")
  .version("0.1.0");

// ---------- dedupe ----------
program
  .command("dedupe")
  .description("Deduplicate records in one or more files")
  .argument("<files...>", "input file paths (.csv, .tsv, .json, .jsonl)")
  .option("-c, --config <path>", "path to YAML config file")
  .option("-e, --exact <fields>", "comma-separated exact match fields")
  .option(
    "-f, --fuzzy <fields>",
    "fuzzy match fields, e.g. 'name:0.85,email:0.9'",
  )
  .option("-b, --blocking <fields>", "comma-separated blocking keys")
  .option("-t, --threshold <value>", "overall fuzzy threshold", parseFloat)
  .option("-o, --output <path>", "output path for golden records")
  .option("--format <format>", "output format: csv or json", "csv")
  .action(async (files: string[], opts: SharedMatchOpts) => {
    const rows = loadFilesWithSource(files);
    const options = buildOptionsFromFlags(opts);
    const result = dedupe(rows, options);
    const pct = (result.stats.matchRate * 100).toFixed(1);
    process.stdout.write(
      `Dedupe complete: ${result.stats.totalRecords} records -> ${result.stats.totalClusters} clusters (${pct}% match rate)\n`,
    );
    if (opts.output) {
      writeOutputRows(
        opts.output,
        result.goldenRecords,
        opts.format ?? "csv",
      );
      process.stdout.write(
        `Wrote ${result.goldenRecords.length} golden records to ${opts.output}\n`,
      );
    }
  });

// ---------- match ----------
program
  .command("match")
  .description("Match target records against a reference dataset")
  .argument("<target>", "target file path")
  .argument("<reference>", "reference file path")
  .option("-c, --config <path>", "path to YAML config file")
  .option("-e, --exact <fields>", "comma-separated exact match fields")
  .option(
    "-f, --fuzzy <fields>",
    "fuzzy match fields, e.g. 'name:0.85,email:0.9'",
  )
  .option("-b, --blocking <fields>", "comma-separated blocking keys")
  .option("-t, --threshold <value>", "overall fuzzy threshold", parseFloat)
  .option("-o, --output <path>", "output path for matched records")
  .option("--format <format>", "output format: csv or json", "csv")
  .action(
    async (targetPath: string, referencePath: string, opts: SharedMatchOpts) => {
      const targetRows = readFile(targetPath).map((row) => ({
        ...row,
        __source__: "target",
      }));
      const referenceRows = readFile(referencePath).map((row) => ({
        ...row,
        __source__: "reference",
      }));
      const options = buildOptionsFromFlags(opts);
      const result = match(targetRows, referenceRows, options);
      process.stdout.write(
        `Match complete: ${result.matched.length} matched, ${result.unmatched.length} unmatched\n`,
      );
      if (opts.output) {
        writeOutputRows(
          opts.output,
          result.matched,
          opts.format ?? "csv",
        );
        process.stdout.write(
          `Wrote ${result.matched.length} matched records to ${opts.output}\n`,
        );
      }
    },
  );

// ---------- score ----------
program
  .command("score")
  .description("Score similarity between two strings")
  .argument("<a>", "first string")
  .argument("<b>", "second string")
  .option(
    "-s, --scorer <name>",
    "scorer: exact, jaro_winkler, levenshtein, token_sort, soundex_match, dice, jaccard, ensemble",
    "jaro_winkler",
  )
  .action((a: string, b: string, opts: { scorer: string }) => {
    const score = scoreStrings(a, b, opts.scorer);
    process.stdout.write(`${opts.scorer}: ${score.toFixed(4)}\n`);
  });

// ---------- info ----------
program
  .command("info")
  .description("Show information about the package")
  .action(() => {
    process.stdout.write("GoldenMatch JS v0.1.0\n");
    process.stdout.write(
      "Scorers: exact, jaro_winkler, levenshtein, token_sort, soundex_match, dice, jaccard, ensemble\n",
    );
    process.stdout.write(
      "Strategies: most_complete, majority_vote, source_priority, most_recent, first_non_null\n",
    );
    process.stdout.write(
      "Blocking: static, multi_pass, sorted_neighborhood, adaptive\n",
    );
    process.stdout.write(
      "Transforms: lowercase, uppercase, strip, soundex, metaphone, digits_only, alpha_only, token_sort\n",
    );
  });

// ---------- profile ----------
program
  .command("profile")
  .description("Profile a dataset (column stats, nulls, cardinality)")
  .argument("<file>", "input file")
  .action((file: string) => {
    const rows = readFile(file);
    const total = rows.length;
    process.stdout.write(`File: ${file}\n`);
    process.stdout.write(`Rows: ${total}\n`);
    if (total === 0) return;
    const columns = new Set<string>();
    for (const r of rows) for (const k of Object.keys(r)) columns.add(k);
    process.stdout.write(`Columns: ${columns.size}\n`);
    process.stdout.write("\n");
    const colList = [...columns];
    const nameWidth = Math.max(6, ...colList.map((c) => c.length));
    const pad = (s: string, w: number) => s + " ".repeat(Math.max(0, w - s.length));
    process.stdout.write(
      `${pad("column", nameWidth)}  ${pad("nulls", 8)}  ${pad("null%", 7)}  ${pad("distinct", 9)}  sample\n`,
    );
    process.stdout.write(
      `${"-".repeat(nameWidth)}  ${"-".repeat(8)}  ${"-".repeat(7)}  ${"-".repeat(9)}  ------\n`,
    );
    for (const col of colList) {
      let nulls = 0;
      const distinct = new Set<string>();
      let sample: string | null = null;
      for (const row of rows) {
        const v = row[col];
        if (v === null || v === undefined || v === "") {
          nulls++;
        } else {
          const s = String(v);
          distinct.add(s);
          if (sample === null) sample = s;
        }
      }
      const nullPct = ((nulls / total) * 100).toFixed(1);
      const sampleStr = sample === null ? "-" : sample.length > 30 ? sample.slice(0, 27) + "..." : sample;
      process.stdout.write(
        `${pad(col, nameWidth)}  ${pad(String(nulls), 8)}  ${pad(nullPct + "%", 7)}  ${pad(String(distinct.size), 9)}  ${sampleStr}\n`,
      );
    }
  });

// ---------- demo ----------
program
  .command("demo")
  .description("Run a quick demo on synthetic data")
  .action(() => {
    const rows: Row[] = [
      { id: 1, name: "John Smith", email: "john@example.com", zip: "01234" },
      { id: 2, name: "Jon Smith", email: "john@example.com", zip: "01234" },
      { id: 3, name: "Jane Doe", email: "jane@example.com", zip: "02139" },
      { id: 4, name: "J. Doe", email: "jane@example.com", zip: "02139" },
      { id: 5, name: "Bob Jones", email: "bob@example.com", zip: "10001" },
    ];
    process.stdout.write(`Input: ${rows.length} synthetic records\n`);
    const result = dedupe(rows, {
      exact: ["email"],
      fuzzy: { name: 0.8 },
      blocking: ["zip"],
      threshold: 0.8,
    });
    process.stdout.write(
      `Dedupe: ${result.stats.totalRecords} records -> ${result.stats.totalClusters} clusters\n`,
    );
    process.stdout.write(
      `Match rate: ${(result.stats.matchRate * 100).toFixed(1)}%\n`,
    );
    process.stdout.write(`Golden records: ${result.goldenRecords.length}\n`);
    for (const g of result.goldenRecords) {
      process.stdout.write(`  ${JSON.stringify(g)}\n`);
    }
  });

// ---------- mcp-serve ----------
program
  .command("mcp-serve")
  .description("Start MCP server over stdio (JSON-RPC 2.0)")
  .action(async () => {
    const { startMcpServer } = await import("./node/mcp/server.js");
    startMcpServer();
  });

// ---------- serve (REST API) ----------
program
  .command("serve")
  .description("Start the REST API server")
  .option("-p, --port <port>", "port", "8000")
  .option("-h, --host <host>", "host", "127.0.0.1")
  .action(async (opts: { port: string; host: string }) => {
    const { startApiServer } = await import("./node/api/server.js");
    startApiServer({ port: parseInt(opts.port, 10), host: opts.host });
  });

// ---------- agent-serve (A2A) ----------
program
  .command("agent-serve")
  .description("Start the A2A agent-to-agent server")
  .option("-p, --port <port>", "port", "8200")
  .option("-h, --host <host>", "host", "127.0.0.1")
  .action(async (opts: { port: string; host: string }) => {
    const { startA2aServer } = await import("./node/a2a/server.js");
    startA2aServer({ port: parseInt(opts.port, 10), host: opts.host });
  });

// ---------- tui ----------
program
  .command("tui")
  .description("Launch interactive TUI (requires optional peer deps: ink + react)")
  .argument("[files...]", "input files to load on startup")
  .option("-c, --config <path>", "path to YAML config file")
  .action(async (files: string[], opts: { config?: string }) => {
    try {
      const { startTui } = await import("./node/tui/app.js");
      const tuiOpts: { files?: string[]; config?: ReturnType<typeof loadConfigFile> } = {};
      if (files && files.length > 0) tuiOpts.files = files;
      if (opts.config) tuiOpts.config = loadConfigFile(opts.config);
      await startTui(tuiOpts);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      process.stderr.write(`TUI error: ${message}\n`);
      process.exit(1);
    }
  });

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

program.parseAsync(process.argv).catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err);
  process.stderr.write(`Error: ${message}\n`);
  process.exit(1);
});
