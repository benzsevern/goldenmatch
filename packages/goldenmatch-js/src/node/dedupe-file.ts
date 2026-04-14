/**
 * dedupe-file.ts -- File-based convenience wrappers around dedupe() / match().
 *
 * Node-only: reads from disk, tags rows with __source__, delegates to the
 * edge-safe core API.
 */

import { readFile, writeCsv, writeJson } from "./connectors/file.js";
import { dedupe, match } from "../core/api.js";
import type {
  Row,
  DedupeResult,
  MatchResult,
  GoldenMatchConfig,
} from "../core/types.js";
import { extname, basename } from "node:path";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/**
 * File specification. Either a bare path (source name is derived from the
 * file's basename without extension) or a tuple `[path, sourceName]`.
 */
export type FileSpec = string | readonly [string, string];

export interface FileDedupeOptions {
  /** Input files. Required when calling `dedupeFile(opts)`. */
  readonly files?: readonly FileSpec[];
  /** Full config -- takes precedence over shorthand fields below. */
  readonly config?: GoldenMatchConfig;
  readonly exact?: readonly string[];
  readonly fuzzy?: Readonly<Record<string, number>>;
  readonly blocking?: readonly string[];
  readonly threshold?: number;
  /** Enable LLM scorer for borderline pairs (not yet implemented in JS). */
  readonly llmScorer?: boolean;
  /** Write golden records to this path (.csv or .json). */
  readonly outputPath?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function defaultSourceName(path: string, index: number): string {
  const base = basename(path, extname(path));
  return base && base.length > 0 ? base : `file_${index}`;
}

function resolveFileSpec(
  spec: FileSpec,
  index: number,
): { path: string; sourceName: string } {
  if (typeof spec === "string") {
    return { path: spec, sourceName: defaultSourceName(spec, index) };
  }
  const [path, sourceName] = spec;
  return { path, sourceName };
}

function loadRowsWithSource(files: readonly FileSpec[]): Row[] {
  const all: Row[] = [];
  for (let i = 0; i < files.length; i++) {
    const { path, sourceName } = resolveFileSpec(files[i]!, i);
    const rows = readFile(path);
    for (const row of rows) {
      all.push({ ...row, __source__: sourceName });
    }
  }
  return all;
}

function buildCoreOptions(opts: FileDedupeOptions) {
  const core: {
    config?: GoldenMatchConfig;
    exact?: readonly string[];
    fuzzy?: Readonly<Record<string, number>>;
    blocking?: readonly string[];
    threshold?: number;
    llmScorer?: boolean;
  } = {};
  if (opts.config !== undefined) core.config = opts.config;
  if (opts.exact !== undefined) core.exact = opts.exact;
  if (opts.fuzzy !== undefined) core.fuzzy = opts.fuzzy;
  if (opts.blocking !== undefined) core.blocking = opts.blocking;
  if (opts.threshold !== undefined) core.threshold = opts.threshold;
  if (opts.llmScorer !== undefined) core.llmScorer = opts.llmScorer;
  return core;
}

function writeGoldenRecords(
  outputPath: string,
  rows: readonly Row[],
): void {
  const ext = extname(outputPath).toLowerCase();
  if (ext === ".json" || ext === ".jsonl" || ext === ".ndjson") {
    writeJson(outputPath, rows);
  } else {
    // Default to CSV (includes `.csv`, `.tsv`, or unknown extensions).
    const delimiter = ext === ".tsv" ? "\t" : ",";
    writeCsv(outputPath, rows, { delimiter });
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Deduplicate records across one or more files.
 *
 * Each file's rows are tagged with `__source__ = <sourceName>` before being
 * concatenated and passed to `dedupe()`.
 *
 * @throws if no files are provided or any file cannot be read.
 */
export function dedupeFile(opts: FileDedupeOptions): DedupeResult {
  if (!opts.files || opts.files.length === 0) {
    throw new Error("dedupeFile: at least one input file is required");
  }
  const rows = loadRowsWithSource(opts.files);
  const result = dedupe(rows, buildCoreOptions(opts));
  if (opts.outputPath) {
    writeGoldenRecords(opts.outputPath, result.goldenRecords);
  }
  return result;
}

/**
 * Match target records against a reference file.
 *
 * Reads both files, tags with `__source__`, and delegates to `match()`.
 * `opts.files` (if provided) is ignored in favor of the explicit paths.
 */
export function matchFiles(
  targetPath: string,
  referencePath: string,
  opts?: FileDedupeOptions,
): MatchResult {
  const targetRows = readFile(targetPath).map((row) => ({
    ...row,
    __source__: "target",
  }));
  const referenceRows = readFile(referencePath).map((row) => ({
    ...row,
    __source__: "reference",
  }));

  const result = match(targetRows, referenceRows, buildCoreOptions(opts ?? {}));
  if (opts?.outputPath) {
    writeGoldenRecords(opts.outputPath, result.matched);
  }
  return result;
}
