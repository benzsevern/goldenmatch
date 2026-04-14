/**
 * mcp/server.ts -- GoldenMatch MCP server (stdio transport, JSON-RPC).
 *
 * Node-only: uses node:fs, node:path, node:readline. NOT edge-safe.
 *
 * Exposes ~20 tools covering dedupe, match, scoring, explanation,
 * profiling, auto-config (shorthand), evaluation, and listings.
 *
 * Every tool dispatch is wrapped in try/catch so a single failure never
 * crashes the JSON-RPC loop; errors come back as `{ error: "<msg>" }`.
 *
 * Ports ideas from goldenmatch/mcp/server.py.
 */

import { readFileSync } from "node:fs";
import { resolve, isAbsolute } from "node:path";
import { createInterface } from "node:readline";

import { dedupe, match, scoreStrings } from "../../core/api.js";
import { readFile, writeCsv, writeJson } from "../connectors/file.js";
import { loadConfigFile } from "../config-file.js";
import type { Row, MatchkeyField } from "../../core/types.js";
import {
  makeMatchkeyConfig,
  makeMatchkeyField,
  VALID_SCORERS,
  VALID_TRANSFORMS,
  VALID_STRATEGIES,
} from "../../core/types.js";
import {
  scoreField,
  findExactMatches,
  findFuzzyMatches,
  scorePair,
} from "../../core/scorer.js";
import { addRowIds } from "../../core/matchkey.js";
import { buildClusters } from "../../core/cluster.js";
import { explainPair, explainCluster } from "../../core/explain.js";
import { profileRows } from "../../core/profiler.js";
import { evaluatePairs, loadGroundTruthPairs } from "../../core/evaluate.js";

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

interface Tool {
  readonly name: string;
  readonly description: string;
  readonly inputSchema: Readonly<Record<string, unknown>>;
}

const pathArg = { type: "string", description: "File path (csv/tsv/json/jsonl)" };
const optionalConfigArg = {
  type: "string",
  description: "Optional path to YAML config file",
};
const optionalFieldsArg = {
  type: "array",
  items: { type: "string" },
  description: "Column names",
};
const stringArg = { type: "string" };
const rowArg = {
  type: "object",
  additionalProperties: true,
  description: "Record object (column -> value)",
};

export const TOOLS: readonly Tool[] = [
  {
    name: "dedupe",
    description:
      "Deduplicate records in a file. Returns cluster counts and optional output path.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        config: optionalConfigArg,
        exact: optionalFieldsArg,
        fuzzy: {
          type: "object",
          additionalProperties: { type: "number" },
          description: "Map of field -> fuzzy threshold",
        },
        blocking: optionalFieldsArg,
        threshold: { type: "number", description: "Overall fuzzy threshold" },
        output: { type: "string", description: "Optional output path for golden records" },
      },
      required: ["path"],
    },
  },
  {
    name: "match",
    description:
      "Match a target file against a reference file. Returns matched/unmatched counts.",
    inputSchema: {
      type: "object",
      properties: {
        target: pathArg,
        reference: pathArg,
        config: optionalConfigArg,
        exact: optionalFieldsArg,
        fuzzy: {
          type: "object",
          additionalProperties: { type: "number" },
        },
        blocking: optionalFieldsArg,
        threshold: { type: "number" },
        output: { type: "string" },
      },
      required: ["target", "reference"],
    },
  },
  {
    name: "score_strings",
    description:
      "Score similarity between two strings using the requested scorer.",
    inputSchema: {
      type: "object",
      properties: {
        a: stringArg,
        b: stringArg,
        scorer: {
          type: "string",
          description:
            "Scorer name (exact, jaro_winkler, levenshtein, token_sort, soundex_match, dice, jaccard, ensemble)",
        },
      },
      required: ["a", "b"],
    },
  },
  {
    name: "score_pair",
    description:
      "Score two record objects across weighted fields. Returns a combined score.",
    inputSchema: {
      type: "object",
      properties: {
        row_a: rowArg,
        row_b: rowArg,
        fields: {
          type: "array",
          items: {
            type: "object",
            properties: {
              field: { type: "string" },
              scorer: { type: "string" },
              weight: { type: "number" },
              transforms: { type: "array", items: { type: "string" } },
            },
            required: ["field"],
          },
        },
      },
      required: ["row_a", "row_b", "fields"],
    },
  },
  {
    name: "explain_pair",
    description:
      "Explain why two records match (or don't) using a matchkey definition.",
    inputSchema: {
      type: "object",
      properties: {
        row_a: rowArg,
        row_b: rowArg,
        fields: {
          type: "array",
          items: {
            type: "object",
            properties: {
              field: { type: "string" },
              scorer: { type: "string" },
              weight: { type: "number" },
              transforms: { type: "array", items: { type: "string" } },
            },
            required: ["field"],
          },
        },
        threshold: { type: "number" },
      },
      required: ["row_a", "row_b", "fields"],
    },
  },
  {
    name: "explain_cluster",
    description:
      "Run dedupe on a file and explain the cluster containing the given row id.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        config: optionalConfigArg,
        exact: optionalFieldsArg,
        fuzzy: {
          type: "object",
          additionalProperties: { type: "number" },
        },
        blocking: optionalFieldsArg,
        row_id: { type: "number" },
      },
      required: ["path", "row_id"],
    },
  },
  {
    name: "profile",
    description:
      "Profile a dataset: per-column null rate, cardinality, inferred type, samples.",
    inputSchema: {
      type: "object",
      properties: { path: pathArg },
      required: ["path"],
    },
  },
  {
    name: "suggest_config",
    description:
      "Suggest a shorthand dedupe config based on a profile of the dataset.",
    inputSchema: {
      type: "object",
      properties: { path: pathArg },
      required: ["path"],
    },
  },
  {
    name: "evaluate",
    description:
      "Evaluate predicted pairs from a dedupe run against ground truth pairs.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        ground_truth: pathArg,
        id_col_a: { type: "string", description: "Ground truth id column A (default id_a)" },
        id_col_b: { type: "string", description: "Ground truth id column B (default id_b)" },
        config: optionalConfigArg,
        exact: optionalFieldsArg,
        fuzzy: {
          type: "object",
          additionalProperties: { type: "number" },
        },
        blocking: optionalFieldsArg,
        threshold: { type: "number" },
      },
      required: ["path", "ground_truth"],
    },
  },
  {
    name: "find_exact_matches",
    description: "Find exact matches on a field in a file. Returns pairs.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        field: { type: "string" },
        transforms: {
          type: "array",
          items: { type: "string" },
          description: "Transforms applied before matching (default lowercase, strip)",
        },
      },
      required: ["path", "field"],
    },
  },
  {
    name: "find_fuzzy_matches",
    description: "Find fuzzy matches in a block of rows. Returns scored pairs.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        field: { type: "string" },
        scorer: { type: "string", description: "Scorer (default jaro_winkler)" },
        threshold: { type: "number", description: "Threshold (default 0.85)" },
        transforms: { type: "array", items: { type: "string" } },
      },
      required: ["path", "field"],
    },
  },
  {
    name: "build_clusters",
    description:
      "Group records into clusters given a file and matchkey definition.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        exact: optionalFieldsArg,
        fuzzy: {
          type: "object",
          additionalProperties: { type: "number" },
        },
        blocking: optionalFieldsArg,
        threshold: { type: "number" },
      },
      required: ["path"],
    },
  },
  {
    name: "list_scorers",
    description: "List all available similarity scorers.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "list_transforms",
    description: "List all available field transforms.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "list_strategies",
    description: "List all golden-record survivorship strategies.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "list_blocking_strategies",
    description: "List all blocking strategy names.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "server_info",
    description: "Return metadata about this GoldenMatch MCP server.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "read_file",
    description: "Read a CSV/JSON file and return the first N records.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        limit: { type: "number", description: "Max rows to return (default 100)" },
      },
      required: ["path"],
    },
  },
  {
    name: "write_csv",
    description: "Write a list of record objects to a CSV file.",
    inputSchema: {
      type: "object",
      properties: {
        path: pathArg,
        rows: { type: "array", items: { type: "object", additionalProperties: true } },
      },
      required: ["path", "rows"],
    },
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sanitizePath(raw: string): string {
  if (typeof raw !== "string" || raw.length === 0) {
    throw new Error("path must be a non-empty string");
  }
  const resolved = isAbsolute(raw) ? resolve(raw) : resolve(process.cwd(), raw);
  const cwd = resolve(process.cwd());
  if (!resolved.startsWith(cwd)) {
    throw new Error(`Path '${raw}' is outside the working directory`);
  }
  return resolved;
}

function asStringArray(v: unknown): string[] | undefined {
  if (v === undefined || v === null) return undefined;
  if (!Array.isArray(v)) return undefined;
  return v.map((x) => String(x));
}

function asNumberMap(v: unknown): Record<string, number> | undefined {
  if (v === undefined || v === null) return undefined;
  if (typeof v !== "object" || Array.isArray(v)) return undefined;
  const out: Record<string, number> = {};
  for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
    const n = typeof val === "number" ? val : Number(val);
    if (Number.isFinite(n)) out[k] = n;
  }
  return out;
}

interface ShorthandArgs {
  exact?: readonly string[];
  fuzzy?: Readonly<Record<string, number>>;
  blocking?: readonly string[];
  threshold?: number;
  configPath?: string;
}

function buildDedupeOptions(args: Record<string, unknown>): {
  config?: ReturnType<typeof loadConfigFile>;
  exact?: readonly string[];
  fuzzy?: Readonly<Record<string, number>>;
  blocking?: readonly string[];
  threshold?: number;
} {
  const opts: {
    config?: ReturnType<typeof loadConfigFile>;
    exact?: readonly string[];
    fuzzy?: Readonly<Record<string, number>>;
    blocking?: readonly string[];
    threshold?: number;
  } = {};

  if (typeof args["config"] === "string" && args["config"]) {
    opts.config = loadConfigFile(sanitizePath(args["config"] as string));
  }
  const exact = asStringArray(args["exact"]);
  if (exact) opts.exact = exact;
  const fuzzy = asNumberMap(args["fuzzy"]);
  if (fuzzy) opts.fuzzy = fuzzy;
  const blocking = asStringArray(args["blocking"]);
  if (blocking) opts.blocking = blocking;
  if (typeof args["threshold"] === "number") opts.threshold = args["threshold"];

  return opts;
}

function buildFieldsFromArg(raw: unknown): MatchkeyField[] {
  if (!Array.isArray(raw)) {
    throw new Error("fields must be an array of field configs");
  }
  const out: MatchkeyField[] = [];
  for (const entry of raw) {
    if (entry === null || typeof entry !== "object") continue;
    const e = entry as Record<string, unknown>;
    if (typeof e["field"] !== "string") {
      throw new Error("each field entry needs a string 'field' property");
    }
    const transforms = asStringArray(e["transforms"]) ?? ["lowercase", "strip"];
    const scorer = typeof e["scorer"] === "string" ? (e["scorer"] as string) : "jaro_winkler";
    const weight = typeof e["weight"] === "number" ? (e["weight"] as number) : 1.0;
    out.push(
      makeMatchkeyField({
        field: e["field"] as string,
        transforms,
        scorer,
        weight,
      }),
    );
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

export async function handleTool(
  name: string,
  rawArgs: Record<string, unknown>,
): Promise<unknown> {
  const args = rawArgs ?? {};
  try {
    switch (name) {
      case "dedupe": {
        const path = sanitizePath(String(args["path"]));
        const rows = readFile(path);
        const options = buildDedupeOptions(args);
        const result = dedupe(rows, options);
        let output_written: string | null = null;
        if (typeof args["output"] === "string" && args["output"]) {
          const outPath = sanitizePath(args["output"] as string);
          try {
            writeCsv(outPath, result.goldenRecords);
            output_written = outPath;
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            return {
              stats: result.stats,
              total_clusters: result.stats.totalClusters,
              total_records: result.stats.totalRecords,
              match_rate: result.stats.matchRate,
              output_error: msg,
            };
          }
        }
        return {
          total_records: result.stats.totalRecords,
          total_clusters: result.stats.totalClusters,
          match_rate: result.stats.matchRate,
          matched_records: result.stats.matchedRecords,
          unique_records: result.stats.uniqueRecords,
          golden_records_count: result.goldenRecords.length,
          output_written,
        };
      }

      case "match": {
        const targetPath = sanitizePath(String(args["target"]));
        const referencePath = sanitizePath(String(args["reference"]));
        const targetRows = readFile(targetPath);
        const referenceRows = readFile(referencePath);
        const options = buildDedupeOptions(args);
        const result = match(
          targetRows.map((r) => ({ ...r, __source__: "target" })),
          referenceRows.map((r) => ({ ...r, __source__: "reference" })),
          options,
        );
        let output_written: string | null = null;
        if (typeof args["output"] === "string" && args["output"]) {
          const outPath = sanitizePath(args["output"] as string);
          try {
            writeCsv(outPath, result.matched);
            output_written = outPath;
          } catch (err) {
            return {
              matched: result.matched.length,
              unmatched: result.unmatched.length,
              output_error: err instanceof Error ? err.message : String(err),
            };
          }
        }
        return {
          matched: result.matched.length,
          unmatched: result.unmatched.length,
          output_written,
        };
      }

      case "score_strings": {
        const a = String(args["a"] ?? "");
        const b = String(args["b"] ?? "");
        const scorer =
          typeof args["scorer"] === "string" ? (args["scorer"] as string) : "jaro_winkler";
        const score = scoreStrings(a, b, scorer);
        return { scorer, score };
      }

      case "score_pair": {
        const rowA = args["row_a"] as Row;
        const rowB = args["row_b"] as Row;
        if (!rowA || !rowB) throw new Error("row_a and row_b are required");
        const fields = buildFieldsFromArg(args["fields"]);
        const score = scorePair(rowA, rowB, fields);
        return { score, field_count: fields.length };
      }

      case "explain_pair": {
        const rowA = args["row_a"] as Row;
        const rowB = args["row_b"] as Row;
        if (!rowA || !rowB) throw new Error("row_a and row_b are required");
        const fields = buildFieldsFromArg(args["fields"]);
        const threshold =
          typeof args["threshold"] === "number" ? (args["threshold"] as number) : 0.85;
        const mk = makeMatchkeyConfig({
          name: "adhoc",
          type: "weighted",
          fields,
          threshold,
        });
        const explanation = explainPair(rowA, rowB, mk);
        return {
          score: explanation.score,
          confidence: explanation.confidence,
          explanation: explanation.explanation,
          field_scores: explanation.fieldScores,
        };
      }

      case "explain_cluster": {
        const path = sanitizePath(String(args["path"]));
        const rowId = Number(args["row_id"]);
        if (!Number.isFinite(rowId)) {
          throw new Error("row_id must be a number");
        }
        const rows = readFile(path);
        const options = buildDedupeOptions(args);
        const result = dedupe(rows, options);
        // Find cluster containing rowId
        let foundId: number | null = null;
        let found: typeof result.clusters extends ReadonlyMap<number, infer V> ? V : never;
        found = undefined as unknown as typeof found;
        for (const [cid, info] of result.clusters.entries()) {
          if (info.members.includes(rowId)) {
            foundId = cid;
            found = info;
            break;
          }
        }
        if (foundId === null || !found) {
          return { error: `row_id ${rowId} not found in any cluster` };
        }
        // Get matchkey
        const mks = (result.config.matchkeys ?? []) as readonly ReturnType<
          typeof makeMatchkeyConfig
        >[];
        const mk =
          mks.length > 0
            ? mks[0]!
            : makeMatchkeyConfig({
                name: "placeholder",
                type: "weighted",
                fields: [
                  makeMatchkeyField({
                    field: Object.keys(rows[0] ?? {})[0] ?? "",
                    transforms: ["lowercase", "strip"],
                    scorer: "jaro_winkler",
                  }),
                ],
              });
        const withIds = addRowIds(rows);
        const explanation = explainCluster(foundId, found, withIds, mk);
        return {
          cluster_id: explanation.clusterId,
          size: explanation.size,
          confidence: explanation.confidence,
          quality: explanation.quality,
          summary: explanation.summary,
        };
      }

      case "profile": {
        const path = sanitizePath(String(args["path"]));
        const rows = readFile(path);
        const profile = profileRows(rows);
        return {
          row_count: profile.rowCount,
          columns: profile.columns.map((c) => ({
            name: c.name,
            inferred_type: c.inferredType,
            null_count: c.nullCount,
            null_rate: c.nullRate,
            distinct_count: c.distinctCount,
            cardinality_ratio: c.cardinalityRatio,
            avg_length: c.avgLength,
            max_length: c.maxLength,
            sample_values: c.sampleValues,
          })),
        };
      }

      case "suggest_config": {
        const path = sanitizePath(String(args["path"]));
        const rows = readFile(path);
        const profile = profileRows(rows);
        const exact: string[] = [];
        const fuzzy: Record<string, number> = {};
        const blocking: string[] = [];

        for (const col of profile.columns) {
          if (col.nullRate > 0.2) continue;
          if (col.inferredType === "email") {
            if (col.cardinalityRatio >= 0.5) exact.push(col.name);
          } else if (col.inferredType === "zip") {
            blocking.push(col.name);
          } else if (col.inferredType === "name") {
            fuzzy[col.name] = 0.85;
          } else if (col.inferredType === "phone") {
            if (col.cardinalityRatio >= 0.5) exact.push(col.name);
          } else if (col.inferredType === "geo") {
            blocking.push(col.name);
          } else if (col.inferredType === "text" && col.avgLength > 4) {
            fuzzy[col.name] = 0.8;
          }
        }

        return {
          row_count: profile.rowCount,
          suggested: {
            exact,
            fuzzy,
            blocking,
            threshold: 0.85,
          },
        };
      }

      case "evaluate": {
        const path = sanitizePath(String(args["path"]));
        const gtPath = sanitizePath(String(args["ground_truth"]));
        const idColA =
          typeof args["id_col_a"] === "string" ? (args["id_col_a"] as string) : "id_a";
        const idColB =
          typeof args["id_col_b"] === "string" ? (args["id_col_b"] as string) : "id_b";
        const rows = readFile(path);
        const gtRows = readFile(gtPath);
        const options = buildDedupeOptions(args);
        const result = dedupe(rows, options);
        const truth = loadGroundTruthPairs(gtRows, idColA, idColB);
        const metrics = evaluatePairs(result.scoredPairs, truth);
        return {
          tp: metrics.truePositives,
          fp: metrics.falsePositives,
          fn: metrics.falseNegatives,
          precision: metrics.precision,
          recall: metrics.recall,
          f1: metrics.f1,
          total_predicted: result.scoredPairs.length,
          total_truth: truth.length,
        };
      }

      case "find_exact_matches": {
        const path = sanitizePath(String(args["path"]));
        const field = String(args["field"]);
        const transforms = asStringArray(args["transforms"]) ?? ["lowercase", "strip"];
        const rows = addRowIds(readFile(path));
        const mk = makeMatchkeyConfig({
          name: "adhoc_exact",
          type: "exact",
          fields: [makeMatchkeyField({ field, transforms, scorer: "exact" })],
        });
        const pairs = findExactMatches(rows, mk);
        return {
          pair_count: pairs.length,
          pairs: pairs.slice(0, 100).map((p) => [p.idA, p.idB, p.score]),
        };
      }

      case "find_fuzzy_matches": {
        const path = sanitizePath(String(args["path"]));
        const field = String(args["field"]);
        const scorer =
          typeof args["scorer"] === "string" ? (args["scorer"] as string) : "jaro_winkler";
        const threshold =
          typeof args["threshold"] === "number" ? (args["threshold"] as number) : 0.85;
        const transforms = asStringArray(args["transforms"]) ?? ["lowercase", "strip"];
        const rows = addRowIds(readFile(path));
        const mk = makeMatchkeyConfig({
          name: "adhoc_fuzzy",
          type: "weighted",
          fields: [makeMatchkeyField({ field, transforms, scorer })],
          threshold,
        });
        const pairs = findFuzzyMatches(rows, mk);
        return {
          pair_count: pairs.length,
          pairs: pairs.slice(0, 100).map((p) => [p.idA, p.idB, p.score]),
        };
      }

      case "build_clusters": {
        const path = sanitizePath(String(args["path"]));
        const options = buildDedupeOptions(args);
        const rows = readFile(path);
        const result = dedupe(rows, options);
        const clusters: Array<{
          cluster_id: number;
          size: number;
          confidence: number;
          quality: string;
          members: readonly number[];
        }> = [];
        for (const [cid, info] of result.clusters.entries()) {
          clusters.push({
            cluster_id: cid,
            size: info.size,
            confidence: info.confidence,
            quality: info.clusterQuality,
            members: info.members,
          });
        }
        return {
          cluster_count: clusters.length,
          clusters: clusters.slice(0, 200),
        };
      }

      case "list_scorers":
        return { scorers: [...VALID_SCORERS] };

      case "list_transforms":
        return { transforms: [...VALID_TRANSFORMS] };

      case "list_strategies":
        return { strategies: [...VALID_STRATEGIES] };

      case "list_blocking_strategies":
        return {
          strategies: [
            "static",
            "adaptive",
            "sorted_neighborhood",
            "multi_pass",
            "ann",
            "canopy",
            "ann_pairs",
            "learned",
          ],
        };

      case "server_info":
        return {
          name: "goldenmatch-js",
          version: "0.1.0",
          tool_count: TOOLS.length,
          description:
            "Node-only GoldenMatch MCP server over stdio (JSON-RPC 2.0)",
        };

      case "read_file": {
        const path = sanitizePath(String(args["path"]));
        const limit =
          typeof args["limit"] === "number" ? Math.max(0, Math.floor(args["limit"] as number)) : 100;
        const rows = readFile(path);
        return {
          total: rows.length,
          returned: Math.min(rows.length, limit),
          rows: rows.slice(0, limit),
        };
      }

      case "write_csv": {
        const path = sanitizePath(String(args["path"]));
        const rowsArg = args["rows"];
        if (!Array.isArray(rowsArg)) {
          throw new Error("rows must be an array of objects");
        }
        writeCsv(path, rowsArg as Row[]);
        return { written: rowsArg.length, path };
      }

      default:
        return { error: `Unknown tool: ${name}` };
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { error: msg };
  }
}

// ---------------------------------------------------------------------------
// JSON-RPC over stdio
// ---------------------------------------------------------------------------

interface JsonRpcRequest {
  jsonrpc?: string;
  id?: number | string | null;
  method?: string;
  params?: Record<string, unknown>;
}

function writeMessage(msg: Record<string, unknown>): void {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

/**
 * Start the MCP server reading JSON-RPC messages one per line from stdin
 * and writing responses to stdout. Intended for Claude Desktop / any MCP
 * client using stdio transport.
 *
 * Unknown methods return a JSON-RPC error. Bad JSON is logged to stderr
 * (via console.warn) but does not crash the loop.
 */
export function startMcpServer(): void {
  const rl = createInterface({ input: process.stdin, terminal: false });

  rl.on("line", (line: string) => {
    if (line.trim() === "") return;
    let req: JsonRpcRequest;
    try {
      req = JSON.parse(line) as JsonRpcRequest;
    } catch (err) {
      console.warn(
        "MCP parse error:",
        err instanceof Error ? err.message : String(err),
      );
      return;
    }

    const id = req.id ?? null;

    void (async () => {
      try {
        if (req.method === "initialize") {
          writeMessage({
            jsonrpc: "2.0",
            id,
            result: {
              protocolVersion: "2024-11-05",
              serverInfo: { name: "goldenmatch-js", version: "0.1.0" },
              capabilities: { tools: {} },
            },
          });
          return;
        }

        if (req.method === "tools/list") {
          writeMessage({
            jsonrpc: "2.0",
            id,
            result: { tools: TOOLS },
          });
          return;
        }

        if (req.method === "tools/call") {
          const params = req.params ?? {};
          const toolName = String(params["name"] ?? "");
          const toolArgs =
            (params["arguments"] as Record<string, unknown> | undefined) ?? {};
          const result = await handleTool(toolName, toolArgs);
          writeMessage({
            jsonrpc: "2.0",
            id,
            result: {
              content: [
                { type: "text", text: JSON.stringify(result) },
              ],
            },
          });
          return;
        }

        if (
          req.method === "notifications/initialized" ||
          req.method === "notifications/cancelled"
        ) {
          // No response to notifications.
          return;
        }

        writeMessage({
          jsonrpc: "2.0",
          id,
          error: { code: -32601, message: `Method not found: ${req.method}` },
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        writeMessage({
          jsonrpc: "2.0",
          id,
          error: { code: -32603, message: msg },
        });
      }
    })();
  });

  rl.on("close", () => {
    // Clean exit when stdin closes.
    process.exit(0);
  });
}

// Re-export for callers that want to pre-warm / test
export { readFileSync, isAbsolute };
export { writeJson };
