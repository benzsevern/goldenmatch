/**
 * a2a/server.ts -- GoldenMatch A2A (Agent-to-Agent) protocol server.
 *
 * Node-only: uses node:http, node:crypto. NOT edge-safe.
 *
 * Endpoints:
 *   GET  /.well-known/agent.json   - agent card (10+ skills)
 *   POST /tasks                    - create a task (skill + input)
 *   GET  /tasks/{id}               - fetch task status/result
 *
 * Ports ideas from goldenmatch/a2a/server.py. This is a simpler
 * synchronous variant (no SSE streaming, no persistent store).
 */

import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { randomUUID } from "node:crypto";
import { dedupe, match, scoreStrings } from "../../core/api.js";
import { profileRows } from "../../core/profiler.js";
import { explainPair } from "../../core/explain.js";
import type { Row } from "../../core/types.js";
import {
  makeMatchkeyConfig,
  makeMatchkeyField,
  VALID_SCORERS,
  VALID_TRANSFORMS,
  VALID_STRATEGIES,
} from "../../core/types.js";

// ---------------------------------------------------------------------------
// Agent card
// ---------------------------------------------------------------------------

export interface AgentSkill {
  readonly name: string;
  readonly description: string;
  readonly inputModes: readonly string[];
  readonly outputModes: readonly string[];
}

export const AGENT_CARD: {
  readonly name: string;
  readonly description: string;
  readonly version: string;
  readonly provider: {
    readonly organization: string;
    readonly url: string;
  };
  readonly capabilities: Readonly<Record<string, boolean>>;
  readonly skills: readonly AgentSkill[];
} = {
  name: "goldenmatch-js",
  description:
    "Entity resolution agent -- dedupe, match, profile, score, explain, evaluate.",
  version: "0.1.0",
  provider: {
    organization: "goldenmatch",
    url: "https://github.com/benzsevern/goldenmatch",
  },
  capabilities: {
    streaming: false,
    pushNotifications: false,
    stateTransitionHistory: false,
  },
  skills: [
    {
      name: "dedupe",
      description: "Deduplicate a list of records and return golden records plus clusters.",
      inputModes: ["data/json"],
      outputModes: ["data/json"],
    },
    {
      name: "match",
      description: "Match target records against reference records.",
      inputModes: ["data/json"],
      outputModes: ["data/json"],
    },
    {
      name: "score",
      description: "Score similarity between two strings.",
      inputModes: ["text"],
      outputModes: ["text"],
    },
    {
      name: "profile",
      description: "Profile a dataset (types, null rates, cardinality).",
      inputModes: ["data/json"],
      outputModes: ["data/json"],
    },
    {
      name: "suggest_config",
      description: "Auto-generate a shorthand dedupe config from a dataset profile.",
      inputModes: ["data/json"],
      outputModes: ["data/json"],
    },
    {
      name: "explain_pair",
      description: "Explain why two records match using weighted field scorers.",
      inputModes: ["data/json"],
      outputModes: ["data/json"],
    },
    {
      name: "evaluate",
      description: "Evaluate predicted pairs vs ground truth (precision/recall/F1).",
      inputModes: ["data/json"],
      outputModes: ["data/json"],
    },
    {
      name: "list_scorers",
      description: "List all available similarity scorers.",
      inputModes: ["text"],
      outputModes: ["data/json"],
    },
    {
      name: "list_transforms",
      description: "List all available field transforms.",
      inputModes: ["text"],
      outputModes: ["data/json"],
    },
    {
      name: "list_strategies",
      description: "List all golden-record survivorship strategies.",
      inputModes: ["text"],
      outputModes: ["data/json"],
    },
  ],
};

// ---------------------------------------------------------------------------
// Task store
// ---------------------------------------------------------------------------

interface Task {
  readonly id: string;
  readonly skill: string;
  status: "pending" | "running" | "completed" | "failed";
  readonly createdAt: string;
  completedAt?: string;
  result?: unknown;
  error?: string;
}

// ---------------------------------------------------------------------------
// Skill dispatch
// ---------------------------------------------------------------------------

async function dispatchSkill(
  skill: string,
  input: Record<string, unknown>,
): Promise<unknown> {
  switch (skill) {
    case "dedupe": {
      if (!Array.isArray(input["rows"])) throw new Error("rows must be an array");
      const rows = input["rows"] as Row[];
      const opts: {
        exact?: readonly string[];
        fuzzy?: Readonly<Record<string, number>>;
        blocking?: readonly string[];
        threshold?: number;
      } = {};
      if (Array.isArray(input["exact"])) opts.exact = input["exact"].map(String);
      if (Array.isArray(input["blocking"])) opts.blocking = input["blocking"].map(String);
      if (input["fuzzy"] && typeof input["fuzzy"] === "object" && !Array.isArray(input["fuzzy"])) {
        const f: Record<string, number> = {};
        for (const [k, v] of Object.entries(input["fuzzy"] as Record<string, unknown>)) {
          const n = typeof v === "number" ? v : Number(v);
          if (Number.isFinite(n)) f[k] = n;
        }
        opts.fuzzy = f;
      }
      if (typeof input["threshold"] === "number") opts.threshold = input["threshold"];
      const result = dedupe(rows, opts);
      return {
        stats: {
          total_records: result.stats.totalRecords,
          total_clusters: result.stats.totalClusters,
          match_rate: result.stats.matchRate,
        },
        golden_records: result.goldenRecords,
      };
    }

    case "match": {
      if (!Array.isArray(input["target"])) throw new Error("target must be an array");
      if (!Array.isArray(input["reference"])) throw new Error("reference must be an array");
      const target = (input["target"] as Row[]).map((r) => ({ ...r, __source__: "target" }));
      const reference = (input["reference"] as Row[]).map((r) => ({
        ...r,
        __source__: "reference",
      }));
      const opts: {
        exact?: readonly string[];
        fuzzy?: Readonly<Record<string, number>>;
        blocking?: readonly string[];
        threshold?: number;
      } = {};
      if (Array.isArray(input["exact"])) opts.exact = input["exact"].map(String);
      if (Array.isArray(input["blocking"])) opts.blocking = input["blocking"].map(String);
      if (input["fuzzy"] && typeof input["fuzzy"] === "object" && !Array.isArray(input["fuzzy"])) {
        const f: Record<string, number> = {};
        for (const [k, v] of Object.entries(input["fuzzy"] as Record<string, unknown>)) {
          const n = typeof v === "number" ? v : Number(v);
          if (Number.isFinite(n)) f[k] = n;
        }
        opts.fuzzy = f;
      }
      if (typeof input["threshold"] === "number") opts.threshold = input["threshold"];
      const result = match(target, reference, opts);
      return {
        matched: result.matched,
        unmatched: result.unmatched,
      };
    }

    case "score": {
      const a = String(input["a"] ?? "");
      const b = String(input["b"] ?? "");
      const scorer = typeof input["scorer"] === "string" ? (input["scorer"] as string) : "jaro_winkler";
      return { scorer, score: scoreStrings(a, b, scorer) };
    }

    case "profile": {
      if (!Array.isArray(input["rows"])) throw new Error("rows must be an array");
      const profile = profileRows(input["rows"] as Row[]);
      return {
        row_count: profile.rowCount,
        columns: profile.columns.map((c) => ({
          name: c.name,
          inferred_type: c.inferredType,
          null_rate: c.nullRate,
          cardinality_ratio: c.cardinalityRatio,
        })),
      };
    }

    case "suggest_config": {
      if (!Array.isArray(input["rows"])) throw new Error("rows must be an array");
      const profile = profileRows(input["rows"] as Row[]);
      const exact: string[] = [];
      const fuzzy: Record<string, number> = {};
      const blocking: string[] = [];
      for (const col of profile.columns) {
        if (col.nullRate > 0.2) continue;
        if (col.inferredType === "email" && col.cardinalityRatio >= 0.5) exact.push(col.name);
        else if (col.inferredType === "phone" && col.cardinalityRatio >= 0.5) exact.push(col.name);
        else if (col.inferredType === "zip" || col.inferredType === "geo") blocking.push(col.name);
        else if (col.inferredType === "name") fuzzy[col.name] = 0.85;
        else if (col.inferredType === "text" && col.avgLength > 4) fuzzy[col.name] = 0.8;
      }
      return { suggested: { exact, fuzzy, blocking, threshold: 0.85 } };
    }

    case "explain_pair": {
      const rowA = input["row_a"] as Row | undefined;
      const rowB = input["row_b"] as Row | undefined;
      if (!rowA || !rowB) throw new Error("row_a and row_b are required");
      const fieldsRaw = input["fields"];
      if (!Array.isArray(fieldsRaw)) throw new Error("fields must be an array");
      const fields = fieldsRaw.map((entry) => {
        const e = entry as Record<string, unknown>;
        return makeMatchkeyField({
          field: String(e["field"]),
          transforms: Array.isArray(e["transforms"])
            ? (e["transforms"] as unknown[]).map(String)
            : ["lowercase", "strip"],
          scorer: typeof e["scorer"] === "string" ? (e["scorer"] as string) : "jaro_winkler",
          weight: typeof e["weight"] === "number" ? (e["weight"] as number) : 1.0,
        });
      });
      const mk = makeMatchkeyConfig({
        name: "adhoc",
        type: "weighted",
        fields,
        threshold: typeof input["threshold"] === "number" ? (input["threshold"] as number) : 0.85,
      });
      const result = explainPair(rowA, rowB, mk);
      return {
        score: result.score,
        confidence: result.confidence,
        explanation: result.explanation,
      };
    }

    case "evaluate": {
      // Accept pre-computed predicted/truth pairs for simplicity.
      const predicted = Array.isArray(input["predicted"])
        ? (input["predicted"] as unknown[]).map((p) => {
            const pair = p as Record<string, unknown>;
            return [Number(pair["id_a"]), Number(pair["id_b"])] as const;
          })
        : [];
      const truth = Array.isArray(input["truth"])
        ? (input["truth"] as unknown[]).map((p) => {
            const pair = p as Record<string, unknown>;
            return [Number(pair["id_a"]), Number(pair["id_b"])] as const;
          })
        : [];
      const truthSet = new Set(truth.map(([a, b]) => `${Math.min(a, b)}:${Math.max(a, b)}`));
      const predSet = new Set(
        predicted.map(([a, b]) => `${Math.min(a, b)}:${Math.max(a, b)}`),
      );
      let tp = 0;
      let fp = 0;
      for (const p of predSet) {
        if (truthSet.has(p)) tp++;
        else fp++;
      }
      let fn = 0;
      for (const t of truthSet) {
        if (!predSet.has(t)) fn++;
      }
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
      return { tp, fp, fn, precision, recall, f1 };
    }

    case "list_scorers":
      return { scorers: [...VALID_SCORERS] };

    case "list_transforms":
      return { transforms: [...VALID_TRANSFORMS] };

    case "list_strategies":
      return { strategies: [...VALID_STRATEGIES] };

    default:
      throw new Error(`Unknown skill: ${skill}`);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  let body = "";
  for await (const chunk of req) {
    body += typeof chunk === "string" ? chunk : (chunk as Buffer).toString("utf8");
  }
  if (!body) return {};
  const parsed = JSON.parse(body);
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("body must be a JSON object");
  }
  return parsed as Record<string, unknown>;
}

function sendJson(res: ServerResponse, status: number, data: unknown): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(data));
}

// ---------------------------------------------------------------------------
// Public: startA2aServer
// ---------------------------------------------------------------------------

export interface StartA2aOptions {
  readonly port?: number;
  readonly host?: string;
}

export function startA2aServer(options: StartA2aOptions = {}): ReturnType<typeof createServer> {
  const port = options.port ?? 8200;
  const host = options.host ?? "127.0.0.1";
  const tasks = new Map<string, Task>();

  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    const pathname = url.pathname;
    const methodName = req.method ?? "GET";

    try {
      if (pathname === "/.well-known/agent.json" && methodName === "GET") {
        sendJson(res, 200, AGENT_CARD);
        return;
      }

      if (pathname === "/health" && methodName === "GET") {
        sendJson(res, 200, { status: "ok", agent: "goldenmatch-js" });
        return;
      }

      if (pathname === "/tasks" && methodName === "POST") {
        const body = await readJsonBody(req);
        const skill = String(body["skill"] ?? "");
        const input =
          (body["input"] as Record<string, unknown> | undefined) ??
          (body["params"] as Record<string, unknown> | undefined) ??
          {};
        if (!skill) {
          sendJson(res, 400, { error: "skill is required" });
          return;
        }
        const id = randomUUID();
        const createdAt = new Date().toISOString();
        const task: Task = {
          id,
          skill,
          status: "running",
          createdAt,
        };
        tasks.set(id, task);

        try {
          const result = await dispatchSkill(skill, input);
          task.status = "completed";
          task.completedAt = new Date().toISOString();
          task.result = result;
          sendJson(res, 200, {
            id,
            status: task.status,
            skill,
            created_at: createdAt,
            completed_at: task.completedAt,
            result,
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          task.status = "failed";
          task.completedAt = new Date().toISOString();
          task.error = msg;
          sendJson(res, 200, {
            id,
            status: task.status,
            skill,
            created_at: createdAt,
            completed_at: task.completedAt,
            error: msg,
          });
        }
        return;
      }

      if (pathname.startsWith("/tasks/") && methodName === "GET") {
        const id = pathname.slice("/tasks/".length);
        const task = tasks.get(id);
        if (!task) {
          sendJson(res, 404, { error: `Task not found: ${id}` });
          return;
        }
        sendJson(res, 200, {
          id: task.id,
          skill: task.skill,
          status: task.status,
          created_at: task.createdAt,
          completed_at: task.completedAt ?? null,
          result: task.result ?? null,
          error: task.error ?? null,
        });
        return;
      }

      sendJson(res, 404, { error: `Not found: ${methodName} ${pathname}` });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      sendJson(res, 500, { error: msg });
    }
  });

  server.listen(port, host, () => {
    // eslint-disable-next-line no-console
    console.log(`GoldenMatch A2A agent listening on http://${host}:${port}`);
  });
  return server;
}
