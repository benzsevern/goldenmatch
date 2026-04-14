/**
 * api/server.ts -- GoldenMatch REST API server (node:http).
 *
 * Node-only: uses node:http, node:path. NOT edge-safe.
 *
 * Endpoints:
 *   GET  /health                     - liveness check
 *   POST /dedupe                     - dedupe a batch of rows (JSON body)
 *   POST /match                      - match target vs reference
 *   POST /score                      - score two strings
 *   POST /explain                    - explain a pair
 *   POST /profile                    - profile a batch of rows
 *   POST /clusters                   - return clusters from dedupe
 *   GET  /reviews                    - list pending review items
 *   POST /reviews/decide             - accept/reject a review item
 *
 * Ports ideas from goldenmatch/api/server.py.
 */

import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { resolve, isAbsolute } from "node:path";
import { dedupe, match, scoreStrings } from "../../core/api.js";
import type { Row } from "../../core/types.js";
import {
  makeMatchkeyConfig,
  makeMatchkeyField,
} from "../../core/types.js";
import { explainPair } from "../../core/explain.js";
import { profileRows } from "../../core/profiler.js";

// ---------------------------------------------------------------------------
// In-memory review queue
// ---------------------------------------------------------------------------

interface ReviewItem {
  readonly id: string;
  readonly idA: number;
  readonly idB: number;
  readonly score: number;
  readonly rowA: Row;
  readonly rowB: Row;
  status: "pending" | "accepted" | "rejected";
  decidedAt?: string;
}

class ReviewQueue {
  private items = new Map<string, ReviewItem>();

  enqueue(item: Omit<ReviewItem, "status" | "id"> & { id?: string }): ReviewItem {
    const id = item.id ?? `${item.idA}:${item.idB}`;
    const rec: ReviewItem = {
      id,
      idA: item.idA,
      idB: item.idB,
      score: item.score,
      rowA: item.rowA,
      rowB: item.rowB,
      status: "pending",
    };
    this.items.set(id, rec);
    return rec;
  }

  pending(): ReviewItem[] {
    return [...this.items.values()].filter((r) => r.status === "pending");
  }

  decide(id: string, accept: boolean): ReviewItem | null {
    const existing = this.items.get(id);
    if (!existing) return null;
    existing.status = accept ? "accepted" : "rejected";
    existing.decidedAt = new Date().toISOString();
    return existing;
  }

  all(): ReviewItem[] {
    return [...this.items.values()];
  }
}

const reviewQueue = new ReviewQueue();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function sanitizePath(raw: string): string {
  const resolved = isAbsolute(raw) ? resolve(raw) : resolve(process.cwd(), raw);
  const cwd = resolve(process.cwd());
  if (!resolved.startsWith(cwd)) {
    throw new Error(`Path '${raw}' is outside the working directory`);
  }
  return resolved;
}

async function readBody(req: IncomingMessage): Promise<string> {
  let body = "";
  for await (const chunk of req) {
    body += typeof chunk === "string" ? chunk : (chunk as Buffer).toString("utf8");
  }
  return body;
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const raw = await readBody(req);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("request body must be a JSON object");
    }
    return parsed as Record<string, unknown>;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`invalid JSON body: ${msg}`);
  }
}

function sendJson(res: ServerResponse, status: number, data: unknown): void {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(data));
}

function asRowArray(v: unknown, label: string): Row[] {
  if (!Array.isArray(v)) throw new Error(`${label} must be an array of objects`);
  return v as Row[];
}

interface ShorthandOpts {
  exact?: readonly string[];
  fuzzy?: Readonly<Record<string, number>>;
  blocking?: readonly string[];
  threshold?: number;
}

function extractShorthand(body: Record<string, unknown>): ShorthandOpts {
  const out: {
    exact?: readonly string[];
    fuzzy?: Readonly<Record<string, number>>;
    blocking?: readonly string[];
    threshold?: number;
  } = {};
  if (Array.isArray(body["exact"])) out.exact = body["exact"].map(String);
  if (Array.isArray(body["blocking"])) out.blocking = body["blocking"].map(String);
  if (body["fuzzy"] && typeof body["fuzzy"] === "object" && !Array.isArray(body["fuzzy"])) {
    const f: Record<string, number> = {};
    for (const [k, v] of Object.entries(body["fuzzy"] as Record<string, unknown>)) {
      const n = typeof v === "number" ? v : Number(v);
      if (Number.isFinite(n)) f[k] = n;
    }
    out.fuzzy = f;
  }
  if (typeof body["threshold"] === "number") out.threshold = body["threshold"];
  return out;
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async function handleRequest(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
  const pathname = url.pathname;
  const method = req.method ?? "GET";

  try {
    if (pathname === "/health" && method === "GET") {
      sendJson(res, 200, { status: "ok", service: "goldenmatch-js" });
      return;
    }

    if (pathname === "/dedupe" && method === "POST") {
      const body = await readJsonBody(req);
      const rows = asRowArray(body["rows"], "rows");
      const options = extractShorthand(body);
      const result = dedupe(rows, options);
      sendJson(res, 200, {
        stats: {
          total_records: result.stats.totalRecords,
          total_clusters: result.stats.totalClusters,
          match_rate: result.stats.matchRate,
          matched_records: result.stats.matchedRecords,
          unique_records: result.stats.uniqueRecords,
        },
        golden_records: result.goldenRecords,
        dupes: result.dupes,
        unique: result.unique,
      });
      return;
    }

    if (pathname === "/match" && method === "POST") {
      const body = await readJsonBody(req);
      const target = asRowArray(body["target"], "target");
      const reference = asRowArray(body["reference"], "reference");
      const options = extractShorthand(body);
      const result = match(
        target.map((r) => ({ ...r, __source__: "target" })),
        reference.map((r) => ({ ...r, __source__: "reference" })),
        options,
      );
      sendJson(res, 200, {
        matched: result.matched,
        unmatched: result.unmatched,
        stats: result.stats,
      });
      return;
    }

    if (pathname === "/score" && method === "POST") {
      const body = await readJsonBody(req);
      const a = String(body["a"] ?? "");
      const b = String(body["b"] ?? "");
      const scorer = typeof body["scorer"] === "string" ? (body["scorer"] as string) : "jaro_winkler";
      sendJson(res, 200, { scorer, score: scoreStrings(a, b, scorer) });
      return;
    }

    if (pathname === "/explain" && method === "POST") {
      const body = await readJsonBody(req);
      const rowA = body["row_a"] as Row | undefined;
      const rowB = body["row_b"] as Row | undefined;
      if (!rowA || !rowB) throw new Error("row_a and row_b are required");
      const fieldsRaw = body["fields"];
      if (!Array.isArray(fieldsRaw)) {
        throw new Error("fields must be an array");
      }
      const fields = fieldsRaw.map((entry) => {
        if (!entry || typeof entry !== "object") {
          throw new Error("each field must be an object");
        }
        const e = entry as Record<string, unknown>;
        if (typeof e["field"] !== "string") {
          throw new Error("each field needs a 'field' property");
        }
        return makeMatchkeyField({
          field: e["field"] as string,
          transforms: Array.isArray(e["transforms"])
            ? (e["transforms"] as unknown[]).map(String)
            : ["lowercase", "strip"],
          scorer: typeof e["scorer"] === "string" ? (e["scorer"] as string) : "jaro_winkler",
          weight: typeof e["weight"] === "number" ? (e["weight"] as number) : 1.0,
        });
      });
      const threshold = typeof body["threshold"] === "number" ? (body["threshold"] as number) : 0.85;
      const mk = makeMatchkeyConfig({
        name: "adhoc",
        type: "weighted",
        fields,
        threshold,
      });
      const explanation = explainPair(rowA, rowB, mk);
      sendJson(res, 200, {
        score: explanation.score,
        confidence: explanation.confidence,
        explanation: explanation.explanation,
        field_scores: explanation.fieldScores,
      });
      return;
    }

    if (pathname === "/profile" && method === "POST") {
      const body = await readJsonBody(req);
      const rows = asRowArray(body["rows"], "rows");
      const profile = profileRows(rows);
      sendJson(res, 200, {
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
      });
      return;
    }

    if (pathname === "/clusters" && method === "POST") {
      const body = await readJsonBody(req);
      const rows = asRowArray(body["rows"], "rows");
      const options = extractShorthand(body);
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
      sendJson(res, 200, {
        cluster_count: clusters.length,
        clusters,
      });
      return;
    }

    if (pathname === "/reviews" && method === "GET") {
      sendJson(res, 200, { pending: reviewQueue.pending() });
      return;
    }

    if (pathname === "/reviews/decide" && method === "POST") {
      const body = await readJsonBody(req);
      const id = String(body["id"] ?? "");
      const accept = Boolean(body["accept"]);
      if (id === "") throw new Error("id is required");
      const decided = reviewQueue.decide(id, accept);
      if (!decided) {
        sendJson(res, 404, { error: `review item ${id} not found` });
        return;
      }
      sendJson(res, 200, { decided });
      return;
    }

    if (pathname === "/reviews/enqueue" && method === "POST") {
      const body = await readJsonBody(req);
      const idA = Number(body["id_a"]);
      const idB = Number(body["id_b"]);
      const score = Number(body["score"]);
      const rowA = body["row_a"] as Row | undefined;
      const rowB = body["row_b"] as Row | undefined;
      if (!Number.isFinite(idA) || !Number.isFinite(idB) || !rowA || !rowB) {
        throw new Error("id_a, id_b, row_a, row_b are required");
      }
      const item = reviewQueue.enqueue({
        idA,
        idB,
        score: Number.isFinite(score) ? score : 0,
        rowA,
        rowB,
      });
      sendJson(res, 200, { item });
      return;
    }

    sendJson(res, 404, { error: `Not found: ${method} ${pathname}` });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    sendJson(res, 500, { error: msg });
  }
}

// ---------------------------------------------------------------------------
// Public: startApiServer
// ---------------------------------------------------------------------------

export interface StartApiOptions {
  readonly port?: number;
  readonly host?: string;
}

/**
 * Start the REST API server.
 * Default: http://127.0.0.1:8000.
 *
 * Returns the http.Server so tests can close it.
 */
export function startApiServer(options: StartApiOptions = {}): ReturnType<typeof createServer> {
  const port = options.port ?? 8000;
  const host = options.host ?? "127.0.0.1";
  const server = createServer((req, res) => {
    handleRequest(req, res).catch((err: unknown) => {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn("Unhandled request error:", msg);
      try {
        if (!res.headersSent) {
          res.statusCode = 500;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: msg }));
        }
      } catch {
        // ignore
      }
    });
  });
  server.listen(port, host, () => {
    // eslint-disable-next-line no-console
    console.log(`GoldenMatch API listening on http://${host}:${port}`);
  });
  return server;
}

export { reviewQueue, ReviewQueue };
