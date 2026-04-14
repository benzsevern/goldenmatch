import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { Server } from "node:http";
import { startApiServer } from "../../src/node/api/server.js";

let server: Server;
let baseUrl: string;

beforeAll(async () => {
  server = startApiServer({ port: 0, host: "127.0.0.1" });
  // Wait for listen to complete.
  await new Promise<void>((resolveFn) => {
    if (server.listening) {
      resolveFn();
      return;
    }
    server.once("listening", () => resolveFn());
  });
  const addr = server.address();
  const port =
    typeof addr === "object" && addr !== null && "port" in addr ? addr.port : 8000;
  baseUrl = `http://127.0.0.1:${port}`;
});

afterAll(async () => {
  if (server) {
    await new Promise<void>((resolveFn, rejectFn) => {
      server.close((err) => (err ? rejectFn(err) : resolveFn()));
    });
  }
});

describe("REST API server", () => {
  it("GET /health returns 200 with status ok", async () => {
    const res = await fetch(baseUrl + "/health");
    expect(res.status).toBe(200);
    const body = (await res.json()) as { status: string };
    expect(body.status).toBe("ok");
  });

  it("POST /dedupe returns DedupeResult-like shape", async () => {
    const res = await fetch(baseUrl + "/dedupe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rows: [
          { email: "a@x.com", name: "Alice" },
          { email: "a@x.com", name: "A." },
          { email: "b@x.com", name: "Bob" },
        ],
        exact: ["email"],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      stats: { total_records: number; total_clusters: number };
      golden_records: unknown[];
      dupes: unknown[];
      unique: unknown[];
    };
    expect(body.stats.total_records).toBe(3);
    expect(typeof body.stats.total_clusters).toBe("number");
    expect(Array.isArray(body.golden_records)).toBe(true);
    expect(Array.isArray(body.dupes)).toBe(true);
    expect(Array.isArray(body.unique)).toBe(true);
  });

  it("POST /match returns matched/unmatched", async () => {
    const res = await fetch(baseUrl + "/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target: [{ email: "a@x.com", name: "Alice" }],
        reference: [
          { email: "a@x.com", name: "A." },
          { email: "z@x.com", name: "Zack" },
        ],
        exact: ["email"],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { matched: unknown[]; unmatched: unknown[] };
    expect(Array.isArray(body.matched)).toBe(true);
    expect(Array.isArray(body.unmatched)).toBe(true);
  });

  it("POST /score returns numeric score", async () => {
    const res = await fetch(baseUrl + "/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a: "John", b: "Jon", scorer: "jaro_winkler" }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { score: number; scorer: string };
    expect(body.scorer).toBe("jaro_winkler");
    expect(typeof body.score).toBe("number");
    expect(body.score).toBeGreaterThan(0.9);
  });

  it("POST /explain returns PairExplanation shape", async () => {
    const res = await fetch(baseUrl + "/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        row_a: { name: "John Smith" },
        row_b: { name: "Jon Smith" },
        fields: [{ field: "name", scorer: "jaro_winkler", weight: 1.0 }],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      score: number;
      confidence: number;
      explanation: string;
      field_scores: unknown;
    };
    expect(typeof body.score).toBe("number");
    // confidence may be numeric or categorical ("high"/"medium"/"low")
    expect(["number", "string"]).toContain(typeof body.confidence);
    expect(typeof body.explanation).toBe("string");
  });

  it("POST /profile returns column profiles", async () => {
    const res = await fetch(baseUrl + "/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rows: [
          { email: "a@x.com", age: 20 },
          { email: "b@x.com", age: 30 },
          { email: "c@x.com", age: 40 },
        ],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      row_count: number;
      columns: Array<{ name: string; inferred_type: string }>;
    };
    expect(body.row_count).toBe(3);
    expect(Array.isArray(body.columns)).toBe(true);
    expect(body.columns.length).toBeGreaterThan(0);
  });

  it("POST /clusters returns clusters object", async () => {
    const res = await fetch(baseUrl + "/clusters", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rows: [
          { email: "a@x.com", name: "Alice" },
          { email: "a@x.com", name: "A." },
          { email: "b@x.com", name: "Bob" },
        ],
        exact: ["email"],
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      cluster_count: number;
      clusters: Array<{ cluster_id: number; members: number[] }>;
    };
    expect(typeof body.cluster_count).toBe("number");
    expect(Array.isArray(body.clusters)).toBe(true);
  });

  it("GET /reviews returns { pending: [] }", async () => {
    const res = await fetch(baseUrl + "/reviews");
    expect(res.status).toBe(200);
    const body = (await res.json()) as { pending: unknown[] };
    expect(Array.isArray(body.pending)).toBe(true);
  });

  it("invalid JSON body returns 500 with error", async () => {
    const res = await fetch(baseUrl + "/dedupe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{not valid json",
    });
    expect([400, 500]).toContain(res.status);
    const body = (await res.json()) as { error: string };
    expect(typeof body.error).toBe("string");
  });

  it("unknown route returns 404", async () => {
    const res = await fetch(baseUrl + "/no-such-route");
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error: string };
    expect(typeof body.error).toBe("string");
  });

  it("POST /reviews/decide with missing id returns error", async () => {
    const res = await fetch(baseUrl + "/reviews/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    // Missing id -> handler throws -> 500 with error. If id exists but missing -> 404.
    expect([400, 404, 500]).toContain(res.status);
    const body = (await res.json()) as { error: string };
    expect(typeof body.error).toBe("string");
  });

  it("POST /reviews/decide with unknown id returns 404", async () => {
    const res = await fetch(baseUrl + "/reviews/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: "never-enqueued", accept: true }),
    });
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error: string };
    expect(typeof body.error).toBe("string");
  });

  it("enqueue + decide round-trip works", async () => {
    const enq = await fetch(baseUrl + "/reviews/enqueue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id_a: 1,
        id_b: 2,
        score: 0.8,
        row_a: { name: "A" },
        row_b: { name: "A." },
      }),
    });
    expect(enq.status).toBe(200);
    const enqBody = (await enq.json()) as { item: { id: string } };
    expect(typeof enqBody.item.id).toBe("string");

    const dec = await fetch(baseUrl + "/reviews/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: enqBody.item.id, accept: true }),
    });
    expect(dec.status).toBe(200);
    const decBody = (await dec.json()) as { decided: { status: string } };
    expect(decBody.decided.status).toBe("accepted");
  });
});
