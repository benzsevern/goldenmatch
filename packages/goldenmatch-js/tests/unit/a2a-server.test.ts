import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { Server } from "node:http";
import { startA2aServer, AGENT_CARD } from "../../src/node/a2a/server.js";

let server: Server;
let baseUrl: string;

beforeAll(async () => {
  server = startA2aServer({ port: 0, host: "127.0.0.1" });
  await new Promise<void>((resolveFn) => {
    if (server.listening) {
      resolveFn();
      return;
    }
    server.once("listening", () => resolveFn());
  });
  const addr = server.address();
  const port =
    typeof addr === "object" && addr !== null && "port" in addr ? addr.port : 8200;
  baseUrl = `http://127.0.0.1:${port}`;
});

afterAll(async () => {
  if (server) {
    await new Promise<void>((resolveFn, rejectFn) => {
      server.close((err) => (err ? rejectFn(err) : resolveFn()));
    });
  }
});

describe("A2A agent card (exported constant)", () => {
  it("has name, description, version, provider, skills", () => {
    expect(typeof AGENT_CARD.name).toBe("string");
    expect(typeof AGENT_CARD.description).toBe("string");
    expect(typeof AGENT_CARD.version).toBe("string");
    expect(AGENT_CARD.provider).toBeDefined();
    expect(typeof AGENT_CARD.provider.organization).toBe("string");
    expect(Array.isArray(AGENT_CARD.skills)).toBe(true);
  });

  it("has at least 5 skills", () => {
    expect(AGENT_CARD.skills.length).toBeGreaterThanOrEqual(5);
  });

  it("every skill has name, description, inputModes, outputModes", () => {
    for (const skill of AGENT_CARD.skills) {
      expect(typeof skill.name).toBe("string");
      expect(skill.name.length).toBeGreaterThan(0);
      expect(typeof skill.description).toBe("string");
      expect(Array.isArray(skill.inputModes)).toBe(true);
      expect(skill.inputModes.length).toBeGreaterThan(0);
      expect(Array.isArray(skill.outputModes)).toBe(true);
      expect(skill.outputModes.length).toBeGreaterThan(0);
    }
  });
});

describe("A2A server HTTP endpoints", () => {
  it("GET /.well-known/agent.json returns the AgentCard", async () => {
    const res = await fetch(baseUrl + "/.well-known/agent.json");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      name: string;
      description: string;
      version: string;
      provider: { organization: string };
      skills: Array<{
        name: string;
        description: string;
        inputModes: string[];
        outputModes: string[];
      }>;
    };
    expect(body.name).toBe("goldenmatch-js");
    expect(typeof body.description).toBe("string");
    expect(typeof body.version).toBe("string");
    expect(body.provider.organization).toBe("goldenmatch");
    expect(body.skills.length).toBeGreaterThanOrEqual(5);
    for (const skill of body.skills) {
      expect(typeof skill.name).toBe("string");
      expect(typeof skill.description).toBe("string");
      expect(Array.isArray(skill.inputModes)).toBe(true);
      expect(Array.isArray(skill.outputModes)).toBe(true);
    }
  });

  it("POST /tasks with skill=dedupe completes and returns result", async () => {
    const res = await fetch(baseUrl + "/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        skill: "dedupe",
        input: {
          rows: [
            { email: "a@x.com", name: "Alice" },
            { email: "a@x.com", name: "A." },
            { email: "b@x.com", name: "Bob" },
          ],
          exact: ["email"],
        },
      }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      id: string;
      status: string;
      skill: string;
      result?: unknown;
    };
    expect(typeof body.id).toBe("string");
    expect(body.id.length).toBeGreaterThan(0);
    expect(body.skill).toBe("dedupe");
    expect(["completed", "running", "pending"]).toContain(body.status);
    if (body.status === "completed") {
      expect(body.result).toBeDefined();
    }
  });

  it("GET /tasks/{id} returns task status after creation", async () => {
    const postRes = await fetch(baseUrl + "/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        skill: "score",
        input: { a: "John", b: "Jon", scorer: "jaro_winkler" },
      }),
    });
    const postBody = (await postRes.json()) as { id: string };
    expect(typeof postBody.id).toBe("string");

    const getRes = await fetch(baseUrl + "/tasks/" + postBody.id);
    expect(getRes.status).toBe(200);
    const getBody = (await getRes.json()) as { id: string; status: string; skill: string };
    expect(getBody.id).toBe(postBody.id);
    expect(getBody.skill).toBe("score");
    expect(["completed", "running", "pending", "failed"]).toContain(getBody.status);
  });

  it("GET /tasks/nonexistent returns 404", async () => {
    const res = await fetch(baseUrl + "/tasks/does-not-exist-xyz");
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error: string };
    expect(typeof body.error).toBe("string");
  });

  it("POST /tasks with unknown skill returns failed task (or error)", async () => {
    const res = await fetch(baseUrl + "/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skill: "not_a_real_skill", input: {} }),
    });
    // Implementation returns 200 with status=failed. Some implementations use 400.
    expect([200, 400]).toContain(res.status);
    const body = (await res.json()) as {
      status?: string;
      error?: string;
    };
    // Either body.error is set or body.status === "failed".
    const hasFailure =
      (typeof body.error === "string" && body.error.length > 0) ||
      body.status === "failed";
    expect(hasFailure).toBe(true);
  });

  it("POST /tasks without skill returns 400", async () => {
    const res = await fetch(baseUrl + "/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: {} }),
    });
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error: string };
    expect(typeof body.error).toBe("string");
  });
});
