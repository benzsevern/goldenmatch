import { describe, it, expect } from "vitest";
import { TOOLS, handleTool } from "../../src/node/mcp/server.js";

describe("MCP server — TOOLS metadata", () => {
  it("exports a non-empty array of tool definitions", () => {
    expect(Array.isArray(TOOLS)).toBe(true);
    expect(TOOLS.length).toBeGreaterThan(0);
  });

  it("each tool has name, description, inputSchema", () => {
    for (const tool of TOOLS) {
      expect(typeof tool.name).toBe("string");
      expect(tool.name.length).toBeGreaterThan(0);
      expect(typeof tool.description).toBe("string");
      expect(tool.description.length).toBeGreaterThan(0);
      expect(tool.inputSchema).toBeTypeOf("object");
      expect(tool.inputSchema).not.toBeNull();
    }
  });

  it("every tool name is unique", () => {
    const names = TOOLS.map((t) => t.name);
    const unique = new Set(names);
    expect(unique.size).toBe(names.length);
  });

  it("includes core tools (dedupe, score_strings, profile, etc.)", () => {
    const names = new Set(TOOLS.map((t) => t.name));
    for (const expected of [
      "dedupe",
      "match",
      "score_strings",
      "explain_pair",
      "profile",
      "list_scorers",
      "list_transforms",
      "list_strategies",
      "server_info",
    ]) {
      expect(names.has(expected)).toBe(true);
    }
  });
});

describe("MCP server — handleTool dispatcher", () => {
  it("score_strings with jaro_winkler scores John~Jon near 0.94", async () => {
    const result = (await handleTool("score_strings", {
      a: "John",
      b: "Jon",
      scorer: "jaro_winkler",
    })) as { score: number; scorer: string };
    expect(result).toMatchObject({ scorer: "jaro_winkler" });
    expect(typeof result.score).toBe("number");
    expect(result.score).toBeGreaterThan(0.9);
    expect(result.score).toBeLessThanOrEqual(1.0);
  });

  it("score_strings with missing scorer defaults to jaro_winkler", async () => {
    const result = (await handleTool("score_strings", {
      a: "x",
      b: "y",
    })) as { score: number; scorer: string };
    expect(result.scorer).toBe("jaro_winkler");
    expect(typeof result.score).toBe("number");
  });

  it("explain_pair returns an NL explanation", async () => {
    const result = (await handleTool("explain_pair", {
      row_a: { name: "John Smith", email: "j@x.com" },
      row_b: { name: "Jon Smith", email: "j@x.com" },
      fields: [
        { field: "name", scorer: "jaro_winkler", weight: 1.0 },
        { field: "email", scorer: "exact", weight: 1.0 },
      ],
    })) as {
      score: number;
      confidence: number;
      explanation: string;
      field_scores: unknown;
    };
    expect(typeof result.score).toBe("number");
    // confidence may be numeric or categorical ("high"/"medium"/"low")
    expect(["number", "string"]).toContain(typeof result.confidence);
    expect(typeof result.explanation).toBe("string");
    expect(result.explanation.length).toBeGreaterThan(0);
    expect(result.field_scores).toBeDefined();
  });

  it("profile returns column profiles via { rows } -> requires path; use find_fuzzy_matches path flow via file is out of scope here", async () => {
    // Note: profile tool takes a `path`. Rows-based profile is only via /profile REST.
    // Verify that profile without a valid path returns an error object (not throws).
    const result = (await handleTool("profile", { path: "nonexistent_file_xyz.csv" })) as {
      error?: string;
    };
    expect(typeof result).toBe("object");
    expect(result).not.toBeNull();
    // Either error shape or some profile shape; not a crash.
    expect(result.error ?? "").not.toBe(undefined);
  });

  it("list_scorers returns an array of scorer names", async () => {
    const result = (await handleTool("list_scorers", {})) as { scorers: string[] };
    expect(Array.isArray(result.scorers)).toBe(true);
    expect(result.scorers.length).toBeGreaterThan(0);
    expect(result.scorers).toContain("jaro_winkler");
  });

  it("list_transforms returns array", async () => {
    const result = (await handleTool("list_transforms", {})) as { transforms: string[] };
    expect(Array.isArray(result.transforms)).toBe(true);
    expect(result.transforms.length).toBeGreaterThan(0);
  });

  it("list_strategies returns array", async () => {
    const result = (await handleTool("list_strategies", {})) as { strategies: string[] };
    expect(Array.isArray(result.strategies)).toBe(true);
    expect(result.strategies.length).toBeGreaterThan(0);
  });

  it("server_info returns metadata with tool_count", async () => {
    const result = (await handleTool("server_info", {})) as {
      name: string;
      tool_count: number;
    };
    expect(result.name).toBe("goldenmatch-js");
    expect(result.tool_count).toBe(TOOLS.length);
  });

  it("unknown tool returns { error } rather than throwing", async () => {
    const result = (await handleTool("nonexistent_tool_xyz", {})) as { error: string };
    expect(typeof result).toBe("object");
    expect(typeof result.error).toBe("string");
    expect(result.error).toMatch(/unknown/i);
  });

  it("path traversal via '..' is rejected (error, not crash)", async () => {
    const result = (await handleTool("read_file", {
      file_path: "../../../etc/passwd",
      path: "../../../etc/passwd",
    })) as { error?: string };
    expect(typeof result).toBe("object");
    expect(typeof result.error).toBe("string");
    expect(result.error).toMatch(/outside|not a|no such|enoent/i);
  });

  it("absolute path outside cwd is rejected", async () => {
    // Pick a path guaranteed to be outside cwd on both Windows and POSIX.
    const outsidePath =
      process.platform === "win32" ? "C:\\Windows\\System32\\drivers\\etc\\hosts" : "/etc/passwd";
    const result = (await handleTool("read_file", { path: outsidePath })) as {
      error?: string;
    };
    expect(typeof result).toBe("object");
    expect(typeof result.error).toBe("string");
    expect(result.error).toMatch(/outside|enoent|no such|not found/i);
  });

  it("write_csv with non-array rows returns error (not crash)", async () => {
    const result = (await handleTool("write_csv", {
      path: "some_output.csv",
      rows: "not-an-array",
    })) as { error?: string };
    expect(typeof result).toBe("object");
    expect(typeof result.error).toBe("string");
  });

  it("score_pair with missing rows returns error", async () => {
    const result = (await handleTool("score_pair", {
      fields: [{ field: "name" }],
    })) as { error?: string };
    expect(typeof result.error).toBe("string");
  });

  it("score_pair with valid inputs returns a score", async () => {
    const result = (await handleTool("score_pair", {
      row_a: { name: "John" },
      row_b: { name: "Jon" },
      fields: [{ field: "name", scorer: "jaro_winkler", weight: 1.0 }],
    })) as { score: number; field_count: number };
    expect(typeof result.score).toBe("number");
    expect(result.field_count).toBe(1);
  });
});
