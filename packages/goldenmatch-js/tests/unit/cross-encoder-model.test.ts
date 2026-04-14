import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  CrossEncoderModel,
  rerankPair,
  rerankTopPairs,
  _resetCrossEncoderModelCache,
} from "../../src/core/index.js";
import type { MatchkeyConfig, Row, ScoredPair } from "../../src/core/index.js";

describe("CrossEncoderModel", () => {
  beforeEach(() => {
    _resetCrossEncoderModelCache();
    // Silence the expected "cross-encoder failed" console.warn noise.
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("throws a clear error when @huggingface/transformers is not installed", async () => {
    const model = new CrossEncoderModel();
    await expect(model.score("a", "b")).rejects.toThrow(
      /@huggingface\/transformers/,
    );
  });

  it("the same error is raised on subsequent calls (cache does not lock into a rejected promise)", async () => {
    const model = new CrossEncoderModel();
    await expect(model.score("a", "b")).rejects.toThrow(
      /@huggingface\/transformers/,
    );
    // Second call should still reject with a fresh load attempt.
    await expect(model.score("a", "b")).rejects.toThrow(
      /@huggingface\/transformers/,
    );
  });
});

describe("rerankPair with reranker option", () => {
  beforeEach(() => {
    _resetCrossEncoderModelCache();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("reranker='cross-encoder' falls back to LLM path when package missing", async () => {
    // No apiKey, no env var, no transformers package -> fallback returns NaN.
    const env = (globalThis as { process?: { env?: Record<string, string | undefined> } })
      .process?.env;
    const savedOpen = env?.OPENAI_API_KEY;
    const savedAnt = env?.ANTHROPIC_API_KEY;
    if (env) {
      delete env.OPENAI_API_KEY;
      delete env.ANTHROPIC_API_KEY;
    }
    try {
      const score = await rerankPair(
        { name: "John", email: "john@x.com" },
        { name: "Jon", email: "jon@x.com" },
        ["name", "email"],
        { reranker: "cross-encoder" },
      );
      // Fallback path returns NaN without apiKey.
      expect(Number.isNaN(score)).toBe(true);
    } finally {
      if (env) {
        if (savedOpen !== undefined) env.OPENAI_API_KEY = savedOpen;
        if (savedAnt !== undefined) env.ANTHROPIC_API_KEY = savedAnt;
      }
    }
  });

  it("reranker='cross-encoder' with apiKey falls back to LLM and produces a number", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => "",
      json: async () => ({
        choices: [{ message: { content: JSON.stringify({ score: 0.77 }) } }],
        usage: { prompt_tokens: 10, completion_tokens: 4 },
      }),
    } as Response);

    const score = await rerankPair(
      { name: "John" },
      { name: "Jon" },
      ["name"],
      { reranker: "cross-encoder", apiKey: "sk-test", maxRetries: 0 },
    );
    expect(typeof score).toBe("number");
    // Fallback exercised the LLM path -> fetch was called.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(score).toBeCloseTo(0.77, 6);
  });

  it("reranker='llm' (default) uses the LLM path", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => "",
      json: async () => ({
        choices: [{ message: { content: JSON.stringify({ score: 0.91 }) } }],
        usage: { prompt_tokens: 10, completion_tokens: 4 },
      }),
    } as Response);

    const score = await rerankPair(
      { name: "John" },
      { name: "Jon" },
      ["name"],
      { reranker: "llm", apiKey: "sk-test", maxRetries: 0 },
    );
    expect(typeof score).toBe("number");
    expect(score).toBeCloseTo(0.91, 6);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("rerankTopPairs with reranker='cross-encoder' fallback", () => {
  const mk: MatchkeyConfig = {
    name: "name_match",
    type: "weighted",
    threshold: 0.85,
    fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1 }],
  };

  const rows: Row[] = [
    { __row_id__: 0, name: "Alice" },
    { __row_id__: 1, name: "Alyce" },
  ];

  beforeEach(() => {
    _resetCrossEncoderModelCache();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("falls back to LLM per-pair when cross-encoder load fails", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockResolvedValueOnce({
      ok: true,
      status: 200,
      text: async () => "",
      json: async () => ({
        choices: [{ message: { content: JSON.stringify({ score: 1.0 }) } }],
        usage: { prompt_tokens: 10, completion_tokens: 4 },
      }),
    } as Response);

    const pairs: ScoredPair[] = [{ idA: 0, idB: 1, score: 0.9 }];
    const out = await rerankTopPairs(pairs, rows, mk, {
      reranker: "cross-encoder",
      apiKey: "sk-test",
      maxRetries: 0,
    });
    expect(out.length).toBe(1);
    // (1 - 0.5) * 0.9 + 0.5 * 1.0 = 0.95
    expect(out[0]!.score).toBeCloseTo(0.95, 6);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("passes pairs through unchanged when cross-encoder load fails and no apiKey", async () => {
    const env = (globalThis as { process?: { env?: Record<string, string | undefined> } })
      .process?.env;
    const savedOpen = env?.OPENAI_API_KEY;
    const savedAnt = env?.ANTHROPIC_API_KEY;
    if (env) {
      delete env.OPENAI_API_KEY;
      delete env.ANTHROPIC_API_KEY;
    }
    try {
      const pairs: ScoredPair[] = [{ idA: 0, idB: 1, score: 0.9 }];
      const out = await rerankTopPairs(pairs, rows, mk, {
        reranker: "cross-encoder",
      });
      // No backend available -> original scores survive (>= threshold).
      expect(out.length).toBe(1);
      expect(out[0]!.score).toBeCloseTo(0.9, 6);
    } finally {
      if (env) {
        if (savedOpen !== undefined) env.OPENAI_API_KEY = savedOpen;
        if (savedAnt !== undefined) env.ANTHROPIC_API_KEY = savedAnt;
      }
    }
  });
});
