import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
} from "vitest";
import { rerankPair, rerankTopPairs } from "../../src/core/index.js";
import type {
  MatchkeyConfig,
  Row,
  ScoredPair,
} from "../../src/core/index.js";

let fetchMock: ReturnType<typeof vi.fn>;

function mockOpenAIChat(score: number) {
  return {
    ok: true,
    status: 200,
    text: async () => "",
    json: async () => ({
      choices: [{ message: { content: JSON.stringify({ score }) } }],
      usage: { prompt_tokens: 50, completion_tokens: 8 },
    }),
  } as Response;
}

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("rerankPair", () => {
  it("returns parsed 0..1 score on success", async () => {
    fetchMock.mockResolvedValueOnce(mockOpenAIChat(0.92));
    const score = await rerankPair(
      { name: "Bob Smith" },
      { name: "Robert Smith" },
      ["name"],
      { apiKey: "sk-test", maxRetries: 0 },
    );
    expect(score).toBeCloseTo(0.92, 6);
  });

  it("returns NaN when no apiKey provided", async () => {
    // Ensure env var not set.
    const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
    const savedOpen = env?.OPENAI_API_KEY;
    const savedAnt = env?.ANTHROPIC_API_KEY;
    if (env) {
      delete env.OPENAI_API_KEY;
      delete env.ANTHROPIC_API_KEY;
    }
    try {
      const score = await rerankPair({ a: 1 }, { a: 1 }, ["a"], {});
      expect(Number.isNaN(score)).toBe(true);
    } finally {
      if (env) {
        if (savedOpen !== undefined) env.OPENAI_API_KEY = savedOpen;
        if (savedAnt !== undefined) env.ANTHROPIC_API_KEY = savedAnt;
      }
    }
  });

  it("returns NaN on HTTP failure", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: async () => "unauthorized",
      json: async () => ({}),
    } as Response);
    const score = await rerankPair({}, {}, [], {
      apiKey: "sk-test",
      maxRetries: 0,
    });
    expect(Number.isNaN(score)).toBe(true);
  });
});

describe("rerankTopPairs", () => {
  const mk: MatchkeyConfig = {
    name: "name_match",
    type: "weighted",
    threshold: 0.85,
    fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1 }],
  };

  const rows: Row[] = [
    { __row_id__: 0, name: "Alice" },
    { __row_id__: 1, name: "Alyce" },
    { __row_id__: 2, name: "Bob" },
    { __row_id__: 3, name: "Robert" },
  ];

  it("filters pairs to within band [threshold-band, 1.0] and caps at topN", async () => {
    // Band default 0.1 → only pairs >= 0.75 are candidates.
    const pairs: ScoredPair[] = [
      { idA: 0, idB: 1, score: 0.95 },
      { idA: 2, idB: 3, score: 0.80 },
      { idA: 0, idB: 2, score: 0.50 }, // below band - not reranked
    ];
    // Mock 1 LLM response (topN=1)
    fetchMock.mockResolvedValueOnce(mockOpenAIChat(1.0));

    const out = await rerankTopPairs(pairs, rows, mk, {
      apiKey: "sk-test",
      maxRetries: 0,
      topN: 1,
      band: 0.1,
    });

    // Only one fetch (capped at topN=1)
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // Output should drop pairs below threshold
    // - (0,1): combined=0.5*0.95+0.5*1.0=0.975 -> kept
    // - (2,3): not reranked, original 0.80 < threshold 0.85 -> dropped
    // - (0,2): not reranked, 0.50 < 0.85 -> dropped
    const ids = out.map((p) => `${p.idA}-${p.idB}`);
    expect(ids).toContain("0-1");
    expect(ids).not.toContain("0-2");
    expect(ids).not.toContain("2-3");
  });

  it("returns input unchanged when no apiKey", async () => {
    const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
    const savedOpen = env?.OPENAI_API_KEY;
    const savedAnt = env?.ANTHROPIC_API_KEY;
    if (env) {
      delete env.OPENAI_API_KEY;
      delete env.ANTHROPIC_API_KEY;
    }
    try {
      const pairs: ScoredPair[] = [{ idA: 0, idB: 1, score: 0.9 }];
      const out = await rerankTopPairs(pairs, rows, mk, {});
      expect(out).toBe(pairs);
    } finally {
      if (env) {
        if (savedOpen !== undefined) env.OPENAI_API_KEY = savedOpen;
        if (savedAnt !== undefined) env.ANTHROPIC_API_KEY = savedAnt;
      }
    }
  });

  it("on HTTP failure pair keeps original score", async () => {
    // Borderline pair: 0.86 (just above threshold 0.85). LLM call will fail.
    const pairs: ScoredPair[] = [{ idA: 0, idB: 1, score: 0.86 }];
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 500,
      text: async () => "boom",
      json: async () => ({}),
    } as Response);

    const out = await rerankTopPairs(pairs, rows, mk, {
      apiKey: "sk-test",
      maxRetries: 0,
    });
    // Pair kept with original score (0.86 >= threshold 0.85)
    expect(out.length).toBe(1);
    expect(out[0]!.score).toBeCloseTo(0.86, 6);
  });

  it("empty input -> empty output", async () => {
    const out = await rerankTopPairs([], rows, mk, { apiKey: "sk-test" });
    expect(out).toEqual([]);
  });
});
