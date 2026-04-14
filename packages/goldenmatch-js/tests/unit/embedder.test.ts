import {
  describe,
  it,
  expect,
  vi,
  beforeEach,
  afterEach,
} from "vitest";
import { Embedder, getEmbedder, EmbedderError } from "../../src/core/index.js";
import { _clearEmbedderCache } from "../../src/core/embedder.js";

let fetchMock: ReturnType<typeof vi.fn>;

function vec(...nums: number[]): Float32Array {
  return new Float32Array(nums);
}

function mockOpenAIResponse(embeddings: number[][], totalTokens = 10) {
  return {
    ok: true,
    status: 200,
    text: async () => "",
    json: async () => ({
      data: embeddings.map((embedding) => ({ embedding })),
      usage: { total_tokens: totalTokens },
    }),
  } as Response;
}

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
  _clearEmbedderCache();
  // Avoid accidental env apiKey leakage between tests.
  delete (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.OPENAI_API_KEY;
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Embedder.cosineSimilarity", () => {
  it("known orthogonal vectors -> 0", () => {
    const e = new Embedder({ apiKey: "sk-test" });
    expect(e.cosineSimilarity(vec(1, 0, 0), vec(0, 1, 0))).toBe(0);
  });

  it("identical vectors -> 1", () => {
    const e = new Embedder({ apiKey: "sk-test" });
    expect(e.cosineSimilarity(vec(1, 2, 3), vec(1, 2, 3))).toBeCloseTo(1, 6);
  });

  it("zero vector -> 0", () => {
    const e = new Embedder({ apiKey: "sk-test" });
    expect(e.cosineSimilarity(vec(0, 0), vec(1, 1))).toBe(0);
  });
});

describe("Embedder.cosineSimilarityMatrix", () => {
  it("returns NxN matrix with diagonal of 1", () => {
    const e = new Embedder({ apiKey: "sk-test" });
    const embeddings = [vec(1, 0, 0), vec(0, 1, 0), vec(1, 0, 0)];
    const matrix = e.cosineSimilarityMatrix(embeddings);
    expect(matrix.length).toBe(3);
    expect(matrix[0]!.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      expect(matrix[i]![i]).toBe(1);
    }
    // Symmetric off-diagonals
    expect(matrix[0]![1]).toBeCloseTo(matrix[1]![0]!, 9);
    // 0 and 2 are identical
    expect(matrix[0]![2]).toBeCloseTo(1, 6);
  });
});

describe("Embedder.embedBatch — unique-text dedup", () => {
  it("dedupes identical texts before calling API", async () => {
    fetchMock.mockResolvedValueOnce(
      mockOpenAIResponse([
        [1, 0, 0],
        [0, 1, 0],
      ]),
    );
    const e = new Embedder({ apiKey: "sk-test" });
    const result = await e.embedBatch(["foo", "bar", "foo", "foo"]);

    // Only one API call
    expect(fetchMock).toHaveBeenCalledTimes(1);
    // Body should contain only unique inputs ["foo","bar"]
    const callArgs = fetchMock.mock.calls[0];
    const body = JSON.parse((callArgs![1] as RequestInit).body as string);
    expect(body.input).toEqual(["foo", "bar"]);

    // 4 outputs in original order
    expect(result.embeddings.length).toBe(4);
    // foo positions (0, 2, 3) share the same embedding object identity
    expect(result.embeddings[0]).toBe(result.embeddings[2]);
    expect(result.embeddings[0]).toBe(result.embeddings[3]);
  });
});

describe("Embedder.embedOne", () => {
  it("returns single Float32Array", async () => {
    fetchMock.mockResolvedValueOnce(mockOpenAIResponse([[3, 4]]));
    const e = new Embedder({ apiKey: "sk-test" });
    const v = await e.embedOne("hello");
    expect(v).toBeInstanceOf(Float32Array);
    expect(v.length).toBe(2);
  });
});

describe("Embedder error handling", () => {
  it("throws EmbedderError when no apiKey and no env var", async () => {
    const e = new Embedder({ provider: "openai" });
    // Make sure there's truly no apiKey.
    await expect(e.embedBatch(["hello"])).rejects.toThrow(EmbedderError);
  });

  it("throws EmbedderError when API returns non-2xx", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 401,
      text: async () => "unauthorized",
      json: async () => ({}),
    } as Response);
    const e = new Embedder({ apiKey: "sk-test", maxRetries: 0 });
    await expect(e.embedBatch(["hi"])).rejects.toThrow(EmbedderError);
  });
});

describe("getEmbedder caching", () => {
  it("returns same instance for same provider+model", () => {
    _clearEmbedderCache();
    const a = getEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    const b = getEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    expect(a).toBe(b);
  });

  it("returns different instance for different model", () => {
    _clearEmbedderCache();
    const a = getEmbedder({ apiKey: "sk-test", model: "text-embedding-3-small" });
    const b = getEmbedder({ apiKey: "sk-test", model: "text-embedding-3-large" });
    expect(a).not.toBe(b);
  });

  it("accepts string shorthand for model", () => {
    _clearEmbedderCache();
    const a = getEmbedder("text-embedding-3-small");
    expect(a).toBeInstanceOf(Embedder);
  });
});
