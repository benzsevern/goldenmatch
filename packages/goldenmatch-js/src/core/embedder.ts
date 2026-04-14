/**
 * embedder.ts — Embedding API client (OpenAI / Vertex AI / Voyage).
 *
 * Edge-safe: uses global `fetch()` only. No `node:` imports.
 *
 * Ports `goldenmatch/core/embedder.py` and `goldenmatch/core/vertex_embedder.py`,
 * but replaces sentence-transformers / google-cloud-aiplatform with HTTP calls
 * so the module runs in Edge / Workers / browser-like runtimes.
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type EmbedderProvider = "openai" | "vertex" | "voyage";

export interface EmbedderOptions {
  readonly provider?: EmbedderProvider;
  readonly model?: string;
  readonly apiKey?: string;
  /** Override the default endpoint URL. */
  readonly endpoint?: string;
  /** Batch size for API calls (default 64 for OpenAI, 50 for Vertex). */
  readonly batchSize?: number;
  /** Cache embeddings by text hash within an Embedder instance. */
  readonly cache?: boolean;
  /**
   * For OpenAI text-embedding-3+ this requests a smaller embedding
   * dimension (e.g. 512 instead of 1536).
   */
  readonly dimensions?: number;
  /** GCP project ID (required for Vertex). */
  readonly project?: string;
  /** GCP region (Vertex). Default: us-central1. */
  readonly location?: string;
  /** Pre-fetched OAuth bearer token for Vertex. */
  readonly bearerToken?: string;
  /** Maximum HTTP retries for transient failures (default 3). */
  readonly maxRetries?: number;
}

export interface EmbeddingResult {
  readonly embeddings: readonly Float32Array[];
  readonly model: string;
  readonly tokensUsed: number;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class EmbedderError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly body?: string,
  ) {
    super(message);
    this.name = "EmbedderError";
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function readEnv(key: string): string | undefined {
  // Optional, soft-read env without hard `process` reference.
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.[key];
}

async function sleep(ms: number): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

/** Cheap stable hash for cache keys. */
function hashText(text: string): string {
  // FNV-1a 32-bit. Stable, no collisions matter much for cache use.
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

/** L2-normalize an embedding in place. */
function l2Normalize(vec: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i]! * vec[i]!;
  const norm = Math.sqrt(s);
  if (norm === 0) return vec;
  for (let i = 0; i < vec.length; i++) vec[i] = vec[i]! / norm;
  return vec;
}

function toFloat32(arr: readonly number[]): Float32Array {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = arr[i] ?? 0;
  return out;
}

// ---------------------------------------------------------------------------
// Provider defaults
// ---------------------------------------------------------------------------

function defaultModelFor(provider: EmbedderProvider): string {
  switch (provider) {
    case "openai":
      return "text-embedding-3-small";
    case "vertex":
      return "text-embedding-004";
    case "voyage":
      return "voyage-3";
  }
}

function defaultBatchSizeFor(provider: EmbedderProvider): number {
  switch (provider) {
    case "openai":
      return 64;
    case "vertex":
      return 50;
    case "voyage":
      return 64;
  }
}

// ---------------------------------------------------------------------------
// Retry wrapper
// ---------------------------------------------------------------------------

async function fetchWithRetry(
  url: string,
  init: RequestInit,
  maxRetries: number,
): Promise<Response> {
  let lastErr: unknown = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const resp = await fetch(url, init);
      if (resp.status === 429 || (resp.status >= 500 && resp.status < 600)) {
        if (attempt < maxRetries) {
          // Exponential backoff: 0.5s, 1s, 2s ...
          await sleep(500 * Math.pow(2, attempt));
          continue;
        }
      }
      return resp;
    } catch (err) {
      lastErr = err;
      if (attempt < maxRetries) {
        await sleep(500 * Math.pow(2, attempt));
        continue;
      }
    }
  }
  throw new EmbedderError(
    `Network error after ${maxRetries + 1} attempts: ${String(lastErr)}`,
  );
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

export class Embedder {
  private readonly cacheMap = new Map<string, Float32Array>();
  private readonly provider: EmbedderProvider;
  private readonly model: string;
  private readonly batchSize: number;
  private readonly maxRetries: number;
  private readonly cacheEnabled: boolean;

  constructor(private readonly options: EmbedderOptions = {}) {
    this.provider = options.provider ?? "openai";
    this.model = options.model ?? defaultModelFor(this.provider);
    this.batchSize = options.batchSize ?? defaultBatchSizeFor(this.provider);
    this.maxRetries = options.maxRetries ?? 3;
    this.cacheEnabled = options.cache ?? true;
  }

  // ──────────────────────────────────────────────────────────
  // Public API
  // ──────────────────────────────────────────────────────────

  /** Embed a batch of texts in one or more API calls. */
  async embedBatch(texts: readonly string[]): Promise<EmbeddingResult> {
    if (texts.length === 0) {
      return { embeddings: [], model: this.model, tokensUsed: 0 };
    }

    // Deduplicate while preserving original order mapping.
    const uniqueOrder: string[] = [];
    const uniqueIndex = new Map<string, number>();
    const indexFor: number[] = new Array(texts.length).fill(0);
    for (let i = 0; i < texts.length; i++) {
      const t = texts[i] ?? "";
      let idx = uniqueIndex.get(t);
      if (idx === undefined) {
        idx = uniqueOrder.length;
        uniqueOrder.push(t);
        uniqueIndex.set(t, idx);
      }
      indexFor[i] = idx;
    }

    // Resolve from cache where possible.
    const uniqueEmbeddings: (Float32Array | null)[] = new Array(uniqueOrder.length).fill(null);
    const toFetchIdx: number[] = [];
    const toFetchText: string[] = [];
    for (let i = 0; i < uniqueOrder.length; i++) {
      const text = uniqueOrder[i]!;
      if (this.cacheEnabled) {
        const hit = this.cacheMap.get(hashText(text));
        if (hit) {
          uniqueEmbeddings[i] = hit;
          continue;
        }
      }
      toFetchIdx.push(i);
      toFetchText.push(text);
    }

    let tokensUsed = 0;

    if (toFetchText.length > 0) {
      // Batch the API calls.
      for (let start = 0; start < toFetchText.length; start += this.batchSize) {
        const end = Math.min(start + this.batchSize, toFetchText.length);
        const slice = toFetchText.slice(start, end);
        const result = await this.callProvider(slice);
        tokensUsed += result.tokensUsed;
        for (let j = 0; j < result.embeddings.length; j++) {
          const targetIdx = toFetchIdx[start + j]!;
          const emb = result.embeddings[j]!;
          uniqueEmbeddings[targetIdx] = emb;
          if (this.cacheEnabled) {
            this.cacheMap.set(hashText(uniqueOrder[targetIdx]!), emb);
          }
        }
      }
    }

    // Re-expand back to original order.
    const embeddings: Float32Array[] = new Array(texts.length);
    for (let i = 0; i < texts.length; i++) {
      const u = uniqueEmbeddings[indexFor[i]!];
      if (!u) {
        // Should not happen; fall back to zero vector of last-known dim.
        const dim = this.firstDim(uniqueEmbeddings) ?? 0;
        embeddings[i] = new Float32Array(dim);
      } else {
        embeddings[i] = u;
      }
    }

    return { embeddings, model: this.model, tokensUsed };
  }

  /** Embed a single text. */
  async embedOne(text: string): Promise<Float32Array> {
    const r = await this.embedBatch([text]);
    return r.embeddings[0] ?? new Float32Array(0);
  }

  /**
   * Embed a column of (possibly null) values. Null/empty get a zero vector.
   * Identical text values are de-duplicated automatically.
   */
  async embedColumn(
    values: readonly (string | null | undefined)[],
    _cacheKey?: string,
  ): Promise<readonly Float32Array[]> {
    if (values.length === 0) return [];

    // Substitute null/empty with a sentinel; we replace with zero vectors after.
    const ZERO_SENTINEL = "\u0000__GM_NULL__\u0000";
    const inputs: string[] = values.map((v) => {
      if (v === null || v === undefined) return ZERO_SENTINEL;
      const s = String(v).trim();
      return s === "" ? ZERO_SENTINEL : s;
    });

    // Embed only the non-null subset.
    const nonNullTexts: string[] = [];
    const positions: number[] = [];
    for (let i = 0; i < inputs.length; i++) {
      if (inputs[i] !== ZERO_SENTINEL) {
        nonNullTexts.push(inputs[i]!);
        positions.push(i);
      }
    }

    let dim = 0;
    let realEmbeddings: readonly Float32Array[] = [];
    if (nonNullTexts.length > 0) {
      const r = await this.embedBatch(nonNullTexts);
      realEmbeddings = r.embeddings;
      dim = realEmbeddings[0]?.length ?? 0;
    }

    const out: Float32Array[] = new Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = new Float32Array(dim);
    for (let k = 0; k < positions.length; k++) {
      out[positions[k]!] = realEmbeddings[k] ?? new Float32Array(dim);
    }
    return out;
  }

  // ──────────────────────────────────────────────────────────
  // Similarity helpers
  // ──────────────────────────────────────────────────────────

  cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    let na = 0;
    let nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const av = a[i]!;
      const bv = b[i]!;
      dot += av * bv;
      na += av * av;
      nb += bv * bv;
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  }

  cosineSimilarityMatrix(embeddings: readonly Float32Array[]): number[][] {
    const n = embeddings.length;
    const out: number[][] = new Array(n);
    for (let i = 0; i < n; i++) out[i] = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      out[i]![i] = 1;
      for (let j = i + 1; j < n; j++) {
        const s = this.cosineSimilarity(embeddings[i]!, embeddings[j]!);
        out[i]![j] = s;
        out[j]![i] = s;
      }
    }
    return out;
  }

  // ──────────────────────────────────────────────────────────
  // Internals
  // ──────────────────────────────────────────────────────────

  private firstDim(arr: readonly (Float32Array | null)[]): number | null {
    for (const v of arr) if (v) return v.length;
    return null;
  }

  private async callProvider(
    texts: readonly string[],
  ): Promise<EmbeddingResult> {
    switch (this.provider) {
      case "openai":
        return this.callOpenAI(texts);
      case "vertex":
        return this.callVertex(texts);
      case "voyage":
        return this.callVoyage(texts);
    }
  }

  // ── OpenAI ────────────────────────────────────────────────
  private async callOpenAI(texts: readonly string[]): Promise<EmbeddingResult> {
    const apiKey = this.options.apiKey ?? readEnv("OPENAI_API_KEY");
    if (!apiKey) {
      throw new EmbedderError(
        "OpenAI API key required. Pass options.apiKey or set OPENAI_API_KEY.",
      );
    }
    const url = this.options.endpoint ?? "https://api.openai.com/v1/embeddings";
    const body: Record<string, unknown> = {
      model: this.model,
      input: texts,
    };
    if (this.options.dimensions !== undefined) {
      body.dimensions = this.options.dimensions;
    }
    const resp = await fetchWithRetry(
      url,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      },
      this.maxRetries,
    );
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new EmbedderError(
        `OpenAI embeddings ${resp.status}`,
        resp.status,
        text.slice(0, 500),
      );
    }
    const data = (await resp.json()) as {
      data?: Array<{ embedding?: number[] }>;
      usage?: { total_tokens?: number };
    };
    const arr = data.data ?? [];
    const embeddings: Float32Array[] = arr.map((d) => {
      const v = d.embedding ?? [];
      return l2Normalize(toFloat32(v));
    });
    return {
      embeddings,
      model: this.model,
      tokensUsed: data.usage?.total_tokens ?? 0,
    };
  }

  // ── Vertex AI ─────────────────────────────────────────────
  private async callVertex(texts: readonly string[]): Promise<EmbeddingResult> {
    const project = this.options.project ?? readEnv("GOOGLE_CLOUD_PROJECT");
    if (!project) {
      throw new EmbedderError(
        "Vertex requires options.project or GOOGLE_CLOUD_PROJECT.",
      );
    }
    const location =
      this.options.location ??
      readEnv("GOOGLE_CLOUD_LOCATION") ??
      "us-central1";
    const token =
      this.options.bearerToken ??
      this.options.apiKey ??
      readEnv("GOOGLE_OAUTH_TOKEN");
    if (!token) {
      throw new EmbedderError(
        "Vertex requires options.bearerToken (OAuth access token). " +
          "Service-account JWT signing is unavailable in edge runtime — " +
          "fetch a token out-of-band and pass it in.",
      );
    }
    const url =
      this.options.endpoint ??
      `https://${location}-aiplatform.googleapis.com/v1/projects/${project}/locations/${location}/publishers/google/models/${this.model}:predict`;

    const body = JSON.stringify({
      instances: texts.map((t) => ({ content: t })),
    });
    const resp = await fetchWithRetry(
      url,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body,
      },
      this.maxRetries,
    );
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new EmbedderError(
        `Vertex embeddings ${resp.status}`,
        resp.status,
        text.slice(0, 500),
      );
    }
    const data = (await resp.json()) as {
      predictions?: Array<{
        embeddings?: { values?: number[]; statistics?: { token_count?: number } };
      }>;
    };
    let tokens = 0;
    const embeddings: Float32Array[] = (data.predictions ?? []).map((p) => {
      const v = p.embeddings?.values ?? [];
      tokens += p.embeddings?.statistics?.token_count ?? 0;
      return l2Normalize(toFloat32(v));
    });
    return { embeddings, model: this.model, tokensUsed: tokens };
  }

  // ── Voyage AI ─────────────────────────────────────────────
  private async callVoyage(texts: readonly string[]): Promise<EmbeddingResult> {
    const apiKey = this.options.apiKey ?? readEnv("VOYAGE_API_KEY");
    if (!apiKey) {
      throw new EmbedderError(
        "Voyage API key required. Pass options.apiKey or set VOYAGE_API_KEY.",
      );
    }
    const url = this.options.endpoint ?? "https://api.voyageai.com/v1/embeddings";
    const resp = await fetchWithRetry(
      url,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ model: this.model, input: texts }),
      },
      this.maxRetries,
    );
    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new EmbedderError(
        `Voyage embeddings ${resp.status}`,
        resp.status,
        text.slice(0, 500),
      );
    }
    const data = (await resp.json()) as {
      data?: Array<{ embedding?: number[] }>;
      usage?: { total_tokens?: number };
    };
    const arr = data.data ?? [];
    const embeddings: Float32Array[] = arr.map((d) => {
      const v = d.embedding ?? [];
      return l2Normalize(toFloat32(v));
    });
    return {
      embeddings,
      model: this.model,
      tokensUsed: data.usage?.total_tokens ?? 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Singleton factory
// ---------------------------------------------------------------------------

const embedderCache = new Map<string, Embedder>();

/**
 * Return a cached Embedder instance keyed by provider+model.
 * Pass a string to use a model name with default provider, or full options.
 */
export function getEmbedder(modelOrOptions?: string | EmbedderOptions): Embedder {
  const opts: EmbedderOptions =
    typeof modelOrOptions === "string"
      ? { model: modelOrOptions }
      : (modelOrOptions ?? {});
  const provider = opts.provider ?? "openai";
  const model = opts.model ?? defaultModelFor(provider);
  const key = `${provider}::${model}`;
  let e = embedderCache.get(key);
  if (!e) {
    e = new Embedder(opts);
    embedderCache.set(key, e);
  }
  return e;
}

/** Test-only: clear the embedder cache. */
export function _clearEmbedderCache(): void {
  embedderCache.clear();
}
