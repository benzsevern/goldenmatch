/**
 * cluster.ts — In-context LLM clustering: send blocks of borderline records
 * to an LLM for direct cluster assignment. Ports `goldenmatch/core/llm_cluster.py`.
 *
 * Flow:
 *   1. Pairs with candidateLo <= score < autoThreshold form the borderline band.
 *   2. Build connected components over the borderline graph.
 *   3. Oversized components split by dropping weakest edges first.
 *   4. Each component (block) sent to LLM with a JSON cluster schema.
 *   5. Pair scores synthesized from cluster membership + confidence.
 *
 * Degrades: cluster call fails -> pairwise fallback -> return input pairs.
 * Edge-safe: fetch-only, no `node:` imports.
 */

import type { Row, ScoredPair, LLMScorerConfig } from "../types.js";
import { BudgetTracker, countTokensApprox } from "./budget.js";
import type { BudgetSnapshot } from "./budget.js";
import { llmScorePairs, LLMHttpError } from "./scorer.js";
import type { LLMScoreResult } from "./scorer.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ClusterBlock {
  readonly records: readonly number[];
  readonly pairs: readonly ScoredPair[];
}

interface LLMClusterResponse {
  readonly clusters: ReadonlyArray<{
    readonly members: readonly number[];
    readonly confidence: number;
  }>;
  readonly singletons: readonly number[];
}

// ---------------------------------------------------------------------------
// Public: llmClusterPairs
// ---------------------------------------------------------------------------

export async function llmClusterPairs(
  pairs: readonly ScoredPair[],
  rows: readonly Row[],
  config: LLMScorerConfig,
  apiKey?: string,
): Promise<LLMScoreResult> {
  const budget = new BudgetTracker(
    config.budget ?? {},
    config.model ?? "gpt-4o-mini",
  );

  if (pairs.length === 0) {
    return { pairs: [], budget: budget.snapshot() };
  }

  // Tier partition.
  const autoAccept: ScoredPair[] = [];
  const candidates: ScoredPair[] = [];
  const below: ScoredPair[] = [];
  for (const p of pairs) {
    if (p.score >= config.autoThreshold) autoAccept.push(p);
    else if (p.score >= config.candidateLo && p.score < config.candidateHi) {
      candidates.push(p);
    } else below.push(p);
  }

  // Result scaffold.
  const result: ScoredPair[] = [];
  for (const p of autoAccept) result.push({ idA: p.idA, idB: p.idB, score: 1.0 });
  for (const p of below) result.push(p);

  if (candidates.length === 0) {
    return { pairs: result, budget: budget.snapshot() };
  }

  // Build row lookup.
  const rowById = new Map<number, Row>();
  for (const r of rows) {
    const id = r["__row_id__"];
    if (typeof id === "number") rowById.set(id, r);
  }

  // Display columns: first 6 non-internal columns.
  const cols = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (!k.startsWith("__")) cols.add(k);
    }
  }
  const displayCols = [...cols].slice(0, 6);

  // Build connected components over the borderline graph.
  const clusterMax = config.clusterMaxSize ?? 20;
  const clusterMin = config.clusterMinSize ?? 2;
  const components = buildComponents(candidates);

  const provider = (config.provider === "anthropic" ? "anthropic" : "openai") as
    | "openai"
    | "anthropic";
  const model =
    config.model ??
    (provider === "openai" ? "gpt-4o-mini" : "claude-haiku-4-5-20251001");

  // Pairs we still need to emit: start with all candidates, remove as we resolve.
  const unresolved = new Set<ScoredPair>(candidates);

  // No API key -> degrade: keep candidates at original scores.
  if (!apiKey) {
    for (const p of candidates) result.push(p);
    return { pairs: result, budget: budget.snapshot() };
  }

  for (const component of components) {
    if (budget.exhausted) break;

    // Tiny components: fall back to pairwise scoring.
    if (component.records.length < clusterMin) {
      const fallback = await llmScorePairs(component.pairs, rows, config, apiKey);
      for (const p of fallback.pairs) result.push(p);
      for (const p of component.pairs) unresolved.delete(p);
      continue;
    }

    // Split oversized components by trimming weakest edges first.
    const blocks = splitComponent(component, clusterMax);

    for (const block of blocks) {
      if (budget.exhausted) break;

      let clusterResult: LLMClusterResponse | null = null;
      try {
        clusterResult = await callLlmCluster(
          block.records,
          rowById,
          displayCols,
          provider,
          model,
          apiKey,
          budget,
        );
      } catch (err) {
        // eslint-disable-next-line no-console
        console.warn(
          "llm_cluster call failed, falling back to pairwise:",
          err instanceof Error ? err.message : String(err),
        );
        clusterResult = null;
      }

      if (clusterResult === null) {
        // Pairwise fallback.
        const fallback = await llmScorePairs(block.pairs, rows, config, apiKey);
        for (const p of fallback.pairs) result.push(p);
      } else {
        const synthesized = applyClusterResults(clusterResult, block.pairs);
        for (const p of synthesized) result.push(p);
      }
      for (const p of block.pairs) unresolved.delete(p);
    }
  }

  // Anything still unresolved (e.g. budget exhausted early): keep original.
  for (const p of unresolved) result.push(p);

  return { pairs: result, budget: budget.snapshot() };
}

// ---------------------------------------------------------------------------
// Component graph construction
// ---------------------------------------------------------------------------

function buildComponents(pairs: readonly ScoredPair[]): ClusterBlock[] {
  const adj = new Map<number, Set<number>>();
  const recordPairs = new Map<number, ScoredPair[]>();

  for (const p of pairs) {
    if (!adj.has(p.idA)) adj.set(p.idA, new Set());
    if (!adj.has(p.idB)) adj.set(p.idB, new Set());
    adj.get(p.idA)!.add(p.idB);
    adj.get(p.idB)!.add(p.idA);
    if (!recordPairs.has(p.idA)) recordPairs.set(p.idA, []);
    if (!recordPairs.has(p.idB)) recordPairs.set(p.idB, []);
    recordPairs.get(p.idA)!.push(p);
    recordPairs.get(p.idB)!.push(p);
  }

  const visited = new Set<number>();
  const components: ClusterBlock[] = [];

  for (const start of adj.keys()) {
    if (visited.has(start)) continue;
    const members: number[] = [];
    const stack = [start];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (visited.has(node)) continue;
      visited.add(node);
      members.push(node);
      const neighbors = adj.get(node);
      if (neighbors) {
        for (const nb of neighbors) {
          if (!visited.has(nb)) stack.push(nb);
        }
      }
    }

    // Collect pairs that live entirely within this component.
    const memberSet = new Set(members);
    const seen = new Set<string>();
    const compPairs: ScoredPair[] = [];
    for (const rec of members) {
      const ps = recordPairs.get(rec);
      if (!ps) continue;
      for (const p of ps) {
        if (!memberSet.has(p.idA) || !memberSet.has(p.idB)) continue;
        const key = `${p.idA}:${p.idB}`;
        if (seen.has(key)) continue;
        seen.add(key);
        compPairs.push(p);
      }
    }

    components.push({ records: members.sort((a, b) => a - b), pairs: compPairs });
  }

  return components;
}

// ---------------------------------------------------------------------------
// Component splitting (oversized blocks)
// ---------------------------------------------------------------------------

function splitComponent(
  component: ClusterBlock,
  maxSize: number,
): ClusterBlock[] {
  if (component.records.length <= maxSize) return [component];

  // Work with a mutable adjacency map.
  const adj = new Map<number, Set<number>>();
  for (const rec of component.records) adj.set(rec, new Set());
  for (const p of component.pairs) {
    adj.get(p.idA)!.add(p.idB);
    adj.get(p.idB)!.add(p.idA);
  }

  // Edges sorted by score ascending (weakest first).
  const edges = [...component.pairs].sort((a, b) => a.score - b.score);
  const removed = new Set<ScoredPair>();

  for (const e of edges) {
    adj.get(e.idA)?.delete(e.idB);
    adj.get(e.idB)?.delete(e.idA);
    removed.add(e);
    const max = largestComponentSize(adj, component.records);
    if (max <= maxSize) break;
  }

  // Rebuild components with the surviving edges.
  const remainingPairs = component.pairs.filter((p) => !removed.has(p));
  const remainingAdj = new Map<number, Set<number>>();
  for (const rec of component.records) remainingAdj.set(rec, new Set());
  for (const p of remainingPairs) {
    remainingAdj.get(p.idA)!.add(p.idB);
    remainingAdj.get(p.idB)!.add(p.idA);
  }

  const visited = new Set<number>();
  const blocks: ClusterBlock[] = [];
  for (const start of component.records) {
    if (visited.has(start)) continue;
    const comp: number[] = [];
    const stack = [start];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (visited.has(node)) continue;
      visited.add(node);
      comp.push(node);
      const nbs = remainingAdj.get(node);
      if (nbs) {
        for (const nb of nbs) if (!visited.has(nb)) stack.push(nb);
      }
    }
    const memberSet = new Set(comp);
    const compPairs = remainingPairs.filter(
      (p) => memberSet.has(p.idA) && memberSet.has(p.idB),
    );
    blocks.push({ records: comp.sort((a, b) => a - b), pairs: compPairs });
  }

  return blocks;
}

function largestComponentSize(
  adj: ReadonlyMap<number, ReadonlySet<number>>,
  records: readonly number[],
): number {
  const visited = new Set<number>();
  let max = 0;
  for (const start of records) {
    if (visited.has(start)) continue;
    let size = 0;
    const stack = [start];
    while (stack.length > 0) {
      const node = stack.pop()!;
      if (visited.has(node)) continue;
      visited.add(node);
      size++;
      const nbs = adj.get(node);
      if (nbs) {
        for (const nb of nbs) if (!visited.has(nb)) stack.push(nb);
      }
    }
    if (size > max) max = size;
  }
  return max;
}

// ---------------------------------------------------------------------------
// LLM cluster call
// ---------------------------------------------------------------------------

async function callLlmCluster(
  recordIds: readonly number[],
  rowById: ReadonlyMap<number, Row>,
  displayCols: readonly string[],
  provider: "openai" | "anthropic",
  model: string,
  apiKey: string,
  budget: BudgetTracker,
): Promise<LLMClusterResponse> {
  const lines: string[] = [
    "Group these records into clusters of duplicates. Return JSON only.",
    "",
    "Records:",
  ];
  for (const rid of recordIds) {
    const row = rowById.get(rid) ?? {};
    const parts = displayCols.map((c) => String(row[c] ?? ""));
    lines.push(`  [${rid}] ${parts.join(" | ")}`);
  }
  lines.push("");
  lines.push(
    'Return JSON: {"clusters": [{"members": [id1, id2, ...], "confidence": 0.0-1.0}, ...], "singletons": [id1, ...]}',
  );
  lines.push("Rules:");
  lines.push("- Each record appears in exactly one cluster or as a singleton");
  lines.push("- confidence = how certain you are that all members are the same entity");
  lines.push("- Only group records that are clearly the same real-world entity");

  const prompt = lines.join("\n");
  const estTokens = countTokensApprox(prompt);
  if (!budget.canSend(estTokens)) {
    throw new Error("Budget insufficient for this block");
  }

  const maxTokens = Math.min(2000, Math.max(200, recordIds.length * 30));
  const { text, inputTokens, outputTokens } =
    provider === "openai"
      ? await openaiJson(prompt, apiKey, model, maxTokens)
      : await anthropicJson(prompt, apiKey, model, maxTokens);

  budget.record(inputTokens || estTokens, outputTokens || maxTokens, model);
  return parseClusterResponse(text, recordIds);
}

async function openaiJson(
  prompt: string,
  apiKey: string,
  model: string,
  maxTokens: number,
): Promise<{ text: string; inputTokens: number; outputTokens: number }> {
  const resp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      max_tokens: maxTokens,
    }),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new LLMHttpError(resp.status, `OpenAI ${resp.status}: ${body.slice(0, 200)}`);
  }
  const data = (await resp.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
    usage?: { prompt_tokens?: number; completion_tokens?: number };
  };
  return {
    text: data.choices?.[0]?.message?.content ?? "",
    inputTokens: data.usage?.prompt_tokens ?? 0,
    outputTokens: data.usage?.completion_tokens ?? 0,
  };
}

async function anthropicJson(
  prompt: string,
  apiKey: string,
  model: string,
  maxTokens: number,
): Promise<{ text: string; inputTokens: number; outputTokens: number }> {
  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "content-type": "application/json",
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: maxTokens,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new LLMHttpError(resp.status, `Anthropic ${resp.status}: ${body.slice(0, 200)}`);
  }
  const data = (await resp.json()) as {
    content?: Array<{ text?: string }>;
    usage?: { input_tokens?: number; output_tokens?: number };
  };
  return {
    text: data.content?.[0]?.text ?? "",
    inputTokens: data.usage?.input_tokens ?? 0,
    outputTokens: data.usage?.output_tokens ?? 0,
  };
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

function parseClusterResponse(
  response: string,
  validIds: readonly number[],
): LLMClusterResponse {
  const validSet = new Set(validIds);
  const fallback: LLMClusterResponse = {
    clusters: [],
    singletons: [...validIds],
  };

  let text = response.trim();

  // Strip markdown code fences if present.
  if (text.includes("```")) {
    const parts = text.split("```");
    for (const raw of parts) {
      let p = raw.trim();
      if (p.startsWith("json")) p = p.slice(4).trim();
      if (p.startsWith("{")) {
        text = p;
        break;
      }
    }
  }

  // Extract first balanced JSON object.
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        parsed = JSON.parse(text.slice(start, end + 1));
      } catch {
        return fallback;
      }
    } else {
      return fallback;
    }
  }

  if (!parsed || typeof parsed !== "object") return fallback;
  const obj = parsed as {
    clusters?: Array<{ members?: unknown; confidence?: unknown }>;
  };

  const clusters: Array<{ members: number[]; confidence: number }> = [];
  const assigned = new Set<number>();

  for (const c of obj.clusters ?? []) {
    const membersRaw = Array.isArray(c.members) ? c.members : [];
    const conf = typeof c.confidence === "number" ? c.confidence : 0.5;
    const clamped = Math.max(0, Math.min(1, conf));
    const validMembers = membersRaw
      .filter((m): m is number => typeof m === "number")
      .filter((m) => validSet.has(m) && !assigned.has(m));
    if (validMembers.length >= 2) {
      clusters.push({ members: validMembers, confidence: clamped });
      for (const m of validMembers) assigned.add(m);
    }
  }

  const singletons = validIds.filter((rid) => !assigned.has(rid));
  return { clusters, singletons };
}

// ---------------------------------------------------------------------------
// Synthesize pair_scores from cluster membership
// ---------------------------------------------------------------------------

function applyClusterResults(
  result: LLMClusterResponse,
  pairs: readonly ScoredPair[],
): ScoredPair[] {
  // record_id -> (cluster_index, confidence)
  const recordCluster = new Map<number, { idx: number; conf: number }>();
  result.clusters.forEach((c, idx) => {
    for (const m of c.members) {
      recordCluster.set(m, { idx, conf: c.confidence });
    }
  });

  const out: ScoredPair[] = [];
  for (const p of pairs) {
    const ca = recordCluster.get(p.idA);
    const cb = recordCluster.get(p.idB);
    if (ca !== undefined && cb !== undefined && ca.idx === cb.idx) {
      // Same cluster: use cluster confidence.
      out.push({ idA: p.idA, idB: p.idB, score: ca.conf });
    } else {
      // Different cluster or singleton: rejected.
      out.push({ idA: p.idA, idB: p.idB, score: 0 });
    }
  }
  return out;
}

// Re-export for convenience.
export type { BudgetSnapshot, LLMScoreResult };
