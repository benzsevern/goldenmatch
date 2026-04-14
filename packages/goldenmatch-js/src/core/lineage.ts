/**
 * lineage.ts — Provenance tracking for golden records.
 * Edge-safe: no `node:` imports.
 *
 * Ports goldenmatch/core/lineage.py. Records which source rows contributed
 * each golden-record field, with the survivorship strategy and confidence.
 */

import type { ClusterInfo, DedupeResult, Row } from "./types.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FieldProvenanceEntry {
  readonly sourceRowId: number;
  readonly strategy: string;
  readonly confidence: number;
}

export interface LineageEdge {
  readonly clusterId: number;
  readonly sourceRowIds: readonly number[];
  readonly goldenRowId: number;
  readonly fieldProvenance: Readonly<Record<string, FieldProvenanceEntry>>;
  readonly timestamp: string;
}

export interface LineageBundle {
  readonly edges: readonly LineageEdge[];
  readonly timestamp: string;
  readonly recordCount: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isoTimestamp(): string {
  // Edge-safe: Date works in browser/workers/edge runtimes.
  return new Date().toISOString();
}

function getRowId(row: Row): number | null {
  const raw = row["__row_id__"];
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function getClusterId(row: Row): number | null {
  const raw = row["__cluster_id__"];
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function computeConfidence(cluster: ClusterInfo, strategy: string): number {
  // Base confidence is the cluster confidence; minor bonus for
  // most_complete / first_non_null which are safer picks.
  const base = cluster.confidence;
  if (strategy === "source_priority" || strategy === "most_complete") {
    return Math.min(1, base + 0.05);
  }
  return base;
}

// ---------------------------------------------------------------------------
// Build lineage
// ---------------------------------------------------------------------------

export interface BuildLineageOptions {
  readonly naturalLanguage?: boolean;
  readonly defaultStrategy?: string;
}

/**
 * Build a lineage bundle from a DedupeResult.
 *
 * The resulting bundle has one edge per golden record, with field-level
 * provenance keyed by column name. Source row IDs include every member of
 * the cluster the golden record came from.
 */
export function buildLineage(
  result: DedupeResult,
  options?: BuildLineageOptions,
): LineageBundle {
  const defaultStrategy =
    options?.defaultStrategy ??
    result.config.goldenRules?.defaultStrategy ??
    "most_complete";

  const timestamp = isoTimestamp();
  const edges: LineageEdge[] = [];

  // Pre-index dupes by row_id → row for quick field lookup.
  const rowById = new Map<number, Row>();
  for (const r of result.dupes) {
    const id = getRowId(r);
    if (id !== null) rowById.set(id, r);
  }
  for (const r of result.unique) {
    const id = getRowId(r);
    if (id !== null) rowById.set(id, r);
  }

  for (const golden of result.goldenRecords) {
    const clusterId = getClusterId(golden);
    const goldenRowId = getRowId(golden) ?? -1;
    if (clusterId === null) continue;

    const cluster = result.clusters.get(clusterId);
    if (!cluster) continue;

    const fieldProvenance: Record<string, FieldProvenanceEntry> = {};
    const confidence = computeConfidence(cluster, defaultStrategy);

    for (const [key, value] of Object.entries(golden)) {
      if (key.startsWith("__")) continue;
      if (value === null || value === undefined) continue;

      // Locate which member row contributed this value.
      let sourceRowId = goldenRowId;
      for (const memberId of cluster.members) {
        const memberRow = rowById.get(memberId);
        if (memberRow && memberRow[key] === value) {
          sourceRowId = memberId;
          break;
        }
      }

      const fieldStrategy =
        result.config.goldenRules?.fieldRules?.[key]?.strategy ??
        defaultStrategy;

      fieldProvenance[key] = {
        sourceRowId,
        strategy: fieldStrategy,
        confidence,
      };
    }

    edges.push({
      clusterId,
      sourceRowIds: [...cluster.members],
      goldenRowId,
      fieldProvenance,
      timestamp,
    });
  }

  return {
    edges,
    timestamp,
    recordCount: edges.length,
  };
}

// ---------------------------------------------------------------------------
// (De)serialization
// ---------------------------------------------------------------------------

/** Serialize a lineage bundle to stable, human-readable JSON. */
export function lineageToJson(bundle: LineageBundle): string {
  return JSON.stringify(bundle, null, 2);
}

/** Parse a lineage bundle from JSON. Does not validate schema. */
export function lineageFromJson(json: string): LineageBundle {
  const parsed = JSON.parse(json) as unknown;
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    !("edges" in parsed) ||
    !("timestamp" in parsed) ||
    !("recordCount" in parsed)
  ) {
    throw new Error("Invalid lineage bundle: missing required fields");
  }
  return parsed as LineageBundle;
}
