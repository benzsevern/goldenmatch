import { describe, it, expect } from "vitest";
import {
  buildLineage,
  lineageToJson,
  lineageFromJson,
} from "../../src/core/lineage.js";
import { runDedupePipeline } from "../../src/core/pipeline.js";
import {
  makeConfig,
  makeMatchkeyConfig,
  makeMatchkeyField,
} from "../../src/core/types.js";
import type { Row } from "../../src/core/types.js";

function buildTinyDedupeResult() {
  const rows: Row[] = [
    { email: "a@x.com", name: "Alice Brown" },
    { email: "a@x.com", name: "Alice B." },
    { email: "b@x.com", name: "Bob Smith" },
  ];
  const mk = makeMatchkeyConfig({
    name: "email_exact",
    type: "exact",
    fields: [
      makeMatchkeyField({
        field: "email",
        transforms: ["lowercase"],
        scorer: "exact",
      }),
    ],
  });
  const config = makeConfig({ matchkeys: [mk] });
  return runDedupePipeline(rows, config);
}

describe("buildLineage", () => {
  it("produces one edge per cluster in the DedupeResult", () => {
    const result = buildTinyDedupeResult();
    const bundle = buildLineage(result);
    // There is at least one multi-member cluster -> at least one edge.
    expect(bundle.edges.length).toBeGreaterThan(0);
    expect(bundle.recordCount).toBe(bundle.edges.length);
  });

  it("edges carry cluster_id, source_row_ids, golden_row_id, and field provenance", () => {
    const result = buildTinyDedupeResult();
    const bundle = buildLineage(result);
    const edge = bundle.edges[0]!;
    expect(typeof edge.clusterId).toBe("number");
    expect(Array.isArray(edge.sourceRowIds)).toBe(true);
    expect(edge.sourceRowIds.length).toBeGreaterThanOrEqual(2);
    expect(typeof edge.goldenRowId).toBe("number");
    // Field provenance should include non-internal fields like email and name.
    const keys = Object.keys(edge.fieldProvenance);
    expect(keys.length).toBeGreaterThan(0);
    for (const k of keys) {
      const entry = edge.fieldProvenance[k]!;
      expect(typeof entry.sourceRowId).toBe("number");
      expect(typeof entry.strategy).toBe("string");
      expect(typeof entry.confidence).toBe("number");
    }
  });

  it("does not emit provenance entries for internal __-prefixed keys", () => {
    const result = buildTinyDedupeResult();
    const bundle = buildLineage(result);
    for (const edge of bundle.edges) {
      for (const k of Object.keys(edge.fieldProvenance)) {
        expect(k.startsWith("__")).toBe(false);
      }
    }
  });

  it("defaultStrategy override propagates into field provenance", () => {
    const result = buildTinyDedupeResult();
    const bundle = buildLineage(result, { defaultStrategy: "first_non_null" });
    const edge = bundle.edges[0];
    if (edge) {
      const anyEntry = Object.values(edge.fieldProvenance)[0];
      if (anyEntry) expect(anyEntry.strategy).toBe("first_non_null");
    }
  });
});

describe("lineageToJson / lineageFromJson", () => {
  it("round-trips a lineage bundle", () => {
    const result = buildTinyDedupeResult();
    const original = buildLineage(result);
    const json = lineageToJson(original);
    const parsed = lineageFromJson(json);
    expect(parsed.edges.length).toBe(original.edges.length);
    expect(parsed.recordCount).toBe(original.recordCount);
    expect(parsed.timestamp).toBe(original.timestamp);
  });

  it("lineageFromJson throws on malformed input", () => {
    expect(() => lineageFromJson("{}")).toThrow(/Invalid lineage bundle/);
    expect(() => lineageFromJson("null")).toThrow(/Invalid lineage bundle/);
  });
});
