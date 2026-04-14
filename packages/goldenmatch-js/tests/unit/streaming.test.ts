import { describe, it, expect } from "vitest";
import { StreamProcessor } from "../../src/core/streaming.js";
import { makeMatchkeyConfig, makeMatchkeyField } from "../../src/core/types.js";
import type { Row } from "../../src/core/types.js";

function nameMk() {
  return makeMatchkeyConfig({
    name: "name_fuzzy",
    type: "weighted",
    fields: [
      makeMatchkeyField({
        field: "name",
        scorer: "jaro_winkler",
        transforms: ["lowercase"],
      }),
    ],
  });
}

describe("StreamProcessor", () => {
  it("first add with no existing cluster state creates a singleton", () => {
    const sp = new StreamProcessor({ matchkey: nameMk(), threshold: 0.85 });
    const result = sp.add({ __row_id__: 0, name: "Alice" });
    expect(result.rowId).toBe(0);
    expect(result.matchedIds).toEqual([]);
    expect(result.clusterId).toBeGreaterThanOrEqual(0);
    expect(sp.size).toBe(1);
  });

  it("matching record joins the existing cluster", () => {
    const sp = new StreamProcessor({ matchkey: nameMk(), threshold: 0.85 });
    const a = sp.add({ __row_id__: 0, name: "John Smith" });
    const b = sp.add({ __row_id__: 1, name: "John Smith" });
    expect(b.matchedIds).toContain(0);
    expect(b.clusterId).toBe(a.clusterId);
    expect(sp.size).toBe(2);
  });

  it("non-matching records get their own singleton clusters", () => {
    const sp = new StreamProcessor({ matchkey: nameMk(), threshold: 0.95 });
    const a = sp.add({ __row_id__: 0, name: "Alice Brown" });
    const b = sp.add({ __row_id__: 1, name: "Zoltan Xiong" });
    expect(b.matchedIds).toEqual([]);
    expect(b.clusterId).not.toBe(a.clusterId);
  });

  it("size increments with each add", () => {
    const sp = new StreamProcessor({ matchkey: nameMk(), threshold: 0.85 });
    expect(sp.size).toBe(0);
    sp.add({ __row_id__: 0, name: "A" });
    expect(sp.size).toBe(1);
    sp.add({ __row_id__: 1, name: "B" });
    expect(sp.size).toBe(2);
    sp.add({ __row_id__: 2, name: "C" });
    expect(sp.size).toBe(3);
  });

  it("snapshot returns current clusters and rows", () => {
    const sp = new StreamProcessor({ matchkey: nameMk(), threshold: 0.85 });
    sp.add({ __row_id__: 0, name: "Alice" });
    sp.add({ __row_id__: 1, name: "Bob" });
    const snap = sp.snapshot();
    expect(snap.rows.length).toBe(2);
    expect(snap.clusters.size).toBeGreaterThan(0);
    // Cluster members collectively cover all added row ids.
    const allMembers = new Set<number>();
    for (const info of snap.clusters.values()) {
      for (const m of info.members) allMembers.add(m);
    }
    expect(allMembers.has(0)).toBe(true);
    expect(allMembers.has(1)).toBe(true);
  });

  it("assigns __row_id__ automatically when absent on input row", () => {
    const sp = new StreamProcessor({ matchkey: nameMk(), threshold: 0.85 });
    const a = sp.add({ name: "X" });
    const b = sp.add({ name: "Y" });
    expect(a.rowId).toBe(0);
    expect(b.rowId).toBe(1);
    expect(sp.size).toBe(2);
  });
});
