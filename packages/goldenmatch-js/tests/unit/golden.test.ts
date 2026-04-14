import { describe, it, expect } from "vitest";
import { mergeField, buildGoldenRecord, makeGoldenRulesConfig } from "../../src/core/index.js";
import type { GoldenFieldRule, Row } from "../../src/core/index.js";

describe("mergeField strategies", () => {
  it("most_complete picks longest string", () => {
    const rule: GoldenFieldRule = { strategy: "most_complete" };
    const result = mergeField(["Jon", "John Smith", "Joh"], rule);
    expect(result.value).toBe("John Smith");
  });

  it("majority_vote picks most frequent", () => {
    const rule: GoldenFieldRule = { strategy: "majority_vote" };
    const result = mergeField(["A", "B", "A", "A"], rule);
    expect(result.value).toBe("A");
  });

  it("source_priority picks according to priority list", () => {
    const rule: GoldenFieldRule = {
      strategy: "source_priority",
      sourcePriority: ["crm", "erp"],
    };
    const result = mergeField(["valA", "valB"], rule, { sources: ["erp", "crm"] });
    // crm has higher priority, so "valB" wins
    expect(result.value).toBe("valB");
  });

  it("most_recent picks latest by date", () => {
    const rule: GoldenFieldRule = { strategy: "most_recent" };
    const result = mergeField(["old", "new"], rule, {
      dates: ["2020-01-01", "2024-01-01"],
    });
    expect(result.value).toBe("new");
  });

  it("first_non_null picks first non-null", () => {
    const rule: GoldenFieldRule = { strategy: "first_non_null" };
    const result = mergeField([null, "first", "second"], rule);
    expect(result.value).toBe("first");
  });

  it("all-identical values -> confidence 1.0", () => {
    const rule: GoldenFieldRule = { strategy: "most_complete" };
    const result = mergeField(["same", "same", "same"], rule);
    expect(result.confidence).toBe(1.0);
    expect(result.value).toBe("same");
  });

  it("all-null -> value null, confidence 0, sourceIndex null", () => {
    const rule: GoldenFieldRule = { strategy: "most_complete" };
    const result = mergeField([null, null, null], rule);
    expect(result.value).toBe(null);
    expect(result.confidence).toBe(0);
    expect(result.sourceIndex).toBe(null);
  });
});

describe("buildGoldenRecord", () => {
  it("produces a merged record for the cluster", () => {
    const clusterRows: Row[] = [
      { __row_id__: 0, name: "Jon", email: "a@x.com" },
      { __row_id__: 1, name: "John", email: "a@x.com" },
    ];
    const rules = makeGoldenRulesConfig({ defaultStrategy: "most_complete" });
    const golden = buildGoldenRecord(clusterRows, rules);

    // email is identical -> confidence 1.0
    expect(golden.fields.email?.value).toBe("a@x.com");
    expect(golden.fields.email?.confidence).toBe(1.0);

    // name: most_complete picks longest -> "John"
    expect(golden.fields.name?.value).toBe("John");

    // goldenConfidence averages field confidences
    expect(golden.goldenConfidence).toBeGreaterThan(0);
    expect(golden.goldenConfidence).toBeLessThanOrEqual(1);
  });

  it("empty cluster returns empty record", () => {
    const rules = makeGoldenRulesConfig();
    const golden = buildGoldenRecord([], rules);
    expect(Object.keys(golden.fields).length).toBe(0);
    expect(golden.goldenConfidence).toBe(0);
  });

  it("ignores internal columns", () => {
    const clusterRows: Row[] = [
      { __row_id__: 0, __cluster_id__: 1, name: "A" },
    ];
    const rules = makeGoldenRulesConfig();
    const golden = buildGoldenRecord(clusterRows, rules);
    // __row_id__ should NOT appear in output fields
    expect(golden.fields.__row_id__).toBeUndefined();
    expect(golden.fields.__cluster_id__).toBeUndefined();
    expect(golden.fields.name).toBeDefined();
  });
});
