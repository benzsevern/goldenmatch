import { describe, it, expect } from "vitest";
import { runDedupePipeline, runMatchPipeline, makeConfig, makeBlockingConfig } from "../../src/core/index.js";
import type { MatchkeyConfig, Row } from "../../src/core/index.js";

describe("runDedupePipeline", () => {
  it("with exact matchkey catches identical emails", () => {
    const rows: Row[] = [
      { id: 1, email: "a@x.com", name: "Alice" },
      { id: 2, email: "a@x.com", name: "A." },
      { id: 3, email: "b@x.com", name: "Bob" },
    ];
    const mk: MatchkeyConfig = {
      name: "email_exact",
      type: "exact",
      fields: [{ field: "email", transforms: ["lowercase"], scorer: "exact", weight: 1.0 }],
    };
    const config = makeConfig({ matchkeys: [mk] });
    const result = runDedupePipeline(rows, config);
    expect(result.stats.totalRecords).toBe(3);
    expect(result.scoredPairs.length).toBeGreaterThanOrEqual(1);
    expect(result.dupes.length).toBeGreaterThanOrEqual(2);
  });

  it("with weighted matchkey + blocking", () => {
    const rows: Row[] = [
      { id: 1, name: "John Smith", zip: "111" },
      { id: 2, name: "Jon Smith", zip: "111" },
      { id: 3, name: "Zeke Xavier", zip: "222" },
    ];
    const mk: MatchkeyConfig = {
      name: "name_fuzzy",
      type: "weighted",
      threshold: 0.7,
      fields: [{ field: "name", transforms: ["lowercase"], scorer: "jaro_winkler", weight: 1.0 }],
    };
    const blocking = makeBlockingConfig({
      strategy: "static",
      keys: [{ fields: ["zip"], transforms: [] }],
    });
    const config = makeConfig({ matchkeys: [mk], blocking });
    const result = runDedupePipeline(rows, config);
    expect(result.stats.totalRecords).toBe(3);
    // John/Jon should match, Zeke should not
    const hasMatch = result.scoredPairs.some((p) =>
      (p.idA === 0 && p.idB === 1) || (p.idA === 1 && p.idB === 0),
    );
    expect(hasMatch).toBe(true);
  });

  it("empty input returns empty result", () => {
    const result = runDedupePipeline([], makeConfig());
    expect(result.stats.totalRecords).toBe(0);
    expect(result.stats.totalClusters).toBe(0);
  });

  it("stats are computed correctly", () => {
    const rows: Row[] = [
      { id: 1, email: "a@x.com" },
      { id: 2, email: "a@x.com" },
      { id: 3, email: "b@x.com" },
    ];
    const mk: MatchkeyConfig = {
      name: "email",
      type: "exact",
      fields: [{ field: "email", transforms: [], scorer: "exact", weight: 1.0 }],
    };
    const config = makeConfig({ matchkeys: [mk] });
    const result = runDedupePipeline(rows, config);
    // totalRecords == matchedRecords + uniqueRecords
    expect(result.stats.matchedRecords + result.stats.uniqueRecords).toBe(
      result.stats.totalRecords,
    );
    // matchRate = matchedRecords / totalRecords
    expect(result.stats.matchRate).toBeCloseTo(
      result.stats.matchedRecords / result.stats.totalRecords,
      5,
    );
  });
});

describe("runMatchPipeline", () => {
  it("finds cross-dataset matches", () => {
    const target: Row[] = [{ id: 1, email: "a@x.com" }];
    const reference: Row[] = [
      { id: 10, email: "a@x.com" },
      { id: 11, email: "b@x.com" },
    ];
    const mk: MatchkeyConfig = {
      name: "email_exact",
      type: "exact",
      fields: [{ field: "email", transforms: ["lowercase"], scorer: "exact", weight: 1.0 }],
    };
    const config = makeConfig({ matchkeys: [mk] });
    const result = runMatchPipeline(target, reference, config);
    expect(result.matched.length).toBe(1);
    expect(result.unmatched.length).toBe(0);
  });

  it("empty target yields no matches", () => {
    const result = runMatchPipeline([], [{ id: 1, email: "a@x.com" }], makeConfig());
    expect(result.matched).toEqual([]);
    expect(result.unmatched).toEqual([]);
  });

  it("records with no reference match go to unmatched", () => {
    const target: Row[] = [{ id: 1, email: "no-match@x.com" }];
    const reference: Row[] = [{ id: 10, email: "a@x.com" }];
    const mk: MatchkeyConfig = {
      name: "email_exact",
      type: "exact",
      fields: [{ field: "email", transforms: [], scorer: "exact", weight: 1.0 }],
    };
    const config = makeConfig({ matchkeys: [mk] });
    const result = runMatchPipeline(target, reference, config);
    expect(result.matched.length).toBe(0);
    expect(result.unmatched.length).toBe(1);
  });
});
