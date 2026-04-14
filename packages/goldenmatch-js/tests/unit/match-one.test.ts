import { describe, it, expect } from "vitest";
import { matchOne, findExactMatchesOne } from "../../src/core/match-one.js";
import { makeMatchkeyConfig, makeMatchkeyField } from "../../src/core/types.js";
import type { Row } from "../../src/core/types.js";

function r(rowId: number, name: string, email?: string): Row {
  return { __row_id__: rowId, name, ...(email !== undefined ? { email } : {}) };
}

describe("matchOne", () => {
  it("returns matches sorted by descending score, above threshold", () => {
    const record: Row = { name: "John Smith" };
    const rows: Row[] = [
      r(0, "John Smith"),
      r(1, "Jon Smith"),
      r(2, "Zxqwer Zxqwer"),
    ];
    const mk = makeMatchkeyConfig({
      name: "name",
      type: "weighted",
      threshold: 0.7,
      fields: [
        makeMatchkeyField({
          field: "name",
          scorer: "jaro_winkler",
          transforms: ["lowercase"],
        }),
      ],
    });
    const hits = matchOne(record, rows, mk);
    expect(hits.length).toBeGreaterThanOrEqual(2);
    // Sorted: first hit is the exact match on row 0 (score 1.0).
    expect(hits[0]!.rowId).toBe(0);
    expect(hits[0]!.score).toBeCloseTo(1.0, 5);
    // All hits respect threshold.
    for (const h of hits) expect(h.score).toBeGreaterThanOrEqual(0.7);
    // Very different row not included.
    expect(hits.some((h) => h.rowId === 2)).toBe(false);
  });

  it("returns empty array on empty dataset", () => {
    const mk = makeMatchkeyConfig({
      name: "n",
      type: "weighted",
      fields: [makeMatchkeyField({ field: "name", scorer: "jaro_winkler" })],
    });
    const hits = matchOne({ name: "Alice" }, [], mk);
    expect(hits).toEqual([]);
  });

  it("returns empty when nothing is above threshold", () => {
    const record: Row = { name: "Alice Brown" };
    const rows: Row[] = [r(0, "Zoltan Xiong"), r(1, "Yuri Nakamura")];
    const mk = makeMatchkeyConfig({
      name: "n",
      type: "weighted",
      threshold: 0.95,
      fields: [makeMatchkeyField({ field: "name", scorer: "jaro_winkler" })],
    });
    const hits = matchOne(record, rows, mk);
    expect(hits).toEqual([]);
  });

  it("threshold defaults to 0 when unset (returns all rows)", () => {
    const record: Row = { name: "Alice" };
    const rows: Row[] = [r(0, "Alice"), r(1, "Zoltan")];
    // Construct without a threshold so matchOne's default-of-0 kicks in.
    const mkNoThreshold = {
      name: "n",
      type: "weighted",
      fields: [makeMatchkeyField({ field: "name", scorer: "jaro_winkler" })],
    } as unknown as Parameters<typeof matchOne>[2];
    const hits = matchOne(record, rows, mkNoThreshold);
    expect(hits.length).toBe(2);
  });
});

describe("findExactMatchesOne", () => {
  it("finds exact composite-key matches only, with score 1.0", () => {
    const record: Row = { email: "alice@example.com" };
    const rows: Row[] = [
      { __row_id__: 0, email: "alice@example.com" },
      { __row_id__: 1, email: "Alice@Example.com" }, // matches after lowercase
      { __row_id__: 2, email: "bob@example.com" },
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
    const hits = findExactMatchesOne(record, rows, mk);
    const ids = hits.map((h) => h.rowId).sort();
    expect(ids).toEqual([0, 1]);
    for (const h of hits) expect(h.score).toBe(1.0);
  });

  it("returns empty array when probe has null transform for any field", () => {
    const record: Row = { email: null };
    const rows: Row[] = [{ __row_id__: 0, email: "alice@example.com" }];
    const mk = makeMatchkeyConfig({
      name: "email_exact",
      type: "exact",
      fields: [makeMatchkeyField({ field: "email", transforms: [], scorer: "exact" })],
    });
    const hits = findExactMatchesOne(record, rows, mk);
    expect(hits).toEqual([]);
  });

  it("skips rows where any field transforms to null", () => {
    const record: Row = { email: "alice@example.com" };
    const rows: Row[] = [
      { __row_id__: 0, email: null },
      { __row_id__: 1, email: "alice@example.com" },
    ];
    const mk = makeMatchkeyConfig({
      name: "email_exact",
      type: "exact",
      fields: [makeMatchkeyField({ field: "email", scorer: "exact" })],
    });
    const hits = findExactMatchesOne(record, rows, mk);
    expect(hits.map((h) => h.rowId)).toEqual([1]);
  });
});
