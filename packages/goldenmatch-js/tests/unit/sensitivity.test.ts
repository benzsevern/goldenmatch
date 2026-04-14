import { describe, it, expect } from "vitest";
import { runSensitivity, stabilityReport } from "../../src/core/sensitivity.js";
import {
  makeConfig,
  makeBlockingConfig,
  makeMatchkeyConfig,
  makeMatchkeyField,
} from "../../src/core/types.js";
import type { Row } from "../../src/core/types.js";

function makeRows(): Row[] {
  // 20 rows, five near-duplicate pairs (so threshold sweeps actually change clustering).
  const rows: Row[] = [];
  let id = 0;
  const dupes: Array<[string, string]> = [
    ["John Smith", "Jon Smith"],
    ["Mary Jones", "Marie Jones"],
    ["Alice Brown", "Alicia Brown"],
    ["Bob Miller", "Robert Miller"],
    ["Carol Davis", "Caroline Davis"],
  ];
  for (const [a, b] of dupes) {
    rows.push({ __row_id__: id++, name: a });
    rows.push({ __row_id__: id++, name: b });
  }
  // 10 distinct distractors.
  for (const nm of [
    "Zygmunt Petrov", "Xiong Wei", "Nakamura Taro", "Kowalski Jan",
    "Oduya Esther", "Rasmussen Ole", "Tanaka Yui", "Vasquez Maria",
    "Wojcik Piotr", "Yamamoto Ken",
  ]) {
    rows.push({ __row_id__: id++, name: nm });
  }
  return rows;
}

function baselineConfig() {
  const mk = makeMatchkeyConfig({
    name: "name_fuzzy",
    type: "weighted",
    threshold: 0.85,
    fields: [
      makeMatchkeyField({
        field: "name",
        scorer: "jaro_winkler",
        transforms: ["lowercase"],
      }),
    ],
  });
  const blocking = makeBlockingConfig({
    strategy: "static",
    keys: [{ fields: ["name"], transforms: ["lowercase", "substring:0:1"] }],
  });
  return makeConfig({ matchkeys: [mk], blocking, threshold: 0.85 });
}

describe("runSensitivity", () => {
  it("produces one SweepPoint per value with stats and TWI vs baseline", () => {
    const rows = makeRows();
    const cfg = baselineConfig();
    const result = runSensitivity(rows, cfg, [
      { path: "threshold", values: [0.75, 0.85, 0.95] },
    ]);
    expect(result.points.length).toBe(3);
    for (const p of result.points) {
      // Either error path or stats path; in a clean run we expect stats.
      if (p.error === undefined) {
        expect(typeof p.stats["totalClusters"]).toBe("number");
        expect(typeof p.stats["totalRecords"]).toBe("number");
        // TWI is an optional number in [0, 1]-ish when present.
        if (p.twi !== undefined) {
          expect(p.twi).toBeGreaterThanOrEqual(0);
          expect(p.twi).toBeLessThanOrEqual(1);
        }
      }
    }
    // Baseline has TWI = 1 by construction.
    expect(result.baseline.twi).toBe(1.0);
  });

  it("writes the sweep-param value into each point's params object (dot-path at root)", () => {
    const rows = makeRows();
    const cfg = baselineConfig();
    const result = runSensitivity(rows, cfg, [
      { path: "threshold", values: [0.7, 0.9] },
    ]);
    const values = result.points.map((p) => p.params["threshold"]);
    expect(values).toEqual([0.7, 0.9]);
  });

  it("preserves partial results when one point errors", () => {
    const rows = makeRows();
    const cfg = baselineConfig();
    // Inject a non-existent path that will parse fine but using an invalid
    // matchkey object forces the pipeline to throw.
    const result = runSensitivity(rows, cfg, [
      {
        path: "matchkeys",
        values: [
          // Valid: single matchkey -> runs fine
          [
            makeMatchkeyConfig({
              name: "m",
              type: "weighted",
              fields: [makeMatchkeyField({ field: "name", scorer: "jaro_winkler" })],
              threshold: 0.85,
            }),
          ],
          // Invalid: matchkey referring to a non-existent scorer triggers an
          // exception during scoring for some inputs. Use a malformed shape
          // that throws early: field array contains non-objects.
          [
            {
              name: "bad",
              type: "weighted",
              fields: [null],
              threshold: 0.85,
            } as unknown,
          ],
        ],
      },
    ]);
    expect(result.points.length).toBe(2);
    // At least one good point, at least one error point.
    const errs = result.points.filter((p) => p.error !== undefined);
    const ok = result.points.filter((p) => p.error === undefined);
    expect(ok.length).toBeGreaterThanOrEqual(1);
    expect(errs.length).toBeGreaterThanOrEqual(1);
  });

  it("empty sweep params yields zero points but still a baseline", () => {
    const rows = makeRows();
    const cfg = baselineConfig();
    const result = runSensitivity(rows, cfg, []);
    expect(result.points.length).toBe(0);
    expect(result.baseline.twi).toBe(1.0);
    expect(result.stable).toBe(true);
  });
});

describe("stabilityReport", () => {
  it("returns a string summarizing the sweep", () => {
    const rows = makeRows();
    const cfg = baselineConfig();
    const result = runSensitivity(rows, cfg, [
      { path: "threshold", values: [0.85, 0.95] },
    ]);
    const report = stabilityReport(result);
    expect(typeof report).toBe("string");
    expect(report).toContain("Sensitivity sweep");
    expect(report).toContain("Points");
    expect(report).toContain("Stable");
  });
});
