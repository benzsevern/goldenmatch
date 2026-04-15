import { describe, it, expect } from "vitest";
import { autoConfigureRows } from "../../src/core/autoconfig.js";

describe("year classification", () => {
  it("birth_year detected by name -> routed to blocking, not scoring", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      birth_year: String(1980 + (i % 40)),
      name: `person ${i}`,
    }));
    const cfg = autoConfigureRows(rows);
    const weightedFields = (cfg.matchkeys ?? [])
      .filter((mk) => mk.type === "weighted")
      .flatMap((mk) => mk.fields ?? []);
    expect(weightedFields.some((f) => f.field === "birth_year")).toBe(false);
  });

  it("4-digit years in [1900, 2100] detected by data", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      year_col: String(1999 + (i % 20)),
      name: `person ${i}`,
    }));
    const cfg = autoConfigureRows(rows);
    const weightedFields = (cfg.matchkeys ?? [])
      .filter((mk) => mk.type === "weighted")
      .flatMap((mk) => mk.fields ?? []);
    expect(weightedFields.some((f) => f.field === "year_col")).toBe(false);
  });

  it("float-promoted '1999.0' values classify as year", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      year_col: `${1999 + (i % 20)}.0`,
      name: `person ${i}`,
    }));
    const cfg = autoConfigureRows(rows);
    const weightedFields = (cfg.matchkeys ?? [])
      .filter((mk) => mk.type === "weighted")
      .flatMap((mk) => mk.fields ?? []);
    expect(weightedFields.some((f) => f.field === "year_col")).toBe(false);
  });

  it("year column gets classified as year (not id/numeric/date)", async () => {
    const { profileRows } = await import("../../src/core/profiler.js");
    const rows = Array.from({ length: 20 }, (_, i) => ({
      birth_year: String(1980 + (i % 40)),
    }));
    const profile = profileRows(rows);
    expect(profile.byName["birth_year"]!.inferredType).toBe("year");
  });
});
