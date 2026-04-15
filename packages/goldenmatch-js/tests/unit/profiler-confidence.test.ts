import { describe, it, expect } from "vitest";
import { profileRows } from "../../src/core/profiler.js";

describe("ColumnProfile.confidence", () => {
  it("high-confidence email column gets >= 0.7", () => {
    const rows = [
      { email: "a@b.com" },
      { email: "c@d.com" },
      { email: "e@f.com" },
    ];
    const profile = profileRows(rows);
    const email = profile.byName["email"]!;
    expect(email.confidence).toBeGreaterThanOrEqual(0.7);
    expect(email.confidence).toBeLessThanOrEqual(1.0);
  });

  it("mystery string column gets low confidence (<= 0.5)", () => {
    const rows = [
      { mystery: "xyz1abc" },
      { mystery: "xyz2abc" },
      { mystery: "xyz3abc" },
    ];
    const profile = profileRows(rows);
    const mystery = profile.byName["mystery"]!;
    expect(mystery.confidence).toBeLessThanOrEqual(0.5);
    expect(mystery.confidence).toBeGreaterThan(0);
  });
});
