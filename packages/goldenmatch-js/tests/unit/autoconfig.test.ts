import { describe, it, expect } from "vitest";
import { autoConfigureRows } from "../../src/core/autoconfig.js";
import type { Row } from "../../src/core/types.js";

// Small synthetic person dataset.
function makePeople(n: number): Row[] {
  const first = ["John", "Jane", "Bob", "Alice", "Carol", "David", "Eve", "Frank"];
  const last = ["Smith", "Jones", "Brown", "Miller", "Davis", "Wilson", "Moore", "Taylor"];
  const cities = ["Boston", "Seattle", "Austin", "Denver"];
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    rows.push({
      __row_id__: i,
      first_name: first[i % first.length]!,
      last_name: last[i % last.length]!,
      email: `user${i}@example.com`,
      phone: `555-010-${String(1000 + i).padStart(4, "0")}`,
      zip: String(10000 + (i % 50)),
      city: cities[i % cities.length]!,
    });
  }
  return rows;
}

describe("autoConfigureRows", () => {
  it("produces a weighted matchkey; preflight drops near-unique exact matchkeys", () => {
    const rows = makePeople(40);
    const cfg = autoConfigureRows(rows);
    const names = (cfg.matchkeys ?? []).map((m) => m.name);
    // email/phone are 100% unique in this fixture -> preflight Check 2
    // drops them as "near-unique, never agree".
    expect(names).not.toContain("exact_email");
    expect(names).not.toContain("exact_phone");
    // Weighted matchkey survives.
    expect(names).toContain("weighted_identity");
    // The drop is surfaced as a repaired warning in the report.
    const report = cfg._preflightReport;
    expect(report).toBeDefined();
    expect(
      report!.findings.some(
        (f) => f.check === "cardinality_high" && f.repaired,
      ),
    ).toBe(true);
  });

  it("zip and geo columns do NOT back exact matchkeys (blocking signal only)", () => {
    const rows = makePeople(40);
    const cfg = autoConfigureRows(rows);
    const names = (cfg.matchkeys ?? []).map((m) => m.name);
    expect(names).not.toContain("exact_zip");
    expect(names).not.toContain("exact_city");
  });

  it("exact matchkey skipped for columns with cardinality_ratio < 0.01", () => {
    // 200 rows, one constant id-like column with only 1 distinct value.
    const rows: Row[] = [];
    for (let i = 0; i < 200; i++) {
      rows.push({
        __row_id__: i,
        email: `user${i}@example.com`,
        account_id: "ACME-123", // constant -> cardinality ratio 1/200 = 0.005
      });
    }
    const cfg = autoConfigureRows(rows);
    const names = (cfg.matchkeys ?? []).map((m) => m.name);
    expect(names).not.toContain("exact_account_id");
  });

  it("skips blocking on columns with >20% null rate", () => {
    // zip is mostly null, city has values -> blocking should prefer city
    const rows: Row[] = [];
    for (let i = 0; i < 30; i++) {
      rows.push({
        __row_id__: i,
        email: `user${i}@example.com`,
        first_name: `First${i % 5}`,
        zip: i % 5 === 0 ? String(10000 + i) : null, // 80% null
        city: i % 3 === 0 ? "Boston" : i % 3 === 1 ? "Austin" : "Denver",
      });
    }
    const cfg = autoConfigureRows(rows);
    const keyFields = (cfg.blocking?.keys ?? []).map((k) => k.fields[0]);
    // zip must not be chosen as a blocking key.
    expect(keyFields).not.toContain("zip");
  });

  it("skips blocking on columns with cardinality_ratio >= 0.95 (near-unique)", () => {
    // email is near-unique; don't block on it.
    const rows: Row[] = [];
    for (let i = 0; i < 30; i++) {
      rows.push({
        __row_id__: i,
        email: `user${i}@example.com`, // fully unique
        last_name: `Smith${i % 3}`, // low-cardinality name-ish column
      });
    }
    const cfg = autoConfigureRows(rows);
    const keyFields = (cfg.blocking?.keys ?? []).map((k) => k.fields[0]);
    expect(keyFields).not.toContain("email");
  });

  it("email column: 100%-unique fixtures have their exact matchkey dropped by preflight", () => {
    // With every email unique, preflight's cardinality_high check repairs
    // the config by dropping exact_email. Autoconfig still builds it — the
    // preflight report records the repair.
    const rows: Row[] = [];
    for (let i = 0; i < 20; i++) {
      rows.push({ __row_id__: i, email: `user${i}@x.com` });
    }
    const cfg = autoConfigureRows(rows);
    const names = (cfg.matchkeys ?? []).map((m) => m.name);
    expect(names).not.toContain("exact_email");
    const report = cfg._preflightReport;
    expect(report).toBeDefined();
    expect(
      report!.findings.some(
        (f) =>
          f.check === "cardinality_high" &&
          f.repaired &&
          f.subject === "exact_email",
      ),
    ).toBe(true);
  });
});
