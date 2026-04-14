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
  it("picks exact matchkeys on email/phone and produces a weighted matchkey", () => {
    const rows = makePeople(40);
    const cfg = autoConfigureRows(rows);
    const names = (cfg.matchkeys ?? []).map((m) => m.name);
    // Exact matchkeys for identifier columns.
    expect(names).toContain("exact_email");
    // Phone column has phone-shaped values and is near-unique -> exact allowed.
    expect(names).toContain("exact_phone");
    // There should be a weighted matchkey for fuzzy fields.
    expect(names).toContain("weighted_identity");
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

  it("email column detected as exact identifier candidate when cardinality is high", () => {
    const rows: Row[] = [];
    for (let i = 0; i < 20; i++) {
      rows.push({ __row_id__: i, email: `user${i}@x.com` });
    }
    const cfg = autoConfigureRows(rows);
    const names = (cfg.matchkeys ?? []).map((m) => m.name);
    expect(names).toContain("exact_email");
  });
});
