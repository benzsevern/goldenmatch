import { describe, it, expect } from "vitest";
import { autoConfigureRows } from "../../src/core/autoconfig.js";

describe("cardinality guard", () => {
  it("unique-value column routes to id, not phone/zip/numeric", () => {
    const rows = Array.from({ length: 10 }, (_, i) => ({
      voter_reg_num: String(9000000 + i), // phone-shaped, all unique
    }));
    const cfg = autoConfigureRows(rows);
    for (const mk of cfg.matchkeys ?? []) {
      for (const field of mk.fields ?? []) {
        expect(field.field).not.toBe("voter_reg_num");
      }
    }
  });
});

describe("ID_NAME_PATTERNS extension", () => {
  it("voter_reg_num routes to id", () => {
    const rows = Array.from({ length: 10 }, (_, i) => ({
      voter_reg_num: `REG${i}`,
      name: `person ${i}`,
    }));
    const cfg = autoConfigureRows(rows);
    for (const mk of cfg.matchkeys ?? []) {
      for (const field of mk.fields ?? []) {
        expect(field.field).not.toBe("voter_reg_num");
      }
    }
  });

  it("num_kids does NOT false-positive as id", () => {
    const rows = Array.from({ length: 10 }, (_, i) => ({
      name: `person ${i}`,
      num_kids: String(i % 5),
    }));
    const cfg = autoConfigureRows(rows);
    for (const mk of cfg.matchkeys ?? []) {
      if (mk.type === "exact") {
        for (const field of mk.fields ?? []) {
          expect(field.field).not.toBe("num_kids");
        }
      }
    }
  });

  it("account_no routes to id", () => {
    const rows = Array.from({ length: 10 }, (_, i) => ({
      account_no: `A${i}`,
      name: `person ${i}`,
    }));
    const cfg = autoConfigureRows(rows);
    for (const mk of cfg.matchkeys ?? []) {
      for (const field of mk.fields ?? []) {
        expect(field.field).not.toBe("account_no");
      }
    }
  });
});
