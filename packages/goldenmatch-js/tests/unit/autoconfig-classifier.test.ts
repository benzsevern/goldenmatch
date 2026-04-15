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
