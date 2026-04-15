import { describe, it, expect } from "vitest";
import { autoConfigureRows } from "../../src/core/autoconfig.js";

describe("confidence-gated weight cap", () => {
  it("low-confidence field caps at 0.3", () => {
    const rows = Array.from({ length: 30 }, (_, i) => ({
      mystery: `xyz${i}abc`,
      name: `person ${i}`,
    }));
    const cfg = autoConfigureRows(rows);
    const mysteryField = (cfg.matchkeys ?? [])
      .flatMap((mk) => mk.fields ?? [])
      .find((f) => f.field === "mystery");
    if (mysteryField !== undefined) {
      expect(mysteryField.weight ?? 0).toBeLessThanOrEqual(0.3);
    }
  });
});
