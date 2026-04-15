import { describe, it, expect } from "vitest";
import { autoConfigureRows } from "../../src/core/autoconfig.js";

describe("autoConfigureRows — preflight integration", () => {
  it("attaches _preflightReport to the returned config", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      name: `person ${i}`,
      email: `p${i}@example.com`,
    }));
    const cfg = autoConfigureRows(rows);
    expect(cfg._preflightReport).toBeDefined();
    expect(cfg._preflightReport?.hasErrors).toBe(false);
  });

  it("strict=true stamps _strictAutoconfig on the returned config", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      name: `person ${i}`,
      email: `p${i}@example.com`,
    }));
    const cfg = autoConfigureRows(rows, { strict: true });
    expect(cfg._strictAutoconfig).toBe(true);
  });

  it("strict=false (default) does not stamp _strictAutoconfig", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      name: `person ${i}`,
      email: `p${i}@example.com`,
    }));
    const cfg = autoConfigureRows(rows);
    expect(cfg._strictAutoconfig).toBeUndefined();
  });

  it("empty row set still produces a config with a preflight report", () => {
    const cfg = autoConfigureRows([]);
    expect(cfg._preflightReport).toBeDefined();
  });
});
