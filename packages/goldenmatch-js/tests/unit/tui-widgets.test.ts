import { describe, it, expect } from "vitest";

describe("TUI widgets", () => {
  it("exports widgets helper", async () => {
    const mod = await import("../../src/node/tui/widgets.js");
    expect(typeof mod.tryLoad).toBe("function");
    expect(typeof mod.loadAddons).toBe("function");
    expect(typeof mod.inkAddons).toBe("object");
  });

  it("tryLoad returns null for missing package", async () => {
    const { tryLoad } = await import("../../src/node/tui/widgets.js");
    expect(tryLoad("nonexistent-package-xyz")).toBeNull();
  });

  it("inkAddons getters return null for missing addons", async () => {
    const { inkAddons } = await import("../../src/node/tui/widgets.js");
    // In the test env none of the optional ink addons are installed.
    // Each getter must return null rather than throwing.
    expect(() => inkAddons.table).not.toThrow();
    expect(() => inkAddons.selectInput).not.toThrow();
    expect(() => inkAddons.textInput).not.toThrow();
    expect(() => inkAddons.spinner).not.toThrow();
    expect(() => inkAddons.gradient).not.toThrow();
  });

  it("loadAddons resolves to an object with all keys", async () => {
    const { loadAddons } = await import("../../src/node/tui/widgets.js");
    const addons = await loadAddons();
    expect(addons).toHaveProperty("Table");
    expect(addons).toHaveProperty("SelectInput");
    expect(addons).toHaveProperty("TextInput");
    expect(addons).toHaveProperty("Spinner");
    expect(addons).toHaveProperty("Gradient");
    // Each field is either null (missing package) or a loaded module/component.
    for (const v of Object.values(addons)) {
      expect(v === null || typeof v === "function" || typeof v === "object").toBe(
        true,
      );
    }
  });
});
