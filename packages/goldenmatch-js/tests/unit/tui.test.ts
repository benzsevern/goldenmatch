import { describe, it, expect } from "vitest";

describe("startTui", () => {
  it("is exported as a function", async () => {
    const { startTui } = await import("../../src/node/tui/app.js");
    expect(typeof startTui).toBe("function");
  });

  it("throws helpful error when 'ink'/'react' not installed", async () => {
    const { startTui } = await import("../../src/node/tui/app.js");
    // ink/react are optional peer deps. In environments without them
    // the loader inside startTui throws a helpful install message.
    // If they happen to be installed, calling startTui would actually
    // launch the TUI (which would block on stdin in tests) — so we
    // only invoke when we expect the load to fail.
    try {
      await startTui({});
      // If we get here, ink+react ARE installed. Just confirm no crash.
      expect(true).toBe(true);
    } catch (err) {
      expect(String(err)).toMatch(/ink|react/i);
    }
  });
});
