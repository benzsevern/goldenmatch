import { describe, it, expect } from "vitest";
import { autoConfigureRows } from "../../src/core/autoconfig.js";

describe("multi_name classification", () => {
  it("multi-author field -> token_sort weight 1.0", () => {
    const rows = [
      { authors: "Alice Smith, Bob Jones, Carol White, Dave Brown", title: "p1" },
      { authors: "Eve Green; Frank Blue; Grace Hopper; Alan Turing", title: "p2" },
      { authors: "Hector Gomez, Iris Liu, Jasper Nguyen, Kara Patel", title: "p3" },
    ].flatMap((r) => Array.from({ length: 10 }, () => r));
    const cfg = autoConfigureRows(rows);
    const authorsField = (cfg.matchkeys ?? [])
      .flatMap((mk) => mk.fields ?? [])
      .find((f) => f.field === "authors");
    expect(authorsField).toBeDefined();
    expect(authorsField!.scorer).toBe("token_sort");
    expect(authorsField!.weight).toBe(1.0);
  });

  it("short comma tags are NOT multi_name (avgLen threshold)", () => {
    const rows = Array.from({ length: 30 }, (_, i) => ({
      tags: "red, blue",
      id: String(i),
    }));
    const cfg = autoConfigureRows(rows);
    const tagsField = (cfg.matchkeys ?? [])
      .flatMap((mk) => mk.fields ?? [])
      .find(
        (f) => f.field === "tags" && f.scorer === "token_sort" && f.weight === 1.0,
      );
    expect(tagsField).toBeUndefined();
  });
});
