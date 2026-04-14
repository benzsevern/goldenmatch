import { describe, it, expect } from "vitest";
import { parseConfig } from "../../src/core/index.js";

describe("parseConfig", () => {
  it("accepts snake_case keys", () => {
    const raw = {
      match_settings: [
        {
          name: "email_mk",
          type: "exact",
          fields: [{ field: "email", transforms: ["lowercase"], scorer: "exact", weight: 1.0 }],
        },
      ],
      threshold: 0.9,
    };
    const config = parseConfig(raw);
    expect(config.matchkeys?.length).toBe(1);
    expect(config.matchkeys?.[0]?.name).toBe("email_mk");
    expect(config.threshold).toBe(0.9);
  });

  it("accepts camelCase keys", () => {
    const raw = {
      matchkeys: [
        {
          name: "mk1",
          type: "weighted",
          fields: [{ field: "name", transforms: [], scorer: "jaro_winkler", weight: 1.0 }],
          threshold: 0.85,
        },
      ],
    };
    const config = parseConfig(raw);
    expect(config.matchkeys?.[0]?.threshold).toBe(0.85);
  });

  it("parses matchkeys fields array", () => {
    const raw = {
      matchkeys: [
        {
          name: "m",
          type: "weighted",
          fields: [
            { field: "first", transforms: ["lowercase"], scorer: "jaro_winkler", weight: 0.5 },
            { field: "last", transforms: [], scorer: "jaro_winkler", weight: 1.0 },
          ],
        },
      ],
    };
    const config = parseConfig(raw);
    expect(config.matchkeys?.[0]?.fields.length).toBe(2);
    expect(config.matchkeys?.[0]?.fields[0]?.weight).toBe(0.5);
  });

  it("parses blocking config", () => {
    const raw = {
      blocking: {
        strategy: "static",
        keys: [{ fields: ["zip"], transforms: ["lowercase"] }],
        max_block_size: 1000,
        skip_oversized: true,
      },
    };
    const config = parseConfig(raw);
    expect(config.blocking?.strategy).toBe("static");
    expect(config.blocking?.maxBlockSize).toBe(1000);
    expect(config.blocking?.skipOversized).toBe(true);
    expect(config.blocking?.keys.length).toBe(1);
  });

  it("normalizes golden_rules.default -> defaultStrategy", () => {
    const raw = {
      golden_rules: {
        default: "most_complete",
      },
    };
    const config = parseConfig(raw);
    expect(config.goldenRules?.defaultStrategy).toBe("most_complete");
  });

  it("accepts goldenRules.defaultStrategy directly", () => {
    const raw = {
      goldenRules: {
        defaultStrategy: "majority_vote",
      },
    };
    const config = parseConfig(raw);
    expect(config.goldenRules?.defaultStrategy).toBe("majority_vote");
  });

  it("throws on invalid config (not an object)", () => {
    expect(() => parseConfig("not-an-object")).toThrow();
    expect(() => parseConfig(null)).toThrow();
  });

  it("throws on invalid nested config (matchkey without name)", () => {
    const raw = {
      matchkeys: [{ type: "exact", fields: [] }],
    };
    expect(() => parseConfig(raw)).toThrow();
  });
});
