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
    const mk0 = config.matchkeys?.[0];
    expect(mk0?.type).toBe("weighted");
    if (mk0?.type === "weighted") {
      expect(mk0.threshold).toBe(0.85);
    }
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

  // -------------------------------------------------------------------------
  // String-union validation
  // -------------------------------------------------------------------------

  describe("string-union validation", () => {
    it("throws on invalid matchkey type with clear message listing valid options", () => {
      const raw = {
        matchkeys: [
          {
            name: "bad",
            type: "garbage",
            fields: [{ field: "x", transforms: [], scorer: "exact", weight: 1 }],
          },
        ],
      };
      try {
        parseConfig(raw);
        throw new Error("should have thrown");
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        expect(msg).toContain("garbage");
        expect(msg).toContain("exact");
        expect(msg).toContain("weighted");
        expect(msg).toContain("probabilistic");
      }
    });

    it("throws on invalid transform with valid options listed", () => {
      const raw = {
        matchkeys: [
          {
            name: "mk",
            type: "weighted",
            fields: [
              {
                field: "name",
                transforms: ["not_a_real_transform"],
                scorer: "jaro_winkler",
                weight: 1,
              },
            ],
          },
        ],
      };
      try {
        parseConfig(raw);
        throw new Error("should have thrown");
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        expect(msg).toContain("not_a_real_transform");
        expect(msg).toContain("lowercase");
        expect(msg).toContain("substring");
        expect(msg).toContain("qgram");
      }
    });

    it("throws on invalid blocking strategy", () => {
      const raw = {
        blocking: {
          strategy: "nonsense",
          keys: [{ fields: ["zip"], transforms: [] }],
        },
      };
      expect(() => parseConfig(raw)).toThrow(/nonsense/);
    });

    it("throws on invalid golden_rules field strategy", () => {
      const raw = {
        golden_rules: {
          default: "most_complete",
          field_rules: {
            email: { strategy: "pick_worst" },
          },
        },
      };
      expect(() => parseConfig(raw)).toThrow(/pick_worst/);
    });

    it("throws on invalid standardizer", () => {
      const raw = {
        standardization: {
          rules: {
            name: ["scramble"],
          },
        },
      };
      expect(() => parseConfig(raw)).toThrow(/scramble/);
    });

    it("throws on invalid memory backend", () => {
      const raw = {
        memory: {
          enabled: true,
          backend: "redis",
        },
      };
      expect(() => parseConfig(raw)).toThrow(/redis/);
    });

    it("accepts parametric transforms (substring, qgram, bloom_filter)", () => {
      const raw = {
        matchkeys: [
          {
            name: "mk",
            type: "weighted",
            fields: [
              {
                field: "a",
                transforms: ["substring:0:3", "qgram:3", "bloom_filter"],
                scorer: "jaro_winkler",
                weight: 1,
              },
              {
                field: "b",
                transforms: ["bloom_filter:high"],
                scorer: "dice",
                weight: 1,
              },
            ],
          },
        ],
      };
      const config = parseConfig(raw);
      expect(config.matchkeys?.[0]?.fields[0]?.transforms).toEqual([
        "substring:0:3",
        "qgram:3",
        "bloom_filter",
      ]);
      expect(config.matchkeys?.[0]?.fields[1]?.transforms).toEqual([
        "bloom_filter:high",
      ]);
    });

    it("unknown scorer only warns (does not throw)", () => {
      const warnings: string[] = [];
      const origWarn = console.warn;
      console.warn = (msg: unknown) => {
        warnings.push(String(msg));
      };
      try {
        const raw = {
          matchkeys: [
            {
              name: "mk",
              type: "weighted",
              fields: [
                {
                  field: "a",
                  transforms: ["lowercase"],
                  scorer: "my_plugin_scorer",
                  weight: 1,
                },
              ],
            },
          ],
        };
        const config = parseConfig(raw);
        expect(config.matchkeys?.[0]?.fields[0]?.scorer).toBe("my_plugin_scorer");
        expect(warnings.join(" ")).toContain("my_plugin_scorer");
      } finally {
        console.warn = origWarn;
      }
    });

    it("accepts valid known scorers without warning", () => {
      const warnings: string[] = [];
      const origWarn = console.warn;
      console.warn = (msg: unknown) => {
        warnings.push(String(msg));
      };
      try {
        const raw = {
          matchkeys: [
            {
              name: "mk",
              type: "weighted",
              fields: [
                {
                  field: "a",
                  transforms: ["lowercase"],
                  scorer: "jaro_winkler",
                  weight: 1,
                },
              ],
            },
          ],
        };
        parseConfig(raw);
        expect(warnings).toHaveLength(0);
      } finally {
        console.warn = origWarn;
      }
    });
  });
});
