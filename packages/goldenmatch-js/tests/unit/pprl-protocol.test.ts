import { describe, it, expect } from "vitest";
import {
  runPPRL,
  autoConfigurePPRL,
  linkTrustedThirdParty,
  linkSMC,
  type PPRLConfig,
} from "../../src/core/pprl/protocol.js";
import type { Row } from "../../src/core/index.js";

// ---------------------------------------------------------------------------
// Deterministic synthetic person generator
// ---------------------------------------------------------------------------

const FIRST_NAMES = [
  "Alice", "Bob", "Carol", "David", "Eve",
  "Frank", "Grace", "Hank", "Ivy", "Jack",
  "Karen", "Leo", "Mary", "Noah", "Olive",
  "Paul", "Quinn", "Ruth", "Steve", "Tina",
];
const LAST_NAMES = [
  "Smith", "Jones", "Brown", "Miller", "Davis",
  "Wilson", "Moore", "Taylor", "Anderson", "Thomas",
  "Jackson", "White", "Harris", "Martin", "Young",
];

function personDataset(n: number, seed: number): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    const fi = (seed * 7 + i * 3) % FIRST_NAMES.length;
    const li = (seed * 11 + i * 5) % LAST_NAMES.length;
    const first = FIRST_NAMES[fi]!;
    const last = LAST_NAMES[li]!;
    rows.push({
      __row_id__: i,
      id: `SEED${seed}-ROW${i}`, // near-unique, should be skipped by auto-config
      first_name: first,
      last_name: last,
      email: `${first.toLowerCase()}.${last.toLowerCase()}${i}@example.com`,
      city: ["NYC", "LA", "CHI", "BOS", "SEA"][i % 5]!,
    });
  }
  return rows;
}

// Introduce typos to simulate overlap between two parties.
function typo(s: string): string {
  if (s.length < 3) return s;
  // Swap two adjacent middle chars.
  const i = Math.floor(s.length / 2);
  return s.slice(0, i - 1) + s[i]! + s[i - 1]! + s.slice(i + 1);
}

// ---------------------------------------------------------------------------
// autoConfigurePPRL
// ---------------------------------------------------------------------------

describe("autoConfigurePPRL", () => {
  it("picks sensible defaults, skipping ID-like and out-of-range fields", () => {
    const a = personDataset(30, 1);
    const b = personDataset(30, 2);

    const cfg = autoConfigurePPRL(a, b);

    // Basic invariants.
    expect(cfg.fields.length).toBeLessThanOrEqual(4);
    expect(cfg.threshold).toBeGreaterThanOrEqual(0.85);
    expect(cfg.protocol).toBe("trusted_third_party");
    expect(cfg.securityLevel).toBe("standard");

    // `id` has cardinality_ratio=1.0 (unique per-row) => must be skipped.
    expect(cfg.fields).not.toContain("id");
    // `__row_id__` is also unique => must be skipped.
    expect(cfg.fields).not.toContain("__row_id__");
  });

  it("handles a field with mixed null + real values (still considered if null rate < 30%)", () => {
    const a: Row[] = [];
    const b: Row[] = [];
    // 20 rows each: first_name always present, middle_name null 20% of the time.
    for (let i = 0; i < 20; i++) {
      a.push({
        __row_id__: i,
        first_name: FIRST_NAMES[i % FIRST_NAMES.length]!,
        last_name: LAST_NAMES[i % LAST_NAMES.length]!,
        middle_name: i % 5 === 0 ? null : `Middle${i % 4}`,
      });
      b.push({
        __row_id__: i,
        first_name: FIRST_NAMES[(i + 3) % FIRST_NAMES.length]!,
        last_name: LAST_NAMES[(i + 2) % LAST_NAMES.length]!,
        middle_name: i % 7 === 0 ? null : `Middle${i % 4}`,
      });
    }
    const cfg = autoConfigurePPRL(a, b);
    // Middle name with ~20% nulls passes (<30%), last_name/first_name should both be considered.
    expect(cfg.fields.length).toBeGreaterThan(0);
    expect(cfg.fields.every((f) => typeof f === "string")).toBe(true);
  });

  it("drops high-null fields (> 30% null rate)", () => {
    const a: Row[] = [];
    const b: Row[] = [];
    for (let i = 0; i < 30; i++) {
      a.push({
        __row_id__: i,
        first_name: FIRST_NAMES[i % FIRST_NAMES.length]!,
        last_name: LAST_NAMES[i % LAST_NAMES.length]!,
        optional: i % 2 === 0 ? null : `val${i}`, // 50% null
      });
      b.push({
        __row_id__: i,
        first_name: FIRST_NAMES[(i + 2) % FIRST_NAMES.length]!,
        last_name: LAST_NAMES[(i + 1) % LAST_NAMES.length]!,
        optional: i % 2 === 0 ? null : `val${i}`,
      });
    }
    const cfg = autoConfigurePPRL(a, b);
    expect(cfg.fields).not.toContain("optional");
  });
});

// ---------------------------------------------------------------------------
// runPPRL
// ---------------------------------------------------------------------------

describe("runPPRL", () => {
  function twoPartiesWithOverlap() {
    // Dataset A: 50 people.
    const a: Row[] = [];
    for (let i = 0; i < 50; i++) {
      const first = FIRST_NAMES[i % FIRST_NAMES.length]!;
      const last = LAST_NAMES[i % LAST_NAMES.length]!;
      a.push({
        __row_id__: i,
        first_name: first,
        last_name: last,
        email: `${first.toLowerCase()}.${last.toLowerCase()}${i}@x.com`,
      });
    }

    // Dataset B: 50 people, first 10 are "the same people" with typos.
    const b: Row[] = [];
    for (let i = 0; i < 10; i++) {
      const first = FIRST_NAMES[i % FIRST_NAMES.length]!;
      const last = LAST_NAMES[i % LAST_NAMES.length]!;
      b.push({
        __row_id__: i,
        // Use typos on last name to simulate noisy overlap.
        first_name: first,
        last_name: typo(last),
        email: `${first.toLowerCase()}.${last.toLowerCase()}${i}@x.com`,
      });
    }
    for (let i = 10; i < 50; i++) {
      b.push({
        __row_id__: i,
        first_name: `NonOverlap${i}`,
        last_name: `Different${i}`,
        email: `novel${i}@other.org`,
      });
    }
    return { a, b };
  }

  it("finds most shared entities in two datasets with partial overlap", () => {
    const { a, b } = twoPartiesWithOverlap();
    const config: PPRLConfig = {
      fields: ["first_name", "last_name", "email"],
      securityLevel: "standard",
      protocol: "trusted_third_party",
      threshold: 0.5,
    };
    const result = runPPRL(a, b, config);

    // Every match has correct shape.
    for (const m of result.matches) {
      expect(m).toHaveProperty("idA");
      expect(m).toHaveProperty("idB");
      expect(m).toHaveProperty("score");
      expect(m.score).toBeGreaterThanOrEqual(config.threshold);
      expect(m.score).toBeLessThanOrEqual(1);
    }

    // Stats reflect the pass.
    expect(result.stats["comparedPairs"]).toBe(50 * 50);
    expect(result.stats["matchCount"]).toBe(result.matches.length);
    expect(result.stats["protocol"]).toBe("trusted_third_party");

    // True pairs should surface: a[i] ~ b[i] for i in 0..9.
    const truePairs = new Set<string>();
    for (let i = 0; i < 10; i++) truePairs.add(`${i}:${i}`);

    let hits = 0;
    for (const m of result.matches) {
      if (truePairs.has(`${m.idA}:${m.idB}`)) hits++;
    }
    // We expect most of the 10 shared entities to be recovered at threshold 0.5.
    expect(hits).toBeGreaterThanOrEqual(7);
  });

  it("runs end-to-end with security_level standard/high/paranoid and finds same true pairs", () => {
    const { a, b } = twoPartiesWithOverlap();
    const baseFields: string[] = ["first_name", "last_name", "email"];

    const runAt = (level: "standard" | "high" | "paranoid") => {
      const cfg: PPRLConfig =
        level === "standard"
          ? {
              fields: baseFields,
              securityLevel: level,
              protocol: "trusted_third_party",
              threshold: 0.4,
            }
          : {
              fields: baseFields,
              securityLevel: level,
              protocol: "trusted_third_party",
              threshold: 0.4,
              salt: "shared-secret",
            };
      return runPPRL(a, b, cfg);
    };

    const rStandard = runAt("standard");
    const rHigh = runAt("high");
    const rParanoid = runAt("paranoid");

    // All produce matches without throwing.
    expect(rStandard.matches.length).toBeGreaterThan(0);
    expect(rHigh.matches.length).toBeGreaterThan(0);
    expect(rParanoid.matches.length).toBeGreaterThan(0);

    // The true pairs (a[i] ~ b[i] for i in 0..9) should be found in all three.
    for (const result of [rStandard, rHigh, rParanoid]) {
      const keys = new Set<string>();
      for (const m of result.matches) keys.add(`${m.idA}:${m.idB}`);
      let hits = 0;
      for (let i = 0; i < 10; i++) if (keys.has(`${i}:${i}`)) hits++;
      expect(hits).toBeGreaterThanOrEqual(5);
    }
  });

  it("deterministic CLK: running twice on same data gives identical matches", () => {
    const { a, b } = twoPartiesWithOverlap();
    const config: PPRLConfig = {
      fields: ["first_name", "last_name", "email"],
      securityLevel: "standard",
      protocol: "trusted_third_party",
      threshold: 0.5,
    };
    const r1 = runPPRL(a, b, config);
    const r2 = runPPRL(a, b, config);
    expect(r1.matches.length).toBe(r2.matches.length);
    for (let i = 0; i < r1.matches.length; i++) {
      expect(r1.matches[i]!.idA).toBe(r2.matches[i]!.idA);
      expect(r1.matches[i]!.idB).toBe(r2.matches[i]!.idB);
      expect(r1.matches[i]!.score).toBeCloseTo(r2.matches[i]!.score, 10);
    }
  });

  it("empty rowsA or rowsB returns empty matches", () => {
    const { a } = twoPartiesWithOverlap();
    const cfg: PPRLConfig = {
      fields: ["first_name", "last_name"],
      securityLevel: "standard",
      protocol: "trusted_third_party",
      threshold: 0.5,
    };
    expect(runPPRL([], a, cfg).matches).toEqual([]);
    expect(runPPRL(a, [], cfg).matches).toEqual([]);
    expect(runPPRL([], [], cfg).matches).toEqual([]);
  });

  it("skips rows that encode to empty strings (all fields null)", () => {
    const a: Row[] = [
      { __row_id__: 0, first_name: "Alice", last_name: "Smith" },
      { __row_id__: 1, first_name: null, last_name: null },
    ];
    const b: Row[] = [
      { __row_id__: 0, first_name: "Alice", last_name: "Smith" },
    ];
    const cfg: PPRLConfig = {
      fields: ["first_name", "last_name"],
      securityLevel: "standard",
      protocol: "trusted_third_party",
      threshold: 0.5,
    };
    const result = runPPRL(a, b, cfg);
    // Only the one non-null row on each side produces a match.
    expect(result.matches.length).toBe(1);
    expect(result.matches[0]!.idA).toBe(0);
    expect(result.matches[0]!.idB).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Bloom filter output format (via low-level transform)
// ---------------------------------------------------------------------------

describe("bloom filter hex output", () => {
  it("hex length differs across security levels (512/1024/2048 bits per current presets)", async () => {
    // runPPRL does not expose raw encodings, so we import the transform directly.
    const { applyTransform } = await import("../../src/core/transforms.js");
    const value = "alice smith";

    const std = applyTransform(value, "bloom_filter:standard")!;
    const high = applyTransform(value, "bloom_filter:high:secret")!;
    const paranoid = applyTransform(value, "bloom_filter:paranoid:secret")!;

    // hex => 2 chars per byte.
    // Active presets: standard=512 bits, high=1024, paranoid=2048.
    expect(std.length).toBe(128);
    expect(high.length).toBe(256);
    expect(paranoid.length).toBe(512);

    // Strictly increasing lengths confirm the "larger filter = higher security" invariant.
    expect(std.length).toBeLessThan(high.length);
    expect(high.length).toBeLessThan(paranoid.length);

    // All valid hex.
    for (const s of [std, high, paranoid]) {
      expect(/^[0-9a-f]+$/.test(s)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// linkTrustedThirdParty / linkSMC
// ---------------------------------------------------------------------------

describe("link protocol wrappers", () => {
  const a = personDataset(10, 1);
  const b = personDataset(10, 2);

  it("linkTrustedThirdParty returns a PPRLResult shape", () => {
    const result = linkTrustedThirdParty(a, b, {
      fields: ["first_name", "last_name"],
      securityLevel: "standard",
      protocol: "smc", // intentionally wrong; wrapper must normalize it.
      threshold: 0.5,
    });
    expect(Array.isArray(result.matches)).toBe(true);
    expect(result.stats["protocol"]).toBe("trusted_third_party");
  });

  it("linkSMC requires a salt and non-standard security level", () => {
    // Missing salt => throws.
    expect(() =>
      linkSMC(a, b, {
        fields: ["first_name", "last_name"],
        securityLevel: "high",
        protocol: "smc",
        threshold: 0.5,
      }),
    ).toThrow(/salt/);

    // standard security => throws.
    expect(() =>
      linkSMC(a, b, {
        fields: ["first_name", "last_name"],
        securityLevel: "standard",
        protocol: "smc",
        threshold: 0.5,
        salt: "shhh",
      }),
    ).toThrow(/high.*paranoid|paranoid/i);

    // Happy path.
    const result = linkSMC(a, b, {
      fields: ["first_name", "last_name"],
      securityLevel: "high",
      protocol: "smc",
      threshold: 0.5,
      salt: "shhh",
    });
    expect(Array.isArray(result.matches)).toBe(true);
    expect(result.stats["protocol"]).toBe("smc");
    expect(result.stats["securityLevel"]).toBe("high");
  });
});
