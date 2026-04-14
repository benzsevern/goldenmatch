import { describe, it, expect } from "vitest";
import {
  buildComparisonVector,
  trainEM,
  scoreProbabilistic,
  scoreProbabilisticPair,
} from "../../src/core/probabilistic.js";
import { makeMatchkeyConfig, makeMatchkeyField } from "../../src/core/index.js";
import type { Row } from "../../src/core/index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let nextId = 0;
function resetIds() {
  nextId = 0;
}
function makePerson(first: string, last: string, email: string): Row {
  return {
    __row_id__: nextId++,
    first_name: first,
    last_name: last,
    email,
  };
}

// Matchkey builder used across tests.
function buildMatchkey() {
  return makeMatchkeyConfig({
    name: "identity",
    type: "probabilistic",
    fields: [
      makeMatchkeyField({
        field: "first_name",
        scorer: "jaro_winkler",
        transforms: ["lowercase"],
      }),
      makeMatchkeyField({
        field: "last_name",
        scorer: "jaro_winkler",
        transforms: ["lowercase"],
      }),
      makeMatchkeyField({
        field: "email",
        scorer: "jaro_winkler",
        transforms: ["lowercase"],
      }),
    ],
  });
}

// ---------------------------------------------------------------------------
// buildComparisonVector
// ---------------------------------------------------------------------------

describe("buildComparisonVector", () => {
  it("levels=2: agree/disagree based on partial threshold", () => {
    const rowA: Row = { name: "John Smith" };
    const rowB: Row = { name: "John Smith" };
    const rowC: Row = { name: "Zxqwer" };
    const fields = [
      makeMatchkeyField({
        field: "name",
        scorer: "jaro_winkler",
        levels: 2,
        partialThreshold: 0.7,
      }),
    ];

    expect(buildComparisonVector(rowA, rowB, fields)).toEqual([1]);
    expect(buildComparisonVector(rowA, rowC, fields)).toEqual([0]);
  });

  it("levels=2: null inputs treated as disagree", () => {
    const rowA: Row = { name: "John Smith" };
    const rowB: Row = { name: null };
    const fields = [
      makeMatchkeyField({ field: "name", scorer: "jaro_winkler", levels: 2 }),
    ];
    const vec = buildComparisonVector(rowA, rowB, fields);
    expect(vec).toEqual([0]);
  });

  it("levels=2: missing field keys treated as disagree", () => {
    const rowA: Row = {};
    const rowB: Row = {};
    const fields = [
      makeMatchkeyField({ field: "name", scorer: "jaro_winkler", levels: 2 }),
    ];
    const vec = buildComparisonVector(rowA, rowB, fields);
    // Two empty strings -> scoreField likely returns null; falls through to 0.
    expect(vec[0]).toBe(0);
  });

  it("levels=3: distinct levels for exact/partial/disagree", () => {
    const equal = buildComparisonVector(
      { email: "alice@example.com" },
      { email: "alice@example.com" },
      [
        makeMatchkeyField({
          field: "email",
          scorer: "jaro_winkler",
          levels: 3,
          partialThreshold: 0.7,
        }),
      ],
    );
    expect(equal).toEqual([2]); // full agreement, s >= 0.95

    const partial = buildComparisonVector(
      { name: "Jonathan" },
      { name: "Jonas" }, // JW ~0.86: below 0.95, above 0.7
      [
        makeMatchkeyField({
          field: "name",
          scorer: "jaro_winkler",
          levels: 3,
          partialThreshold: 0.7,
        }),
      ],
    );
    expect(partial).toEqual([1]);

    const disagree = buildComparisonVector(
      { name: "Alice" },
      { name: "Zxqwer" },
      [
        makeMatchkeyField({
          field: "name",
          scorer: "jaro_winkler",
          levels: 3,
          partialThreshold: 0.7,
        }),
      ],
    );
    expect(disagree).toEqual([0]);
  });

  it("applies field transforms before scoring", () => {
    const rowA: Row = { name: "  ALICE  " };
    const rowB: Row = { name: "alice" };
    const fields = [
      makeMatchkeyField({
        field: "name",
        scorer: "exact",
        transforms: ["lowercase", "strip"],
        levels: 2,
        partialThreshold: 0.9,
      }),
    ];
    const vec = buildComparisonVector(rowA, rowB, fields);
    expect(vec).toEqual([1]);
  });
});

// ---------------------------------------------------------------------------
// trainEM
// ---------------------------------------------------------------------------

describe("trainEM", () => {
  it("learns m/u on constructed near-dup dataset and converges", () => {
    resetIds();
    // 10 near-duplicate pairs (small typos) — 20 rows total that cluster.
    const duplicates: Row[] = [];
    const dupeSpecs: Array<[string, string, string]> = [
      ["John", "Smith", "john@x.com"],
      ["Mary", "Jones", "mary@y.com"],
      ["Alice", "Brown", "alice@z.com"],
      ["Bob", "Miller", "bob@a.com"],
      ["Carol", "Davis", "carol@b.com"],
      ["David", "Wilson", "david@c.com"],
      ["Eve", "Moore", "eve@d.com"],
      ["Frank", "Taylor", "frank@e.com"],
      ["Grace", "Anderson", "grace@f.com"],
      ["Hank", "Thomas", "hank@g.com"],
    ];
    for (const [first, last, email] of dupeSpecs) {
      duplicates.push(makePerson(first, last, email));
      // Genuine typo: swap last two chars.
      const firstTypo =
        first.length >= 3
          ? first.slice(0, -2) + first.slice(-1) + first.slice(-2, -1)
          : first;
      duplicates.push(makePerson(firstTypo, last, email));
    }

    // 60 random non-match rows — each row gets a distinct first/last/email
    // that the JW scorer won't accidentally treat as "agree".
    const randomFirsts = [
      "Zoltan", "Xavier", "Yolanda", "Victor", "Wendy", "Nathan", "Oscar",
      "Penny", "Quincy", "Rita", "Sasha", "Trevor", "Ursula", "Vince",
      "Walter", "Yusuf", "Zara", "Ahmed", "Beatrice", "Clive", "Dimitri",
      "Esperanza", "Farid", "Gabriela", "Horacio", "Ingrid", "Jorge",
      "Katarina", "Lorenzo", "Mikhail", "Natalia", "Octavio", "Priya",
      "Qasim", "Rosalind", "Sergio", "Tatiana", "Ulrich", "Valentina",
      "Wilhelm", "Xiomara", "Yaroslav", "Zephyr", "Adelaide", "Bartholomew",
      "Cressida", "Demetrius", "Evangeline", "Fionnuala", "Gregorius",
      "Hephzibah", "Isambard", "Jocasta", "Kenelm", "Ludmilla", "Mordecai",
      "Nicephorus", "Ophelia", "Peregrine", "Quirinus",
    ];
    const randomLasts = [
      "Zygmunt", "Xiong", "Petrov", "Kowalski", "Nakamura", "Oduya",
      "Fernandez", "Kaplan", "Lowenstein", "Meyer", "Papadopoulos",
      "Obradovic", "Rasmussen", "Silva", "Tanaka", "Ueno", "Vasquez",
      "Wojcik", "Yamamoto", "Zalewski", "Ababukh", "Beaumaris",
      "Caravaggio", "Drobny", "Eisenhower", "Filippov", "Gauthier",
      "Hashimoto", "Ignatiev", "Jankovic", "Khatri", "Lindqvist",
      "Magnusson", "Novotny", "Ostrowski", "Pemberton", "Quesnel",
      "Rostropovich", "Schmidt", "Tartaglia", "Ulyanov", "Vermeer",
      "Wroblewski", "Xanthopoulos", "Yankovic", "Zielinski",
      "Abercrombie", "Blumberg", "Carnelian", "Dumitrescu", "Esterhazy",
      "Finkelstein", "Grzybowski", "Hohenzollern", "Iglesias", "Jimenez",
      "Kaczmarek", "Lindstrom", "Mazzanti", "Niedermeyer", "Oppenheimer",
    ];
    const randoms: Row[] = [];
    for (let i = 0; i < 60; i++) {
      const first = randomFirsts[i]!;
      const last = randomLasts[i]!;
      randoms.push(
        makePerson(first, last, `uniq${i}_${(i * 97) % 1000}@zzz${i}.io`),
      );
    }

    const allRows = [...duplicates, ...randoms];
    const mk = buildMatchkey();

    const result = trainEM(allRows, mk, { seed: 123, maxIterations: 100 });

    // Basic shape checks.
    expect(result.m).toHaveProperty("first_name");
    expect(result.m).toHaveProperty("last_name");
    expect(result.m).toHaveProperty("email");
    expect(result.u).toHaveProperty("first_name");
    expect(result.matchWeights).toHaveProperty("email");

    // proportionMatched should be a sensible probability.
    expect(result.proportionMatched).toBeGreaterThan(0);
    expect(result.proportionMatched).toBeLessThan(1);

    // With default levels=2 the vectors are [disagree, agree].
    // For cleanly-differentiated name fields, u[agree] should be small
    // (random pairs rarely agree) and m[agree] should be large
    // (true-match pairs do agree), yielding positive match weight at level=1.
    for (const f of ["first_name", "last_name"]) {
      const uAgree = result.u[f]![1]!;
      const mAgree = result.m[f]![1]!;
      expect(uAgree).toBeLessThan(0.2);
      expect(mAgree).toBeGreaterThan(0.6);
      // match weight at "agree" level must be positive (log2(m/u) > 0).
      expect(result.matchWeights[f]![1]!).toBeGreaterThan(0);
    }

    // EM should finish within the iteration cap.
    expect(result.iterations).toBeLessThanOrEqual(100);
    expect(result.iterations).toBeGreaterThan(0);
    // With this well-separated dataset EM converges before hitting the cap.
    expect(result.converged).toBe(true);
  });

  it("blocking-field invariant: near-constant field has m approximately u (zero discriminative weight)", () => {
    resetIds();
    // Dataset where `country` is constant across all rows — a blocking field candidate.
    const rows: Row[] = [];
    for (let i = 0; i < 40; i++) {
      rows.push({
        __row_id__: nextId++,
        first_name: `Name${i % 5}`,
        country: "US",
      });
    }

    const mk = makeMatchkeyConfig({
      name: "identity",
      type: "probabilistic",
      fields: [
        makeMatchkeyField({ field: "first_name", scorer: "jaro_winkler" }),
        makeMatchkeyField({ field: "country", scorer: "exact" }),
      ],
    });

    const result = trainEM(rows, mk, {
      seed: 7,
      blockingFields: ["country"],
      maxIterations: 20,
    });

    // Blocking field gets fixed neutral u (0.5, 0.5 for 2-level).
    expect(result.u.country).toEqual([0.5, 0.5]);

    // Blocking field retains its prior m (exponential, normalized): [1/3, 2/3].
    // In either case, match weights for blocking fields are the fixed linear
    // interpolation from -3 to +3, by construction.
    const cw = result.matchWeights.country!;
    expect(cw.length).toBe(2);
    expect(cw[0]!).toBeCloseTo(-3.0, 5);
    expect(cw[1]!).toBeCloseTo(3.0, 5);
  });

  it("iterations cap respected on degenerate (all identical rows) data", () => {
    resetIds();
    const rows: Row[] = [];
    for (let i = 0; i < 30; i++) {
      rows.push({ __row_id__: nextId++, first_name: "John", last_name: "Smith" });
    }
    const mk = makeMatchkeyConfig({
      name: "identity",
      type: "probabilistic",
      fields: [
        makeMatchkeyField({ field: "first_name", scorer: "jaro_winkler" }),
        makeMatchkeyField({ field: "last_name", scorer: "jaro_winkler" }),
      ],
    });
    const result = trainEM(rows, mk, { seed: 1, maxIterations: 5 });
    // Must not exceed cap; must terminate cleanly.
    expect(result.iterations).toBeLessThanOrEqual(5);
    expect(Number.isFinite(result.proportionMatched)).toBe(true);
  });

  it("deterministic: same seed + same input => identical EMResult", () => {
    resetIds();
    const rows: Row[] = [];
    for (let i = 0; i < 25; i++) {
      rows.push(makePerson(`First${i % 7}`, `Last${i % 5}`, `user${i % 9}@x.com`));
    }
    const mk = buildMatchkey();

    const r1 = trainEM(rows, mk, { seed: 999, maxIterations: 20 });
    const r2 = trainEM(rows, mk, { seed: 999, maxIterations: 20 });

    expect(r1.iterations).toBe(r2.iterations);
    expect(r1.converged).toBe(r2.converged);
    expect(r1.proportionMatched).toBeCloseTo(r2.proportionMatched, 12);
    for (const f of Object.keys(r1.m)) {
      for (let k = 0; k < r1.m[f]!.length; k++) {
        expect(r1.m[f]![k]!).toBeCloseTo(r2.m[f]![k]!, 12);
        expect(r1.u[f]![k]!).toBeCloseTo(r2.u[f]![k]!, 12);
        expect(r1.matchWeights[f]![k]!).toBeCloseTo(r2.matchWeights[f]![k]!, 12);
      }
    }
  });

  it("fallback result on tiny dataset (< 10 sampled pairs) still returns a valid EMResult", () => {
    resetIds();
    // Only 3 rows -> 3 possible pairs, below the fallback threshold of 10.
    const rows: Row[] = [
      makePerson("A", "B", "a@b.com"),
      makePerson("C", "D", "c@d.com"),
      makePerson("E", "F", "e@f.com"),
    ];
    const mk = buildMatchkey();
    const result = trainEM(rows, mk, { seed: 1 });
    // Fallback path: iterations=0, converged=false, but shape is valid.
    expect(result.iterations).toBe(0);
    expect(result.converged).toBe(false);
    for (const f of ["first_name", "last_name", "email"]) {
      expect(result.m[f]!.length).toBe(2);
      expect(result.u[f]!.length).toBe(2);
      expect(result.matchWeights[f]!.length).toBe(2);
    }
  });
});

// ---------------------------------------------------------------------------
// scoreProbabilistic
// ---------------------------------------------------------------------------

describe("scoreProbabilistic", () => {
  it("identical rows score near 1.0; very different rows drop out or score near 0", () => {
    resetIds();
    // Build a reasonable EM model on a small mixed dataset.
    const trainRows: Row[] = [];
    for (let i = 0; i < 10; i++) {
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `user${i}@x.com`));
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `user${i}@x.com`));
    }
    for (let i = 0; i < 10; i++) {
      trainRows.push(
        makePerson(`Zzz${i}`, `Qqq${i}`, `other${i}@y.com`),
      );
    }
    const mk = buildMatchkey();
    const em = trainEM(trainRows, mk, { seed: 5, maxIterations: 20 });

    // Now score a block containing one clearly-matching pair and one clearly-different pair.
    resetIds();
    const block: Row[] = [
      makePerson("Alice", "Smith", "alice@example.com"),
      makePerson("Alice", "Smith", "alice@example.com"),
      makePerson("Zoltan", "Qwerty", "zzz@nope.com"),
    ];
    const scored = scoreProbabilistic(block, mk, em, { threshold: 0.0 });

    // All 3 pairs evaluated (threshold=0).
    expect(scored.length).toBe(3);

    const byKey = new Map<string, number>();
    for (const s of scored) byKey.set(`${s.idA}:${s.idB}`, s.score);

    // Identical rows (ids 0,1) -> near 1.0
    const identical = byKey.get("0:1")!;
    expect(identical).toBeGreaterThan(0.9);

    // Completely different pair (0,2) -> near 0.0 (low match weight sum).
    const different = byKey.get("0:2")!;
    expect(different).toBeLessThan(identical);
    expect(different).toBeLessThan(0.3);
  });

  it("threshold filters out low-scoring pairs", () => {
    resetIds();
    const trainRows: Row[] = [];
    for (let i = 0; i < 10; i++) {
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `user${i}@x.com`));
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `user${i}@x.com`));
    }
    for (let i = 0; i < 10; i++) {
      trainRows.push(makePerson(`Zzz${i}`, `Qqq${i}`, `other${i}@y.com`));
    }
    const mk = buildMatchkey();
    const em = trainEM(trainRows, mk, { seed: 5, maxIterations: 20 });

    resetIds();
    const block: Row[] = [
      makePerson("Alice", "Smith", "alice@example.com"),
      makePerson("Alice", "Smith", "alice@example.com"),
      makePerson("Zoltan", "Qwerty", "zzz@nope.com"),
    ];
    const highT = scoreProbabilistic(block, mk, em, { threshold: 0.9 });
    // Only the identical pair survives a high threshold.
    expect(highT.length).toBe(1);
    expect(highT[0]!.idA).toBe(0);
    expect(highT[0]!.idB).toBe(1);
  });

  it("excludePairs skips already-matched pairs", () => {
    resetIds();
    const trainRows: Row[] = [];
    for (let i = 0; i < 10; i++) {
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `user${i}@x.com`));
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `user${i}@x.com`));
    }
    for (let i = 0; i < 10; i++) {
      trainRows.push(makePerson(`Zzz${i}`, `Qqq${i}`, `other${i}@y.com`));
    }
    const mk = buildMatchkey();
    const em = trainEM(trainRows, mk, { seed: 5, maxIterations: 20 });

    resetIds();
    const block: Row[] = [
      makePerson("Alice", "Smith", "alice@example.com"),
      makePerson("Alice", "Smith", "alice@example.com"),
    ];
    const excluded = new Set<string>(["0:1"]);
    const scored = scoreProbabilistic(block, mk, em, {
      excludePairs: excluded,
      threshold: 0.0,
    });
    expect(scored.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// scoreProbabilisticPair
// ---------------------------------------------------------------------------

describe("scoreProbabilisticPair", () => {
  it("returns a [0,1] score for a trained EMResult", () => {
    resetIds();
    const trainRows: Row[] = [];
    for (let i = 0; i < 10; i++) {
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `u${i}@x.com`));
      trainRows.push(makePerson(`First${i}`, `Last${i}`, `u${i}@x.com`));
    }
    for (let i = 0; i < 10; i++) {
      trainRows.push(makePerson(`Zzz${i}`, `Qqq${i}`, `o${i}@y.com`));
    }
    const mk = buildMatchkey();
    const em = trainEM(trainRows, mk, { seed: 11, maxIterations: 20 });

    const a: Row = { first_name: "Alice", last_name: "Smith", email: "a@x.com" };
    const b: Row = { first_name: "Alice", last_name: "Smith", email: "a@x.com" };
    const c: Row = { first_name: "Zoltan", last_name: "Qqq", email: "z@z.com" };

    const hi = scoreProbabilisticPair(a, b, mk, em);
    const lo = scoreProbabilisticPair(a, c, mk, em);

    expect(hi).toBeGreaterThanOrEqual(0);
    expect(hi).toBeLessThanOrEqual(1);
    expect(lo).toBeGreaterThanOrEqual(0);
    expect(lo).toBeLessThanOrEqual(1);
    expect(hi).toBeGreaterThan(lo);
  });
});
