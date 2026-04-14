/**
 * Fellegi-Sunter probabilistic matching with Splink-style EM training.
 *
 * The F-S model learns per-field "agreement" probabilities under match
 * vs. non-match hypotheses, producing match weights (log-likelihood
 * ratios). Train on unlabeled data via EM -- no ground truth required.
 *
 * Run: npx tsx examples/06-probabilistic-fs.ts
 */
import {
  trainEM,
  scoreProbabilistic,
  makeMatchkeyConfig,
  makeMatchkeyField,
} from "goldenmatch";

// Synthetic labeled-ish data (row ids must be on __row_id__)
const rows = [
  { __row_id__: 0, first_name: "John",  last_name: "Smith",  zip: "12345" },
  { __row_id__: 1, first_name: "Jon",   last_name: "Smith",  zip: "12345" },
  { __row_id__: 2, first_name: "John",  last_name: "Smyth",  zip: "12345" },
  { __row_id__: 3, first_name: "Jane",  last_name: "Doe",    zip: "54321" },
  { __row_id__: 4, first_name: "Janet", last_name: "Doe",    zip: "54321" },
  { __row_id__: 5, first_name: "Bob",   last_name: "Jones",  zip: "99999" },
  { __row_id__: 6, first_name: "Alice", last_name: "Miller", zip: "11111" },
  { __row_id__: 7, first_name: "Alice", last_name: "Miller", zip: "11111" },
];

// Build a probabilistic matchkey
const mk = makeMatchkeyConfig({
  name: "fs_identity",
  type: "probabilistic",
  threshold: 0.5,
  linkThreshold: 0.5,
  fields: [
    makeMatchkeyField({ field: "first_name", transforms: ["lowercase"], scorer: "jaro_winkler", levels: 3 }),
    makeMatchkeyField({ field: "last_name",  transforms: ["lowercase"], scorer: "jaro_winkler", levels: 3 }),
    makeMatchkeyField({ field: "zip",        transforms: [],            scorer: "exact",        levels: 2 }),
  ],
});

// Train the EM model
const em = trainEM(rows, mk, {
  maxIterations: 25,
  convergence: 1e-4,
  blockingFields: ["zip"],  // zip is used for blocking; fix neutral priors
  seed: 42,
  nSamplePairs: 200,
});

console.log(`EM converged: ${em.converged} (${em.iterations} iterations)`);
console.log(`Estimated p(match): ${em.proportionMatched.toFixed(3)}\n`);

console.log("Match weights (log2 m/u) per field per level:");
for (const [field, weights] of Object.entries(em.matchWeights)) {
  console.log(`  ${field.padEnd(12)} [${weights.map((w) => w.toFixed(2)).join(", ")}]`);
}

// Score all pairs in the block
const matches = scoreProbabilistic(rows, mk, em, { threshold: 0.5 });

console.log(`\nFound ${matches.length} probabilistic matches:`);
for (const m of matches) {
  console.log(`  (${m.idA}, ${m.idB}) -> ${m.score.toFixed(3)}`);
}

/**
 * Python -> TS differences:
 *   - Python `train_em()` vs. TS `trainEM()` (camelCase).
 *   - Python returns numpy arrays; TS returns `readonly number[]`.
 *   - TS requires `__row_id__` to be attached; Python adds it in its DF pipeline.
 */
