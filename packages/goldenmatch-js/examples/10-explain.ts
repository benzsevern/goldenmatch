/**
 * Explain why two records matched (or didn't).
 *
 * `explainPair()` re-runs per-field scoring with the matchkey config and
 * produces an NL explanation plus per-field scores. Zero LLM cost.
 *
 * Run: npx tsx examples/10-explain.ts
 */
import {
  explainPair,
  makeMatchkeyConfig,
  makeMatchkeyField,
} from "goldenmatch";

const mk = makeMatchkeyConfig({
  name: "identity",
  type: "weighted",
  threshold: 0.85,
  fields: [
    makeMatchkeyField({ field: "first_name", transforms: ["lowercase"], scorer: "jaro_winkler", weight: 0.3 }),
    makeMatchkeyField({ field: "last_name",  transforms: ["lowercase"], scorer: "jaro_winkler", weight: 0.4 }),
    makeMatchkeyField({ field: "email",      transforms: ["lowercase"], scorer: "exact",        weight: 0.3 }),
  ],
});

// --- Case 1: strong match ---
const a1 = { first_name: "John", last_name: "Smith", email: "john@example.com" };
const b1 = { first_name: "Jon",  last_name: "Smith", email: "john@example.com" };

const exp1 = explainPair(a1, b1, mk);
console.log("=== Case 1: strong match ===");
console.log(`Overall score: ${exp1.score.toFixed(3)} (confidence: ${exp1.confidence})`);
console.log(`Explanation: ${exp1.explanation}`);
console.log("Per-field scores:");
for (const [field, score] of Object.entries(exp1.fieldScores)) {
  console.log(`  ${field.padEnd(12)} ${score === null ? "missing" : score.toFixed(3)}`);
}

// --- Case 2: weak match ---
const a2 = { first_name: "John",  last_name: "Smith", email: "john@example.com" };
const b2 = { first_name: "Johan", last_name: "Smyth", email: "j.smith@other.com" };

const exp2 = explainPair(a2, b2, mk);
console.log("\n=== Case 2: weak match ===");
console.log(`Overall score: ${exp2.score.toFixed(3)} (confidence: ${exp2.confidence})`);
console.log(`Explanation: ${exp2.explanation}`);
console.log("Reasoning steps:");
for (const step of exp2.reasoning) {
  console.log(`  - ${step}`);
}

/**
 * `details` also contains the full `FieldScoreDetail` array (normalized values
 * after transforms, diff classification). Useful for building review-queue UIs.
 */
for (const d of exp2.details) {
  console.log(
    `  [details] ${d.field}: "${d.valueA}" vs "${d.valueB}"  (${d.diffType})`,
  );
}
