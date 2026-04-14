/**
 * Evaluate predicted pairs against ground truth: precision, recall, F1.
 *
 * Two evaluation modes:
 *   - evaluatePairs: compare the predicted ScoredPair[] to ground truth
 *   - evaluateClusters: expand clusters into pairs, then compare
 *
 * Run: npx tsx examples/11-evaluate.ts
 */
import { dedupe, evaluatePairs, evaluateClusters } from "goldenmatch";

// Dataset with known ground truth
const rows = [
  { id: 0, name: "John Smith",  zip: "12345" },
  { id: 1, name: "Jon Smith",   zip: "12345" },  // dupe of 0
  { id: 2, name: "Johnny Smth", zip: "12345" },  // dupe of 0
  { id: 3, name: "Jane Doe",    zip: "54321" },
  { id: 4, name: "Janet Doe",   zip: "54321" },  // dupe of 3
  { id: 5, name: "Bob Jones",   zip: "99999" },
];

// Ground truth: pairs that SHOULD match (canonicalized min:max)
const truth: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 2],
  [3, 4],
];

// Run dedupe
const result = dedupe(rows, {
  fuzzy: { name: 0.85 },
  blocking: ["zip"],
  threshold: 0.85,
});

// --- Evaluate via pair set ---
const pairEval = evaluatePairs(result.scoredPairs, truth);
console.log("=== Pair-based evaluation ===");
console.log(`  Precision: ${pairEval.precision.toFixed(3)}`);
console.log(`  Recall:    ${pairEval.recall.toFixed(3)}`);
console.log(`  F1:        ${pairEval.f1.toFixed(3)}`);
console.log(`  TP=${pairEval.truePositives}  FP=${pairEval.falsePositives}  FN=${pairEval.falseNegatives}`);

// --- Evaluate via cluster expansion ---
const allIds = rows.map((r) => r.id);
const clusterEval = evaluateClusters(result.clusters, truth, allIds);
console.log("\n=== Cluster-based evaluation ===");
console.log(`  Precision: ${clusterEval.precision.toFixed(3)}`);
console.log(`  Recall:    ${clusterEval.recall.toFixed(3)}`);
console.log(`  F1:        ${clusterEval.f1.toFixed(3)}`);

/**
 * Cluster-based eval is typically more favorable than pair-based because
 * transitive closures pick up pairs the direct scorer missed (if A~B and
 * B~C cluster together, A~C counts as a TP even if A~C scored below
 * threshold directly).
 *
 * Python parity: `evaluate_pairs()` / `evaluate_clusters()` in
 * `goldenmatch.core.evaluate`. CLI: `goldenmatch evaluate --ground-truth gt.csv`.
 */
