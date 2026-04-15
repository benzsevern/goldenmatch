/**
 * Inspect auto-config verification signals — preflight + postflight.
 *
 * Shows how to:
 *   1. Auto-configure a config from raw rows.
 *   2. Read `cfg._preflightReport` before dedupe runs.
 *   3. Run dedupe with the auto-configured config.
 *   4. Read `result.postflightReport` produced by the pipeline.
 *
 * Run: npx tsx examples/verificationInspection.ts
 */
import { autoConfigureRows, dedupe } from "goldenmatch";

// A small synthetic dataset with deliberate duplicates + noisy fields.
const rows = [
  { id: 1, name: "John Smith",   email: "john@x.com",   zip: "12345" },
  { id: 2, name: "Jon  Smith",   email: "JOHN@X.COM",   zip: "12345" },
  { id: 3, name: "Jane Doe",     email: "jane@y.com",   zip: "54321" },
  { id: 4, name: "Jane Doh",     email: "jane@y.com",   zip: "54321" },
  { id: 5, name: "Robert Brown", email: "bob@z.com",    zip: "99999" },
  { id: 6, name: "Rob Brown",    email: "bob@z.com",    zip: "99999" },
  { id: 7, name: "Alice Jones",  email: "alice@a.com",  zip: "11111" },
  { id: 8, name: "Alicia Jones", email: "alice@a.com",  zip: "11111" },
];

console.log("=".repeat(60));
console.log("STEP 1 — auto-configure from rows");
console.log("=".repeat(60));

const cfg = autoConfigureRows(rows);
console.log(`matchkeys: ${cfg.matchkeys?.length ?? 0}`);
console.log(`blocking strategy: ${cfg.blocking?.strategy}`);
console.log(`threshold: ${cfg.threshold}`);

console.log("\n" + "=".repeat(60));
console.log("STEP 2 — preflight report (config-time checks)");
console.log("=".repeat(60));

const pre = cfg._preflightReport;
if (pre === undefined) {
  console.log("no preflight report attached");
} else {
  console.log(`findings: ${pre.findings.length}`);
  console.log(`configWasModified: ${pre.configWasModified}`);
  console.log(`hasErrors: ${pre.hasErrors}`);
  for (const f of pre.findings) {
    console.log(
      `  - [${f.severity}] ${f.check} / ${f.subject}: ${f.message}` +
        (f.repaired ? ` (repaired: ${f.repairNote ?? "auto"})` : ""),
    );
  }
}

console.log("\n" + "=".repeat(60));
console.log("STEP 3 — run dedupe with the auto-configured config");
console.log("=".repeat(60));

const result = dedupe(rows, { config: cfg });
console.log(`input: ${result.stats.totalRecords} rows`);
console.log(`clusters: ${result.stats.totalClusters}`);
console.log(`match rate: ${(result.stats.matchRate * 100).toFixed(1)}%`);

console.log("\n" + "=".repeat(60));
console.log("STEP 4 — postflight report (runtime signals)");
console.log("=".repeat(60));

const post = result.postflightReport;
if (post === undefined) {
  console.log("no postflight report attached (expected when no preflight ran)");
} else {
  const s = post.signals;
  console.log(`totalPairsScored: ${s.totalPairsScored}`);
  console.log(`currentThreshold: ${s.currentThreshold}`);
  console.log(`blockingRecall: ${s.blockingRecall}`);
  console.log(
    `blockSize p50/p95/p99/max: ${s.blockSizePercentiles.p50}/` +
      `${s.blockSizePercentiles.p95}/${s.blockSizePercentiles.p99}/` +
      `${s.blockSizePercentiles.max}`,
  );
  console.log(
    `clusterSize count/p50/p95/max: ${s.preliminaryClusterSizes.count}/` +
      `${s.preliminaryClusterSizes.p50}/${s.preliminaryClusterSizes.p95}/` +
      `${s.preliminaryClusterSizes.max}`,
  );
  console.log(
    `thresholdOverlapPct: ${(s.thresholdOverlapPct * 100).toFixed(2)}%`,
  );
  console.log(`oversizedClusters: ${s.oversizedClusters.length}`);
  console.log(`adjustments applied: ${post.adjustments.length}`);
  for (const adj of post.adjustments) {
    console.log(
      `  - ${adj.field}: ${adj.fromValue} -> ${adj.toValue} ` +
        `(${adj.signal}: ${adj.reason})`,
    );
  }
  for (const adv of post.advisories) {
    console.log(`  advisory: ${adv}`);
  }
}
