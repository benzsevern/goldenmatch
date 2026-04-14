/**
 * Basic deduplication: find duplicate people in a small array.
 * Run: npx tsx examples/01-basic-dedupe.ts
 */
import { dedupe } from "goldenmatch";

const people = [
  { id: 1, first_name: "John",   last_name: "Smith", email: "john@example.com",    zip: "12345" },
  { id: 2, first_name: "Jon",    last_name: "Smith", email: "john@example.com",    zip: "12345" },
  { id: 3, first_name: "Johnny", last_name: "Smith", email: "j.smith@example.com", zip: "12345" },
  { id: 4, first_name: "Jane",   last_name: "Doe",   email: "jane@example.com",    zip: "54321" },
  { id: 5, first_name: "Janet",  last_name: "Doe",   email: "janet@example.com",   zip: "54321" },
];

const result = dedupe(people, {
  exact: ["email"],
  fuzzy: { first_name: 0.8, last_name: 0.85 },
  blocking: ["zip"],
  threshold: 0.85,
});

console.log(`Records:    ${result.stats.totalRecords}`);
console.log(`Clusters:   ${result.stats.totalClusters}`);
console.log(`Match rate: ${(result.stats.matchRate * 100).toFixed(1)}%\n`);

console.log("Golden records:");
for (const rec of result.goldenRecords) {
  console.log(" ", rec);
}

console.log("\nDuplicate groups:");
for (const [cid, cluster] of result.clusters) {
  if (cluster.size < 2) continue;
  console.log(
    `  Cluster ${cid} (${cluster.size} members, confidence ${cluster.confidence.toFixed(2)}):`,
  );
  for (const mid of cluster.members) {
    console.log(`    row ${mid}`);
  }
}

/**
 * Expected output (approximate):
 *
 *   Records:    5
 *   Clusters:   3
 *   Match rate: 40.0%
 *
 *   Golden records:
 *     { __cluster_id__: 0, id: 1, first_name: 'John', ... }
 *     ...
 *
 *   Duplicate groups:
 *     Cluster 0 (2 members, confidence 0.95): row 0, row 1
 *     Cluster 1 (...): ...
 *
 * Python -> TS differences:
 *   - Python returns DataFrames; TS returns plain arrays + ReadonlyMap of clusters.
 *   - Python `result.stats["total_records"]`; TS `result.stats.totalRecords`.
 */
