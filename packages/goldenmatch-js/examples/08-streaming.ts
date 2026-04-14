/**
 * Streaming / incremental record matching.
 *
 * Add records one at a time to a running cluster state. Useful for:
 *   - Kafka / Kinesis consumers
 *   - Continuous CDC ingest
 *   - Web forms (match new signup vs. existing customers on submit)
 *
 * Each `add()` does one `matchOne()` against the running set, then
 * updates the cluster map in-place.
 *
 * Run: npx tsx examples/08-streaming.ts
 */
import {
  StreamProcessor,
  makeMatchkeyConfig,
  makeMatchkeyField,
} from "goldenmatch";

const mk = makeMatchkeyConfig({
  name: "identity",
  type: "weighted",
  threshold: 0.85,
  fields: [
    makeMatchkeyField({ field: "name",  transforms: ["lowercase", "strip"], scorer: "jaro_winkler", weight: 0.6 }),
    makeMatchkeyField({ field: "email", transforms: ["lowercase", "strip"], scorer: "exact",        weight: 0.4 }),
  ],
});

const stream = new StreamProcessor({
  matchkey: mk,
  threshold: 0.85,
  maxClusterSize: 50,
});

// Simulated record stream
const records = [
  { name: "John Smith",  email: "john@example.com" },
  { name: "Jane Doe",    email: "jane@example.com" },
  { name: "Jon Smith",   email: "john@example.com" },  // dupe of #0
  { name: "Bob Jones",   email: "bob@example.com" },
  { name: "Janet Doe",   email: "janet@example.com" },
  { name: "J. Smith",    email: "john@example.com" },  // dupe of #0
  { name: "Alice Chen",  email: "alice@example.com" },
  // ... imagine 43 more records
];

console.log(`Streaming ${records.length} records...\n`);

for (let i = 0; i < records.length; i++) {
  const rec = records[i]!;
  const result = stream.add(rec);
  if (result.matchedIds.length > 0) {
    console.log(
      `  #${i} "${rec.name}" -> cluster ${result.clusterId}, matched ${result.matchedIds.length} existing (ids: ${result.matchedIds.join(", ")})`,
    );
  } else {
    console.log(`  #${i} "${rec.name}" -> new cluster ${result.clusterId}`);
  }
}

// Final snapshot
const snap = stream.snapshot();
console.log(`\nFinal state: ${stream.size} records in ${snap.clusters.size} clusters`);

for (const [cid, info] of snap.clusters) {
  if (info.size < 2) continue;
  console.log(
    `  Cluster ${cid}: ${info.size} members ${JSON.stringify(info.members)} (confidence ${info.confidence.toFixed(2)})`,
  );
}

/**
 * Python -> TS differences:
 *   - Python `StreamProcessor.add(df_row)`; TS `stream.add(rowObject)`.
 *   - `matchedIds` is the list of pre-existing rows the new record joined.
 *   - `clusterId` is the cluster the record ultimately landed in (may be
 *     a brand-new cluster or an existing one that got merged into).
 */
