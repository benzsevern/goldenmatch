/**
 * CSV file pipeline: read CSV -> dedupe -> write golden records back to CSV.
 *
 * This example uses `goldenmatch/node` (the Node-only subpackage) for
 * file I/O. The core `goldenmatch` package is edge-safe and doesn't
 * import `node:fs`.
 *
 * Run: npx tsx examples/03-csv-file-pipeline.ts
 *
 * Prereq: a file `customers.csv` in the working directory. We create one
 * inline for demo purposes.
 */
import { writeFileSync, unlinkSync } from "node:fs";
import { dedupe } from "goldenmatch";
import { readFile, writeCsv } from "goldenmatch/node";

// --- Step 0: create a demo CSV (you'd skip this step in real life) ---
const DEMO_PATH = "customers_demo.csv";
const GOLDEN_PATH = "golden_demo.csv";
writeFileSync(
  DEMO_PATH,
  [
    "id,name,email,zip",
    "1,John Smith,john@example.com,12345",
    "2,Jon Smith,john@example.com,12345",
    "3,Jane Doe,jane@example.com,54321",
    "4,J. Smith,john@example.com,12345",
    "5,Janet Doe,janet@example.com,54321",
  ].join("\n"),
);

// --- Step 1: read the CSV (auto-coerces numbers, handles BOM) ---
const rows = readFile(DEMO_PATH);
console.log(`Read ${rows.length} rows from ${DEMO_PATH}`);

// --- Step 2: dedupe ---
const result = dedupe(rows, {
  exact: ["email"],
  fuzzy: { name: 0.8 },
  blocking: ["zip"],
  threshold: 0.8,
});

console.log(
  `Found ${result.stats.totalClusters} clusters from ${result.stats.totalRecords} records`,
);

// --- Step 3: write golden records back to CSV ---
writeCsv(GOLDEN_PATH, result.goldenRecords);
console.log(`Wrote ${result.goldenRecords.length} golden records to ${GOLDEN_PATH}`);

// Cleanup demo files
unlinkSync(DEMO_PATH);
unlinkSync(GOLDEN_PATH);

/**
 * `readFile()` auto-detects CSV vs JSON by extension. Use `readCsv()` /
 * `readJson()` directly if you want to pass options (delimiter, encoding).
 *
 * `writeCsv()` serializes an array of objects; columns are inferred from
 * the union of keys across all rows.
 */
