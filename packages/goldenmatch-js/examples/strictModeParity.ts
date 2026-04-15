/**
 * Strict-mode auto-config: show how `_strictAutoconfig: true` disables
 * runtime postflight threshold adjustments, yielding byte-identical-shape
 * configs suitable for CI / reproducible pipelines.
 *
 * Compares two runs on the same data:
 *   (a) normal auto-config — postflight may shift threshold
 *   (b) strict auto-config — postflight reports signals but adjusts nothing
 *
 * Run: npx tsx examples/strictModeParity.ts
 */
import { autoConfigureRows, dedupe } from "goldenmatch";
import type { Row } from "goldenmatch";

// Bimodal synthetic data: three tight duplicate clusters + scattered singletons.
// The bimodal score distribution is what tempts postflight to shift threshold.
const rows: Row[] = [
  { id: 1,  name: "John Smith",    email: "john@a.com",  zip: "10001" },
  { id: 2,  name: "Jon Smith",     email: "john@a.com",  zip: "10001" },
  { id: 3,  name: "Johnny Smith",  email: "JOHN@a.com",  zip: "10001" },
  { id: 4,  name: "Jane Doe",      email: "jane@b.com",  zip: "20002" },
  { id: 5,  name: "Jane Doh",      email: "jane@b.com",  zip: "20002" },
  { id: 6,  name: "Janet Doe",     email: "jane@b.com",  zip: "20002" },
  { id: 7,  name: "Bob Jones",     email: "bob@c.com",   zip: "30003" },
  { id: 8,  name: "Robert Jones",  email: "bob@c.com",   zip: "30003" },
  { id: 9,  name: "Alice Zhang",   email: "alice@d.com", zip: "40004" },
  { id: 10, name: "Carlos Ruiz",   email: "c@e.com",     zip: "50005" },
  { id: 11, name: "Dana White",    email: "dana@f.com",  zip: "60006" },
  { id: 12, name: "Eve Black",     email: "eve@g.com",   zip: "70007" },
];

function run(
  label: string,
  strict: boolean,
): void {
  console.log("\n" + "=".repeat(60));
  console.log(`${label} (strict=${strict})`);
  console.log("=".repeat(60));

  const cfg = autoConfigureRows(rows, { strict });
  console.log(`_strictAutoconfig flag on config: ${cfg._strictAutoconfig === true}`);

  const result = dedupe(rows, { config: cfg });
  const post = result.postflightReport;

  console.log(`clusters: ${result.stats.totalClusters}`);
  console.log(`match rate: ${(result.stats.matchRate * 100).toFixed(1)}%`);

  if (post === undefined) {
    console.log("no postflight report");
    return;
  }
  console.log(`threshold used: ${post.signals.currentThreshold}`);
  console.log(`adjustments proposed: ${post.adjustments.length}`);
  for (const adj of post.adjustments) {
    console.log(
      `  - ${adj.field}: ${adj.fromValue} -> ${adj.toValue} (${adj.reason})`,
    );
  }
  if (strict && post.adjustments.length === 0) {
    console.log("  -> strict mode: no adjustments applied (as expected)");
  }
  for (const adv of post.advisories) {
    console.log(`  advisory: ${adv}`);
  }
}

run("A. normal auto-config", false);
run("B. strict auto-config", true);

console.log("\n" + "=".repeat(60));
console.log("takeaway");
console.log("=".repeat(60));
console.log(
  "Use strict=true when you need reproducible, deterministic configs " +
    "across environments (CI, prod). Postflight still emits diagnostic " +
    "signals, but the pipeline will not silently shift your threshold.",
);
