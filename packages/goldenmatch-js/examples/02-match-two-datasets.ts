/**
 * Match records in a "target" dataset against a "reference" dataset.
 * Useful for: incoming leads vs. CRM, transactions vs. customers, etc.
 *
 * Run: npx tsx examples/02-match-two-datasets.ts
 */
import { match } from "goldenmatch";

// Reference dataset: known customers
const customers = [
  { id: "C001", name: "Acme Corp",     city: "Seattle",  phone: "555-1000" },
  { id: "C002", name: "Globex Inc",    city: "Portland", phone: "555-2000" },
  { id: "C003", name: "Initech LLC",   city: "Austin",   phone: "555-3000" },
  { id: "C004", name: "Umbrella Co",   city: "Boston",   phone: "555-4000" },
];

// Target dataset: incoming leads (possibly dupes of customers, possibly new)
const leads = [
  { id: "L1", name: "ACME Corporation", city: "Seattle",  phone: "555-1000" },
  { id: "L2", name: "Globex, Inc.",     city: "Portland", phone: "555-2000" },
  { id: "L3", name: "Stark Industries", city: "New York", phone: "555-9000" },
  { id: "L4", name: "Initech",          city: "Austin",   phone: "555-3000" },
];

const result = match(leads, customers, {
  fuzzy: { name: 0.75 },
  blocking: ["city"],
  threshold: 0.75,
});

console.log(`Matched leads:   ${result.matched.length}`);
console.log(`Unmatched leads: ${result.unmatched.length}\n`);

console.log("Matched (likely existing customers):");
for (const row of result.matched) {
  console.log(" ", row);
}

console.log("\nUnmatched (likely new leads):");
for (const row of result.unmatched) {
  console.log(" ", row);
}

/**
 * Python -> TS differences:
 *   - Python `match()` returns a MatchResult with DataFrames; TS returns arrays.
 *   - `result.stats` is a generic record; shape differs slightly across modes.
 */
