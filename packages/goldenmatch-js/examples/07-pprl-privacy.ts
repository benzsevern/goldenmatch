/**
 * Privacy-preserving record linkage (PPRL) via bloom filter CLKs.
 *
 * Two parties want to find common records without revealing plaintext.
 * Each side encodes rows as bloom filters (CLKs) over agreed fields,
 * exchanges only the encoded bit-vectors, and scores via Dice similarity.
 *
 * Three security levels:
 *   - standard:  bloom filter only (fast, demo/low-risk data)
 *   - high:      HMAC-SHA256 per-party salt (requires coordinated salt)
 *   - paranoid:  balanced padding + HMAC (resists frequency analysis)
 *
 * Run: npx tsx examples/07-pprl-privacy.ts
 */
import { runPPRL, autoConfigurePPRL } from "goldenmatch";

// Party A (e.g., hospital)
const partyA = [
  { name: "John Smith",   dob: "1980-01-15", city: "Seattle" },
  { name: "Jane Doe",     dob: "1975-06-22", city: "Portland" },
  { name: "Bob Johnson",  dob: "1990-11-03", city: "Austin" },
  { name: "Alice Miller", dob: "1985-03-18", city: "Boston" },
];

// Party B (e.g., insurance) -- some overlap with A
const partyB = [
  { name: "Jon Smith",    dob: "1980-01-15", city: "Seattle" },    // same as A[0]
  { name: "Jane Doe",     dob: "1975-06-22", city: "Portland" },   // same as A[1]
  { name: "Carol Young",  dob: "1992-08-09", city: "Denver" },     // new
];

// --- Standard security: trusted-third-party protocol ---
console.log("=== Standard security (trusted third party) ===");
const stdResult = runPPRL(partyA, partyB, {
  fields: ["name", "dob", "city"],
  securityLevel: "standard",
  protocol: "trusted_third_party",
  threshold: 0.85,
});
for (const m of stdResult.matches) {
  console.log(`  A[${m.idA}] <-> B[${m.idB}]  score=${m.score.toFixed(3)}`);
}

// --- High security: salted HMAC ---
console.log("\n=== High security (HMAC salt) ===");
const highResult = runPPRL(partyA, partyB, {
  fields: ["name", "dob", "city"],
  securityLevel: "high",
  protocol: "trusted_third_party",
  threshold: 0.85,
  salt: "shared-secret-agreed-upon-out-of-band",
});
console.log(`  ${highResult.matches.length} matches found`);

// --- Paranoid + SMC stub: salted, balanced padding, SMC protocol ---
console.log("\n=== Paranoid + SMC protocol ===");
const smcResult = runPPRL(partyA, partyB, {
  fields: ["name", "dob", "city"],
  securityLevel: "paranoid",
  protocol: "smc",
  threshold: 0.85,
  salt: "shared-secret-agreed-upon-out-of-band",
});
console.log(`  ${smcResult.matches.length} matches found`);

// --- Auto-configure: let GoldenMatch pick the fields / threshold ---
console.log("\n=== Auto-configured ===");
const autoConfig = autoConfigurePPRL(partyA, partyB);
console.log(`  Auto-picked fields: ${autoConfig.fields.join(", ")}`);
console.log(`  Threshold: ${autoConfig.threshold}`);

/**
 * NB: standard level leaks frequency info -- names appearing many times
 * produce identifiable bit patterns. Use "high" or "paranoid" for
 * anything beyond demos.
 */
