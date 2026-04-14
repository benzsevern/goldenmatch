/**
 * String scoring: compare every scorer on the same string pairs.
 *
 * This is pedagogical — each scorer has different strengths:
 *   - jaro_winkler: short strings, common-prefix bonus (great for names)
 *   - levenshtein:  edit distance (typos)
 *   - token_sort:   order-independent (word reordering)
 *   - soundex_match: phonetic (Smith/Smyth)
 *   - dice / jaccard: bigram/set overlap
 *   - ensemble:     weighted combination (default for names)
 *
 * Run: npx tsx examples/04-string-scoring.ts
 */
import { scoreStrings } from "goldenmatch";

const pairs: [string, string, string][] = [
  ["John Smith",    "Jon Smith",        "typo / common name variant"],
  ["John Smith",    "Smith, John",      "word reorder"],
  ["Smith",         "Smyth",            "phonetic equivalent"],
  ["Robert",        "Bob",              "nickname (no scorer handles well)"],
  ["123 Main St",   "123 Main Street",  "abbreviation"],
  ["apple inc",     "Apple, Inc.",      "punctuation/case noise"],
  ["totally",       "different",        "no similarity"],
];

const scorers = [
  "exact",
  "jaro_winkler",
  "levenshtein",
  "token_sort",
  "soundex_match",
  "dice",
  "jaccard",
  "ensemble",
];

// Header
const pad = (s: string, n: number) => s.padEnd(n);
const colWidths = [28, 28, ...scorers.map(() => 14)];

process.stdout.write(pad("A", colWidths[0]!));
process.stdout.write(pad("B", colWidths[1]!));
for (let i = 0; i < scorers.length; i++) {
  process.stdout.write(pad(scorers[i]!, colWidths[i + 2]!));
}
process.stdout.write("\n");
process.stdout.write("-".repeat(colWidths.reduce((a, b) => a + b, 0)) + "\n");

for (const [a, b, _label] of pairs) {
  process.stdout.write(pad(a, colWidths[0]!));
  process.stdout.write(pad(b, colWidths[1]!));
  for (let i = 0; i < scorers.length; i++) {
    const score = scoreStrings(a, b, scorers[i]!);
    process.stdout.write(pad(score.toFixed(2), colWidths[i + 2]!));
  }
  process.stdout.write("\n");
}

/**
 * Expected: jaro_winkler typically wins on short names, token_sort crushes
 * word-reorder cases, soundex_match catches Smith/Smyth, and ensemble is
 * the most balanced default.
 */
