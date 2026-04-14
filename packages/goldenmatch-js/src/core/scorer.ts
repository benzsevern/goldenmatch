/**
 * scorer.ts — Fuzzy scoring module for GoldenMatch.
 * Edge-safe: no Node.js imports, pure TypeScript only.
 *
 * Ports goldenmatch/core/scorer.py. The Python version uses `rapidfuzz`
 * for vectorized NxN scoring. Here we implement all algorithms in pure TS.
 */

import type {
  Row,
  MatchkeyField,
  MatchkeyConfig,
  PairKey,
  ScoredPair,
  BlockResult,
} from "./types.js";
import { makeScoredPair } from "./types.js";
import { pairKey } from "./cluster.js";
import { applyTransforms, soundex } from "./transforms.js";

// ---------------------------------------------------------------------------
// Helper: coerce unknown to string | null
// ---------------------------------------------------------------------------

/** Convert unknown value to string or null. */
export function asString(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "string") return v;
  return String(v);
}

// ---------------------------------------------------------------------------
// Scoring algorithms — pure TS
// ---------------------------------------------------------------------------

/**
 * Jaro similarity between two strings.
 *
 * matchWindow = floor(max(lenA, lenB) / 2) - 1
 * Count matches (chars within window) and transpositions.
 * jaro = (m/lenA + m/lenB + (m - t/2) / m) / 3
 */
export function jaro(a: string, b: string): number {
  if (a === b) return 1.0;
  const lenA = a.length;
  const lenB = b.length;
  if (lenA === 0 || lenB === 0) return 0.0;

  const matchWindow = Math.max(Math.floor(Math.max(lenA, lenB) / 2) - 1, 0);

  const aMatched = new Uint8Array(lenA); // 0 = unmatched
  const bMatched = new Uint8Array(lenB);
  let matches = 0;

  // Find matching characters
  for (let i = 0; i < lenA; i++) {
    const lo = Math.max(0, i - matchWindow);
    const hi = Math.min(lenB - 1, i + matchWindow);
    for (let j = lo; j <= hi; j++) {
      if (bMatched[j] !== 0 || a[i] !== b[j]) continue;
      aMatched[i] = 1;
      bMatched[j] = 1;
      matches++;
      break;
    }
  }

  if (matches === 0) return 0.0;

  // Count transpositions
  let transpositions = 0;
  let k = 0;
  for (let i = 0; i < lenA; i++) {
    if (aMatched[i] === 0) continue;
    while (bMatched[k] === 0) k++;
    if (a[i] !== b[k]) transpositions++;
    k++;
  }

  return (
    (matches / lenA + matches / lenB + (matches - transpositions / 2) / matches) / 3
  );
}

/**
 * Jaro-Winkler similarity.
 * Adds a bonus for a common prefix of up to 4 characters, scaling factor 0.1.
 */
export function jaroWinkler(a: string, b: string): number {
  const jaroSim = jaro(a, b);
  if (jaroSim === 0.0) return 0.0;

  // Common prefix up to 4 chars
  const maxPrefix = Math.min(4, Math.min(a.length, b.length));
  let prefix = 0;
  for (let i = 0; i < maxPrefix; i++) {
    if (a[i] === b[i]) prefix++;
    else break;
  }

  return jaroSim + prefix * 0.1 * (1 - jaroSim);
}

/**
 * Levenshtein edit distance (classic DP, 2-row optimization).
 */
export function levenshteinDistance(a: string, b: string): number {
  const lenA = a.length;
  const lenB = b.length;
  if (lenA === 0) return lenB;
  if (lenB === 0) return lenA;

  // Two-row DP
  let prev = new Uint32Array(lenB + 1);
  let curr = new Uint32Array(lenB + 1);

  for (let j = 0; j <= lenB; j++) prev[j] = j;

  for (let i = 1; i <= lenA; i++) {
    curr[0] = i;
    for (let j = 1; j <= lenB; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j]! + 1,      // deletion
        curr[j - 1]! + 1,  // insertion
        prev[j - 1]! + cost, // substitution
      );
    }
    // Swap rows
    [prev, curr] = [curr, prev];
  }

  return prev[lenB]!;
}

/**
 * Normalized Levenshtein similarity: 1 - distance / max(lenA, lenB).
 */
export function levenshteinSimilarity(a: string, b: string): number {
  if (a === b) return 1.0;
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1.0;
  return 1 - levenshteinDistance(a, b) / maxLen;
}

/**
 * Indel (insertion+deletion) edit distance.
 *
 * Like Levenshtein but without substitutions — a substitution costs 2
 * (one delete + one insert) instead of 1. This matches the distance
 * metric used by rapidfuzz's Indel ratio, which underlies
 * `rapidfuzz.fuzz.token_sort_ratio` in Python.
 */
export function indelDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  const m = a.length;
  const n = b.length;
  let prev = new Uint32Array(n + 1);
  let curr = new Uint32Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      if (a.charCodeAt(i - 1) === b.charCodeAt(j - 1)) {
        curr[j] = prev[j - 1]!;
      } else {
        // Only insert or delete allowed — cost 1 each. No substitution.
        curr[j] = Math.min(prev[j]! + 1, curr[j - 1]! + 1);
      }
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n]!;
}

/**
 * Indel normalized similarity: `1 - d_indel / (len_a + len_b)`.
 * Matches rapidfuzz's `Indel.normalized_similarity`.
 */
export function indelSimilarity(a: string, b: string): number {
  const total = a.length + b.length;
  if (total === 0) return 1.0;
  return 1 - indelDistance(a, b) / total;
}

/**
 * Token sort ratio, rapidfuzz-compatible.
 *
 * Matches `rapidfuzz.fuzz.token_sort_ratio`:
 * 1. Lowercase both strings.
 * 2. Strip non-alphanumeric characters (replace with whitespace).
 * 3. Split on whitespace, drop empties, sort tokens, rejoin with single space.
 * 4. Compare via Indel normalized similarity (NOT Levenshtein).
 *
 * Python reference: for ("John Smith", "Smith Johnson") returns ~0.8571.
 */
export function tokenSortRatio(a: string, b: string): number {
  const normalize = (s: string): string =>
    s
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .sort()
      .join(" ");
  return indelSimilarity(normalize(a), normalize(b));
}

/**
 * Soundex match: 1.0 if soundex codes equal, else 0.0.
 */
export function soundexMatch(a: string, b: string): number {
  return soundex(a) === soundex(b) ? 1.0 : 0.0;
}

// ---------------------------------------------------------------------------
// Bloom filter / PPRL scorers
// ---------------------------------------------------------------------------

/** Convert a hex string to a Uint8Array of bytes. */
function hexToBytes(hex: string): Uint8Array {
  const len = hex.length >>> 1;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

/** Count the number of set bits (popcount) in a byte array. */
function popcount(bytes: Uint8Array): number {
  let count = 0;
  for (let i = 0; i < bytes.length; i++) {
    let b = bytes[i]!;
    // Brian Kernighan's algorithm
    while (b !== 0) {
      b &= b - 1;
      count++;
    }
  }
  return count;
}

/** Count set bits in bitwise AND of two byte arrays. */
function popcountAnd(a: Uint8Array, b: Uint8Array): number {
  const len = Math.min(a.length, b.length);
  let count = 0;
  for (let i = 0; i < len; i++) {
    let v = (a[i]! & b[i]!);
    while (v !== 0) {
      v &= v - 1;
      count++;
    }
  }
  return count;
}

/** Count set bits in bitwise OR of two byte arrays. */
function popcountOr(a: Uint8Array, b: Uint8Array): number {
  const maxLen = Math.max(a.length, b.length);
  let count = 0;
  for (let i = 0; i < maxLen; i++) {
    let v = ((a[i] ?? 0) | (b[i] ?? 0));
    while (v !== 0) {
      v &= v - 1;
      count++;
    }
  }
  return count;
}

/**
 * Dice coefficient on two hex-encoded bloom filters.
 * 2 * intersection / (popcount_a + popcount_b)
 */
export function diceCoefficient(a: string, b: string): number {
  const bytesA = hexToBytes(a);
  const bytesB = hexToBytes(b);
  const pcA = popcount(bytesA);
  const pcB = popcount(bytesB);
  const total = pcA + pcB;
  if (total === 0) return 0.0;
  const intersection = popcountAnd(bytesA, bytesB);
  return (2 * intersection) / total;
}

/**
 * Jaccard similarity on two hex-encoded bloom filters.
 * intersection / union of bits
 */
export function jaccardSimilarity(a: string, b: string): number {
  const bytesA = hexToBytes(a);
  const bytesB = hexToBytes(b);
  const intersection = popcountAnd(bytesA, bytesB);
  const union = popcountOr(bytesA, bytesB);
  if (union === 0) return 0.0;
  return intersection / union;
}

// ---------------------------------------------------------------------------
// Ensemble scorer
// ---------------------------------------------------------------------------

/**
 * Ensemble scorer: combines jaro_winkler, token_sort, and soundex_match * 0.8.
 * Takes element-wise max of all three.
 */
export function ensembleScore(a: string, b: string): number {
  const jw = jaroWinkler(a, b);
  const ts = tokenSortRatio(a, b);
  const sx = soundexMatch(a, b) * 0.8;
  return Math.max(jw, ts, sx);
}

// ---------------------------------------------------------------------------
// Public: scoreField
// ---------------------------------------------------------------------------

/**
 * Score two field values using the specified scorer.
 * Returns null if either value is null.
 */
export function scoreField(
  valA: string | null,
  valB: string | null,
  scorer: string,
): number | null {
  if (valA === null || valB === null) return null;

  switch (scorer) {
    case "exact":
      return valA === valB ? 1.0 : 0.0;
    case "jaro_winkler":
      return jaroWinkler(valA, valB);
    case "levenshtein":
      return levenshteinSimilarity(valA, valB);
    case "token_sort":
      return tokenSortRatio(valA, valB);
    case "soundex_match":
      return soundexMatch(valA, valB);
    case "dice":
      return diceCoefficient(valA, valB);
    case "jaccard":
      return jaccardSimilarity(valA, valB);
    case "ensemble":
      return ensembleScore(valA, valB);
    default:
      throw new Error(`Unknown scorer: ${JSON.stringify(scorer)}`);
  }
}

// ---------------------------------------------------------------------------
// Public: scorePair
// ---------------------------------------------------------------------------

/**
 * Score a pair of rows across all fields using weighted aggregation.
 * Fields that produce null scores are excluded. If all null -> 0.0.
 */
export function scorePair(
  rowA: Row,
  rowB: Row,
  fields: readonly MatchkeyField[],
): number {
  let weightedSum = 0;
  let weightSum = 0;
  for (const f of fields) {
    const valA = applyTransforms(asString(rowA[f.field]), f.transforms);
    const valB = applyTransforms(asString(rowB[f.field]), f.transforms);
    const fieldScore = scoreField(valA, valB, f.scorer);
    if (fieldScore !== null) {
      weightedSum += fieldScore * f.weight;
      weightSum += f.weight;
    }
  }
  return weightSum === 0 ? 0 : weightedSum / weightSum;
}

// ---------------------------------------------------------------------------
// NxN score matrix
// ---------------------------------------------------------------------------

/**
 * Build an NxN score matrix for a list of values using a scorer.
 * Symmetric: matrix[i][j] === matrix[j][i]. Diagonal is 0.
 */
export function scoreMatrix(
  values: (string | null)[],
  scorerName: string,
): number[][] {
  const n = values.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const s = scoreField(values[i]!, values[j]!, scorerName) ?? 0;
      matrix[i]![j] = s;
      matrix[j]![i] = s;
    }
  }
  return matrix;
}

// ---------------------------------------------------------------------------
// Exact score matrix (hash-based grouping, O(n))
// ---------------------------------------------------------------------------

function exactScoreMatrix(values: (string | null)[]): number[][] {
  const n = values.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  // Group indices by value
  const groups = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v != null) {
      const existing = groups.get(v);
      if (existing !== undefined) {
        existing.push(i);
      } else {
        groups.set(v, [i]);
      }
    }
  }
  groups.forEach((indices) => {
    if (indices.length > 1) {
      for (let a = 0; a < indices.length; a++) {
        for (let b = a + 1; b < indices.length; b++) {
          matrix[indices[a]!]![indices[b]!] = 1.0;
          matrix[indices[b]!]![indices[a]!] = 1.0;
        }
      }
    }
  });
  return matrix;
}

/** Soundex score matrix: group by soundex code, 1.0 for same code. */
function soundexScoreMatrix(values: (string | null)[]): number[][] {
  const codes = values.map((v) => (v !== null ? soundex(v) : null));
  return exactScoreMatrix(codes);
}

/** Ensemble score matrix: max of jaro_winkler, token_sort, soundex*0.8 */
function ensembleScoreMatrix(values: (string | null)[]): number[][] {
  const n = values.length;
  const clean = values.map((v) => v ?? "");
  const jw: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const ts: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const sx = soundexScoreMatrix(values);
  const result: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (values[i] === null || values[j] === null) continue;
      jw[i]![j] = jaroWinkler(clean[i]!, clean[j]!);
      jw[j]![i] = jw[i]![j]!;
      ts[i]![j] = tokenSortRatio(clean[i]!, clean[j]!);
      ts[j]![i] = ts[i]![j]!;
    }
  }

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const val = Math.max(jw[i]![j]!, ts[i]![j]!, sx[i]![j]! * 0.8);
      result[i]![j] = val;
      result[j]![i] = val;
    }
  }
  return result;
}

/**
 * Build an NxN null mask: true where either value is null.
 */
function buildNullMask(values: (string | null)[]): boolean[][] {
  const n = values.length;
  const mask: boolean[][] = Array.from({ length: n }, () => new Array<boolean>(n).fill(false));
  for (let i = 0; i < n; i++) {
    if (values[i] === null) {
      for (let j = 0; j < n; j++) {
        mask[i]![j] = true;
        mask[j]![i] = true;
      }
    }
  }
  return mask;
}

/**
 * Build the appropriate score matrix for a scorer name.
 */
function buildScoreMatrix(values: (string | null)[], scorerName: string): number[][] {
  switch (scorerName) {
    case "exact":
      return exactScoreMatrix(values);
    case "soundex_match":
      return soundexScoreMatrix(values);
    case "ensemble":
      return ensembleScoreMatrix(values);
    default:
      return scoreMatrix(values, scorerName);
  }
}

// ---------------------------------------------------------------------------
// Get transformed values for a field from block rows
// ---------------------------------------------------------------------------

function getTransformedValues(
  rows: readonly Row[],
  field: MatchkeyField,
): (string | null)[] {
  return rows.map((row) => {
    const raw = asString(row[field.field]);
    return applyTransforms(raw, field.transforms);
  });
}

// ---------------------------------------------------------------------------
// Public: findExactMatches
// ---------------------------------------------------------------------------

/**
 * Find exact matches by grouping rows on matchkey columns.
 * Builds a composite key from all matchkey fields (with transforms applied),
 * groups rows sharing the same key, and returns all pairs with score 1.0.
 *
 * Rows must have a `__row_id__` field.
 */
export function findExactMatches(
  rows: readonly Row[],
  mk: MatchkeyConfig,
): ScoredPair[] {
  if (rows.length < 2) return [];

  // Build composite matchkey for each row
  const groups = new Map<string, number[]>();
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    const rowId = row["__row_id__"] as number;
    // Build key from all fields
    let keyParts: (string | null)[] = [];
    let hasNull = false;
    for (const f of mk.fields) {
      const raw = asString(row[f.field]);
      const transformed = applyTransforms(raw, f.transforms);
      if (transformed === null) {
        hasNull = true;
        break;
      }
      keyParts.push(transformed);
    }
    // Skip rows with any null field (nulls don't match)
    if (hasNull) continue;

    const key = keyParts.join("\x00"); // null byte separator
    const existing = groups.get(key);
    if (existing !== undefined) {
      existing.push(rowId);
    } else {
      groups.set(key, [rowId]);
    }
  }

  // Extract pairs from groups
  const pairs: ScoredPair[] = [];
  groups.forEach((members) => {
    if (members.length < 2) return;
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        pairs.push(makeScoredPair(members[i]!, members[j]!, 1.0));
      }
    }
  });
  return pairs;
}

// ---------------------------------------------------------------------------
// Public: findFuzzyMatches
// ---------------------------------------------------------------------------

/**
 * Find fuzzy matches within a block of rows (NxN scoring).
 *
 * Implements early termination:
 * - Score cheap fields (exact/soundex) first
 * - Check if max possible score can reach threshold
 * - Score expensive fuzzy fields only for promising pairs
 *
 * Rows must have a `__row_id__` field.
 */
export function findFuzzyMatches(
  rows: readonly Row[],
  mk: MatchkeyConfig,
  excludePairs?: ReadonlySet<PairKey>,
  preScoredPairs?: readonly ScoredPair[],
): ScoredPair[] {
  // findFuzzyMatches only runs for weighted/probabilistic matchkeys
  // (exact is handled via findExactMatches). Exact has no threshold.
  const threshold = mk.type === "exact" ? 1.0 : (mk.threshold ?? 0.85);

  // Fast path: pre-scored pairs (from ANN blocking)
  if (preScoredPairs !== undefined) {
    const results: ScoredPair[] = [];
    for (const p of preScoredPairs) {
      if (p.score < threshold) continue;
      const idA = Math.min(p.idA, p.idB);
      const idB = Math.max(p.idA, p.idB);
      const key = pairKey(idA, idB);
      if (excludePairs !== undefined && excludePairs.has(key)) continue;
      results.push(makeScoredPair(idA, idB, p.score));
    }
    return results;
  }

  const n = rows.length;
  if (n < 2) return [];

  const rowIds = rows.map((r) => r["__row_id__"] as number);

  // Separate cheap (exact + soundex) from expensive (fuzzy) fields
  const cheapFields = mk.fields.filter(
    (f) => f.scorer === "exact" || f.scorer === "soundex_match",
  );
  const fuzzyFields = mk.fields.filter(
    (f) => f.scorer !== "exact" && f.scorer !== "soundex_match" && f.scorer !== "record_embedding",
  );

  const totalWeight = mk.fields.reduce((sum, f) => sum + f.weight, 0);
  if (totalWeight === 0) return [];

  // Phase 1: Score cheap fields and build null masks
  // cheapNumerator[i][j] = sum(fieldScore * weight) for cheap fields
  // cheapDenominator[i][j] = sum(weight) for non-null cheap fields
  const cheapNumerator: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const cheapDenominator: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));

  for (const f of cheapFields) {
    const values = getTransformedValues(rows, f);
    const nullMask = buildNullMask(values);
    const scores =
      f.scorer === "exact"
        ? exactScoreMatrix(values)
        : soundexScoreMatrix(values);

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (!nullMask[i]![j]!) {
          cheapNumerator[i]![j]! += scores[i]![j]! * f.weight;
          cheapNumerator[j]![i]! = cheapNumerator[i]![j]!;
          cheapDenominator[i]![j]! += f.weight;
          cheapDenominator[j]![i]! = cheapDenominator[i]![j]!;
        }
      }
    }
  }

  // Phase 2: Early termination check
  const fuzzyTotalWeight = fuzzyFields.reduce((sum, f) => sum + f.weight, 0);

  // Track which pairs are impossible (can't reach threshold)
  const impossible: boolean[][] = Array.from({ length: n }, () => new Array<boolean>(n).fill(false));

  let combined: number[][];

  if (fuzzyFields.length === 0) {
    // No fuzzy fields — just use cheap scores
    combined = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        combined[i]![j] =
          cheapDenominator[i]![j]! > 0
            ? cheapNumerator[i]![j]! / cheapDenominator[i]![j]!
            : 0;
        combined[j]![i] = combined[i]![j]!;
      }
    }
  } else {
    // Check which pairs can possibly reach threshold
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const maxNum = cheapNumerator[i]![j]! + fuzzyTotalWeight;
        const maxDen = cheapDenominator[i]![j]! + fuzzyTotalWeight;
        const maxPossible = maxDen > 0 ? maxNum / maxDen : 0;
        if (maxPossible < threshold) {
          impossible[i]![j] = true;
          impossible[j]![i] = true;
        }
      }
    }

    // Phase 3: Score fuzzy fields with intra-field early termination
    const fuzzyNumerator: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    const fuzzyDenominator: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));

    for (let fIdx = 0; fIdx < fuzzyFields.length; fIdx++) {
      const f = fuzzyFields[fIdx]!;
      const values = getTransformedValues(rows, f);
      const nullMask = buildNullMask(values);
      const scores = buildScoreMatrix(values, f.scorer);

      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (!nullMask[i]![j]!) {
            fuzzyNumerator[i]![j]! += scores[i]![j]! * f.weight;
            fuzzyNumerator[j]![i] = fuzzyNumerator[i]![j]!;
            fuzzyDenominator[i]![j]! += f.weight;
            fuzzyDenominator[j]![i] = fuzzyDenominator[i]![j]!;
          }
        }
      }

      // Intra-field early termination: check if any pair can still reach threshold
      const remainingWeight = fuzzyFields
        .slice(fIdx + 1)
        .reduce((sum, ff) => sum + ff.weight, 0);

      if (remainingWeight > 0) {
        let anyCanReach = false;
        for (let i = 0; i < n && !anyCanReach; i++) {
          for (let j = i + 1; j < n && !anyCanReach; j++) {
            if (impossible[i]![j]!) continue;
            const totalNum =
              cheapNumerator[i]![j]! + fuzzyNumerator[i]![j]! + remainingWeight;
            const totalDen =
              cheapDenominator[i]![j]! + fuzzyDenominator[i]![j]! + remainingWeight;
            const bestPossible = totalDen > 0 ? totalNum / totalDen : 0;
            if (bestPossible >= threshold) {
              anyCanReach = true;
            }
          }
        }
        if (!anyCanReach) break; // No pair can reach threshold — skip remaining fields
      }
    }

    // Combine cheap + fuzzy
    combined = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (impossible[i]![j]!) {
          combined[i]![j] = 0;
        } else {
          const totalNum = cheapNumerator[i]![j]! + fuzzyNumerator[i]![j]!;
          const totalDen = cheapDenominator[i]![j]! + fuzzyDenominator[i]![j]!;
          combined[i]![j] = totalDen > 0 ? totalNum / totalDen : 0;
        }
        combined[j]![i] = combined[i]![j]!;
      }
    }
  }

  // Extract upper triangle pairs above threshold
  const results: ScoredPair[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const score = combined[i]![j]!;
      if (score < threshold) continue;
      const idA = Math.min(rowIds[i]!, rowIds[j]!);
      const idB = Math.max(rowIds[i]!, rowIds[j]!);
      const key = pairKey(idA, idB);
      if (excludePairs !== undefined && excludePairs.has(key)) continue;
      results.push(makeScoredPair(idA, idB, score));
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Public: scoreBlocksSequential
// ---------------------------------------------------------------------------

export interface ScoreBlocksOptions {
  /** Filter to cross-source pairs only. */
  readonly acrossFilesOnly?: boolean;
  /** Row ID -> source name mapping (for acrossFilesOnly). */
  readonly sourceLookup?: ReadonlyMap<number, string>;
  /** Target IDs for match mode — filter to target/ref cross pairs. */
  readonly targetIds?: ReadonlySet<number>;
}

/**
 * Score all blocks sequentially.
 *
 * In JS there is no GIL, so we use sequential scoring as the default.
 * For web workers or similar concurrency, the caller can partition blocks.
 */
export function scoreBlocksSequential(
  blocks: readonly BlockResult[],
  mk: MatchkeyConfig,
  matchedPairs: Set<PairKey>,
  options?: ScoreBlocksOptions,
): ScoredPair[] {
  if (blocks.length === 0) return [];

  const acrossFilesOnly = options?.acrossFilesOnly ?? false;
  const sourceLookup = options?.sourceLookup;
  const targetIds = options?.targetIds;

  const allPairs: ScoredPair[] = [];

  for (const block of blocks) {
    // For cross-file mode, check that block has records from multiple sources
    if (acrossFilesOnly && sourceLookup !== undefined) {
      const sourcesInBlock = new Set<string>();
      for (const row of block.rows) {
        const src = sourceLookup.get(row["__row_id__"] as number);
        if (src !== undefined) sourcesInBlock.add(src);
      }
      if (sourcesInBlock.size < 2) continue;
    }

    // Use a frozen copy of matchedPairs for consistency
    const excludeSnapshot: ReadonlySet<PairKey> = new Set(matchedPairs);

    let pairs = findFuzzyMatches(
      block.rows,
      mk,
      excludeSnapshot,
      block.preScoredPairs,
    );

    // Cross-file filter
    if (acrossFilesOnly && sourceLookup !== undefined) {
      pairs = pairs.filter((p) => {
        const srcA = sourceLookup.get(p.idA);
        const srcB = sourceLookup.get(p.idB);
        return srcA !== srcB;
      });
    }

    // Target/ref cross filter for match mode
    if (targetIds !== undefined) {
      pairs = pairs.filter(
        (p) => targetIds.has(p.idA) !== targetIds.has(p.idB),
      );
    }

    for (const p of pairs) {
      allPairs.push(p);
      matchedPairs.add(pairKey(p.idA, p.idB));
    }
  }

  return allPairs;
}

// ---------------------------------------------------------------------------
// Utility: canonicalize pair key
// ---------------------------------------------------------------------------

// Re-export pairKey from cluster.ts — single canonical source of truth.
export { pairKey };
