/**
 * transforms.ts — Pure field transform utilities.
 * Edge-safe: no Node.js imports, no `process`.
 *
 * Ports goldenmatch/utils/transforms.py.
 */

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Apply a single named transform to a value. Returns null if input is null. */
export function applyTransform(
  value: string | null,
  transform: string,
): string | null {
  if (value === null) return null;

  // Handle parameterized transforms (substring:start:end, qgram:n, bloom_filter:...)
  if (transform.startsWith("substring:")) return applySubstring(value, transform);
  if (transform.startsWith("qgram:")) return applyQgram(value, transform);
  if (transform.startsWith("bloom_filter")) return applyBloomFilter(value, transform);

  switch (transform) {
    case "lowercase":
      return value.toLowerCase();
    case "uppercase":
      return value.toUpperCase();
    case "strip":
      return value.trim();
    case "strip_all":
      return value.replace(/\s+/g, "");
    case "digits_only":
      return value.replace(/\D/g, "");
    case "alpha_only":
      return value.replace(/[^a-zA-Z]/g, "");
    case "normalize_whitespace":
      return value.trim().replace(/\s+/g, " ");
    case "token_sort":
      return value
        .trim()
        .split(/\s+/)
        .sort()
        .join(" ");
    case "first_token":
      return value.trim().split(/\s+/)[0] ?? "";
    case "last_token": {
      const tokens = value.trim().split(/\s+/);
      return tokens[tokens.length - 1] ?? "";
    }
    case "soundex":
      return soundex(value);
    case "metaphone":
      return metaphone(value);
    default:
      return value;
  }
}

/** Apply a chain of transforms in order. */
export function applyTransforms(
  value: string | null,
  transforms: readonly string[],
): string | null {
  let result = value;
  for (const t of transforms) {
    result = applyTransform(result, t);
    if (result === null) return null;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Parameterized transforms
// ---------------------------------------------------------------------------

/** substring:start:end */
function applySubstring(value: string, transform: string): string {
  const parts = transform.split(":");
  const start = parseInt(parts[1] ?? "0", 10);
  const end = parts[2] !== undefined ? parseInt(parts[2], 10) : undefined;
  return value.slice(start, end);
}

/** qgram:n — split into character n-grams, sorted and space-separated. */
function applyQgram(value: string, transform: string): string {
  const parts = transform.split(":");
  const n = parseInt(parts[1] ?? "2", 10);
  if (n <= 0 || value.length < n) return value;
  const grams: string[] = [];
  for (let i = 0; i <= value.length - n; i++) {
    grams.push(value.slice(i, i + n));
  }
  return grams.sort().join(" ");
}

// ---------------------------------------------------------------------------
// Soundex — Robert Russell's algorithm
// ---------------------------------------------------------------------------

const SOUNDEX_MAP: Record<string, string> = {
  B: "1", F: "1", P: "1", V: "1",
  C: "2", G: "2", J: "2", K: "2", Q: "2", S: "2", X: "2", Z: "2",
  D: "3", T: "3",
  L: "4",
  M: "5", N: "5",
  R: "6",
};

/**
 * American Soundex (Robert Russell, 1918).
 * 1. Keep first letter
 * 2. Map consonants to digits (B/F/P/V->1, C/G/J/K/Q/S/X/Z->2, D/T->3, L->4, M/N->5, R->6)
 * 3. Remove adjacent duplicates, vowels/H/W
 * 4. Pad/truncate to 4 chars
 *
 * H and W are transparent — they do NOT reset the duplicate suppression.
 * Vowels (A/E/I/O/U/Y) DO reset, so "Pfister" and "Jackson" work correctly.
 */
export function soundex(value: string): string {
  const clean = value.toUpperCase().replace(/[^A-Z]/g, "");
  if (clean.length === 0) return "0000";

  const firstLetter = clean[0]!;
  let code = firstLetter;
  let lastDigit = SOUNDEX_MAP[firstLetter] ?? "0";

  for (let i = 1; i < clean.length && code.length < 4; i++) {
    const ch = clean[i]!;
    const digit = SOUNDEX_MAP[ch];
    if (digit && digit !== lastDigit) {
      code += digit;
      lastDigit = digit;
    } else if (!digit) {
      // Vowel / H / W / Y — only H and W are transparent (do NOT reset)
      if (ch !== "H" && ch !== "W") {
        lastDigit = "0";
      }
    }
  }

  return (code + "0000").slice(0, 4);
}

// ---------------------------------------------------------------------------
// Simplified Metaphone (Lawrence Philips, 1990)
// ---------------------------------------------------------------------------

/**
 * Simplified Metaphone.
 * Returns a phonetic code of up to 4 characters.
 */
export function metaphone(value: string): string {
  let word = value.toUpperCase().replace(/[^A-Z]/g, "");
  if (word.length === 0) return "";

  // Drop initial silent letter pairs
  const dropPrefixes = ["AE", "GN", "KN", "PN", "WR"];
  for (const prefix of dropPrefixes) {
    if (word.startsWith(prefix)) {
      word = word.slice(1);
      break;
    }
  }

  // Drop trailing MB (silent B)
  if (word.endsWith("MB")) {
    word = word.slice(0, -1);
  }

  let code = "";
  let i = 0;

  while (i < word.length && code.length < 4) {
    const ch = word[i]!;
    const next = word[i + 1] ?? "";
    const prev = i > 0 ? word[i - 1]! : "";

    // Skip duplicate adjacent letters (except C)
    if (ch === prev && ch !== "C") {
      i++;
      continue;
    }

    switch (ch) {
      case "A":
      case "E":
      case "I":
      case "O":
      case "U":
        // Vowels only kept at the beginning
        if (i === 0) code += ch;
        break;

      case "B":
        code += "B";
        break;

      case "C":
        if (next === "I" || next === "E" || next === "Y") {
          code += "S";
        } else {
          code += "K";
        }
        break;

      case "D":
        if (next === "G" && "IEY".includes(word[i + 2] ?? "")) {
          code += "J";
        } else {
          code += "T";
        }
        break;

      case "F":
        code += "F";
        break;

      case "G":
        if (next === "H" && i + 2 < word.length && !"AEIOU".includes(word[i + 2] ?? "")) {
          // GH before non-vowel is silent
          i += 2;
          continue;
        } else if (i > 0 && (next === "N" || (next === "N" && word[i + 2] === "E" && i + 2 === word.length - 1))) {
          // GN at end is silent — skip
        } else if (prev === "G") {
          // Already handled double G
        } else if (next === "I" || next === "E" || next === "Y") {
          code += "J";
        } else {
          code += "K";
        }
        break;

      case "H":
        if ("AEIOU".includes(next) && !"AEIOU".includes(prev)) {
          code += "H";
        }
        break;

      case "J":
        code += "J";
        break;

      case "K":
        if (prev !== "C") code += "K";
        break;

      case "L":
        code += "L";
        break;

      case "M":
        code += "M";
        break;

      case "N":
        code += "N";
        break;

      case "P":
        if (next === "H") {
          code += "F";
          i++;
        } else {
          code += "P";
        }
        break;

      case "Q":
        code += "K";
        break;

      case "R":
        code += "R";
        break;

      case "S":
        if (next === "H" || (next === "I" && (word[i + 2] === "O" || word[i + 2] === "A"))) {
          code += "X";
          i++;
        } else if (next === "C" && word[i + 2] === "H") {
          code += "SK";
          i += 2;
        } else {
          code += "S";
        }
        break;

      case "T":
        if (next === "H") {
          code += "0"; // theta
          i++;
        } else if (next === "I" && (word[i + 2] === "O" || word[i + 2] === "A")) {
          code += "X";
        } else {
          code += "T";
        }
        break;

      case "V":
        code += "F";
        break;

      case "W":
      case "Y":
        if ("AEIOU".includes(next)) {
          code += ch;
        }
        break;

      case "X":
        code += "KS";
        break;

      case "Z":
        code += "S";
        break;

      default:
        break;
    }
    i++;
  }

  return code.slice(0, 4);
}

// ---------------------------------------------------------------------------
// Bloom filter transform (pure TS, for PPRL)
// ---------------------------------------------------------------------------

/**
 * Security level presets for bloom filter parameters.
 * standard: 256-bit, 20 hash functions, 2-grams
 * high:     512-bit, 30 hash functions, 2-grams, HMAC salting
 * paranoid: 1024-bit, 40 hash functions, 3-grams, HMAC salting, balanced padding
 */
interface BloomPreset {
  readonly size: number;
  readonly k: number;
  readonly ngram: number;
  readonly salt: boolean;
  readonly padding: boolean;
}

const BLOOM_PRESETS: Record<string, BloomPreset> = {
  standard: { size: 256, k: 20, ngram: 2, salt: false, padding: false },
  high:     { size: 512, k: 30, ngram: 2, salt: true,  padding: false },
  paranoid: { size: 1024, k: 40, ngram: 3, salt: true,  padding: true },
};

/**
 * bloom_filter[:level[:salt]]
 * e.g. "bloom_filter", "bloom_filter:high", "bloom_filter:paranoid:mysecret"
 */
function applyBloomFilter(value: string, transform: string): string {
  const parts = transform.split(":");
  const levelName = parts[1] ?? "standard";
  const salt = parts[2] ?? "";
  const preset = BLOOM_PRESETS[levelName] ?? BLOOM_PRESETS["standard"]!;

  const ngrams = generateNgrams(value.toLowerCase(), preset.ngram);

  // Initialize bit array
  const bits = new Uint8Array(Math.ceil(preset.size / 8));

  for (const gram of ngrams) {
    const input = preset.salt && salt ? `${salt}:${gram}` : gram;
    // Double hashing scheme: h(i) = h1 + i*h2
    const h1 = fnv1a32(input);
    const h2 = fnv1a32(input + "\x00");
    for (let i = 0; i < preset.k; i++) {
      const pos = Math.abs((h1 + i * h2) | 0) % preset.size;
      bits[pos >>> 3]! |= 1 << (pos & 7);
    }
  }

  // Balanced padding: set deterministic extra bits so all filters have similar density
  if (preset.padding) {
    const targetDensity = 0.5;
    let setBits = 0;
    for (let i = 0; i < preset.size; i++) {
      if (bits[i >>> 3]! & (1 << (i & 7))) setBits++;
    }
    const targetSet = Math.floor(preset.size * targetDensity);
    if (setBits < targetSet) {
      let padSeed = fnv1a32("pad:" + value);
      let added = 0;
      while (setBits + added < targetSet) {
        padSeed = (padSeed * 1103515245 + 12345) | 0;
        const pos = Math.abs(padSeed) % preset.size;
        if (!(bits[pos >>> 3]! & (1 << (pos & 7)))) {
          bits[pos >>> 3]! |= 1 << (pos & 7);
          added++;
        }
      }
    }
  }

  return hexEncode(bits);
}

/** Generate character n-grams from a string. */
function generateNgrams(value: string, n: number): string[] {
  if (value.length < n) return [value];
  const result: string[] = [];
  for (let i = 0; i <= value.length - n; i++) {
    result.push(value.slice(i, i + n));
  }
  return result;
}

/**
 * FNV-1a 32-bit hash — fast, well-distributed, pure JS.
 * Used for bloom filter bit positions (not cryptographic).
 */
function fnv1a32(input: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return hash >>> 0;
}

/** Hex-encode a Uint8Array. */
function hexEncode(bytes: Uint8Array): string {
  const hex: string[] = [];
  for (let i = 0; i < bytes.length; i++) {
    hex.push(bytes[i]!.toString(16).padStart(2, "0"));
  }
  return hex.join("");
}
