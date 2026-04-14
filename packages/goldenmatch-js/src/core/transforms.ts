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
// Bloom filter transform (pure TS, for PPRL) — SHA-256 parity with Python
// ---------------------------------------------------------------------------

/**
 * Security level presets for bloom filter parameters (match Python exactly).
 *
 * standard: 512-bit, 20 hash functions, 2-grams
 * high:     1024-bit, 30 hash functions, 2-grams, HMAC-SHA256 salting
 * paranoid: 2048-bit, 40 hash functions, 3-grams, HMAC-SHA256 + balanced padding
 *
 * See goldenmatch/utils/transforms.py::_bloom_filter_transform for the
 * reference algorithm we match.
 */
interface BloomPreset {
  readonly size: number;
  readonly k: number;
  readonly ngram: number;
  readonly hmac: boolean;
  readonly balanced: boolean;
}

const BLOOM_PRESETS: Record<string, BloomPreset> = {
  standard: { size: 512,  k: 20, ngram: 2, hmac: false, balanced: false },
  high:     { size: 1024, k: 30, ngram: 2, hmac: true,  balanced: false },
  paranoid: { size: 2048, k: 40, ngram: 3, hmac: true,  balanced: true },
};

/** Default parameters when called as plain "bloom_filter" (matches Python). */
const BLOOM_DEFAULTS = { size: 1024, k: 20, ngram: 2 };

/** Default HMAC key used by the high/paranoid presets (matches Python). */
const BLOOM_DEFAULT_HMAC_KEY = "default_field_key";

/**
 * Build a CLK (Cryptographic Longterm Key) bloom filter hex string.
 *
 * Forms accepted:
 *   - "bloom_filter"                        -> defaults (1024/20/2, no hmac)
 *   - "bloom_filter:standard"               -> preset
 *   - "bloom_filter:high[:customKey]"       -> preset, optional HMAC key override
 *   - "bloom_filter:paranoid[:customKey]"   -> preset, optional HMAC key override
 *   - "bloom_filter:<ngram>:<k>:<size>[:hmac_key]"  -> fully parametric
 */
function applyBloomFilter(value: string, transform: string): string {
  let ngramSize = BLOOM_DEFAULTS.ngram;
  let numHashes = BLOOM_DEFAULTS.k;
  let filterSize = BLOOM_DEFAULTS.size;
  let hmacKey: string | null = null;
  let balanced = false;

  if (transform === "bloom_filter") {
    // defaults
  } else {
    const parts = transform.split(":");
    const maybeLevel = parts[1];
    if (maybeLevel && (maybeLevel in BLOOM_PRESETS)) {
      const preset = BLOOM_PRESETS[maybeLevel]!;
      ngramSize = preset.ngram;
      numHashes = preset.k;
      filterSize = preset.size;
      balanced = preset.balanced;
      if (preset.hmac) {
        // Allow per-field HMAC key override via bloom_filter:<level>:<key>
        hmacKey = parts[2] && parts[2].length > 0 ? parts[2] : BLOOM_DEFAULT_HMAC_KEY;
      }
    } else {
      // Parametric form: bloom_filter:<ngram>:<k>:<size>[:hmac_key]
      ngramSize = parseInt(parts[1] ?? String(BLOOM_DEFAULTS.ngram), 10);
      numHashes = parseInt(parts[2] ?? String(BLOOM_DEFAULTS.k), 10);
      filterSize = parseInt(parts[3] ?? String(BLOOM_DEFAULTS.size), 10);
      if (parts.length > 4 && parts[4]!.length > 0) {
        hmacKey = parts[4]!;
      }
    }
  }

  const filterBytes = Math.floor(filterSize / 8);
  const bits = new Uint8Array(filterBytes);

  // Match Python: value.lower().strip(), left-pad with '_' up to ngramSize.
  let padded = value.toLowerCase().trim();
  if (padded.length < ngramSize) {
    padded = padded.padEnd(ngramSize, "_");
  }

  // Balanced padding: deterministic salt append to normalize filter density.
  if (balanced && padded.length < 8) {
    const salt = sha256Hex(padded).slice(0, 8);
    padded = padded + salt;
  }

  // Generate character n-grams.
  const ngrams: string[] = [];
  for (let i = 0; i <= padded.length - ngramSize; i++) {
    ngrams.push(padded.slice(i, i + ngramSize));
  }

  // Hash each n-gram k times.
  for (const ngram of ngrams) {
    for (let k = 0; k < numHashes; k++) {
      const hex = hmacKey
        ? hmacSha256Hex(`${hmacKey}:${k}`, ngram)
        : sha256Hex(`${k}:${ngram}`);
      // bit_pos = int(h, 16) % filter_size
      const bitPos = Number(modHexBigInt(hex, filterSize));
      bits[bitPos >> 3]! |= 1 << (bitPos & 7);
    }
  }

  return hexEncode(bits);
}

/** Compute (BigInt hex) mod (Number) and return a non-negative Number result. */
function modHexBigInt(hex: string, modulus: number): number {
  // filterSize fits comfortably in a Number; use BigInt only for the big hex.
  const big = BigInt("0x" + hex);
  const mod = BigInt(modulus);
  const rem = big % mod;
  return Number(rem);
}

/** Hex-encode a Uint8Array. */
function hexEncode(bytes: Uint8Array): string {
  const hex: string[] = [];
  for (let i = 0; i < bytes.length; i++) {
    hex.push(bytes[i]!.toString(16).padStart(2, "0"));
  }
  return hex.join("");
}

// ---------------------------------------------------------------------------
// SHA-256 — pure TS, edge-safe (no node: imports)
// ---------------------------------------------------------------------------

// FIPS 180-4 round constants.
const K256 = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotr32(x: number, n: number): number {
  return ((x >>> n) | (x << (32 - n))) >>> 0;
}

/** UTF-8 encode a string to bytes (edge-safe — uses TextEncoder). */
function utf8Encode(input: string): Uint8Array {
  return new TextEncoder().encode(input);
}

/** SHA-256 core: digest a byte array, return 32-byte digest. */
function sha256Bytes(msg: Uint8Array): Uint8Array {
  // Initial hash values (FIPS 180-4).
  const H = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);

  // Pre-processing: append 1 bit, k zero bits, 64-bit length (big-endian).
  const msgLen = msg.length;
  const bitLen = msgLen * 8;
  const withPadLen = ((msgLen + 9 + 63) >> 6) << 6; // round up to 64-byte block
  const padded = new Uint8Array(withPadLen);
  padded.set(msg);
  padded[msgLen] = 0x80;
  // Length in bits as 64-bit big-endian at the end.
  // High 32 bits are ~0 for any realistic input in JS; write bitLen in low 32.
  const hi = Math.floor(bitLen / 0x100000000);
  const lo = bitLen >>> 0;
  const dv = new DataView(padded.buffer);
  dv.setUint32(withPadLen - 8, hi, false);
  dv.setUint32(withPadLen - 4, lo, false);

  const W = new Uint32Array(64);

  for (let offset = 0; offset < withPadLen; offset += 64) {
    // Schedule
    for (let t = 0; t < 16; t++) {
      W[t] = dv.getUint32(offset + t * 4, false);
    }
    for (let t = 16; t < 64; t++) {
      const w15 = W[t - 15]!;
      const w2 = W[t - 2]!;
      const s0 = rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >>> 3);
      const s1 = rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >>> 10);
      W[t] = (W[t - 16]! + s0 + W[t - 7]! + s1) >>> 0;
    }

    let a = H[0]!, b = H[1]!, c = H[2]!, d = H[3]!;
    let e = H[4]!, f = H[5]!, g = H[6]!, h = H[7]!;

    for (let t = 0; t < 64; t++) {
      const S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
      const ch = (e & f) ^ (~e & g);
      const T1 = (h + S1 + ch + K256[t]! + W[t]!) >>> 0;
      const S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
      const mj = (a & b) ^ (a & c) ^ (b & c);
      const T2 = (S0 + mj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + T1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (T1 + T2) >>> 0;
    }

    H[0] = (H[0]! + a) >>> 0;
    H[1] = (H[1]! + b) >>> 0;
    H[2] = (H[2]! + c) >>> 0;
    H[3] = (H[3]! + d) >>> 0;
    H[4] = (H[4]! + e) >>> 0;
    H[5] = (H[5]! + f) >>> 0;
    H[6] = (H[6]! + g) >>> 0;
    H[7] = (H[7]! + h) >>> 0;
  }

  const out = new Uint8Array(32);
  const outDv = new DataView(out.buffer);
  for (let i = 0; i < 8; i++) outDv.setUint32(i * 4, H[i]!, false);
  return out;
}

/**
 * SHA-256 digest of a UTF-8 string, returned as lowercase 64-char hex.
 *
 * Matches Python `hashlib.sha256(s.encode()).hexdigest()` bit-for-bit.
 */
export function sha256Hex(input: string): string {
  return hexEncode(sha256Bytes(utf8Encode(input)));
}

/**
 * HMAC-SHA256(key, msg) as lowercase 64-char hex.
 *
 * Matches Python `hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()`.
 */
export function hmacSha256Hex(key: string, msg: string): string {
  const blockSize = 64;
  let keyBytes = utf8Encode(key);
  if (keyBytes.length > blockSize) {
    keyBytes = sha256Bytes(keyBytes);
  }
  const kPad = new Uint8Array(blockSize);
  kPad.set(keyBytes);

  const oKeyPad = new Uint8Array(blockSize);
  const iKeyPad = new Uint8Array(blockSize);
  for (let i = 0; i < blockSize; i++) {
    oKeyPad[i] = kPad[i]! ^ 0x5c;
    iKeyPad[i] = kPad[i]! ^ 0x36;
  }

  const msgBytes = utf8Encode(msg);
  const inner = new Uint8Array(blockSize + msgBytes.length);
  inner.set(iKeyPad);
  inner.set(msgBytes, blockSize);
  const innerHash = sha256Bytes(inner);

  const outer = new Uint8Array(blockSize + innerHash.length);
  outer.set(oKeyPad);
  outer.set(innerHash, blockSize);
  return hexEncode(sha256Bytes(outer));
}
