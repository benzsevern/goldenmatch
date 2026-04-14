/**
 * file.ts -- CSV/JSON/JSONL file I/O connector.
 *
 * Node-only: uses node:fs, node:path. NOT edge-safe.
 *
 * CSV parser rules (CRITICAL):
 *   - Quoted fields preserve embedded commas and newlines
 *   - Doubled quotes inside quoted fields unescape to a single quote
 *   - Empty unquoted fields become null
 *   - Leading-zero strings (zip codes "01234", SSNs, phones) are NEVER
 *     coerced to numbers
 *   - Booleans "true"/"false" (case-insensitive) coerce to boolean
 *   - Numeric strings coerce to number only when fully numeric and not
 *     leading-zero
 *   - Supports both `\n` and `\r\n` line endings
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, extname, dirname } from "node:path";
import type { Row } from "../../core/types.js";

// ---------------------------------------------------------------------------
// Value coercion
// ---------------------------------------------------------------------------

/**
 * Coerce a raw CSV field string to string | number | boolean.
 * Preserves leading-zero strings (zip codes, SSNs, phone numbers) as strings.
 */
function coerceValue(raw: string): string | number | boolean {
  // Booleans (case insensitive)
  const lower = raw.toLowerCase();
  if (lower === "true") return true;
  if (lower === "false") return false;

  // Only try numeric coercion for strings that are already trimmed
  if (raw.length === 0 || raw !== raw.trim()) return raw;

  // NEVER coerce leading-zero strings to numbers, except "0" itself and
  // decimals like "0.5".
  if (raw.length > 1 && raw[0] === "0" && raw[1] !== ".") return raw;

  // Also preserve negative leading-zero strings like "-0123" as strings
  if (raw.length > 2 && raw[0] === "-" && raw[1] === "0" && raw[2] !== ".") {
    return raw;
  }

  const n = Number(raw);
  if (Number.isFinite(n)) return n;

  return raw;
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

/**
 * Parse a full CSV document honoring quoted fields, doubled quotes,
 * embedded newlines/commas, and both CRLF/LF line endings.
 *
 * Returns an array of raw string rows. The caller is responsible for
 * interpreting the first row as a header.
 */
function parseCsvDocument(content: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let current = "";
  let row: string[] = [];
  let inQuotes = false;
  let i = 0;
  const n = content.length;

  while (i < n) {
    const ch = content[i]!;

    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < n && content[i + 1] === '"') {
          current += '"';
          i += 2;
          continue;
        }
        inQuotes = false;
        i++;
        continue;
      }
      current += ch;
      i++;
      continue;
    }

    // Not in quotes
    if (ch === '"') {
      inQuotes = true;
      i++;
      continue;
    }
    if (ch === delimiter) {
      row.push(current);
      current = "";
      i++;
      continue;
    }
    if (ch === "\r") {
      // Swallow \r, then handle \n on next iteration
      i++;
      continue;
    }
    if (ch === "\n") {
      row.push(current);
      current = "";
      rows.push(row);
      row = [];
      i++;
      continue;
    }
    current += ch;
    i++;
  }

  // Flush trailing field/row if any
  if (current.length > 0 || row.length > 0) {
    row.push(current);
    rows.push(row);
  }

  return rows;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface ReadCsvOptions {
  readonly delimiter?: string;
  readonly hasHeader?: boolean;
  readonly encoding?: BufferEncoding;
}

/**
 * Read a CSV (or TSV, via `delimiter: "\t"`) file from disk.
 *
 * Returns an array of `Row` objects (header -> coerced value). If
 * `hasHeader` is false, synthetic headers `col_0`, `col_1`, ... are used.
 */
export function readCsv(path: string, options?: ReadCsvOptions): Row[] {
  const resolved = resolve(path);
  if (!existsSync(resolved)) {
    throw new Error(`File not found: ${resolved}`);
  }

  const delimiter = options?.delimiter ?? ",";
  const hasHeader = options?.hasHeader ?? true;
  const encoding = options?.encoding ?? "utf8";

  const content = readFileSync(resolved, encoding);
  // Strip UTF-8 BOM if present (Excel-exported CSVs often include it)
  const cleaned = content.charCodeAt(0) === 0xfeff ? content.slice(1) : content;

  const rawRows = parseCsvDocument(cleaned, delimiter);
  if (rawRows.length === 0) return [];

  // Skip completely empty rows (trailing newlines, blank lines between records)
  const nonEmpty = rawRows.filter(
    (r) => !(r.length === 1 && r[0] === ""),
  );
  if (nonEmpty.length === 0) return [];

  let headers: string[];
  let dataRows: string[][];
  if (hasHeader) {
    headers = nonEmpty[0]!.map((h) => h.trim());
    dataRows = nonEmpty.slice(1);
  } else {
    const width = nonEmpty[0]!.length;
    headers = Array.from({ length: width }, (_, i) => `col_${i}`);
    dataRows = nonEmpty;
  }

  const rows: Row[] = [];
  for (const raw of dataRows) {
    const record: Record<string, unknown> = {};
    for (let j = 0; j < headers.length; j++) {
      const field = raw[j] ?? "";
      record[headers[j]!] = field === "" ? null : coerceValue(field);
    }
    rows.push(record);
  }
  return rows;
}

/**
 * Read a JSON or JSONL file.
 *
 * - `.json`: expects an array of objects at the top level.
 * - `.jsonl` / `.ndjson`: one JSON object per line.
 *
 * Auto-detected based on whether the first non-whitespace character is `[`.
 */
export function readJson(path: string): Row[] {
  const resolved = resolve(path);
  if (!existsSync(resolved)) {
    throw new Error(`File not found: ${resolved}`);
  }
  const content = readFileSync(resolved, "utf8");
  const trimmed = content.trimStart();

  // JSONL: one object per line, detected by lack of opening `[`
  const ext = extname(resolved).toLowerCase();
  const isJsonl =
    ext === ".jsonl" ||
    ext === ".ndjson" ||
    (trimmed.length > 0 && trimmed[0] !== "[");

  if (isJsonl) {
    const rows: Row[] = [];
    const lines = content.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]!.trim();
      if (line === "") continue;
      try {
        const parsed = JSON.parse(line);
        if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
          rows.push(parsed as Row);
        } else {
          throw new Error(
            `JSONL line ${i + 1}: expected an object, got ${Array.isArray(parsed) ? "array" : typeof parsed}`,
          );
        }
      } catch (err) {
        if (err instanceof SyntaxError) {
          throw new Error(`JSONL parse error at line ${i + 1}: ${err.message}`);
        }
        throw err;
      }
    }
    return rows;
  }

  // Standard JSON array
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`JSON parse error in ${resolved}: ${msg}`);
  }
  if (!Array.isArray(parsed)) {
    throw new Error(
      `JSON file ${resolved}: expected an array of objects at the root`,
    );
  }
  return parsed as Row[];
}

/**
 * Dispatch to readCsv / readJson based on file extension.
 *
 * Supported: `.csv`, `.tsv`, `.json`, `.jsonl`, `.ndjson`.
 */
export function readFile(path: string): Row[] {
  const ext = extname(path).toLowerCase();
  if (ext === ".csv") return readCsv(path, { delimiter: "," });
  if (ext === ".tsv") return readCsv(path, { delimiter: "\t" });
  if (ext === ".json" || ext === ".jsonl" || ext === ".ndjson") {
    return readJson(path);
  }
  throw new Error(
    `Unsupported file format: ${ext}. Supported: .csv, .tsv, .json, .jsonl, .ndjson`,
  );
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/** Escape a CSV field: quote if it contains delimiter, quote, or newline. */
function escapeCsvField(value: unknown, delimiter: string): string {
  if (value === null || value === undefined) return "";
  const s = String(value);
  const needsQuoting =
    s.includes(delimiter) ||
    s.includes('"') ||
    s.includes("\n") ||
    s.includes("\r");
  if (!needsQuoting) return s;
  return `"${s.replace(/"/g, '""')}"`;
}

export interface WriteCsvOptions {
  readonly columns?: readonly string[];
  readonly delimiter?: string;
}

/**
 * Write rows to a CSV file. Creates parent directories as needed.
 *
 * If `columns` is not supplied, the union of keys from all rows is used,
 * ordered by first appearance.
 */
export function writeCsv(
  path: string,
  rows: readonly Row[],
  options?: WriteCsvOptions,
): void {
  const resolved = resolve(path);
  const dir = dirname(resolved);
  if (dir && dir !== "." && !existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const delimiter = options?.delimiter ?? ",";

  let columns: string[];
  if (options?.columns && options.columns.length > 0) {
    columns = [...options.columns];
  } else if (rows.length === 0) {
    writeFileSync(resolved, "", "utf8");
    return;
  } else {
    const seen = new Set<string>();
    columns = [];
    for (const row of rows) {
      for (const key of Object.keys(row)) {
        if (!seen.has(key)) {
          seen.add(key);
          columns.push(key);
        }
      }
    }
  }

  const lines: string[] = [];
  lines.push(columns.map((c) => escapeCsvField(c, delimiter)).join(delimiter));
  for (const row of rows) {
    const fields = columns.map((c) => escapeCsvField(row[c], delimiter));
    lines.push(fields.join(delimiter));
  }
  writeFileSync(resolved, lines.join("\n") + "\n", "utf8");
}

/** Write rows to a JSON file as a pretty-printed array. */
export function writeJson(path: string, rows: readonly Row[]): void {
  const resolved = resolve(path);
  const dir = dirname(resolved);
  if (dir && dir !== "." && !existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  writeFileSync(resolved, JSON.stringify(rows, null, 2), "utf8");
}
