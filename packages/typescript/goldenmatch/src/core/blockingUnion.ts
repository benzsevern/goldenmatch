/**
 * blockingUnion.ts — the #1207 strong-identifier blocking union DECISION logic,
 * the TS port of the shared `autoconfig-core` kernel
 * (`autoconfig-core/src/select_blocking.rs`).
 *
 * This is increment 2a of moving blocking-key selection into the shared core: the
 * pure two-phase decision (`assembleStrongIdUnion` + `finalizeStrongIdUnion`),
 * byte-for-byte with the Rust core and pinned to it by the cross-surface golden
 * fixture (`tests/parity/fixtures/select-blocking/`, also checked by the Rust
 * golden test + the wasm shims). Blocking selection is data-dependent, so the
 * split mirrors the core: the host MEASURES the row-level signals (OR-coverage,
 * per-pass block size), these pure functions GATE.
 *
 * **Name detection uses `classifyByName` (increment 2b).** The core's `assemble`
 * derives name-column detection from `classify_by_name` (a name-*pattern*-only
 * classifier — bare `first`/`last` are NOT names, only `first_name`/`surname`
 * are), which differs from TS's data-aware `classifyColumn`. This port therefore
 * calls `classifyByName(c.name)` (the fixture-pinned pure-TS port of the core's
 * `classify_by_name`) for name detection, so it matches Python exactly and does
 * not over-fire when wired into the always-on `buildBlocking`. Strong-id / geo
 * detection still uses the data-aware `colType`, matching the core.
 */

import { classifyByName } from "./classifyByName.js";

/** `autoconfig.py::_STRONG_EXACT_TYPES` — the strong-identifier col_types.
 *  `exact_derived` (#2876/#2881) is a domain-extracted exact-scored column
 *  (e.g. `__title_key__`) — the same strong-identity signal as
 *  identifier/email/phone for blocking purposes. */
const STRONG_EXACT_TYPES = new Set(["identifier", "email", "phone", "exact_derived"]);
/** `_UNION_PASS_MIN_NONNULL` — a per-id pass must block more than a handful. */
const UNION_PASS_MIN_NONNULL = 0.02;
/** `_BLOCKING_UNION_COVERAGE_TARGET`. */
export const BLOCKING_UNION_COVERAGE_TARGET = 0.95;

/** One column's profile signals the union decision needs (mirrors the core's
 *  `BlockingColumnInput`). `colType` is the data-aware classifier's snake_case
 *  verdict (used for strong-id / geo detection); name detection is derived
 *  internally from the column NAME via `classifyByName`, exactly as the core
 *  derives it from `classify_by_name` — so the boundary stays `(name, colType,
 *  nullRate, cardinalityRatio)` with no extra host input. */
export interface UnionColumn {
  readonly name: string;
  readonly colType: string;
  readonly nullRate: number;
  readonly cardinalityRatio: number;
}

/** A candidate pass. `isStrongId` marks the single-field strong-id singletons. */
export interface UnionPass {
  readonly fields: readonly string[];
  readonly transforms: readonly string[];
  readonly isStrongId: boolean;
}

/** The emitted union config shape (mirrors the core's `BlockingConfigOut`). */
export interface UnionConfigOut {
  readonly strategy: "multi_pass";
  readonly keys: readonly UnionPass[];
  readonly passes: readonly UnionPass[];
  readonly maxBlockSize: number;
  readonly skipOversized: true;
}

function isStrong(colType: string): boolean {
  return STRONG_EXACT_TYPES.has(colType);
}

/** `_transforms_for(fields)` — email/exact_derived → `[lowercase, strip]`,
 *  else `[strip]`. */
function transformsForField(field: string, cols: readonly UnionColumn[]): string[] {
  const c = cols.find((x) => x.name === field);
  return c && (c.colType === "email" || c.colType === "exact_derived")
    ? ["lowercase", "strip"]
    : ["strip"];
}

/**
 * Phase 1 — assemble the candidate union passes from column profiles (pure).
 * Faithful port of `assemble_strong_id_union`. `null` unless ≥1 strong-id pass
 * AND ≥2 distinct passes survive assembly.
 */
export function assembleStrongIdUnion(
  cols: readonly UnionColumn[],
): UnionPass[] | null {
  const passes: UnionPass[] = [];
  let strongIdCount = 0;

  for (const c of cols) {
    if (!isStrong(c.colType)) continue;
    const nonnull = 1.0 - c.nullRate;
    if (nonnull < UNION_PASS_MIN_NONNULL) continue;
    // #876 surrogate guard: a perfect-surrogate id (card_ratio >= 1.0) makes
    // singleton blocks. blocking_max_ratio is deliberately NOT applied here.
    if (c.cardinalityRatio >= 1.0) continue;
    passes.push({
      fields: [c.name],
      transforms: transformsForField(c.name, cols),
      isStrongId: true,
    });
    strongIdCount += 1;
  }

  if (strongIdCount < 1) return null;

  // name+geo passes for rows missing every strong id. Name detection uses the
  // name-*pattern*-only `classifyByName` (the core's internal `classify_by_name`,
  // Python `_classify_by_name`), NOT the data-aware `colType` — bare `first`/
  // `last` are NOT names, so the union does not over-fire on them (the #1317
  // 2a→2b crux). Strong-id / geo detection still uses the data-aware `colType`.
  const nameCols = cols.filter((c) => classifyByName(c.name) === "name");
  const first = nameCols.find((c) => c.name.toLowerCase().includes("first"))?.name;
  const last = nameCols.find((c) => {
    const n = c.name.toLowerCase();
    return n.includes("last") || n.includes("surname");
  })?.name;
  const geo = cols.find((c) => c.colType === "zip" || c.colType === "geo")?.name;

  if (first !== undefined && last !== undefined) {
    passes.push({
      fields: [first, last],
      transforms: transformsForField(first, cols),
      isStrongId: false,
    });
  }
  if (last !== undefined && geo !== undefined) {
    passes.push({
      fields: [last, geo],
      transforms: transformsForField(last, cols),
      isStrongId: false,
    });
  }

  if (passes.length < 2) return null;
  return passes;
}

/**
 * Phase 2 — apply the gates and emit the `multi_pass` union config (pure).
 * Faithful port of `finalize_strong_id_union`. `null` (fall through) when the
 * coverage target is not cleared, or < 2 passes survive, or no strong-id
 * survives. `passSurvives[i]` is the host's scale-safety verdict for `passes[i]`.
 */
export function finalizeStrongIdUnion(
  passes: readonly UnionPass[],
  coverage: number,
  passSurvives: readonly boolean[],
  maxSafeBlock: number,
): UnionConfigOut | null {
  if (coverage < BLOCKING_UNION_COVERAGE_TARGET) return null;
  if (passSurvives.length !== passes.length) return null;
  const survivors = passes.filter((_, i) => passSurvives[i]);
  const anyStrongId = survivors.some((p) => p.isStrongId);
  if (!anyStrongId || survivors.length < 2) return null;
  return {
    strategy: "multi_pass",
    keys: [survivors[0]!],
    passes: survivors,
    maxBlockSize: maxSafeBlock,
    skipOversized: true,
  };
}
