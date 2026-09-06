//! Blocking-key selection — the #1207 strong-identifier union decision, shared
//! across surfaces (no pyo3).
//!
//! Parity port of:
//!   - `packages/python/goldenmatch/goldenmatch/core/autoconfig.py`
//!     `_build_strong_identifier_union` (~line 1583) — phase 1 (assembly), and
//!   - the `build_blocking` union call-site survivor filtering (~line 3145) —
//!     phase 2 (gates + config emission).
//!
//! Blocking selection is data-dependent (the gates need row-level OR-coverage +
//! per-pass block-size), unlike the pure-profile planner/classifier. The split
//! (see `docs/superpowers/specs/2026-07-19-blocking-selection-native-core-design.md`):
//! the **host measures** those signals; the **core decides**. So the flow is two
//! phases mirroring Python's helper-then-call-site structure:
//!
//! 1. [`assemble_strong_id_union`] — pure assembly from column profiles.
//! 2. host measures OR-coverage of the assembled passes + per-pass scale-safety.
//! 3. [`finalize_strong_id_union`] — pure gates (coverage, survivor filter,
//!    re-gate) → the emitted `multi_pass` config.
//!
//! All thresholds/branch-orders are reproduced from the Python source.

use serde::{Deserialize, Serialize};

use crate::classify::{classify_by_name, ColType};

/// `autoconfig.py::_STRONG_EXACT_TYPES` — the strong-identifier col_types that
/// back a per-id blocking pass. `ExactDerived` (#2876/#2881) is included for
/// the same reason as the Python side: a domain-extracted exact-scored column
/// (e.g. `__title_key__`) is exactly this kind of strong identity signal for
/// blocking purposes, even though it is not itself an "identifier" column.
pub const STRONG_EXACT_TYPES: [ColType; 4] = [
    ColType::Identifier,
    ColType::Email,
    ColType::Phone,
    ColType::ExactDerived,
];

/// `_UNION_PASS_MIN_NONNULL` — a per-id pass must block more than a trivial
/// handful of rows (non-null fraction floor).
const UNION_PASS_MIN_NONNULL: f64 = 0.02;

/// `_BLOCKING_UNION_COVERAGE_TARGET` — the assembled passes' OR-coverage must
/// clear this for the union to be admitted. The threshold lives in the core; the
/// host only supplies the *measured* coverage.
pub const BLOCKING_UNION_COVERAGE_TARGET: f64 = 0.95;

/// One column's profile signals the union decision needs. `col_type` is the full
/// classifier's verdict (`classify_columns`); name-column detection for the
/// name+geo passes is done in-core via [`classify_by_name`], so no extra host
/// input is required.
#[derive(Debug, Clone, Deserialize)]
pub struct BlockingColumnInput {
    pub name: String,
    pub col_type: ColType,
    pub null_rate: f64,
    pub cardinality_ratio: f64,
}

/// A blocking pass. `is_strong_id` marks the single-field strong-identifier
/// singletons (the ones phase 2 gates on the NON-NULL block size); it is carried
/// on the wire so the host can measure the right scale-safety signal per pass and
/// so phase 2 can re-gate on "≥1 surviving strong-id".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionPass {
    pub fields: Vec<String>,
    pub transforms: Vec<String>,
    pub is_strong_id: bool,
}

fn is_strong(ct: ColType) -> bool {
    STRONG_EXACT_TYPES.contains(&ct)
}

/// `_transforms_for(fields)` — email/exact_derived → `[lowercase, strip]`,
/// else `[strip]`, keyed on the col_type of `fields[0]`.
fn transforms_for_field(field: &str, cols: &[BlockingColumnInput]) -> Vec<String> {
    let is_email = cols
        .iter()
        .find(|c| c.name == field)
        .map(|c| matches!(c.col_type, ColType::Email | ColType::ExactDerived))
        .unwrap_or(false);
    if is_email {
        vec!["lowercase".to_string(), "strip".to_string()]
    } else {
        vec!["strip".to_string()]
    }
}

/// Phase 1 — assemble the candidate union passes from column profiles (pure).
///
/// Returns `None` unless ≥1 strong-id pass is present AND ≥2 distinct passes
/// survive assembly. The ≥1-strong-id requirement keeps this from emitting a
/// name-only "strong-id union" (a name-only shape belongs to the name fallback).
/// The OR-coverage gate is deferred to phase 2 (it needs the host measurement).
///
/// One pass per strong-identifier column above the non-null floor (excluding a
/// #876 perfect-surrogate, `cardinality_ratio >= 1.0`, which makes singleton
/// blocks), then `[first, last]` and `[last, geo]` for rows missing every strong
/// id — matching `_build_strong_identifier_union`.
pub fn assemble_strong_id_union(cols: &[BlockingColumnInput]) -> Option<Vec<UnionPass>> {
    let mut passes: Vec<UnionPass> = Vec::new();
    let mut strong_id_count = 0usize;

    // one pass per strong-identifier field, above the non-null population floor.
    for c in cols {
        if !is_strong(c.col_type) {
            continue;
        }
        let nonnull = 1.0 - c.null_rate;
        if nonnull < UNION_PASS_MIN_NONNULL {
            continue;
        }
        // #876 surrogate guard: a perfect-surrogate id (card_ratio >= 1.0) makes
        // singleton blocks (0 pairs). NOTE: blocking_max_ratio is deliberately
        // NOT applied — the union exists precisely to use near-unique-but-
        // repeating ids the single-key gate rejects.
        if c.cardinality_ratio >= 1.0 {
            continue;
        }
        passes.push(UnionPass {
            fields: vec![c.name.clone()],
            transforms: transforms_for_field(&c.name, cols),
            is_strong_id: true,
        });
        strong_id_count += 1;
    }

    // require ≥1 strong-id pass; a name-only shape belongs to the name fallback.
    if strong_id_count < 1 {
        return None;
    }

    // name+geo passes for rows missing every strong id. name columns are detected
    // by the same name-classifier Python uses (`_classify_by_name(name) == name`).
    let first: Option<String> = cols
        .iter()
        .filter(|c| classify_by_name(&c.name) == Some(ColType::Name))
        .find(|c| c.name.to_lowercase().contains("first"))
        .map(|c| c.name.clone());
    let last: Option<String> = cols
        .iter()
        .filter(|c| classify_by_name(&c.name) == Some(ColType::Name))
        .find(|c| {
            let n = c.name.to_lowercase();
            n.contains("last") || n.contains("surname")
        })
        .map(|c| c.name.clone());
    let geo: Option<String> = cols
        .iter()
        .find(|c| matches!(c.col_type, ColType::Zip | ColType::Geo))
        .map(|c| c.name.clone());

    if let (Some(f), Some(l)) = (&first, &last) {
        passes.push(UnionPass {
            fields: vec![f.clone(), l.clone()],
            transforms: transforms_for_field(f, cols),
            is_strong_id: false,
        });
    }
    if let (Some(l), Some(g)) = (&last, &geo) {
        passes.push(UnionPass {
            fields: vec![l.clone(), g.clone()],
            transforms: transforms_for_field(l, cols),
            is_strong_id: false,
        });
    }

    if passes.len() < 2 {
        return None;
    }
    Some(passes)
}

/// Phase 2 input: the assembled passes plus the host measurements. `coverage` is
/// the OR-coverage of `passes` (a multi-field pass counts a row only when ALL its
/// fields are non-null). `pass_survives[i]` is the per-pass scale-safety verdict
/// the host measured — for a strong-id singleton, the NON-NULL projected block
/// size ≤ `max_safe_block` (the runtime blocker drops null keys); for a name/geo
/// pass, the standard bounded gate. `max_safe_block` is the host's scale budget.
#[derive(Debug, Clone, Deserialize)]
pub struct UnionFinalizeInput {
    pub passes: Vec<UnionPass>,
    pub coverage: f64,
    pub pass_survives: Vec<bool>,
    pub max_safe_block: usize,
}

/// The emitted blocking config (the shape `build_blocking` returns for the
/// union). `keys` is `[primary]` (the first surviving pass).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BlockingConfigOut {
    pub strategy: String,
    pub keys: Vec<UnionPass>,
    pub passes: Vec<UnionPass>,
    pub max_block_size: usize,
    pub skip_oversized: bool,
}

/// Phase 2 — apply the gates and emit the `multi_pass` union config (pure).
///
/// `None` (fall through to the name/compound fallback) when the coverage target
/// is not cleared, or fewer than 2 passes survive scale-safety, or no strong-id
/// pass survives. Mirrors `_build_strong_identifier_union`'s coverage gate + the
/// `build_blocking` call-site survivor filtering. Because the assembled order is
/// already `[strong-id…, name/geo…]`, filtering it in place preserves Python's
/// `surviving_ids + surviving_other` ordering.
pub fn finalize_strong_id_union(input: &UnionFinalizeInput) -> Option<BlockingConfigOut> {
    if input.coverage < BLOCKING_UNION_COVERAGE_TARGET {
        return None;
    }
    // Defensive: a mismatched survives vector can't be trusted — fall through.
    if input.pass_survives.len() != input.passes.len() {
        return None;
    }
    let survivors: Vec<UnionPass> = input
        .passes
        .iter()
        .zip(input.pass_survives.iter())
        .filter(|(_, &survives)| survives)
        .map(|(p, _)| p.clone())
        .collect();
    let any_strong_id = survivors.iter().any(|p| p.is_strong_id);
    if !any_strong_id || survivors.len() < 2 {
        return None;
    }
    Some(BlockingConfigOut {
        strategy: "multi_pass".to_string(),
        keys: vec![survivors[0].clone()],
        passes: survivors,
        max_block_size: input.max_safe_block,
        skip_oversized: true,
    })
}
