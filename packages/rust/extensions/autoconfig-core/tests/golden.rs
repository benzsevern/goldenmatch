//! Golden-vector parity harness: Rust `decide_plan` + `classify_columns` vs
//! the Python oracle (`scripts/gen_autoconfig_golden.py`).
//!
//! Each vector is `{input, expected}` in JSON. We deserialize the input,
//! run the Rust function, re-serialize the Rust output to `serde_json::Value`,
//! and compare it against the `expected` Value — so any serde rename / Option
//! modelling mismatch surfaces as a test failure rather than a silent success.
//!
//! Key serde contracts verified by these tests:
//!   - `BackendName` → `"polars-direct"` / `"bucket"` / `"chunked"` / `"ray"` / `"duckdb"`
//!   - `ClusteringStrategy` → `"in_memory"` / `"partitioned_union_find"` / `"streaming_cc"`
//!   - `ColType` → `"email"` / `"name"` / `"phone"` / `"zip"` / `"address"` / `"geo"` / `"identifier"` / `"description"` / `"numeric"` / `"date"` / `"string"` / `"year"` / `"multi_name"` / `"exact_derived"`
//!   - `pair_spill_threshold` and `chunk_size` serialize to JSON `null` (NOT `"none"`
//!     or absent) for rules that leave them as `None` (pathological / simple /
//!     fast_box / bucket_suggested / user_override).
//!   - `SpillThreshold` → `"ram"` / `"duckdb"` / `"disk_per_worker"`
//!
//! Floating-point comparison:
//!   `confidence` and other f64 fields are compared with a 1e-9 absolute tolerance
//!   because Python's float arithmetic (e.g. `0.7 + 0.2 = 0.8999999999999999`) can
//!   differ from Rust's by ULP. We convert the oracle's `expected.confidence` to f64
//!   and compare against Rust's before re-encoding to JSON.

use goldenmatch_autoconfig_core::{
    assemble_strong_id_union, classify_by_name, classify_columns, decide_plan,
    exact_matchkey_floor, extrapolate_pair_count, finalize_strong_id_union, sparse_match_floor,
    BlockingColumnInput, ColType, ColumnStats, ExtrapolationInput, PlannerInput, UnionFinalizeInput,
};
use serde_json::Value;

// ── Embedded golden files (committed alongside this test) ─────────────────────
const PLANNER_JSON: &str = include_str!("../golden/planner_vectors.json");
const CLASSIFIER_JSON: &str = include_str!("../golden/classifier_vectors.json");
const EXTRAPOLATION_JSON: &str = include_str!("../golden/extrapolation_vectors.json");
const SPARSE_MATCH_FLOOR_JSON: &str = include_str!("../golden/sparse_match_floor_vectors.json");
const EXACT_MATCHKEY_FLOOR_JSON: &str = include_str!("../golden/exact_matchkey_floor_vectors.json");

// ── Planner parity ────────────────────────────────────────────────────────────

#[test]
fn planner_golden_parity() {
    let vectors: Vec<Value> =
        serde_json::from_str(PLANNER_JSON).expect("failed to parse planner_vectors.json");

    assert!(
        vectors.len() >= 40,
        "expected >= 40 planner vectors, got {}",
        vectors.len()
    );

    let mut failures: Vec<String> = Vec::new();

    for (idx, vec) in vectors.iter().enumerate() {
        let input_val = &vec["input"];
        let expected_val = &vec["expected"];

        // Deserialize the input
        let input: PlannerInput = match serde_json::from_value(input_val.clone()) {
            Ok(i) => i,
            Err(e) => {
                failures.push(format!(
                    "vector[{idx}]: failed to deserialize PlannerInput: {e}\ninput={input_val}"
                ));
                continue;
            }
        };

        // Run Rust decide_plan
        let plan = decide_plan(&input);

        // Re-serialize to JSON Value for field-by-field comparison
        let got_val: Value = match serde_json::to_value(&plan) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!(
                    "vector[{idx}]: failed to serialize ExecutionPlan: {e}"
                ));
                continue;
            }
        };

        // Compare field by field (avoids f64 precision issues; no f64 in plan)
        let fields = [
            "backend",
            "max_workers",
            "pair_spill_threshold",
            "clustering_strategy",
            "rule_name",
            "chunk_size",
        ];

        for field in &fields {
            let got = &got_val[field];
            let exp = &expected_val[field];
            if got != exp {
                failures.push(format!(
                    "vector[{idx}] field `{field}` mismatch:\n  got      = {got}\n  expected = {exp}\n  input    = {input_val}\n  rule fired = {}",
                    got_val["rule_name"]
                ));
            }
        }

        // Explicit null checks for pair_spill_threshold and chunk_size
        // so any `Option::None` → absent (vs → null) regression is caught.
        let pst = &got_val["pair_spill_threshold"];
        if pst.is_null() != expected_val["pair_spill_threshold"].is_null() {
            failures.push(format!(
                "vector[{idx}] pair_spill_threshold null mismatch: got={pst} expected={}",
                expected_val["pair_spill_threshold"]
            ));
        }
        let cs = &got_val["chunk_size"];
        if cs.is_null() != expected_val["chunk_size"].is_null() {
            failures.push(format!(
                "vector[{idx}] chunk_size null mismatch: got={cs} expected={}",
                expected_val["chunk_size"]
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} planner vector(s) failed:\n{}",
            failures.len(),
            failures.join("\n---\n")
        );
    }
}

// ── Extrapolation parity (S1) ─────────────────────────────────────────────────

#[test]
fn extrapolation_golden_parity() {
    let vectors: Vec<Value> = serde_json::from_str(EXTRAPOLATION_JSON)
        .expect("failed to parse extrapolation_vectors.json");

    assert!(
        vectors.len() >= 30,
        "expected >= 30 extrapolation vectors, got {}",
        vectors.len()
    );

    let mut failures: Vec<String> = Vec::new();
    for (idx, vec) in vectors.iter().enumerate() {
        let input: ExtrapolationInput = match serde_json::from_value(vec["input"].clone()) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("vec {idx}: bad input json: {e}"));
                continue;
            }
        };
        let got: Value = match serde_json::to_value(extrapolate_pair_count(&input)) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("vec {idx}: serialize failed: {e}"));
                continue;
            }
        };
        for field in ["n_blocks", "total_comparisons", "singleton_block_count"] {
            if got[field] != vec["expected"][field] {
                failures.push(format!(
                    "vec {idx} field {field}: got {} exp {}",
                    got[field], vec["expected"][field]
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} extrapolation mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ── Sparse-match floor parity (S2b) ───────────────────────────────────────────

#[test]
fn sparse_match_floor_golden_parity() {
    let vectors: Vec<Value> = serde_json::from_str(SPARSE_MATCH_FLOOR_JSON)
        .expect("failed to parse sparse_match_floor_vectors.json");

    assert!(
        vectors.len() >= 15,
        "expected >= 15 sparse-match-floor vectors, got {}",
        vectors.len()
    );

    let mut failures: Vec<String> = Vec::new();
    for (idx, vec) in vectors.iter().enumerate() {
        let estimated_pairs = vec["input"]["estimated_pairs"]
            .as_u64()
            .expect("estimated_pairs must be u64");
        let got = sparse_match_floor(estimated_pairs);
        let exp = vec["expected"]["floor"]
            .as_u64()
            .expect("floor must be u64");
        if got != exp {
            failures.push(format!(
                "vec {idx}: estimated_pairs={estimated_pairs} got {got} exp {exp}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} sparse-match-floor mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ── Exact-matchkey floor parity (S3) ──────────────────────────────────────────

#[test]
fn exact_matchkey_floor_golden_parity() {
    let vectors: Vec<Value> = serde_json::from_str(EXACT_MATCHKEY_FLOOR_JSON)
        .expect("failed to parse exact_matchkey_floor_vectors.json");

    assert!(
        vectors.len() >= 13,
        "expected >= 13 exact-matchkey-floor vectors, got {}",
        vectors.len()
    );

    let mut failures: Vec<String> = Vec::new();
    for (idx, vec) in vectors.iter().enumerate() {
        let col_type = vec["input"]["col_type"]
            .as_str()
            .expect("col_type must be a string");
        let got = exact_matchkey_floor(col_type);
        let exp = vec["expected"]["floor"]
            .as_f64()
            .expect("floor must be f64");
        if (got - exp).abs() >= 1e-9 {
            failures.push(format!(
                "vec {idx}: col_type={col_type} got {got} exp {exp}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} exact-matchkey-floor mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ── Classifier parity ─────────────────────────────────────────────────────────

#[test]
fn classifier_golden_parity() {
    let vectors: Vec<Value> =
        serde_json::from_str(CLASSIFIER_JSON).expect("failed to parse classifier_vectors.json");

    assert!(
        vectors.len() >= 30,
        "expected >= 30 classifier vectors, got {}",
        vectors.len()
    );

    let mut failures: Vec<String> = Vec::new();

    for (idx, vec) in vectors.iter().enumerate() {
        let input_val = &vec["input"];
        let expected_val = &vec["expected"];

        // Deserialize the input into ColumnStats
        let col_stats: ColumnStats = match serde_json::from_value(input_val.clone()) {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!(
                    "vector[{idx}]: failed to deserialize ColumnStats: {e}\ninput={input_val}"
                ));
                continue;
            }
        };

        // Run Rust classify_columns (wrap in slice)
        let profiles = classify_columns(&[col_stats]);
        if profiles.len() != 1 {
            failures.push(format!(
                "vector[{idx}]: classify_columns returned {} profiles, expected 1",
                profiles.len()
            ));
            continue;
        }
        let profile = &profiles[0];

        // Re-serialize to JSON Value
        let got_val: Value = match serde_json::to_value(profile) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!(
                    "vector[{idx}]: failed to serialize ColumnProfile: {e}"
                ));
                continue;
            }
        };

        // Compare string / bool fields exactly
        let exact_fields = ["name", "dtype", "col_type", "needs_llm_escalation"];
        for field in &exact_fields {
            let got = &got_val[field];
            let exp = &expected_val[field];
            if got != exp {
                failures.push(format!(
                    "vector[{idx}] field `{field}` mismatch:\n  got      = {got}\n  expected = {exp}\n  col_name = {}",
                    input_val["name"]
                ));
            }
        }

        // Compare float fields with tolerance (Python float arithmetic can differ by ULP)
        let float_fields = ["confidence", "null_rate", "cardinality_ratio", "avg_len"];
        for field in &float_fields {
            let got_f = got_val[field].as_f64().unwrap_or(f64::NAN);
            let exp_f = expected_val[field].as_f64().unwrap_or(f64::NAN);
            if (got_f - exp_f).abs() > 1e-9 {
                failures.push(format!(
                    "vector[{idx}] field `{field}` mismatch:\n  got      = {got_f}\n  expected = {exp_f}\n  diff     = {}\n  col_name = {}",
                    (got_f - exp_f).abs(),
                    input_val["name"]
                ));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} classifier vector(s) failed:\n{}",
            failures.len(),
            failures.join("\n---\n")
        );
    }
}

// ── Explicit serde-shape spot-checks ─────────────────────────────────────────
//
// These tests guard the literal string values used in the JSON contract so a
// typo in a `serde` attribute is caught immediately rather than buried in a
// failure message.

#[test]
fn serde_backend_names() {
    use goldenmatch_autoconfig_core::BackendName;
    assert_eq!(
        serde_json::to_string(&BackendName::PolarsDirect).unwrap(),
        r#""polars-direct""#
    );
    assert_eq!(
        serde_json::to_string(&BackendName::Bucket).unwrap(),
        r#""bucket""#
    );
    assert_eq!(
        serde_json::to_string(&BackendName::Chunked).unwrap(),
        r#""chunked""#
    );
    assert_eq!(
        serde_json::to_string(&BackendName::Duckdb).unwrap(),
        r#""duckdb""#
    );
    assert_eq!(
        serde_json::to_string(&BackendName::Ray).unwrap(),
        r#""ray""#
    );
}

#[test]
fn serde_clustering_strategy_names() {
    use goldenmatch_autoconfig_core::ClusteringStrategy;
    assert_eq!(
        serde_json::to_string(&ClusteringStrategy::InMemory).unwrap(),
        r#""in_memory""#
    );
    assert_eq!(
        serde_json::to_string(&ClusteringStrategy::PartitionedUnionFind).unwrap(),
        r#""partitioned_union_find""#
    );
    assert_eq!(
        serde_json::to_string(&ClusteringStrategy::StreamingCc).unwrap(),
        r#""streaming_cc""#
    );
}

#[test]
fn serde_spill_threshold_names() {
    use goldenmatch_autoconfig_core::SpillThreshold;
    assert_eq!(
        serde_json::to_string(&SpillThreshold::Ram).unwrap(),
        r#""ram""#
    );
    assert_eq!(
        serde_json::to_string(&SpillThreshold::Duckdb).unwrap(),
        r#""duckdb""#
    );
    assert_eq!(
        serde_json::to_string(&SpillThreshold::DiskPerWorker).unwrap(),
        r#""disk_per_worker""#
    );
}

#[test]
fn serde_col_type_names() {
    // Verify each ColType serializes to the Python-oracle string
    let cases: &[(&str, ColType)] = &[
        ("email", ColType::Email),
        ("name", ColType::Name),
        ("phone", ColType::Phone),
        ("zip", ColType::Zip),
        ("address", ColType::Address),
        ("geo", ColType::Geo),
        ("identifier", ColType::Identifier),
        ("description", ColType::Description),
        ("numeric", ColType::Numeric),
        ("date", ColType::Date),
        ("string", ColType::String),
        ("year", ColType::Year),
        ("multi_name", ColType::MultiName),
        ("exact_derived", ColType::ExactDerived),
    ];
    for (expected_str, col_type) in cases {
        let got = serde_json::to_string(col_type).unwrap();
        let expected_json = format!(r#""{expected_str}""#);
        assert_eq!(got, expected_json, "ColType::{col_type:?} serialized wrong");
    }
}

#[test]
fn serde_null_pair_spill_threshold() {
    // ExecutionPlan with pair_spill_threshold=None must serialize to "null" not absent
    use goldenmatch_autoconfig_core::{BackendName, ClusteringStrategy, ExecutionPlan};
    let plan = ExecutionPlan {
        backend: BackendName::PolarsDirect,
        chunk_size: None,
        max_workers: 1,
        pair_spill_threshold: None,
        clustering_strategy: ClusteringStrategy::InMemory,
        rule_name: "test".into(),
    };
    let v: Value = serde_json::to_value(&plan).unwrap();
    assert!(
        v["pair_spill_threshold"].is_null(),
        "pair_spill_threshold=None must serialize to JSON null, got: {}",
        v["pair_spill_threshold"]
    );
    assert!(
        v["chunk_size"].is_null(),
        "chunk_size=None must serialize to JSON null, got: {}",
        v["chunk_size"]
    );
}

#[test]
fn serde_non_null_pair_spill_threshold() {
    // pair_spill_threshold=Some(Ram) must serialize to "ram" (not "Ram" or "RAM")
    use goldenmatch_autoconfig_core::{
        BackendName, ClusteringStrategy, ExecutionPlan, SpillThreshold,
    };
    let plan = ExecutionPlan {
        backend: BackendName::Chunked,
        chunk_size: Some(100_000),
        max_workers: 8,
        pair_spill_threshold: Some(SpillThreshold::Ram),
        clustering_strategy: ClusteringStrategy::InMemory,
        rule_name: "plan_selected_chunked".into(),
    };
    let v: Value = serde_json::to_value(&plan).unwrap();
    assert_eq!(v["pair_spill_threshold"], Value::String("ram".into()));
    assert_eq!(v["chunk_size"], Value::Number(100_000u64.into()));
}

// ── Blocking-selection parity (#1207 strong-identifier union) ─────────────────

const SELECT_BLOCKING_JSON: &str = include_str!("../golden/select_blocking_vectors.json");

#[test]
fn select_blocking_golden_parity() {
    let doc: Value = serde_json::from_str(SELECT_BLOCKING_JSON)
        .expect("failed to parse select_blocking_vectors.json");
    let mut failures: Vec<String> = Vec::new();

    // Phase 1 — assemble
    let assemble = doc["assemble"].as_array().expect("assemble array");
    assert!(
        assemble.len() >= 6,
        "expected >= 6 assemble vectors, got {}",
        assemble.len()
    );
    for case in assemble {
        let name = case["name"].as_str().unwrap_or("?");
        let cols: Vec<BlockingColumnInput> = match serde_json::from_value(case["input"].clone()) {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("assemble[{name}]: bad input: {e}"));
                continue;
            }
        };
        let got: Value = serde_json::to_value(assemble_strong_id_union(&cols)).unwrap();
        if got != case["expected"] {
            failures.push(format!(
                "assemble[{name}] mismatch:\n  expected={}\n  got={got}",
                case["expected"]
            ));
        }
    }

    // Phase 2 — finalize
    let finalize = doc["finalize"].as_array().expect("finalize array");
    assert!(
        finalize.len() >= 5,
        "expected >= 5 finalize vectors, got {}",
        finalize.len()
    );
    for case in finalize {
        let name = case["name"].as_str().unwrap_or("?");
        let input: UnionFinalizeInput = match serde_json::from_value(case["input"].clone()) {
            Ok(i) => i,
            Err(e) => {
                failures.push(format!("finalize[{name}]: bad input: {e}"));
                continue;
            }
        };
        let got: Value = serde_json::to_value(finalize_strong_id_union(&input)).unwrap();
        if got != case["expected"] {
            failures.push(format!(
                "finalize[{name}] mismatch:\n  expected={}\n  got={got}",
                case["expected"]
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "select_blocking golden failures:\n{}",
        failures.join("\n")
    );
}

// ── classify_by_name parity (the strong-id union's name-classification authority) ──

const CLASSIFY_BY_NAME_JSON: &str = include_str!("../golden/classify_by_name_vectors.json");

#[test]
fn classify_by_name_golden_parity() {
    let vectors: Vec<Value> = serde_json::from_str(CLASSIFY_BY_NAME_JSON)
        .expect("failed to parse classify_by_name_vectors.json");
    assert!(
        vectors.len() >= 40,
        "expected >= 40 classify_by_name vectors, got {}",
        vectors.len()
    );
    let mut failures: Vec<String> = Vec::new();
    for vec in &vectors {
        let name = vec["name"].as_str().expect("name");
        let got: Value = serde_json::to_value(classify_by_name(name)).unwrap();
        if got != vec["expected"] {
            failures.push(format!(
                "classify_by_name({name:?}) mismatch: expected={} got={got}",
                vec["expected"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "classify_by_name golden failures:\n{}",
        failures.join("\n")
    );
}
