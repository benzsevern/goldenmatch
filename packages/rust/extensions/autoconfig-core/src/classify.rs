//! Layer 2 — Column classifier: `guess_type`, `classify_by_name`,
//! `classify_by_data`, `classify_columns`, and supporting types.
//!
//! Parity port of:
//!   - `packages/python/goldenmatch/goldenmatch/core/profiler.py` (_guess_type + regexes)
//!   - `packages/python/goldenmatch/goldenmatch/core/autoconfig.py`
//!     (_classify_by_name, _classify_by_data, profile_columns merge logic, ~lines 111-379)
//!
//! All branch orders, thresholds, and regex strings are reproduced byte-for-byte
//! from the Python source. The >0.6 / >0.4 thresholds use STRICT `>` (not `>=`).

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

// ── ColType ──────────────────────────────────────────────────────────────────

/// The column-type classification. `snake_case` serialisation gives names like
/// `multi_name` that match Python's runtime strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColType {
    Email,
    Name,
    Phone,
    Zip,
    Address,
    Geo,
    Identifier,
    Description,
    Numeric,
    Date,
    String,
    Year,
    MultiName,
    /// `autoconfig.py::_EXACT_DERIVED_COL_TYPE` (#2876/#2881) — given to a
    /// domain-extracted, exact-scored derived column (e.g. `__title_key__`,
    /// `__model_norm__`). Never produced by this crate's own classifier
    /// (`classify_by_name`/`classify_by_data`/`classify_columns`) — it only
    /// ever arrives as an INPUT to the blocking-union decision
    /// (`select_blocking::BlockingColumnInput`), sent by the Python host.
    ExactDerived,
}

// `ColType::as_str()` removed — the cross-surface contract is serde JSON
// (`#[serde(rename_all = "snake_case")]` on the enum), so a Rust-side string
// helper was dead code with no callers anywhere in the crate.

// ── Public structs ────────────────────────────────────────────────────────────

/// Input column statistics (produced by the Arrow profiler in Stage C, or
/// constructed in tests).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub name: std::string::String,
    pub dtype: std::string::String,
    pub sample_values: Vec<std::string::String>,
    pub null_rate: f64,
    pub cardinality_ratio: f64,
    pub avg_len: f64,
}

/// Fully-classified column profile, the output of `classify_columns`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: std::string::String,
    pub dtype: std::string::String,
    pub col_type: ColType,
    pub confidence: f64,
    pub null_rate: f64,
    pub cardinality_ratio: f64,
    pub avg_len: f64,
    pub needs_llm_escalation: bool,
}

// ── profiler.py regexes ───────────────────────────────────────────────────────
//
// All patterns are compiled once via `once_cell::sync::Lazy`.
// fancy-regex handles all patterns (including lookbehind in Layer 3).

/// Strips `[()\-+.\s]` from phone strings before digit-counting.
/// Python: `_PHONE_STRIP_RE = re.compile(r"[()\-+.\s]")`
static PHONE_STRIP_RE: Lazy<fancy_regex::Regex> =
    Lazy::new(|| fancy_regex::Regex::new(r"[()\-+.\s]").expect("PHONE_STRIP_RE"));

/// Date patterns used by `_guess_type`.  Python uses `re.compile(...)` with
/// `re.match`, which anchors at the START only (not a full-string match).
/// The patterns already have `^...$` so `is_match` works correctly.
static DATE_PATTERNS_PROFILER: Lazy<[fancy_regex::Regex; 3]> = Lazy::new(|| {
    [
        fancy_regex::Regex::new(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$")
            .expect("DATE_P0"),
        fancy_regex::Regex::new(r"^\d{4}[/\-]\d{1,2}[/\-]\d{1,2}$")
            .expect("DATE_P1"),
        fancy_regex::Regex::new(r"^\d{1,2}\s\w+\s\d{2,4}$")
            .expect("DATE_P2"),
    ]
});

/// Word-boundary address-word detector.
/// Python: `re.compile(r"\b(...)\b", re.IGNORECASE)`
static ADDRESS_WORDS_RE: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(
        r"(?i)\b(st|street|ave|avenue|rd|road|dr|drive|blvd|boulevard|ln|lane|ct|court|way|pl|place|cir|circle)\b",
    )
    .expect("ADDRESS_WORDS_RE")
});

/// Name pattern for `_guess_type`.
/// Python: `re.compile(r"^[A-Za-z][A-Za-z \-']{0,28}[A-Za-z]$|^[A-Za-z]{2,3}$")`
/// `re.match` anchors at start; the pattern already has `^` so `is_match` is equivalent.
static NAME_RE: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(r"^[A-Za-z][A-Za-z \-']{0,28}[A-Za-z]$|^[A-Za-z]{2,3}$")
        .expect("NAME_RE")
});

// ── autoconfig.py column-name patterns ────────────────────────────────────────

static NAME_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(
        r"(?i)(^name$|first.?name|last.?name|full.?name|fname|lname|surname|given.?name)",
    )
    .expect("NAME_PATTERNS")
});

static EMAIL_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(r"(?i)(email|e.?mail|email.?addr)").expect("EMAIL_PATTERNS")
});

static PHONE_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(r"(?i)(phone|tel|mobile|fax|cell)").expect("PHONE_PATTERNS")
});

static ZIP_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(r"(?i)(zip|postal|postcode|zip.?code)").expect("ZIP_PATTERNS")
});

static PRICE_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(
        r"(?i)(price|cost|amount|revenue|salary|fee|charge|total|balance)",
    )
    .expect("PRICE_PATTERNS")
});

static ADDRESS_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(r"(?i)(address|street|addr|line.?1|line.?2)")
        .expect("ADDRESS_PATTERNS")
});

/// Geo pattern — contains lookbehind `(?<![a-z])` (requires fancy-regex).
/// Python: `re.compile(r"((?<![a-z])city|^state$|state.?cd|^country$|province|region|(?<![a-z])county)", re.IGNORECASE)`
static GEO_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(
        r"(?i)((?<![a-z])city|^state$|state.?cd|^country$|province|region|(?<![a-z])county)",
    )
    .expect("GEO_PATTERNS")
});

static DATE_PATTERNS_NAME: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(r"(?i)(date|_dt$|_date$|registr|created|updated|birth.?d|dob)")
        .expect("DATE_PATTERNS_NAME")
});

static YEAR_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(r"(?i)(^|_)(year|yr)(_|$)").expect("YEAR_PATTERNS")
});

/// ID pattern — embeds inline `(?i:...)` groups (some parts are case-sensitive).
/// Specifically: `(?<=[a-zA-Z])(?:ID|Id)$` is NOT in a `(?i:)` group, so the
/// literal `ID`/`Id` casing matters (matches `recordID`/`recordId` but NOT
/// `recordid`).  Lookbehind requires fancy-regex.
///
/// Python source (autoconfig.py lines 123-131):
///   ^(?i:id|key|code|sku)$
///   |_(?i:id|key)$
///   |(?<=[a-zA-Z])(?:ID|Id)$
///   |(?i:^uuid$|^guid$|_uuid$|_guid$)
///   |(?i:^uuid_|^guid_)
///   |^(?i:account_no|account_num)$
///   |_(?i:ref|ref_num|reg_num|account_no|account_num|account)$
static ID_PATTERNS: Lazy<fancy_regex::Regex> = Lazy::new(|| {
    fancy_regex::Regex::new(
        concat!(
            r"^(?i:id|key|code|sku)$",
            r"|_(?i:id|key)$",
            r"|(?<=[a-zA-Z])(?:ID|Id)$",
            r"|(?i:^uuid$|^guid$|_uuid$|_guid$)",
            r"|(?i:^uuid_|^guid_)",
            r"|^(?i:account_no|account_num)$",
            r"|_(?i:ref|ref_num|reg_num|account_no|account_num|account)$",
        ),
    )
    .expect("ID_PATTERNS")
});

// ── Task B1: guess_type ───────────────────────────────────────────────────────

/// Port of `profiler.py::_guess_type`.
///
/// All thresholds match the Python: `>0.6` for most, `>0.4` for address/date.
/// Python's `.isdigit()` / `.isalpha()` / `.isupper()` return `False` for empty
/// strings, reproduced via `.chars().all(...)` which is vacuously `true` on empty
/// — so we add an explicit `!s.is_empty()` guard where needed.
pub fn guess_type(values: &[std::string::String]) -> &'static str {
    if values.is_empty() {
        return "text";
    }
    let n = values.len() as f64;

    // email: >60% contain @ and a dot after the LAST @
    // Python: "@" in v and "." in v.split("@")[-1]
    // Use rfind('@') to mirror split("@")[-1] — diverges from find('@') on
    // values with multiple @ signs (e.g. "user@domain.com@nodot").
    let email_count = values
        .iter()
        .filter(|v| match v.rfind('@') {
            Some(at_idx) => v[at_idx + 1..].contains('.'),
            None => false,
        })
        .count();
    if email_count as f64 / n > 0.6 {
        return "email";
    }

    // phone: >60% mostly digits after stripping, 7..=15 len
    let phone_count = values
        .iter()
        .filter(|v| {
            let stripped = PHONE_STRIP_RE.replace_all(v, "").into_owned();
            // Python `.isdigit()` is False on empty string
            !stripped.is_empty()
                && stripped.chars().all(|c| c.is_ascii_digit())
                && stripped.len() >= 7
                && stripped.len() <= 15
        })
        .count();
    if phone_count as f64 / n > 0.6 {
        return "phone";
    }

    // zip: >60% 5 or 9 digits after removing '-'
    let zip_count = values
        .iter()
        .filter(|v| {
            let clean: std::string::String = v.replace('-', "");
            // Python `.isdigit()` is False on empty
            !clean.is_empty()
                && clean.chars().all(|c| c.is_ascii_digit())
                && (clean.len() == 5 || clean.len() == 9)
        })
        .count();
    if zip_count as f64 / n > 0.6 {
        return "zip";
    }

    // state: >60% exactly 2 uppercase letters
    // Python: len(v)==2 and v.isalpha() and v.isupper()
    let state_count = values
        .iter()
        .filter(|v| {
            // len==2 implies non-empty; is_ascii_uppercase() == is_ascii_alphabetic() && is_uppercase()
            v.len() == 2 && v.chars().all(|c| c.is_ascii_uppercase())
        })
        .count();
    if state_count as f64 / n > 0.6 {
        return "state";
    }

    // numeric: >60% parse as float (comma-tolerant)
    let numeric_count = values
        .iter()
        .filter(|v| {
            let cleaned = v.replace(',', "");
            cleaned.parse::<f64>().is_ok()
        })
        .count();
    if numeric_count as f64 / n > 0.6 {
        return "numeric";
    }

    // name: >60% match _NAME_RE  (Python uses re.match — anchored at start, but pattern has ^)
    let name_count = values
        .iter()
        .filter(|v| {
            let stripped = v.trim();
            NAME_RE.is_match(stripped).unwrap_or(false)
        })
        .count();
    if name_count as f64 / n > 0.6 {
        return "name";
    }

    // address: >40% contain a digit AND an address word
    let addr_count = values
        .iter()
        .filter(|v| {
            v.chars().any(|c| c.is_ascii_digit())
                && ADDRESS_WORDS_RE.is_match(v).unwrap_or(false)
        })
        .count();
    if addr_count as f64 / n > 0.4 {
        return "address";
    }

    // date: >40% match a date pattern  (Python uses re.match — patterns are ^...$)
    let date_count = values
        .iter()
        .filter(|v| {
            let stripped = v.trim();
            DATE_PATTERNS_PROFILER
                .iter()
                .any(|p| p.is_match(stripped).unwrap_or(false))
        })
        .count();
    if date_count as f64 / n > 0.4 {
        return "date";
    }

    "text"
}

// ── Task B2: classify_by_name ─────────────────────────────────────────────────

/// Port of `autoconfig.py::_classify_by_name`.
///
/// `re.search` is unanchored; `fancy_regex::Regex::is_match` is also unanchored.
/// Order is load-bearing — reproduce exactly.
pub fn classify_by_name(col_name: &str) -> Option<ColType> {
    if DATE_PATTERNS_NAME.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Date);
    }
    if YEAR_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Year);
    }
    if EMAIL_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Email);
    }
    if ID_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Identifier);
    }
    if PRICE_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Numeric);
    }
    if ZIP_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Zip);
    }
    if GEO_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Geo);
    }
    if ADDRESS_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Address);
    }
    if PHONE_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Phone);
    }
    if NAME_PATTERNS.is_match(col_name).unwrap_or(false) {
        return Some(ColType::Name);
    }
    None
}

// ── Task B3: classify_by_data ─────────────────────────────────────────────────

/// Port of `autoconfig.py::_is_year` (inner function of `_classify_by_data`).
///
/// True if `v` looks like a 4-digit year in 1900-2100, tolerating float-promoted
/// integer columns (e.g. `'1999.0'`).
///
/// Python semantics:
///   n = int(float(v))   → parse as f64, truncate toward zero to i64
///   1900 <= n <= 2100
///   str(n) == v.replace(".0","").strip()   (round-trip check)
pub(crate) fn is_year(v: &str) -> bool {
    let trimmed = v.trim();
    let f: f64 = match trimmed.parse() {
        Ok(f) => f,
        Err(_) => return false,
    };
    // Reject inf / NaN (Python's `int(float('inf'))` raises OverflowError)
    if !f.is_finite() {
        return false;
    }
    // Python int(float(v)) truncates toward zero (same as `as i64` for finite values
    // in range, but we must guard against values that exceed i64 range).
    if f.abs() > i64::MAX as f64 {
        return false;
    }
    let n = f as i64;
    if !(1900..=2100).contains(&n) {
        return false;
    }
    // Round-trip: stringified int must equal v with ".0" removed and stripped.
    let cleaned = trimmed.replace(".0", "");
    let cleaned = cleaned.trim();
    n.to_string() == cleaned
}

/// Port of `autoconfig.py::_classify_by_data`.
///
/// Branch order is load-bearing; reproduce exactly.
pub fn classify_by_data(values: &[std::string::String]) -> (ColType, f64) {
    if values.is_empty() {
        return (ColType::String, 0.0);
    }

    let data_type = guess_type(values);

    // Cardinality guard: near-unique numeric-looking columns are identifiers.
    // Gate: data_type in {"phone","zip","numeric"} AND len(values) >= 10.
    if matches!(data_type, "phone" | "zip" | "numeric") && values.len() >= 10 {
        let unique: std::collections::HashSet<&str> =
            values.iter().map(|s| s.as_str()).collect();
        let cardinality_ratio = unique.len() as f64 / values.len() as f64;
        // S2a: identifier floor max(0.95, 1 - 1/sqrt(n)). At scale the floor
        // RISES above the old fixed 0.95 (a 10k-row 0.95-cardinality column is a
        // high-entropy name, not an ID, and is no longer promoted); it never
        // drops below 0.95, so small-n behavior is unchanged (a looser small-n
        // floor reclassified moderately-unique phone/numeric columns and broke
        // established matchkey behavior). `.sqrt()` is correctly-rounded IEEE
        // 754, bit-identical to Python's math.sqrt (oracle parity).
        let floor = (1.0 - 1.0 / (values.len() as f64).sqrt()).max(0.95);
        if cardinality_ratio >= floor {
            return (ColType::Identifier, 0.9);
        }
    }

    // Year detection: all values look like a year in 1900-2100.
    if !values.is_empty() && values.iter().all(|v| is_year(v)) {
        return (ColType::Year, 0.9);
    }

    // Map profiler types to ColType.
    let col_type = match data_type {
        "email" => ColType::Email,
        "phone" => ColType::Phone,
        "zip" => ColType::Zip,
        "state" => ColType::Geo,
        "numeric" => ColType::Numeric,
        "name" => ColType::Name,
        "address" => ColType::Address,
        "date" => ColType::Date,
        _ => ColType::String, // "text" and anything else
    };

    // Shared avg_len for the two String sub-checks below.
    // Uses .max(1) to match Python's max(len(values), 1) in _classify_by_data;
    // values is non-empty here (guarded above), so this is always identical.
    let avg_len: f64 =
        values.iter().map(|v| v.len()).sum::<usize>() as f64 / values.len().max(1) as f64;

    // Multi-value name detection (only when col_type == String).
    if col_type == ColType::String {
        let rows_with_delim: usize = values
            .iter()
            .filter(|v| v.contains(',') || v.contains(';'))
            .count();
        let delim_ratio = rows_with_delim as f64 / values.len().max(1) as f64;
        let avg_delims_in_delim_rows = if rows_with_delim > 0 {
            let total_delims: usize = values
                .iter()
                .filter(|v| v.contains(',') || v.contains(';'))
                .map(|v| v.chars().filter(|&c| c == ',' || c == ';').count())
                .sum();
            total_delims as f64 / rows_with_delim as f64
        } else {
            0.0
        };
        if avg_len > 30.0 && delim_ratio >= 0.7 && avg_delims_in_delim_rows >= 2.0 {
            return (ColType::MultiName, 0.7);
        }
    }

    // Description detection: long freetext (only when col_type == String).
    let col_type = if col_type == ColType::String {
        if avg_len > 50.0 {
            ColType::Description
        } else {
            ColType::String
        }
    } else {
        col_type
    };

    let confidence = if col_type != ColType::String { 0.7 } else { 0.3 };
    (col_type, confidence)
}

// ── Task B4: classify_columns ─────────────────────────────────────────────────

/// Column types where the name heuristic overrides data profiling.
/// Python: `name_authoritative = {Date, Geo, Identifier, Numeric, Year, Zip}`.
/// Zip is authoritative because ZIP+4 codes ('10001-3904') look phone-shaped to
/// the data profiler; a zip misclassified as phone would back an exact matchkey
/// and over-merge same-address records.
const NAME_AUTHORITATIVE: &[ColType] = &[
    ColType::Date,
    ColType::Geo,
    ColType::Identifier,
    ColType::Numeric,
    ColType::Year,
    ColType::Zip,
];

/// Port of `autoconfig.py::profile_columns` merge-precedence logic (lines 350-368).
///
/// Authoritative set: `{Date, Geo, Identifier, Numeric, Year, Zip}`.
/// `needs_llm_escalation` implements the predicate from `_llm_classify_columns`
/// (line 401-405 in autoconfig.py):
///   `(confidence < 0.8 || col_type in {String, Numeric})
///    && col_type not in {Date, Geo, Email, Identifier}`
pub fn classify_columns(cols: &[ColumnStats]) -> Vec<ColumnProfile> {
    cols.iter()
        .map(|cs| {
            let name_type = classify_by_name(&cs.name);
            let (data_type, data_conf) = classify_by_data(&cs.sample_values);

            let (col_type, confidence) = if let Some(nt) = name_type {
                if NAME_AUTHORITATIVE.contains(&nt) {
                    // Name pattern is authoritative for these types.
                    (nt, 0.9)
                } else if data_type != ColType::String {
                    // Both have opinions — Phase 2 wins if types differ.
                    if nt == data_type {
                        (nt, (data_conf + 0.2).min(1.0))
                    } else {
                        (data_type, data_conf)
                    }
                } else {
                    // Data says "string"; name has an opinion — use name at lower confidence.
                    (nt, 0.6)
                }
            } else {
                // No name opinion — use data profiling.
                (data_type, data_conf)
            };

            // LLM escalation predicate (autoconfig.py lines 400-405):
            //   ambiguous = [p for p in profiles
            //     if p.confidence < 0.8 or p.col_type in ("string","numeric")
            //     if p.col_type not in high_confidence_types]
            // where high_confidence_types = {"date","geo","email","identifier"}
            let high_conf = matches!(
                col_type,
                ColType::Date | ColType::Geo | ColType::Email | ColType::Identifier
            );
            let needs_llm_escalation =
                (confidence < 0.8 || matches!(col_type, ColType::String | ColType::Numeric))
                    && !high_conf;

            ColumnProfile {
                name: cs.name.clone(),
                dtype: cs.dtype.clone(),
                col_type,
                confidence,
                null_rate: cs.null_rate,
                cardinality_ratio: cs.cardinality_ratio,
                avg_len: cs.avg_len,
                needs_llm_escalation,
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(v: &[&str]) -> Vec<std::string::String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    // ── B1: guess_type ────────────────────────────────────────────────────────

    #[test]
    fn test_guess_type_empty() {
        assert_eq!(guess_type(&[]), "text");
    }

    #[test]
    fn test_guess_type_email() {
        let vals = sv(&[
            "a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.co", "k@l.dev", "m@n.edu",
        ]);
        assert_eq!(guess_type(&vals), "email");
    }

    #[test]
    fn test_guess_type_email_threshold_boundary() {
        // 6/10 = 0.6 — NOT >0.6, so should fall through
        let mut vals: Vec<std::string::String> =
            (0..6).map(|i| format!("u{}@x.com", i)).collect();
        vals.extend((0..4).map(|i| format!("notanemail{}", i)));
        // 0.6 is not > 0.6, so should not be "email"
        assert_ne!(guess_type(&vals), "email");
    }

    #[test]
    fn test_guess_type_email_uses_last_at() {
        // Regression: "user@domain.com@nodot" has a dot after the FIRST @ but NOT
        // after the LAST @.  Python's `"." in v.split("@")[-1]` evaluates to False
        // ("nodot" has no dot), so these values must NOT classify as "email".
        // The old Rust code used find('@') (first @), which would find "domain.com@nodot"
        // → contains '.' → wrongly counted as email.
        let vals = sv(&[
            "user@domain.com@nodot",
            "alice@foo.org@bar",
            "bob@x.net@baz",
            "carol@y.io@qux",
            "dave@z.co@quux",
            "eve@a.dev@corge",
            "frank@b.edu@grault",
        ]);
        // 7/7 would be >0.6 if the old (first-@) logic ran — must NOT return "email"
        assert_ne!(guess_type(&vals), "email");
        // These values have no digits so not phone/zip, not 2-char uppercase so not
        // state, contain @ so won't match NAME_RE, no address word, no date pattern
        // → they fall through to "text".
        assert_eq!(guess_type(&vals), "text");
    }

    #[test]
    fn test_guess_type_phone() {
        let vals = sv(&[
            "5551234567", "4155556789", "2125559876", "7185554321", "9175551234",
            "3125558765", "8005553456",
        ]);
        assert_eq!(guess_type(&vals), "phone");
    }

    #[test]
    fn test_guess_type_phone_strips_formatting() {
        let vals = sv(&[
            "(555) 123-4567",
            "(415) 555-6789",
            "(212) 555-9876",
            "(718) 555-4321",
            "(917) 555-1234",
            "(312) 555-8765",
            "(800) 555-3456",
        ]);
        assert_eq!(guess_type(&vals), "phone");
    }

    #[test]
    fn test_guess_type_phone_empty_stripped_not_counted() {
        // After stripping, empty string must NOT count as phone
        // Mix: 6 valid phones, 4 non-phones (so 6/10 = 0.6, not > 0.6)
        let mut vals: Vec<std::string::String> =
            (0..6).map(|i| format!("555000{:04}", i)).collect();
        vals.extend((0..4).map(|i| format!("word{}", i)));
        assert_ne!(guess_type(&vals), "phone");
    }

    #[test]
    fn test_guess_type_zip() {
        let vals = sv(&["12345", "90210", "10001", "60601", "33101", "94105", "20001"]);
        assert_eq!(guess_type(&vals), "zip");
    }

    #[test]
    fn test_guess_type_zip_9digit() {
        // NOTE: Python strips phone chars first, and 9-digit strings pass the phone check
        // (7 <= 9 <= 15 and all digits). So pure 9-digit strings classify as "phone" before "zip".
        // This is correct Python parity — the phone check comes before zip.
        let vals = sv(&[
            "123456789", "902104567", "100010001", "606011234", "331015678", "941056789",
            "200010001",
        ]);
        // 9-digit all-digit strings pass the phone check (7-15 len, all digits) before zip
        assert_eq!(guess_type(&vals), "phone");
    }

    #[test]
    fn test_guess_type_state() {
        let vals = sv(&["NY", "CA", "TX", "FL", "WA", "OR", "IL"]);
        assert_eq!(guess_type(&vals), "state");
    }

    #[test]
    fn test_guess_type_numeric() {
        let vals = sv(&["1.5", "2.7", "100", "42.0", "0", "999", "-3.14"]);
        assert_eq!(guess_type(&vals), "numeric");
    }

    #[test]
    fn test_guess_type_numeric_with_commas() {
        let vals = sv(&["1,000", "2,500", "10,000", "5,000", "3,750", "8,200", "1,100"]);
        assert_eq!(guess_type(&vals), "numeric");
    }

    #[test]
    fn test_guess_type_name() {
        let vals = sv(&[
            "Alice Smith",
            "Bob Jones",
            "Carol White",
            "Dave Brown",
            "Eve Green",
            "Frank Hall",
            "Grace Lee",
        ]);
        assert_eq!(guess_type(&vals), "name");
    }

    #[test]
    fn test_guess_type_address() {
        let vals = sv(&[
            "123 Main St",
            "456 Oak Ave",
            "789 Pine Rd",
            "321 Elm Dr",
            "654 Maple Blvd",
            "987 Cedar Ln",
            "111 First Ct",
        ]);
        assert_eq!(guess_type(&vals), "address");
    }

    #[test]
    fn test_guess_type_address_threshold_boundary() {
        // 4/10 = 0.4 — NOT >0.4, so should not be "address"
        let mut vals: Vec<std::string::String> = vec![
            "123 Main St".to_string(),
            "456 Oak Ave".to_string(),
            "789 Pine Rd".to_string(),
            "321 Elm Dr".to_string(),
        ];
        vals.extend((0..6).map(|i| format!("plaintext{}", i)));
        assert_ne!(guess_type(&vals), "address");
    }

    #[test]
    fn test_guess_type_date() {
        let vals = sv(&["01/15/2023", "12/31/2022", "07/04/2021", "03/17/2020", "11/11/2019"]);
        assert_eq!(guess_type(&vals), "date");
    }

    #[test]
    fn test_guess_type_date_iso_format() {
        // NOTE: Python strips "()\-+.\s" from ISO dates, leaving all-digit 8-char strings
        // (e.g. "2023-01-15" -> "20230115"), which pass the phone check (7-15 len, all digits).
        // So ISO dates like "2023-01-15" are classified as "phone" in Python's _guess_type.
        // The DATE_PATTERNS in profiler.py catch DD/MM/YYYY and similar, NOT YYYY-MM-DD.
        // col-name heuristics in _classify_by_name are the correct path for date columns.
        let vals = sv(&["2023-01-15", "2022-12-31", "2021-07-04", "2020-03-17", "2019-11-11"]);
        // ISO dates with dashes stripped become 8-digit strings -> phone (parity with Python)
        assert_eq!(guess_type(&vals), "phone");
    }

    #[test]
    fn test_guess_type_text_fallback() {
        // NOTE: "hello world" etc. match _NAME_RE (alpha+space body), so these classify as "name"
        // not "text". Use values that are truly generic (mixed chars) to get "text".
        // Values with digits / special chars that don't fit any pattern -> text.
        let vals = sv(&[
            "foo123bar", "baz!qux", "hello#world", "abc$def", "test&value",
        ]);
        assert_eq!(guess_type(&vals), "text");
    }

    // ── B2: classify_by_name ──────────────────────────────────────────────────

    #[test]
    fn test_classify_by_name_city_geo() {
        // "city" matches (?<![a-z])city — no preceding lowercase letter
        assert_eq!(classify_by_name("city"), Some(ColType::Geo));
    }

    #[test]
    fn test_classify_by_name_municipality_none() {
        // "municipality" ends in "ity" but has preceding lowercase "c" — lookbehind (?<![a-z])city fails
        assert_eq!(classify_by_name("municipality"), None);
    }

    #[test]
    fn test_classify_by_name_record_id_identifier() {
        // "recordID" — (?<=[a-zA-Z])(?:ID|Id)$ matches (case-sensitive ID)
        assert_eq!(classify_by_name("recordID"), Some(ColType::Identifier));
    }

    #[test]
    fn test_classify_by_name_record_id_lowercase_none() {
        // "recordid" — (?:ID|Id) is case-sensitive, so "id" at end does NOT match that branch.
        // Check: does _(?i:id|key)$ or ^(?i:id|key|code|sku)$ match?
        // "recordid" -> ends with "id", but the suffix branch is `_(?i:id|key)$` which needs underscore.
        // "recordID" suffix branch requires capital ID.
        // So "recordid" should NOT match ID_PATTERNS.
        // (It could match ^(?i:id...)$ if the whole name is "id", but "recordid" is not just "id")
        assert_ne!(classify_by_name("recordid"), Some(ColType::Identifier));
    }

    #[test]
    fn test_classify_by_name_account_no_identifier() {
        assert_eq!(classify_by_name("account_no"), Some(ColType::Identifier));
    }

    #[test]
    fn test_classify_by_name_created_at_date() {
        assert_eq!(classify_by_name("created_at"), Some(ColType::Date));
    }

    #[test]
    fn test_classify_by_name_birth_year_year() {
        // "birth_year" — DATE_PATTERNS looks for "birth.?d" (needs 'd' after 'birth').
        // "birth_year" has no 'd' after 'birth', so DATE misses.
        // YEAR_PATTERNS: (^|_)(year|yr)(_|$) — "_year" at end matches.
        assert_eq!(classify_by_name("birth_year"), Some(ColType::Year));
    }

    #[test]
    fn test_classify_by_name_zip_code_zip() {
        assert_eq!(classify_by_name("zip_code"), Some(ColType::Zip));
    }

    #[test]
    fn test_classify_by_name_first_name_name() {
        assert_eq!(classify_by_name("first_name"), Some(ColType::Name));
    }

    #[test]
    fn test_classify_by_name_email_email() {
        assert_eq!(classify_by_name("email_address"), Some(ColType::Email));
    }

    #[test]
    fn test_classify_by_name_state_geo() {
        // ^state$ is anchored, matches exactly "state"
        assert_eq!(classify_by_name("state"), Some(ColType::Geo));
    }

    #[test]
    fn test_classify_by_name_state_cd_geo() {
        // state.?cd matches "state_cd"
        assert_eq!(classify_by_name("state_cd"), Some(ColType::Geo));
    }

    #[test]
    fn test_classify_by_name_country_geo() {
        assert_eq!(classify_by_name("country"), Some(ColType::Geo));
    }

    #[test]
    fn test_classify_by_name_county_geo() {
        // (?<![a-z])county — no preceding lowercase letter at start
        assert_eq!(classify_by_name("county"), Some(ColType::Geo));
    }

    #[test]
    fn test_classify_by_name_phone_phone() {
        assert_eq!(classify_by_name("phone_number"), Some(ColType::Phone));
    }

    #[test]
    fn test_classify_by_name_address_address() {
        assert_eq!(classify_by_name("street_address"), Some(ColType::Address));
    }

    #[test]
    fn test_classify_by_name_price_numeric() {
        assert_eq!(classify_by_name("price"), Some(ColType::Numeric));
    }

    #[test]
    fn test_classify_by_name_unknown_none() {
        assert_eq!(classify_by_name("foobar_xyz"), None);
    }

    #[test]
    fn test_classify_by_name_dob_date() {
        assert_eq!(classify_by_name("dob"), Some(ColType::Date));
    }

    #[test]
    fn test_classify_by_name_birth_date_date() {
        // "birth_date" — birth.?d matches (birth + _ + d)? Let's check: "birth.?d" with re.search
        // "birth_date" contains "birth" then "_" then "d" — `birth.?d` means birth + optional char + d
        // "_" is a char, so "birth_d" in "birth_date" — yes, matches.
        assert_eq!(classify_by_name("birth_date"), Some(ColType::Date));
    }

    #[test]
    fn test_classify_by_name_user_id_identifier() {
        // "user_id" — _(?i:id|key)$ matches "_id" at end
        assert_eq!(classify_by_name("user_id"), Some(ColType::Identifier));
    }

    #[test]
    fn test_classify_by_name_uuid_identifier() {
        assert_eq!(classify_by_name("uuid"), Some(ColType::Identifier));
    }

    #[test]
    fn test_classify_by_name_guid_col_identifier() {
        // guid_ prefix matches (?i:^guid_)
        assert_eq!(classify_by_name("guid_col"), Some(ColType::Identifier));
    }

    // ── B3: classify_by_data ──────────────────────────────────────────────────

    #[test]
    fn test_classify_by_data_empty() {
        assert_eq!(classify_by_data(&[]), (ColType::String, 0.0));
    }

    #[test]
    fn test_classify_by_data_identifier_via_guard() {
        // 12 unique numeric strings, cardinality = 1.0 >= 0.95 -> Identifier
        let vals: Vec<std::string::String> = (1000..1012).map(|i| i.to_string()).collect();
        assert_eq!(vals.len(), 12);
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::Identifier);
        assert!((conf - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_identifier_guard_needs_10_samples() {
        // Only 9 values — guard does NOT fire even if cardinality is 1.0
        let vals: Vec<std::string::String> = (1000..1009).map(|i| i.to_string()).collect();
        assert_eq!(vals.len(), 9);
        // Should NOT return Identifier via the guard
        let (ct, _) = classify_by_data(&vals);
        assert_ne!(ct, ColType::Identifier);
    }

    #[test]
    fn test_classify_by_data_s2a_adaptive_floor() {
        // S2a: floor = max(0.95, 1 - 1/sqrt(n)). Small-n behavior is UNCHANGED
        // (floor stays 0.95), so a 0.80-cardinality n=10 column is still numeric
        // (not promoted) -- the same as the old fixed 0.95.
        let mut small: Vec<std::string::String> = (1000..1008).map(|i| i.to_string()).collect();
        small.push("1007".into());
        small.push("1007".into());
        assert_eq!(small.len(), 10);
        assert_ne!(classify_by_data(&small).0, ColType::Identifier);

        // Stricter at scale: n=900, card ~0.955 is BELOW floor(900)=max(0.95,
        // 0.9667)=0.9667 -> NOT an identifier (the old fixed 0.95 WOULD have
        // promoted it -- this is S2a's behavioral change). 860 distinct of 900.
        let mut big: Vec<std::string::String> = (0..860).map(|i| (100_000 + i).to_string()).collect();
        for i in 0..40 {
            big.push((100_000 + i).to_string());
        }
        assert_eq!(big.len(), 900);
        let unique: std::collections::HashSet<&std::string::String> = big.iter().collect();
        let card = unique.len() as f64 / big.len() as f64; // 860/900 ~= 0.9556
        let floor = (1.0 - 1.0 / 900_f64.sqrt()).max(0.95); // ~= 0.9667
        assert!(card < floor, "card {card} should be below floor {floor}");
        assert!(card > 0.95, "card {card} would have been promoted under the old 0.95");
        assert_ne!(classify_by_data(&big).0, ColType::Identifier);
    }

    #[test]
    fn test_classify_by_data_year() {
        let vals = sv(&["2001", "2010", "1999", "2023", "1985"]);
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::Year);
        assert!((conf - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_year_float_format() {
        // Float-promoted year columns (e.g., from Polars Int->Float cast)
        let vals = sv(&["2001.0", "2010.0", "1999.0", "2023.0", "1985.0"]);
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::Year);
        assert!((conf - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_multi_name() {
        // Long delimiter-heavy strings
        let base = "Smith, John; Doe, Jane; Bloom, Alice; Roberts, Tom";
        let vals: Vec<std::string::String> = (0..10).map(|_| base.to_string()).collect();
        // avg_len = 50, delim_ratio = 1.0, avg_delims ~ 6 => MultiName
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::MultiName);
        assert!((conf - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_description() {
        // Long freetext, no delimiters, avg_len > 50
        let long = "This is a very long description that exceeds fifty characters easily here.";
        let vals: Vec<std::string::String> = (0..5).map(|_| long.to_string()).collect();
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::Description);
        assert!((conf - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_plain_text_string() {
        // Values must not match any heuristic. Short pure-alpha strings like "hello" match NAME_RE.
        // Use strings with digits/special chars to get through to "text" -> String.
        let vals = sv(&["foo123", "bar456", "baz789", "qux000", "xyz111"]);
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::String);
        assert!((conf - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_email() {
        let vals = sv(&[
            "a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.co", "k@l.dev", "m@n.edu",
        ]);
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::Email);
        assert!((conf - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_name_direct() {
        // "name"-typed sample from guess_type (>60% match NAME_RE) -> ColType::Name, conf=0.7
        let vals = sv(&[
            "John Smith", "Jane Doe", "Alice Brown", "Bob Jones", "Carol White",
            "Dave Green", "Eve Hall",
        ]);
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::Name);
        assert!((conf - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_classify_by_data_phone_small_sample_no_cardinality_guard() {
        // 5 distinct phone numbers — cardinality guard requires >= 10 values, so does NOT fire.
        // All 5 are valid phone format -> guess_type returns "phone" -> ColType::Phone, conf=0.7.
        let vals = sv(&["5551234567", "4155556789", "2125559876", "7185554321", "9175551234"]);
        assert_eq!(vals.len(), 5);
        let (ct, conf) = classify_by_data(&vals);
        assert_eq!(ct, ColType::Phone);
        assert!((conf - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_classify_columns_data_wins_when_name_disagrees_both_non_string() {
        // name="phone_contact" -> classify_by_name returns Some(Phone).
        // sample_values are all emails -> classify_by_data returns (Email, 0.7).
        // Both name_type (Phone) and data_type (Email) are non-String and they disagree.
        // The merge rule: data wins -> col_type=Email, confidence=0.7.
        let cols = vec![make_col(
            "phone_contact",
            &["a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.co", "k@l.dev", "m@n.edu"],
        )];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Email);
        assert!((profiles[0].confidence - 0.7).abs() < 1e-10);
    }

    // ── B4: classify_columns (merge precedence) ───────────────────────────────

    fn make_col(name: &str, values: &[&str]) -> ColumnStats {
        ColumnStats {
            name: name.to_string(),
            dtype: "Utf8".to_string(),
            sample_values: values.iter().map(|s| s.to_string()).collect(),
            null_rate: 0.0,
            cardinality_ratio: 0.5,
            avg_len: values.iter().map(|s| s.len()).sum::<usize>() as f64
                / values.len().max(1) as f64,
        }
    }

    #[test]
    fn test_classify_columns_name_authoritative_date() {
        // name="created_at" -> Date (authoritative); data has non-date strings -> Date wins, conf=0.9
        let cols = vec![make_col(
            "created_at",
            &["hello", "world", "foo", "bar", "baz"],
        )];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Date);
        assert!((profiles[0].confidence - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_classify_columns_name_agrees_with_data() {
        // name="email_address" -> Email; data has emails -> same type, conf = min(0.7+0.2, 1.0)
        let cols = vec![make_col(
            "email_address",
            &["a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.co", "k@l.dev", "m@n.edu"],
        )];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Email);
        assert!((profiles[0].confidence - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_classify_columns_name_disagrees_with_data() {
        // name="phone_num" -> Phone; data has chars-with-digits (not pure alpha, don't match NAME_RE)
        // -> data says String; String means data says nothing opinionated -> name wins at conf=0.6
        // Actually when data_type == String and name_type is Some -> use name at conf=0.6.
        // So: name=Phone, data=String -> col_type=Phone, conf=0.6
        let cols = vec![make_col(
            "phone_num",
            &["foo123", "bar456", "baz789", "qux000", "xyz111"],
        )];
        let profiles = classify_columns(&cols);
        // data says String (alpha+digit), name says Phone -> name wins at conf=0.6
        assert_eq!(profiles[0].col_type, ColType::Phone);
        assert!((profiles[0].confidence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_classify_columns_name_only_string_data() {
        // name="first_name" -> Name; data has alpha+digit strings (don't match NAME_RE) -> String
        // name=Name, data=String -> name wins at conf=0.6
        let cols = vec![make_col(
            "first_name",
            &["foo123", "bar456", "baz789", "qux000", "xyz111"],
        )];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Name);
        assert!((profiles[0].confidence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_classify_columns_no_name_match() {
        // name="foobar_xyz" -> no match; data has emails -> data wins
        let cols = vec![make_col(
            "foobar_xyz",
            &["a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.co", "k@l.dev", "m@n.edu"],
        )];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Email);
        assert!((profiles[0].confidence - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_classify_columns_needs_llm_escalation_string() {
        // String type -> needs_llm_escalation = true
        // Use alpha+digit values so they don't match NAME_RE
        let cols = vec![make_col("foobar", &["foo123", "bar456", "baz789"])];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::String);
        assert!(profiles[0].needs_llm_escalation);
    }

    #[test]
    fn test_classify_columns_no_llm_escalation_date() {
        // Date (high confidence type) -> needs_llm_escalation = false
        let cols = vec![make_col("created_at", &["hello", "world", "foo"])];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Date);
        assert!(!profiles[0].needs_llm_escalation);
    }

    #[test]
    fn test_classify_columns_no_llm_escalation_identifier() {
        // name="user_id" -> Identifier (authoritative), conf=0.9 -> needs_llm_escalation = false
        // The name heuristic drives the classification (Identifier is in the authoritative set).
        let cols = vec![make_col("user_id", &["foo123", "bar456", "baz789"])];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Identifier);
        assert!(!profiles[0].needs_llm_escalation);
    }

    #[test]
    fn test_classify_columns_llm_escalation_numeric() {
        // Numeric type -> always escalates (unless Date/Geo/Email/Identifier)
        let cols = vec![make_col(
            "total_amount",
            &["1.5", "2.7", "100", "42.0", "0", "999", "-3.14"],
        )];
        let profiles = classify_columns(&cols);
        assert_eq!(profiles[0].col_type, ColType::Numeric);
        assert!(profiles[0].needs_llm_escalation);
    }

    // ── is_year edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_is_year_basic() {
        assert!(is_year("2001"));
        assert!(is_year("1999"));
        assert!(is_year("2100"));
        assert!(is_year("1900"));
    }

    #[test]
    fn test_is_year_float_format() {
        assert!(is_year("1999.0"));
        assert!(is_year("2023.0"));
    }

    #[test]
    fn test_is_year_out_of_range() {
        assert!(!is_year("1899"));
        assert!(!is_year("2101"));
    }

    #[test]
    fn test_is_year_scientific_notation_blocked() {
        // "1.999e3" parses to 1999.0, but round-trip str(1999) != "1.999e3".replace(".0","") = "1.999e3"
        assert!(!is_year("1.999e3"));
    }

    #[test]
    fn test_is_year_with_suffix_blocked() {
        // "2001abc" fails to parse as f64
        assert!(!is_year("2001abc"));
    }

    #[test]
    fn test_is_year_empty() {
        assert!(!is_year(""));
    }

    #[test]
    fn test_is_year_inf() {
        assert!(!is_year("inf"));
    }
}
