"""Regression tests for v1.4 auto-config bugs.

Covers three issues reported against dedupe_df auto-config on person data:

1. Learned blocking auto-upgrade fires at 5K rows and trains on 100% of the
   dataset, causing multi-minute runtimes on small inputs.
2. Columns classified as col_type="zip" or "geo" are promoted into exact
   matchkeys, collapsing every record sharing a city/zip into one cluster.
   Combined with (3), this also catches low-cardinality numerics
   (e.g. birth_year misclassified as phone after a date transform).
3. DedupeResult.total_records includes golden canonical records alongside
   dupes + unique, double-counting clusters so total_records > df.height
   whenever duplicates exist.

All checks use auto_configure_df (config shape only) except for a single
small end-to-end invariant check — running dedupe_df on thousands of rows
is out of scope for a regression test because it brings in GoldenCheck,
GoldenFlow, cross-encoder reranking, and minutes of overhead unrelated to
the bugs under test.
"""
from __future__ import annotations

import time

import polars as pl

from goldenmatch import dedupe_df
from goldenmatch._api import _extract_stats
from goldenmatch.core.autoconfig import (
    ColumnProfile,
    auto_configure_df,
    build_matchkeys,
)


# Realistic surname pool that distributes across soundex codes. Using a
# pattern like f"Last{i:05d}" is a trap: every synthetic name starting with
# "Last" collapses to the same soundex code, so soundex(last_name) blocking
# lands every row in ONE giant block and downstream fuzzy scoring becomes
# O(n^2) — a 5K test that was meant to run in 15s hangs for an hour.
_SURNAMES = [
    "Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark",
    "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young",
    "King", "Wright", "Lopez",
]
_FIRSTNAMES = [
    "Alex", "Blair", "Casey", "Dana", "Eli", "Finley", "Gray", "Harper",
    "Indigo", "Jamie", "Kendall", "Logan", "Morgan", "Noel", "Oakley",
    "Parker", "Quinn", "Riley", "Sage", "Taylor", "Umi", "Val", "Wren",
    "Xena", "Yael", "Zane",
]


def _person_df(n: int) -> pl.DataFrame:
    """Synthetic person dataset with the shape that exposes these bugs.

    - 3 distinct cities / 3 distinct zips → low cardinality geo/zip
    - 80 distinct birth years → low cardinality numeric (was misclassified
      as phone after upstream date transforms)
    - surnames drawn from a 30-name pool so soundex blocking produces
      ~30 blocks of n/30 each (keeps fuzzy scoring tractable)
    - distinct addresses per row so no real duplicates
    """
    return pl.DataFrame([
        {
            "last_name": _SURNAMES[i % len(_SURNAMES)],
            "first_name": f"{_FIRSTNAMES[i % len(_FIRSTNAMES)]}{i}",
            "middle_name": f"M{i % 26}",
            "res_street_address": f"{i} Main St",
            "res_city_desc": ["Raleigh", "Durham", "Cary"][i % 3],
            "zip_code": ["27601", "27701", "27511"][i % 3],
            "birth_year": str(1930 + (i % 80)),
        }
        for i in range(n)
    ])


# ── Claim 1: learned blocking gating ───────────────────────────────────────

def test_learned_blocking_not_triggered_below_50k():
    """Auto-config must not upgrade small datasets to learned blocking.

    The old gate was >= 5000 with sample_size=5000 — on a 5K dataset the
    learner trained on 100% of its own input, producing multi-minute
    runtimes. Small datasets should use static/multi_pass blocking.
    """
    cfg = auto_configure_df(_person_df(5_000))
    assert cfg.blocking is not None
    assert cfg.blocking.strategy != "learned", (
        f"5K rows triggered learned blocking: strategy={cfg.blocking.strategy}"
    )


def test_learned_blocking_sample_size_capped_at_quarter():
    """When learned blocking does engage, sample_size must stay below 25% of
    the dataset so the learner always has held-out rows to generalize to."""
    cfg = auto_configure_df(_person_df(60_000))
    assert cfg.blocking is not None
    if cfg.blocking.strategy == "learned":
        assert cfg.blocking.learned_sample_size <= 60_000 // 4, (
            f"sample_size={cfg.blocking.learned_sample_size} exceeds 25% of "
            f"60000 rows — learner would overfit on its own input"
        )


# ── Claim 2: geo/zip/low-cardinality promoted to exact matchkeys ──────────

def test_geo_not_promoted_to_exact_matchkey():
    """col_type='geo' is a blocking signal, not an identity claim.

    Two voters sharing a city are not the same entity. Promoting city_desc
    into an exact matchkey collapses every voter in each city into one
    mega-cluster.
    """
    cfg = auto_configure_df(_person_df(500))
    for mk in cfg.get_matchkeys():
        if mk.type != "exact":
            continue
        for f in mk.fields:
            assert f.field != "res_city_desc", (
                "res_city_desc promoted to exact matchkey"
            )


def test_zip_not_promoted_to_exact_matchkey():
    """col_type='zip' is a blocking signal, not an identity claim."""
    cfg = auto_configure_df(_person_df(500))
    for mk in cfg.get_matchkeys():
        if mk.type != "exact":
            continue
        for f in mk.fields:
            assert f.field != "zip_code", (
                "zip_code promoted to exact matchkey"
            )


def test_low_cardinality_not_promoted_to_exact_matchkey():
    """A column with cardinality_ratio well below 0.5 must not back an
    exact matchkey even if its classifier says 'exact'.

    Birth year is the canonical failure mode: after an upstream date
    transform a 4-digit year can look phone-shaped (8+ stripped digits),
    the phone classifier accepts it, and with ~80 distinct values in 5K
    rows each 'exact birth_year' match collapses ~60 unrelated people.
    """
    cfg = auto_configure_df(_person_df(5_000))
    for mk in cfg.get_matchkeys():
        if mk.type != "exact":
            continue
        for f in mk.fields:
            assert f.field != "birth_year", (
                "birth_year promoted to exact matchkey despite ~80 distinct "
                "values in 5K rows (cardinality_ratio ~0.016)"
            )


# ── Claim 3: total_records double-counts golden rollup ────────────────────

def test_total_records_equals_input_row_count_with_duplicates():
    """End-to-end invariant: total_records must equal df.height regardless
    of how many duplicate clusters are produced.

    Uses a tiny fixture with two known duplicate pairs so the golden rollup
    is non-empty (which is the condition that triggered the old bug — when
    golden was empty the double-count was a no-op).
    """
    rows = [
        # Duplicate pair 1: minor spelling variation
        {"last_name": "Smith", "first_name": "John",  "middle_name": "A",
         "res_street_address": "1 Elm St", "res_city_desc": "Raleigh",
         "zip_code": "27601", "birth_year": "1980"},
        {"last_name": "Smith", "first_name": "Jon",   "middle_name": "A",
         "res_street_address": "1 Elm St", "res_city_desc": "Raleigh",
         "zip_code": "27601", "birth_year": "1980"},
        # Duplicate pair 2
        {"last_name": "Doe",   "first_name": "Jane",  "middle_name": "B",
         "res_street_address": "2 Oak Ave", "res_city_desc": "Durham",
         "zip_code": "27701", "birth_year": "1975"},
        {"last_name": "Doe",   "first_name": "Janet", "middle_name": "B",
         "res_street_address": "2 Oak Ave", "res_city_desc": "Durham",
         "zip_code": "27701", "birth_year": "1975"},
    ]
    # A handful of distinct singletons so auto-config has enough columns to
    # profile without hitting edge cases on tiny frames.
    for i in range(20):
        rows.append({
            "last_name": f"Unique{i:03d}",
            "first_name": f"Person{i:03d}",
            "middle_name": "X",
            "res_street_address": f"{i} Maple Rd",
            "res_city_desc": "Cary",
            "zip_code": "27511",
            "birth_year": str(1950 + i),
        })
    df = pl.DataFrame(rows)
    result = dedupe_df(df)
    assert result.total_records == df.height, (
        f"total_records={result.total_records} != df.height={df.height}. "
        f"Stats aggregation is double-counting the golden rollup "
        f"(golden + dupes + unique instead of dupes + unique)."
    )


# ── Positive cases: the fixes must NOT over-exclude legitimate identifiers ─

def test_high_cardinality_email_still_promoted_to_exact():
    """The raised cardinality guard (0.5) must not accidentally exclude real
    identifier columns. Email with ratio ~0.95 should still back an exact
    matchkey. This is the negative-of-the-fix test: any future tightening
    that breaks email classification must fail CI, because silently losing
    the most common dedupe key destroys recall without warning.
    """
    profiles = [
        ColumnProfile("email", "Utf8", "email", 0.95, cardinality_ratio=0.95),
        ColumnProfile("name",  "Utf8", "name",  0.9,  cardinality_ratio=0.5),
    ]
    matchkeys = build_matchkeys(profiles)
    exact_fields: list[str] = []
    for mk in matchkeys:
        if mk.type == "exact":
            exact_fields.extend(f.field for f in mk.fields)
    assert "email" in exact_fields, (
        "high-cardinality email (ratio=0.95) was not promoted to exact matchkey"
    )


def test_high_cardinality_phone_still_promoted_to_exact():
    """Same as email: phone with high cardinality must still back an exact
    matchkey. Separately tested because phone has a different col_type
    scorer lookup path and a misclassification on phone is the exact
    regression #2 exists to prevent on the *negative* side."""
    profiles = [
        ColumnProfile("phone", "Utf8", "phone", 0.95, cardinality_ratio=0.90),
        ColumnProfile("name",  "Utf8", "name",  0.9,  cardinality_ratio=0.5),
    ]
    matchkeys = build_matchkeys(profiles)
    exact_fields: list[str] = []
    for mk in matchkeys:
        if mk.type == "exact":
            exact_fields.extend(f.field for f in mk.fields)
    assert "phone" in exact_fields, (
        "high-cardinality phone (ratio=0.90) was not promoted to exact matchkey"
    )


# ── Boundary values at the new thresholds ─────────────────────────────────

def test_learned_blocking_exact_50k_boundary_triggers():
    """The gate is `>= 50_000`, so a dataset of exactly 50,000 rows MUST
    upgrade to learned blocking. Guards against an off-by-one refactor
    switching `>=` to `>`.
    """
    cfg = auto_configure_df(_person_df(50_000))
    assert cfg.blocking is not None
    assert cfg.blocking.strategy == "learned", (
        f"total_rows=50_000 did not trigger learned blocking: "
        f"strategy={cfg.blocking.strategy}"
    )


def test_learned_blocking_just_below_50k_does_not_trigger():
    """49,999 rows must stay on static/multi_pass — off-by-one guard on
    the low side of the boundary."""
    cfg = auto_configure_df(_person_df(49_999))
    assert cfg.blocking is not None
    assert cfg.blocking.strategy != "learned", (
        f"total_rows=49_999 triggered learned blocking: "
        f"strategy={cfg.blocking.strategy}"
    )


def test_cardinality_guard_exact_0_5_boundary_included():
    """The matchkey guard is `< 0.5`, so a column at exactly 0.5 MUST be
    included. This pins the contract against a future tightening that
    silently flips the comparator to `<=`.
    """
    profiles = [
        ColumnProfile("email", "Utf8", "email", 0.95, cardinality_ratio=0.5),
        ColumnProfile("name",  "Utf8", "name",  0.9,  cardinality_ratio=0.5),
    ]
    matchkeys = build_matchkeys(profiles)
    exact_fields: list[str] = []
    for mk in matchkeys:
        if mk.type == "exact":
            exact_fields.extend(f.field for f in mk.fields)
    assert "email" in exact_fields, (
        "cardinality_ratio=0.5 was excluded — guard should be strict < 0.5"
    )


def test_cardinality_guard_just_below_0_5_excluded():
    """0.4999 is below the guard and must be excluded."""
    profiles = [
        ColumnProfile("email", "Utf8", "email", 0.95, cardinality_ratio=0.4999),
        ColumnProfile("name",  "Utf8", "name",  0.9,  cardinality_ratio=0.5),
    ]
    matchkeys = build_matchkeys(profiles)
    exact_fields: list[str] = []
    for mk in matchkeys:
        if mk.type == "exact":
            exact_fields.extend(f.field for f in mk.fields)
    assert "email" not in exact_fields, (
        "cardinality_ratio=0.4999 was included — guard should exclude <0.5"
    )


# ── Interaction: single e2e run that exercises all three fixes together ───

def test_dedupe_df_interaction_all_three_fixes_together():
    """One end-to-end run that would hit all three bugs simultaneously if
    any regressed:

    - Bug 1 (learned blocking auto-upgrade): pre-fix this triggered at
      >= 5K rows and hung for minutes. A 500-row run must finish fast.
    - Bug 2 (geo/zip/low-card → exact matchkey): pre-fix this collapsed
      every record per city/zip/birth-year into mega-clusters. We bound
      the largest cluster at 10 members.
    - Bug 3 (total_records double-count): pre-fix this exceeded df.height
      by the number of multi-member clusters. We assert strict equality.

    Uses 500 rows rather than 5000 — big enough that all three guards
    actually engage (blocking falls through geo/zip guards, cardinality
    guard fires on birth_year, config shape is non-trivial) but small
    enough to run in a handful of seconds including cross-encoder warm-up.
    A 5K run on the same synthetic shape would take significantly longer
    because the surname pool is small (by design) and ensemble scoring
    grows quadratically inside each soundex block.
    """
    df = _person_df(500)
    t0 = time.perf_counter()
    result = dedupe_df(df)
    elapsed = time.perf_counter() - t0

    # Bug 1: runtime must not explode. 90s is generous for a 500-row run
    # (actual runtime is ~15s with cold-start cross-encoder loading).
    assert elapsed < 90.0, (
        f"dedupe_df(500) took {elapsed:.1f}s — expected < 90s. "
        f"Likely cause: a blocking / matchkey fix regressed and the "
        f"scorer is doing O(n^2) work inside a single huge block."
    )
    # Bug 2: no mega-clusters. Synthetic data has distinct addresses and
    # rotating surnames/firstnames, so real clusters (if any from ensemble
    # noise) should be tiny. Any cluster > 10 members indicates a
    # geo/zip/low-cardinality column regressed back into an exact matchkey.
    if result.clusters:
        largest = max(len(c["members"]) for c in result.clusters.values())
        assert largest <= 10, (
            f"Largest cluster has {largest} members. Expected <= 10 on "
            f"synthetic data with distinct addresses. Likely cause: "
            f"geo/zip or low-cardinality column promoted back into an "
            f"exact matchkey."
        )
    # Bug 3: row count invariant.
    assert result.total_records == df.height, (
        f"total_records={result.total_records} != df.height={df.height}. "
        f"Likely cause: _extract_stats re-added golden to the sum."
    )


# ── _extract_stats edge cases (contract pinning) ──────────────────────────

def test_extract_stats_golden_only_warns_not_silent_zero():
    """If a pipeline path ever produces golden without dupes/unique, the
    stats helper must surface it as a warning rather than silently returning
    total_records=0 (which would make match_rate a meaningless 0/0).

    This is a contract test — the standard pipeline always materializes
    all three, so if this shape ever starts appearing a refactor changed
    _api.py's output contract and _extract_stats needs revisiting.
    """
    import logging
    caplog_records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            caplog_records.append(record)

    handler = _Capture()
    api_logger = logging.getLogger("goldenmatch._api")
    api_logger.addHandler(handler)
    api_logger.setLevel(logging.WARNING)
    try:
        stats = _extract_stats({
            "golden": pl.DataFrame({"x": [1, 2, 3]}),
            "dupes": None,
            "unique": None,
            "clusters": {0: {"size": 3, "members": [1, 2, 3], "pair_scores": {}}},
        })
    finally:
        api_logger.removeHandler(handler)

    assert stats["total_records"] == 0
    # Must have warned — silent zero is the exact failure mode we want to
    # prevent because match_rate divides by total_records.
    assert any("golden" in r.getMessage() for r in caplog_records), (
        "golden-only result shape returned zeroed stats with no warning"
    )


def test_extract_stats_all_none_returns_empty():
    """All-None input is a valid empty-result shape (no records at all)
    and must return zero stats cleanly without raising or warning about
    the golden-only edge case.
    """
    stats = _extract_stats({
        "golden": None, "dupes": None, "unique": None, "clusters": {},
    })
    assert stats["total_records"] == 0
    assert stats["total_clusters"] == 0
    assert stats["match_rate"] == 0.0
