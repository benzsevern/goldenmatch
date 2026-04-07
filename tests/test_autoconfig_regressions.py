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

import polars as pl

from goldenmatch import dedupe_df
from goldenmatch.core.autoconfig import auto_configure_df


def _person_df(n: int) -> pl.DataFrame:
    """Synthetic person dataset with the shape that exposes these bugs.

    - 3 distinct cities / 3 distinct zips → low cardinality geo/zip
    - 80 distinct birth years → low cardinality numeric (was misclassified
      as phone after upstream date transforms)
    - distinct names per row
    """
    return pl.DataFrame([
        {
            "last_name": f"Last{i:05d}",
            "first_name": f"First{i:05d}",
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
