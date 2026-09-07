"""Cost-aware blocking primary-key selection (#2021).

The exact-blocking "best case" branch picked a small-fixed-domain ``year``/``date``
column (e.g. ``birth_year``, ~65 distinct) as the SOLE primary blocking key on
identifier-poor person data -- a key whose block grows proportional to N, so it
explodes candidate pairs at scale (~7.7B at 1M). ``GOLDENMATCH_BLOCKING_COST_AWARE=1``
demotes such a key from the primary slot when a bounded fallback (a name key or a
bounded compound) exists; it stays available as a recall pass (#438).

Default OFF is byte-identical; these tests lock both states.
"""
import types

import goldenmatch
import polars as pl
from goldenmatch.core.autoconfig import (
    _cost_aware_blocking_enabled,
    _is_bibliographic_dataset,
)


def _year_pathology_df(n_clusters: int = 400, per: int = 5) -> pl.DataFrame:
    """Person-shape dedupe frame with the birth_year pathology: per-cluster names
    (bounded, ideal blocking keys) + a low-cardinality year column shared within a
    cluster + a near-unique email. No clean single identifier."""
    rows = []
    for c in range(n_clusters):
        year = 1950 + (c % 60)  # ~60 distinct years -> low-card blocking key
        for m in range(per):
            rows.append({
                "id": f"{c}-{m}",                       # unique surrogate (card 1.0)
                "first_name": f"First{c}",              # per-cluster, bounded
                "last_name": f"Last{c}",                # per-cluster, bounded
                "birth_year": str(year),                # low-card year
                "email": f"e{c}_{m % 3}@ex.com",        # mid-card: ~3 distinct/cluster
                "city": f"City{c % 8}",                 # geo
            })
    return pl.DataFrame(rows)


def _biblio_year_pathology_df(n_clusters: int = 400, per: int = 5) -> pl.DataFrame:
    """The SAME cardinality pathology as ``_year_pathology_df`` but with
    BIBLIOGRAPHIC column names (title/authors/year/venue): the ``year`` is a
    publication year, a legitimately strong same-year blocking signal. Used to prove
    the cost-aware demotion is domain-routed OFF here (it must NOT demote year)."""
    rows = []
    for c in range(n_clusters):
        year = 1950 + (c % 60)
        for m in range(per):
            rows.append({
                "doi": f"{c}-{m}",                      # unique surrogate
                "title": f"Title{c}",                   # per-cluster
                "authors": f"Authors{c}",               # per-cluster
                "year": str(year),                      # low-card publication year
                "venue": f"Venue{c % 8}",               # low-card venue
            })
    return pl.DataFrame(rows)


def _blocking_key_fields(cfg) -> list[list[str]]:
    bl = cfg.blocking
    return [list(k.fields) for k in (bl.keys or [])]


def _bc_keys(bl) -> list[list[str]]:
    """Key fields off a raw BlockingConfig (build_blocking's return)."""
    return [list(k.fields) for k in (bl.keys or [])]


def test_is_bibliographic_dataset_predicate():
    def _p(*names):
        return [types.SimpleNamespace(name=n) for n in names]

    # Person shape (first_name/last_name/email/birth_year) -> NOT bibliographic
    # (person signals dominate; birth_year's "year"/"birth" tokens don't flip it).
    assert _is_bibliographic_dataset(
        _p("id", "first_name", "last_name", "birth_year", "email", "city")
    ) is False
    # Bibliographic shape -> True.
    assert _is_bibliographic_dataset(
        _p("doi", "title", "authors", "year", "venue")
    ) is True
    # No signal at all -> not bibliographic (fail-safe: demotion applies).
    assert _is_bibliographic_dataset(_p("a", "b", "c")) is False


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv("GOLDENMATCH_BLOCKING_COST_AWARE", raising=False)
    assert _cost_aware_blocking_enabled() is False
    for on in ("1", "true", "YES", "on"):
        monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", on)
        assert _cost_aware_blocking_enabled() is True
    for off in ("0", "false", "no", ""):
        monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", off)
        assert _cost_aware_blocking_enabled() is False


def test_cost_aware_demotes_year_primary(monkeypatch):
    df = _year_pathology_df()

    # Flag OFF: the pathology -- a single low-card year column is the sole primary.
    monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", "0")
    cfg_off = goldenmatch.auto_configure_df(df)
    keys_off = _blocking_key_fields(cfg_off)
    assert keys_off == [["birth_year"]], (
        f"expected the year-only pathology under flag OFF, got {keys_off}"
    )

    # Flag ON: birth_year is demoted from the primary slot; the committed primary
    # is a bounded key (name / geo / email / compound), NOT the sole year column.
    monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", "1")
    cfg_on = goldenmatch.auto_configure_df(df)
    keys_on = _blocking_key_fields(cfg_on)
    assert keys_on and keys_on != [["birth_year"]], (
        f"cost-aware should demote the sole year primary, got {keys_on}"
    )
    # The primary must not be a lone low-cardinality year/date field.
    assert keys_on[0] != ["birth_year"], keys_on


def test_cost_aware_preserves_year_on_bibliographic(monkeypatch):
    # Domain routing: on BIBLIOGRAPHIC data the year is a publication year (a strong
    # same-year blocking signal), so the cost-aware demotion must be SKIPPED -- flag
    # ON must produce the SAME blocking as flag OFF (year preserved, not demoted).
    # Exercise build_blocking directly (the routing site) to avoid the full-pipeline
    # scoring of the synthetic biblio frame.
    from goldenmatch.core.autoconfig import build_blocking, profile_columns

    df = _biblio_year_pathology_df()
    profiles = profile_columns(df)

    monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", "0")
    keys_off = _bc_keys(build_blocking(profiles, df))

    monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", "1")
    keys_on = _bc_keys(build_blocking(profiles, df))

    assert keys_on == keys_off, (
        f"bibliographic data must be exempt from cost-aware demotion: "
        f"OFF={keys_off} ON={keys_on}"
    )


def test_cost_aware_demotes_year_on_person_at_build_blocking(monkeypatch):
    # Contrast to the bibliographic case above, through the SAME build_blocking seam:
    # on PERSON-named columns the year IS demoted under the flag.
    from goldenmatch.core.autoconfig import build_blocking, profile_columns

    df = _year_pathology_df()
    profiles = profile_columns(df)

    monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", "0")
    keys_off = _bc_keys(build_blocking(profiles, df))
    monkeypatch.setenv("GOLDENMATCH_BLOCKING_COST_AWARE", "1")
    keys_on = _bc_keys(build_blocking(profiles, df))

    # OFF picks the lone year; ON demotes it (routing did NOT exempt person data).
    assert keys_off == [["birth_year"]], keys_off
    assert keys_on != [["birth_year"]], keys_on


def _biblio_title_coarser_than_year_df(n_clusters: int = 300, per: int = 5) -> pl.DataFrame:
    """Bibliographic data where `title` (the `__title_key__` stand-in) has
    MORE distinct values than `year` (100 vs. 60, matching real DBLP-ACM
    where the title-derived key beats year on raw cardinality) while still
    being a COARSER grouping than the true cluster: many different papers
    collide on a title bucket, but two papers in the same bucket usually
    disagree on `year`. Each real cluster is internally consistent on both
    fields (every member shares its own title bucket AND year), so `year`
    never breaks up a true match -- it only splits together different
    clusters that happen to collide on the title bucket. This is the
    DBLP-ACM shape: `__title_key__` (the first significant word) collides
    across unrelated papers; `year` is free extra selectivity, not a
    recall risk.
    """
    rows = []
    for c in range(n_clusters):
        title_bucket = c % 100  # coarser than n_clusters, but > year's domain
        year = 1950 + (c % 60)
        for m in range(per):
            rows.append({
                "doi": f"{c}-{m}",
                "title": f"Title{title_bucket}",
                "authors": f"Authors{c}",
                "year": str(year),
                "venue": f"Venue{c % 8}",
            })
    return pl.DataFrame(rows)


def test_exact_pool_compounds_year_instead_of_picking_highest_cardinality_alone():
    """#2633: on real DBLP-ACM, the domain-extracted `__title_key__` column is
    injected into blocking candidates with `col_type="exact_derived"`
    (`autoconfig.py`'s exact-domain-column injection tags every exact-scored
    domain column that way -- see `_EXACT_DERIVED_COL_TYPE`, #2876). That's
    what makes it `exact_cols`-eligible alongside `year` on bibliographic
    data (both survive `_is_scale_safe` into `safe_exact`).

    Once both are in `safe_exact`, the branch used to do
    ``best = max(safe_exact, key=n_unique)`` and commit `best` ALONE --
    title_key (higher cardinality) always beats year (lower cardinality) on
    raw distinct-value count, so `year` was measured, present, and never
    used. On real DBLP-ACM this produces 33,563 candidate pairs where
    `title_key + year` produces 5,749 at IDENTICAL recall (every true match
    shares its publication year -- the same domain-routed trust
    `_is_bibliographic_dataset` already relies on to keep year off the
    demotion path above).

    A compound of two exact keys is a strict refinement (AND) of `best`
    alone -- it can only keep-or-shrink `best`'s blocks, never grow them,
    as long as every true match agrees on both fields (asserted by
    construction in the fixture) -- so this is free once the eligibility
    question is already settled by biblio-domain routing.
    """
    import dataclasses

    from goldenmatch.core.autoconfig import build_blocking, profile_columns

    df = _biblio_title_coarser_than_year_df()
    profiles = profile_columns(df)
    # Stand-in for the real __title_key__ injection: force `title`'s profile
    # to col_type="exact_derived", exactly as the real injection does for any
    # exact-scored domain-extracted column, so it enters exact_cols the same
    # way it does on real DBLP-ACM.
    profiles = [
        dataclasses.replace(p, col_type="exact_derived") if p.name == "title" else p
        for p in profiles
    ]

    cfg = build_blocking(profiles, df)
    keys = _bc_keys(cfg)
    assert keys and set(keys[0]) == {"title", "year"}, (
        f"expected a compound of the two safe_exact candidates (title + "
        f"year); got {keys} -- picking the single highest-cardinality exact "
        f"column measures the defect this issue reports, not a fix for it"
    )

    # And it must actually reduce candidates, not just add a no-op field.
    solo_block = df.group_by("title").len().select(pl.col("len").max()).item()
    compound_block = df.group_by(["title", "year"]).len().select(pl.col("len").max()).item()
    assert compound_block < solo_block, (
        f"fixture doesn't demonstrate the shrink this fix targets: "
        f"solo={solo_block} compound={compound_block}"
    )


def test_off_is_default(monkeypatch):
    # No env set -> OFF (byte-identical legacy behaviour).
    monkeypatch.delenv("GOLDENMATCH_BLOCKING_COST_AWARE", raising=False)
    df = _year_pathology_df()
    cfg = goldenmatch.auto_configure_df(df)
    assert _blocking_key_fields(cfg) == [["birth_year"]]


def test_standardization_config_skips_domain_derived_column():
    """#2876: a domain-extracted exact column (col_type="exact_derived", e.g.
    __title_key__) must NOT get a standardization rule. It previously carried
    col_type="email", so _detect_standardization_config's email branch built
    a rule applying std_email -- which nulls any value lacking "@" -- to a
    column that is never an email address. A real email column must still get
    its rule."""
    from goldenmatch.core.autoconfig import ColumnProfile, _detect_standardization_config

    profiles = [
        ColumnProfile(
            name="__title_key__", dtype="Utf8", col_type="exact_derived",
            confidence=0.9, null_rate=0.0, cardinality_ratio=0.5, avg_len=0,
        ),
        ColumnProfile(
            name="contact_email", dtype="Utf8", col_type="email",
            confidence=0.9, null_rate=0.0, cardinality_ratio=0.9, avg_len=10,
        ),
    ]
    cfg = _detect_standardization_config(profiles)
    assert cfg is not None
    assert "__title_key__" not in cfg.rules, (
        f"domain-derived column must not get a standardization rule; "
        f"got {cfg.rules}"
    )
    assert cfg.rules.get("contact_email") == ["email"], cfg.rules


def test_real_title_key_injection_not_routed_to_email_standardizer():
    """#2876 end-to-end: the REAL __title_key__ injection (auto_configure_df's
    domain-column append) must not emit a StandardizationConfig rule for it,
    and the #2633 title+year compound must still be produced with the
    corrected col_type."""
    df = _biblio_title_coarser_than_year_df()
    cfg = goldenmatch.auto_configure_df(df, allow_red_config=True)

    rules = cfg.standardization.rules if cfg.standardization else {}
    assert "__title_key__" not in rules, (
        f"domain-derived __title_key__ must not get a standardization rule "
        f"(it is not an email); rules={rules}"
    )

    keys = _bc_keys(cfg.blocking)
    assert keys and set(keys[0]) == {"__title_key__", "year"}, (
        f"the #2633 compound must still fire with the corrected col_type; got {keys}"
    )
