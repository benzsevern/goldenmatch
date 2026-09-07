"""#2893: `lookup_entity_ids` must not chunk at SQLite's parameter limit on Postgres.

The 900-id chunk in `lookup_entity_ids` exists for
`SQLITE_MAX_VARIABLE_NUMBER` (#670 -- a single IN-list over 1M+ ids raised
"too many SQL variables"). It was applied to every SQL backend, so on
Postgres it turned `apply_batch`'s ONE bulk pre-flight into a round trip per
900 ids.

Measured at 5M rows (run 34148189911): 50.82 s in that single
`lookup_entity_ids` call on Postgres vs 0.26 s for the same logical work on
SQLite -- 195x, with the network already removed (co-located services
container). ~5,556 round trips.

Postgres takes the whole set as one array parameter (`= ANY($1)`), so the
guard here is on ROUND-TRIP COUNT, not just correctness: a correctness-only
test would pass just as happily if someone reinstated per-900 chunking, which
is the actual regression.
"""

from __future__ import annotations

import pytest
from goldenmatch.identity import IdentityNode, IdentityStore, SourceRecord, new_entity_id

from .._pg_helpers import pg_url_fixture


@pytest.fixture()
def pg_url():
    yield from pg_url_fixture()


@pytest.fixture()
def pg_store(pg_url):
    store = IdentityStore(backend="postgres", connection=pg_url.url())
    yield store
    store.close()


def test_pg_lookup_entity_ids_is_one_roundtrip_per_100k_not_per_900(pg_store):
    """N ids must cost ceil(N/100_000) queries, not ceil(N/900)."""
    n = 2500  # spans 3 chunks under the old 900 cap, 1 under the new one
    eid = new_entity_id()
    pg_store.upsert_identity(IdentityNode(entity_id=eid))
    for i in range(n):
        pg_store.upsert_record(SourceRecord(f"a:{i}", "a", str(i), "h", entity_id=eid))

    calls: list[str] = []
    real_fetchall = pg_store._fetchall

    def counting_fetchall(sql, params):
        calls.append(sql)
        return real_fetchall(sql, params)

    pg_store._fetchall = counting_fetchall  # type: ignore[method-assign]
    try:
        out = pg_store.lookup_entity_ids([f"a:{i}" for i in range(n)] + ["a:missing"])
    finally:
        pg_store._fetchall = real_fetchall  # type: ignore[method-assign]

    # Correctness is unchanged vs the IN-list form: ANY(array) is
    # equality-equivalent, and the absent id must still be absent.
    assert out == {f"a:{i}": eid for i in range(n)}

    # The actual regression guard. Under the old shared 900-chunk this was 3.
    assert len(calls) == 1, (
        f"expected a single round trip for {n} ids on postgres, got {len(calls)} "
        "-- the SQLite 900-parameter chunk has been reapplied to postgres (#2893)"
    )
    assert "= ANY(" in calls[0], f"expected an array parameter, got: {calls[0]}"


def test_pg_lookup_entity_ids_empty_and_missing(pg_store):
    """Empty input short-circuits; unknown ids simply do not appear."""
    assert pg_store.lookup_entity_ids([]) == {}
    assert pg_store.lookup_entity_ids(["nope:1", "nope:2"]) == {}


def test_pg_lookup_entity_ids_skips_records_with_no_entity(pg_store):
    """`entity_id IS NOT NULL` filtering survives the rewrite."""
    eid = new_entity_id()
    pg_store.upsert_identity(IdentityNode(entity_id=eid))
    pg_store.upsert_record(SourceRecord("b:1", "b", "1", "h", entity_id=eid))
    pg_store.upsert_record(SourceRecord("b:2", "b", "2", "h"))  # no entity

    assert pg_store.lookup_entity_ids(["b:1", "b:2"]) == {"b:1": eid}
