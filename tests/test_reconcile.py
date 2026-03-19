"""Tests for reconciliation, cluster management, and golden record versioning."""

from __future__ import annotations

import pytest

try:
    import testing.postgresql
    import psycopg2
    HAS_POSTGRES = True
except (ImportError, Exception):
    HAS_POSTGRES = False


@pytest.fixture
def pg():
    if not HAS_POSTGRES:
        pytest.skip("PostgreSQL not available")
    with testing.postgresql.Postgresql() as postgresql:
        yield postgresql


@pytest.fixture
def connector(pg):
    from goldenmatch.db.connector import PostgresConnector
    conn = PostgresConnector(pg.url())
    conn.connect()
    yield conn
    conn.close()


@pytest.fixture
def setup_tables(connector):
    """Create source table + metadata tables."""
    from goldenmatch.db.metadata import ensure_metadata_tables
    ensure_metadata_tables(connector)

    connector.execute("""
        CREATE TABLE customers (
            id SERIAL PRIMARY KEY,
            name TEXT,
            email TEXT,
            zip TEXT
        )
    """)
    connector.execute("""
        INSERT INTO customers (name, email, zip) VALUES
        ('John Smith', 'john@test.com', '10001'),
        ('Jane Doe', 'jane@test.com', '90210'),
        ('Jon Smith', 'jon@test.com', '10001'),
        ('Bob Johnson', 'bob@test.com', '30301'),
        ('Janet Doe', 'janet@test.com', '90210')
    """)
    return "customers"


# ── Cluster Management Tests ──────────────────────────────────────────────

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not available")
class TestClusterManagement:
    def test_create_cluster(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster, get_cluster_members

        cid = create_cluster(connector, [1, 2], "customers", "run-1")
        assert cid >= 1
        members = get_cluster_members(connector, cid)
        assert sorted(members) == [1, 2]

    def test_add_to_cluster(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster, add_to_cluster, get_cluster_members

        cid = create_cluster(connector, [1], "customers", "run-1")
        add_to_cluster(connector, cid, [2, 3], "customers", "run-2")
        members = get_cluster_members(connector, cid)
        assert sorted(members) == [1, 2, 3]

    def test_merge_clusters(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster, merge_clusters, get_cluster_members

        cid1 = create_cluster(connector, [1, 2], "customers", "run-1")
        cid2 = create_cluster(connector, [3, 4], "customers", "run-1")

        survivor = merge_clusters(connector, [cid1, cid2], "customers", "run-2")
        members = get_cluster_members(connector, survivor)
        assert sorted(members) == [1, 2, 3, 4]

    def test_get_cluster_for_record(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster, get_cluster_for_record

        cid = create_cluster(connector, [1, 2], "customers", "run-1")
        assert get_cluster_for_record(connector, 1, "customers") == cid
        assert get_cluster_for_record(connector, 99, "customers") is None

    def test_cluster_size(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster, get_cluster_size

        cid = create_cluster(connector, [1, 2, 3], "customers", "run-1")
        assert get_cluster_size(connector, cid) == 3

    def test_next_cluster_id(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster, next_cluster_id

        cid1 = create_cluster(connector, [1], "customers", "run-1")
        cid2 = create_cluster(connector, [2], "customers", "run-1")
        assert cid2 > cid1


# ── Reconciliation Tests ─────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not available")
class TestReconciliation:
    def test_new_entity(self, connector, setup_tables):
        from goldenmatch.db.reconcile import reconcile_match

        record = {"id": 1, "name": "John Smith", "email": "john@test.com"}
        result = reconcile_match(
            record, 1, [], {},
            connector, "customers",
            run_id="run-1",
        )
        assert result.action == "new"
        assert result.cluster_id >= 1
        assert result.golden_record["name"] == "John Smith"

    def test_single_match(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster
        from goldenmatch.db.reconcile import reconcile_match

        # Create existing cluster
        cid = create_cluster(connector, [1, 3], "customers", "run-1")

        # New record matches cluster
        record = {"id": 5, "name": "Johnny Smith", "email": "johnny@test.com"}
        result = reconcile_match(
            record, 5, [cid], {cid: 0.9},
            connector, "customers",
            run_id="run-2",
        )
        assert result.action == "merged"
        assert result.cluster_id == cid

    def test_multi_match_under_cap(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster
        from goldenmatch.db.reconcile import reconcile_match

        cid1 = create_cluster(connector, [1], "customers", "run-1")
        cid2 = create_cluster(connector, [2], "customers", "run-1")

        record = {"id": 5, "name": "Bridge Record"}
        result = reconcile_match(
            record, 5, [cid1, cid2], {cid1: 0.8, cid2: 0.7},
            connector, "customers",
            max_cluster_size=100,
            run_id="run-2",
        )
        assert result.action == "merged"
        assert len(result.merged_clusters) == 2

    def test_multi_match_over_cap(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster
        from goldenmatch.db.reconcile import reconcile_match

        cid1 = create_cluster(connector, [1, 2], "customers", "run-1")
        cid2 = create_cluster(connector, [3, 4], "customers", "run-1")

        record = {"id": 5, "name": "Bridge Record"}
        result = reconcile_match(
            record, 5, [cid1, cid2], {cid1: 0.8, cid2: 0.7},
            connector, "customers",
            max_cluster_size=3,  # too small for merge
            run_id="run-2",
        )
        assert result.action == "conflict"
        assert result.cluster_id == cid1  # best scoring


# ── Golden Record Versioning Tests ────────────────────────────────────────

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not available")
class TestGoldenVersioning:
    def test_first_version(self, connector, setup_tables):
        from goldenmatch.db.reconcile import reconcile_match

        record = {"id": 1, "name": "John Smith"}
        result = reconcile_match(
            record, 1, [], {},
            connector, "customers", run_id="run-1",
        )

        df = connector.read_query(
            f"SELECT * FROM gm_golden_records WHERE cluster_id = {result.cluster_id}"
        )
        assert df.height == 1
        assert df["is_current"][0] is True
        assert df["version"][0] == 1

    def test_version_increment(self, connector, setup_tables):
        from goldenmatch.db.clusters import create_cluster
        from goldenmatch.db.reconcile import reconcile_match

        # Create initial cluster + golden record
        result1 = reconcile_match(
            {"id": 1, "name": "John Smith"}, 1, [], {},
            connector, "customers", run_id="run-1",
        )
        cid = result1.cluster_id

        # Add another record → should create version 2
        result2 = reconcile_match(
            {"id": 3, "name": "Jon Smith"}, 3, [cid], {cid: 0.9},
            connector, "customers", run_id="run-2",
        )

        df = connector.read_query(
            f"SELECT * FROM gm_golden_records WHERE cluster_id = {cid} ORDER BY version"
        )
        assert df.height == 2
        assert df["version"].to_list() == [1, 2]
        # Only latest is current
        assert df["is_current"].to_list() == [False, True]

    def test_query_current_only(self, connector, setup_tables):
        from goldenmatch.db.reconcile import reconcile_match

        # Create two versions
        r1 = reconcile_match(
            {"id": 1, "name": "John"}, 1, [], {},
            connector, "customers", run_id="run-1",
        )
        reconcile_match(
            {"id": 2, "name": "Jane"}, 2, [r1.cluster_id], {r1.cluster_id: 0.9},
            connector, "customers", run_id="run-2",
        )

        current = connector.read_query(
            "SELECT * FROM gm_golden_records WHERE is_current = TRUE"
        )
        assert current.height == 1
        assert current["version"][0] == 2
