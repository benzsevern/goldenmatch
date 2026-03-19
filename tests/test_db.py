"""Tests for database connector, metadata, blocking, and sync."""

from __future__ import annotations

import pytest
import polars as pl

# Check if testing.postgresql is available
try:
    import testing.postgresql
    import psycopg2
    HAS_POSTGRES = True
except (ImportError, Exception):
    HAS_POSTGRES = False


@pytest.fixture
def pg():
    """Create a temporary PostgreSQL instance."""
    if not HAS_POSTGRES:
        pytest.skip("testing.postgresql or PostgreSQL not available")
    with testing.postgresql.Postgresql() as postgresql:
        yield postgresql


@pytest.fixture
def connector(pg):
    """Create a PostgresConnector connected to temp Postgres."""
    from goldenmatch.db.connector import PostgresConnector
    conn = PostgresConnector(pg.url())
    conn.connect()
    yield conn
    conn.close()


@pytest.fixture
def sample_table(connector):
    """Create a sample customers table with test data."""
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


# ── Connector Tests ───────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not available")
class TestPostgresConnector:
    def test_connect_and_close(self, pg):
        from goldenmatch.db.connector import PostgresConnector
        conn = PostgresConnector(pg.url())
        conn.connect()
        assert conn._conn is not None
        conn.close()
        assert conn._conn is None

    def test_context_manager(self, pg):
        from goldenmatch.db.connector import PostgresConnector
        with PostgresConnector(pg.url()) as conn:
            assert conn._conn is not None
        assert conn._conn is None

    def test_table_exists(self, connector, sample_table):
        assert connector.table_exists("customers") is True
        assert connector.table_exists("nonexistent") is False

    def test_get_row_count(self, connector, sample_table):
        assert connector.get_row_count("customers") == 5

    def test_read_table(self, connector, sample_table):
        chunks = list(connector.read_table("customers", chunk_size=2))
        total_rows = sum(chunk.height for chunk in chunks)
        assert total_rows == 5
        assert len(chunks) == 3  # 2 + 2 + 1

    def test_read_query(self, connector, sample_table):
        df = connector.read_query("SELECT * FROM customers WHERE zip = '10001'")
        assert df.height == 2

    def test_write_dataframe(self, connector):
        connector.execute("CREATE TABLE test_write (name TEXT, value INT)")
        df = pl.DataFrame({"name": ["a", "b"], "value": [1, 2]})
        rows = connector.write_dataframe(df, "test_write")
        assert rows == 2
        assert connector.get_row_count("test_write") == 2

    def test_execute(self, connector):
        connector.execute("CREATE TABLE test_exec (id SERIAL PRIMARY KEY)")
        assert connector.table_exists("test_exec") is True


# ── Metadata Tests ────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not available")
class TestMetadata:
    def test_ensure_metadata_tables(self, connector):
        from goldenmatch.db.metadata import ensure_metadata_tables
        ensure_metadata_tables(connector)
        assert connector.table_exists("gm_state") is True
        assert connector.table_exists("gm_embeddings") is True
        assert connector.table_exists("gm_match_log") is True
        assert connector.table_exists("gm_golden_records") is True

    def test_idempotent_creation(self, connector):
        from goldenmatch.db.metadata import ensure_metadata_tables
        ensure_metadata_tables(connector)
        ensure_metadata_tables(connector)  # should not fail

    def test_state_roundtrip(self, connector):
        from goldenmatch.db.metadata import ensure_metadata_tables, get_state, update_state
        ensure_metadata_tables(connector)

        assert get_state(connector, "customers") is None

        update_state(connector, "customers", last_row_id=100, cfg_hash="abc123", record_count=100)
        state = get_state(connector, "customers")
        assert state is not None
        assert state["source_table"] == "customers"
        assert state["last_row_id"] == 100
        assert state["config_hash"] == "abc123"

    def test_match_logging(self, connector):
        from goldenmatch.db.metadata import ensure_metadata_tables, log_matches_batch, new_run_id
        ensure_metadata_tables(connector)

        run_id = new_run_id()
        matches = [(1, 2, 0.95, "merged"), (3, 4, 0.30, "skipped")]
        log_matches_batch(connector, matches, run_id)

        df = connector.read_query("SELECT * FROM gm_match_log")
        assert df.height == 2


# ── Blocking Tests ────────────────────────────────────────────────────────

class TestSQLBlocking:
    def test_exact_blocking_query(self):
        from goldenmatch.db.blocking import build_blocking_query
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=["strip"])],
        )
        record = {"zip": "10001", "name": "John"}

        query = build_blocking_query("customers", record, config, exclude_id=1)
        assert "WHERE" in query
        assert "10001" in query
        assert "!= 1" in query

    def test_soundex_blocking_query(self):
        from goldenmatch.db.blocking import build_blocking_query
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["name"], transforms=["soundex"])],
        )
        record = {"name": "Smith"}

        query = build_blocking_query("customers", record, config)
        assert "soundex" in query

    def test_substring_transform(self):
        from goldenmatch.db.blocking import build_blocking_query
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"])],
        )
        record = {"name": "Johnson"}

        query = build_blocking_query("customers", record, config)
        assert "lower" in query
        assert "substring" in query

    def test_null_value_skipped(self):
        from goldenmatch.db.blocking import build_blocking_query
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
        )
        record = {"zip": None}

        query = build_blocking_query("customers", record, config)
        assert query == ""


# ── Integration Test ──────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not available")
class TestSyncIntegration:
    def test_full_scan_sync(self, connector, sample_table):
        from goldenmatch.db.sync import run_sync
        from goldenmatch.db.metadata import ensure_metadata_tables
        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
            BlockingConfig, BlockingKeyConfig, OutputConfig, GoldenRulesConfig,
        )

        config = GoldenMatchConfig(
            matchkeys=[
                MatchkeyConfig(
                    name="name_zip",
                    type="weighted",
                    threshold=0.80,
                    fields=[
                        MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7, transforms=["lowercase"]),
                        MatchkeyField(field="zip", scorer="exact", weight=0.3),
                    ],
                ),
            ],
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["zip"])],
            ),
            golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
            output=OutputConfig(),
        )

        results = run_sync(
            connector=connector,
            source_table="customers",
            config=config,
            full_rescan=True,
            dry_run=True,
        )

        assert results["new_records"] == 5
        assert isinstance(results["matches"], int)
        assert results["run_id"] is not None
