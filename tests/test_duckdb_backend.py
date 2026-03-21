"""Tests for DuckDB backend."""
from __future__ import annotations

import polars as pl
import pytest

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

pytestmark = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


class TestDuckDBBackend:
    def test_create_and_read(self):
        from goldenmatch.backends.duckdb_backend import DuckDBBackend

        backend = DuckDBBackend(":memory:")
        backend.execute("CREATE TABLE test (id INT, name VARCHAR)")
        backend.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")

        lf = backend.read_table("test")
        df = lf.collect()
        assert df.height == 2
        assert "name" in df.columns
        backend.close()

    def test_read_with_query(self):
        from goldenmatch.backends.duckdb_backend import DuckDBBackend

        backend = DuckDBBackend(":memory:")
        backend.execute("CREATE TABLE test (id INT, name VARCHAR, active BOOL)")
        backend.execute("INSERT INTO test VALUES (1, 'Alice', true), (2, 'Bob', false)")

        lf = backend.read_table("test", query="SELECT * FROM test WHERE active")
        df = lf.collect()
        assert df.height == 1
        assert df["name"][0] == "Alice"
        backend.close()

    def test_write_table(self):
        from goldenmatch.backends.duckdb_backend import DuckDBBackend

        backend = DuckDBBackend(":memory:")
        backend.execute("CREATE TABLE results (id INT, cluster_id INT)")

        df = pl.DataFrame({"id": [1, 2, 3], "cluster_id": [1, 1, 2]})
        backend.write_table(df, "results", mode="append")

        result = backend.read_table("results").collect()
        assert result.height == 3
        backend.close()

    def test_write_replace(self):
        from goldenmatch.backends.duckdb_backend import DuckDBBackend

        backend = DuckDBBackend(":memory:")
        backend.execute("CREATE TABLE results (id INT, name VARCHAR)")
        backend.execute("INSERT INTO results VALUES (1, 'Old')")

        df = pl.DataFrame({"id": [2], "name": ["New"]})
        backend.write_table(df, "results", mode="replace")

        result = backend.read_table("results").collect()
        assert result.height == 1
        assert result["name"][0] == "New"
        backend.close()

    def test_list_tables(self):
        from goldenmatch.backends.duckdb_backend import DuckDBBackend

        backend = DuckDBBackend(":memory:")
        backend.execute("CREATE TABLE foo (id INT)")
        backend.execute("CREATE TABLE bar (id INT)")

        tables = backend.list_tables()
        assert "foo" in tables
        assert "bar" in tables
        backend.close()

    def test_file_backend(self, tmp_path):
        from goldenmatch.backends.duckdb_backend import DuckDBBackend

        db_path = str(tmp_path / "test.duckdb")
        backend = DuckDBBackend(db_path)
        backend.execute("CREATE TABLE test (id INT, name VARCHAR)")
        backend.execute("INSERT INTO test VALUES (1, 'Alice')")
        backend.close()

        # Reopen and verify persistence
        backend2 = DuckDBBackend(db_path)
        df = backend2.read_table("test").collect()
        assert df.height == 1
        backend2.close()
