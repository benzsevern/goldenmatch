"""GoldenMatch backends -- alternative data storage engines.

Backends are user-maintained. GoldenMatch reads from and writes to
the backend but does not manage schemas or migrations.
"""
from goldenmatch.backends.duckdb_backend import DuckDBBackend

__all__ = ["DuckDBBackend"]
