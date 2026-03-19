"""Database connector interface and Postgres implementation."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Iterator

import polars as pl

logger = logging.getLogger(__name__)


class DatabaseConnector(ABC):
    """Abstract interface for database connections."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection."""

    @abstractmethod
    def close(self) -> None:
        """Close connection."""

    @abstractmethod
    def read_table(self, table: str, chunk_size: int = 10000) -> Iterator[pl.DataFrame]:
        """Read table in chunks. Yields DataFrames."""

    @abstractmethod
    def read_query(self, query: str) -> pl.DataFrame:
        """Execute a SELECT query and return results as DataFrame."""

    @abstractmethod
    def write_dataframe(self, df: pl.DataFrame, table: str, mode: str = "append") -> int:
        """Write DataFrame to table. Returns rows written."""

    @abstractmethod
    def execute(self, sql: str, params: tuple | None = None) -> None:
        """Execute a SQL statement (DDL/DML)."""

    @abstractmethod
    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""

    @abstractmethod
    def get_row_count(self, table: str) -> int:
        """Get row count for a table."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


class PostgresConnector(DatabaseConnector):
    """PostgreSQL connector using psycopg2."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._conn = None

    def connect(self) -> None:
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "Postgres support requires psycopg2. "
                "Install with: pip install goldenmatch[postgres]"
            )
        self._conn = psycopg2.connect(self.connection_string)
        self._conn.autocommit = False
        logger.info("Connected to PostgreSQL")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._conn

    def read_table(self, table: str, chunk_size: int = 10000) -> Iterator[pl.DataFrame]:
        """Read table in chunks."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM {_quote_ident(table)}")
            columns = [desc[0] for desc in cursor.description]

            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
                yield pl.DataFrame(data)
        finally:
            cursor.close()

    def read_query(self, query: str) -> pl.DataFrame:
        """Execute SELECT and return as DataFrame."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(query)
            if cursor.description is None:
                return pl.DataFrame()
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            if not rows:
                return pl.DataFrame({col: [] for col in columns})
            data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
            return pl.DataFrame(data)
        finally:
            cursor.close()

    def write_dataframe(self, df: pl.DataFrame, table: str, mode: str = "append") -> int:
        """Write DataFrame to table using COPY for performance."""
        import io
        import csv

        if df.height == 0:
            return 0

        cursor = self.conn.cursor()
        try:
            if mode == "replace":
                cursor.execute(f"TRUNCATE TABLE {_quote_ident(table)}")

            # Use COPY FROM for fast bulk insert
            columns = df.columns
            # Build INSERT statements for compatibility
            placeholders = ", ".join(["%s"] * len(columns))
            col_list = ", ".join(_quote_ident(c) for c in columns)
            insert_sql = f"INSERT INTO {_quote_ident(table)} ({col_list}) VALUES ({placeholders})"

            for row in df.iter_rows():
                cursor.execute(insert_sql, row)
            self.conn.commit()

            logger.info("Wrote %d rows to %s", df.height, table)
            return df.height
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def execute(self, sql: str, params: tuple | None = None) -> None:
        """Execute DDL/DML statement."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql, params)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def table_exists(self, table: str) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table,),
            )
            return cursor.fetchone()[0]
        finally:
            cursor.close()

    def get_row_count(self, table: str) -> int:
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}")
            return cursor.fetchone()[0]
        finally:
            cursor.close()


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier to prevent injection."""
    # Simple quoting — replace any double quotes and wrap
    return '"' + name.replace('"', '""') + '"'


def create_connector(config: dict) -> DatabaseConnector:
    """Factory to create a connector from config."""
    source_type = config.get("type", "postgres")
    connection = config.get("connection") or os.environ.get("GOLDENMATCH_DATABASE_URL")

    if not connection:
        raise ValueError(
            "Database connection string required. "
            "Set in config (source.connection) or env var GOLDENMATCH_DATABASE_URL."
        )

    if source_type == "postgres":
        return PostgresConnector(connection)
    else:
        raise ValueError(f"Unsupported database type: {source_type}")
