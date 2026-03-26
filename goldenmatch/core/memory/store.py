"""MemoryStore -- SQLite/Postgres persistence for corrections and adjustments."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger("goldenmatch.memory")


@dataclass
class Correction:
    """A single pair decision stored in memory."""
    id: str
    id_a: int
    id_b: int
    decision: str                # "approve" | "reject"
    source: str                  # "steward" | "boost" | "unmerge" | "agent" | "llm"
    trust: float                 # 1.0 (human) or 0.5 (agent)
    field_hash: str
    record_hash: str
    original_score: float
    matchkey_name: str | None = None
    reason: str | None = None
    dataset: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LearnedAdjustment:
    """Output of the rule learner."""
    matchkey_name: str
    threshold: float | None = None
    field_weights: dict[str, float] | None = None
    sample_size: int = 0
    learned_at: datetime = field(default_factory=datetime.now)


def _canon_pair(id_a: int, id_b: int) -> tuple[int, int]:
    """Canonicalize pair ordering to (min, max)."""
    return (min(id_a, id_b), max(id_a, id_b))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    id_a INTEGER, id_b INTEGER,
    decision TEXT, source TEXT, trust REAL,
    field_hash TEXT, record_hash TEXT,
    original_score REAL,
    matchkey_name TEXT,
    reason TEXT, dataset TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(id_a, id_b, dataset)
);
CREATE INDEX IF NOT EXISTS idx_corrections_pair ON corrections(id_a, id_b, dataset);

CREATE TABLE IF NOT EXISTS adjustments (
    matchkey_name TEXT PRIMARY KEY,
    threshold REAL, field_weights TEXT,
    sample_size INTEGER,
    learned_at TIMESTAMP
);
"""


class MemoryStore:
    """Persistence layer for Learning Memory."""

    def __init__(
        self,
        backend: str = "sqlite",
        path: str = ".goldenmatch/memory.db",
        connection: str | None = None,
    ) -> None:
        self._backend = backend
        if backend == "sqlite":
            import os
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            log.debug("MemoryStore opened: %s", path)
        else:
            raise NotImplementedError(f"Backend '{backend}' not yet implemented")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def add_correction(self, correction: Correction) -> None:
        """Upsert a correction. Higher trust wins; same trust = latest wins.

        Pairs are canonicalized to (min, max) ordering before storage.
        """
        ca, cb = _canon_pair(correction.id_a, correction.id_b)
        existing = self.get_pair_correction(ca, cb, correction.dataset)

        if existing is not None:
            if correction.trust < existing.trust:
                log.debug("Correction ignored (lower trust): (%d, %d)", ca, cb)
                return

        # Atomic upsert: DELETE + INSERT in one transaction
        with self._conn:
            self._conn.execute(
                "DELETE FROM corrections WHERE id_a = ? AND id_b = ? AND dataset IS ?",
                (ca, cb, correction.dataset),
            )
            self._conn.execute(
                "INSERT INTO corrections "
                "(id, id_a, id_b, decision, source, trust, field_hash, record_hash, "
                "original_score, matchkey_name, reason, dataset, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    correction.id, ca, cb,
                    correction.decision, correction.source, correction.trust,
                    correction.field_hash, correction.record_hash,
                    correction.original_score, correction.matchkey_name,
                    correction.reason, correction.dataset,
                    correction.created_at.isoformat(),
                ),
            )
        log.debug("Correction stored: (%d, %d) %s [%s]", ca, cb,
                   correction.decision, correction.source)

    def get_pair_correction(
        self, id_a: int, id_b: int, dataset: str | None = None,
    ) -> Correction | None:
        ca, cb = _canon_pair(id_a, id_b)
        if dataset is not None:
            row = self._conn.execute(
                "SELECT * FROM corrections WHERE id_a = ? AND id_b = ? AND dataset = ?",
                (ca, cb, dataset),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM corrections WHERE id_a = ? AND id_b = ? AND dataset IS NULL",
                (ca, cb),
            ).fetchone()
        return self._row_to_correction(row) if row else None

    def get_pair_corrections_bulk(
        self, pairs: list[tuple[int, int]], dataset: str | None = None,
    ) -> dict[tuple[int, int], Correction]:
        all_corrections = self.get_corrections(dataset=dataset)
        lookup = {(c.id_a, c.id_b): c for c in all_corrections}
        # Canonicalize lookup keys from input pairs
        result = {}
        for a, b in pairs:
            ca, cb = _canon_pair(a, b)
            if (ca, cb) in lookup:
                result[(a, b)] = lookup[(ca, cb)]
        return result

    def get_corrections(self, dataset: str | None = None) -> list[Correction]:
        if dataset is not None:
            rows = self._conn.execute(
                "SELECT * FROM corrections WHERE dataset = ? ORDER BY created_at",
                (dataset,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM corrections ORDER BY created_at",
            ).fetchall()
        return [self._row_to_correction(r) for r in rows]

    def count_corrections(self, dataset: str | None = None) -> int:
        if dataset is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM corrections WHERE dataset = ?", (dataset,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM corrections").fetchone()
        return row[0] if row else 0

    def corrections_since(self, since: datetime) -> list[Correction]:
        rows = self._conn.execute(
            "SELECT * FROM corrections WHERE created_at > ? ORDER BY created_at",
            (since.isoformat(),),
        ).fetchall()
        return [self._row_to_correction(r) for r in rows]

    def save_adjustment(self, adj: LearnedAdjustment) -> None:
        weights_json = json.dumps(adj.field_weights) if adj.field_weights else None
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO adjustments "
                "(matchkey_name, threshold, field_weights, sample_size, learned_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (adj.matchkey_name, adj.threshold, weights_json,
                 adj.sample_size, adj.learned_at.isoformat()),
            )
        log.debug("Adjustment saved: %s threshold=%.3f samples=%d",
                   adj.matchkey_name, adj.threshold or 0, adj.sample_size)

    def get_adjustment(self, matchkey_name: str) -> LearnedAdjustment | None:
        row = self._conn.execute(
            "SELECT * FROM adjustments WHERE matchkey_name = ?",
            (matchkey_name,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_adjustment(row)

    def get_all_adjustments(self) -> list[LearnedAdjustment]:
        rows = self._conn.execute("SELECT * FROM adjustments").fetchall()
        return [self._row_to_adjustment(r) for r in rows]

    def last_learn_time(self) -> datetime | None:
        row = self._conn.execute(
            "SELECT MAX(learned_at) FROM adjustments",
        ).fetchone()
        if row and row[0]:
            return datetime.fromisoformat(row[0])
        return None

    @staticmethod
    def _row_to_correction(row: Any) -> Correction:
        return Correction(
            id=row["id"], id_a=row["id_a"], id_b=row["id_b"],
            decision=row["decision"], source=row["source"],
            trust=row["trust"], field_hash=row["field_hash"],
            record_hash=row["record_hash"],
            original_score=row["original_score"],
            matchkey_name=row["matchkey_name"],
            reason=row["reason"], dataset=row["dataset"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_adjustment(row: Any) -> LearnedAdjustment:
        weights = json.loads(row["field_weights"]) if row["field_weights"] else None
        return LearnedAdjustment(
            matchkey_name=row["matchkey_name"],
            threshold=row["threshold"],
            field_weights=weights,
            sample_size=row["sample_size"],
            learned_at=datetime.fromisoformat(row["learned_at"]),
        )
