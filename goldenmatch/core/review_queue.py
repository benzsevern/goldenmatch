"""Confidence-gated review queue for human-in-the-loop pair decisions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ReviewItem:
    """A single pair awaiting human review."""

    job_name: str
    id_a: int
    id_b: int
    score: float
    explanation: str
    status: str = "pending"
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    reason: Optional[str] = None

    def approve(self, decided_by: str) -> None:
        self.status = "approved"
        self.decided_by = decided_by
        self.decided_at = datetime.now(timezone.utc).isoformat()

    def reject(self, decided_by: str, reason: str = "") -> None:
        self.status = "rejected"
        self.decided_by = decided_by
        self.decided_at = datetime.now(timezone.utc).isoformat()
        self.reason = reason


def gate_pairs(
    pairs: List[Tuple[int, int, float]],
    merge_threshold: float = 0.95,
    review_threshold: float = 0.75,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """Split scored pairs into auto-merged, review, and auto-rejected buckets.

    Parameters
    ----------
    pairs : list of (id_a, id_b, score) tuples
    merge_threshold : scores strictly above this are auto-merged
    review_threshold : scores >= this (and <= merge_threshold) go to review

    Returns
    -------
    (auto_merged, review, auto_rejected) tuple of lists
    """
    auto_merged: list[tuple[int, int, float]] = []
    review: list[tuple[int, int, float]] = []
    auto_rejected: list[tuple[int, int, float]] = []

    for id_a, id_b, score in pairs:
        if score > merge_threshold:
            auto_merged.append((id_a, id_b, score))
        elif score >= review_threshold:
            review.append((id_a, id_b, score))
        else:
            auto_rejected.append((id_a, id_b, score))

    return auto_merged, review, auto_rejected


class _MemoryBackend:
    """In-memory storage for review items."""

    def __init__(self) -> None:
        self._jobs: dict[str, list[ReviewItem]] = {}

    def add(self, item: ReviewItem) -> None:
        self._jobs.setdefault(item.job_name, []).append(item)

    def list_pending(self, job_name: str) -> list[ReviewItem]:
        return [it for it in self._jobs.get(job_name, []) if it.status == "pending"]

    def _find(self, job_name: str, id_a: int, id_b: int) -> Optional[ReviewItem]:
        for it in self._jobs.get(job_name, []):
            if it.id_a == id_a and it.id_b == id_b and it.status == "pending":
                return it
        return None

    def approve(self, job_name: str, id_a: int, id_b: int, decided_by: str) -> None:
        item = self._find(job_name, id_a, id_b)
        if item:
            item.approve(decided_by)

    def reject(self, job_name: str, id_a: int, id_b: int, decided_by: str, reason: str = "") -> None:
        item = self._find(job_name, id_a, id_b)
        if item:
            item.reject(decided_by, reason)

    def stats(self, job_name: str) -> dict[str, int]:
        items = self._jobs.get(job_name, [])
        return {
            "pending": sum(1 for it in items if it.status == "pending"),
            "approved": sum(1 for it in items if it.status == "approved"),
            "rejected": sum(1 for it in items if it.status == "rejected"),
        }


class _SQLiteBackend:
    """SQLite-backed persistent storage for review items."""

    def __init__(self) -> None:
        db_dir = Path(".goldenmatch")
        db_dir.mkdir(exist_ok=True)
        self._db_path = db_dir / "reviews.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.execute(
                """CREATE TABLE IF NOT EXISTS reviews (
                    job_name TEXT NOT NULL,
                    id_a INTEGER NOT NULL,
                    id_b INTEGER NOT NULL,
                    score REAL NOT NULL,
                    explanation TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    decided_by TEXT,
                    decided_at TEXT,
                    reason TEXT,
                    PRIMARY KEY (job_name, id_a, id_b)
                )"""
            )
            con.commit()
        finally:
            con.close()

    def add(self, item: ReviewItem) -> None:
        con = self._connect()
        try:
            con.execute(
                "INSERT OR REPLACE INTO reviews (job_name, id_a, id_b, score, explanation, status, decided_by, decided_at, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item.job_name, item.id_a, item.id_b, item.score, item.explanation, item.status, item.decided_by, item.decided_at, item.reason),
            )
            con.commit()
        finally:
            con.close()

    def list_pending(self, job_name: str) -> list[ReviewItem]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT job_name, id_a, id_b, score, explanation, status, decided_by, decided_at, reason FROM reviews WHERE job_name = ? AND status = 'pending'",
                (job_name,),
            ).fetchall()
            return [
                ReviewItem(
                    job_name=r[0], id_a=r[1], id_b=r[2], score=r[3],
                    explanation=r[4], status=r[5], decided_by=r[6],
                    decided_at=r[7], reason=r[8],
                )
                for r in rows
            ]
        finally:
            con.close()

    def approve(self, job_name: str, id_a: int, id_b: int, decided_by: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        con = self._connect()
        try:
            con.execute(
                "UPDATE reviews SET status = 'approved', decided_by = ?, decided_at = ? WHERE job_name = ? AND id_a = ? AND id_b = ? AND status = 'pending'",
                (decided_by, now, job_name, id_a, id_b),
            )
            con.commit()
        finally:
            con.close()

    def reject(self, job_name: str, id_a: int, id_b: int, decided_by: str, reason: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        con = self._connect()
        try:
            con.execute(
                "UPDATE reviews SET status = 'rejected', decided_by = ?, decided_at = ?, reason = ? WHERE job_name = ? AND id_a = ? AND id_b = ? AND status = 'pending'",
                (decided_by, now, reason, job_name, id_a, id_b),
            )
            con.commit()
        finally:
            con.close()

    def stats(self, job_name: str) -> dict[str, int]:
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT status, COUNT(*) FROM reviews WHERE job_name = ? GROUP BY status",
                (job_name,),
            ).fetchall()
            counts = {"pending": 0, "approved": 0, "rejected": 0}
            for status, cnt in rows:
                if status in counts:
                    counts[status] = cnt
            return counts
        finally:
            con.close()


class ReviewQueue:
    """Confidence-gated review queue with pluggable backends.

    Parameters
    ----------
    backend : str
        "memory" (default) or "sqlite"
    """

    def __init__(self, backend: str = "memory") -> None:
        if backend == "memory":
            self._backend = _MemoryBackend()
        elif backend == "sqlite":
            self._backend = _SQLiteBackend()
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'memory' or 'sqlite'.")
        self._backend_name = backend

    @property
    def storage_tier(self) -> str:
        return self._backend_name

    def add(self, job_name: str, id_a: int, id_b: int, score: float, explanation: str) -> None:
        item = ReviewItem(job_name=job_name, id_a=id_a, id_b=id_b, score=score, explanation=explanation)
        self._backend.add(item)

    def list_pending(self, job_name: str) -> list[ReviewItem]:
        return self._backend.list_pending(job_name)

    def approve(self, job_name: str, id_a: int, id_b: int, decided_by: str) -> None:
        self._backend.approve(job_name, id_a, id_b, decided_by)

    def reject(self, job_name: str, id_a: int, id_b: int, decided_by: str, reason: str = "") -> None:
        self._backend.reject(job_name, id_a, id_b, decided_by, reason)

    def stats(self, job_name: str) -> dict[str, int]:
        return self._backend.stats(job_name)
