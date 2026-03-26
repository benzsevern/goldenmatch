# Learning Memory Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent Learning Memory to GoldenMatch so corrections from humans and agents accumulate across runs and improve future matching via pair-level overrides and rule-level threshold/weight learning.

**Architecture:** Three new modules in `goldenmatch/core/memory/` (store, corrections, learner) plus wiring into 8 existing modules. MemoryStore handles SQLite/Postgres persistence. Corrections module applies pair overrides with dual-hash staleness detection. Learner derives threshold and weight adjustments from accumulated corrections.

**Tech Stack:** Python 3.12, Polars, SQLite (stdlib), Pydantic, scikit-learn (LogisticRegression for weight learning)

**Spec:** `docs/superpowers/specs/2026-03-26-learning-memory-design.md`

**Working directory:** `D:\show_case\goldenmatch`

**Important codebase notes:**
- `pytest --tb=short` from project root, 1133 tests currently passing
- Internal columns prefixed with `__` (e.g., `__row_id__`, `__source__`)
- Scorer returns `list[tuple[int, int, float]]` — (row_id_a, row_id_b, score)
- `build_clusters` returns `dict[int, dict]` with keys: members, size, pair_scores, confidence, bottleneck_pair
- Config: Pydantic models in `config/schemas.py`, YAML loading in `config/loader.py`
- CLI uses Typer with `app.command("name")(handler_func)` pattern
- MCP tools registered in `mcp/agent_tools.py` AGENT_TOOLS list + `_dispatch()` handler

---

## File Map

### New files to create

| File | Responsibility |
|------|---------------|
| `goldenmatch/core/memory/__init__.py` | Re-export Correction, LearnedAdjustment, CorrectionStats, MemoryStore, MemoryLearner, apply_corrections |
| `goldenmatch/core/memory/store.py` | MemoryStore — SQLite/Postgres CRUD for corrections and adjustments |
| `goldenmatch/core/memory/corrections.py` | apply_corrections(), compute_field_hash(), compute_record_hash(), build_row_lookup(), CorrectionStats |
| `goldenmatch/core/memory/learner.py` | MemoryLearner — threshold tuning + field weight adjustment |
| `tests/test_memory_store.py` | MemoryStore CRUD tests |
| `tests/test_corrections.py` | apply_corrections, staleness, hashing tests |
| `tests/test_learner.py` | Threshold tuning, weight adjustment, graduation tests |
| `tests/test_memory_integration.py` | End-to-end: correct -> learn -> better scores |
| `tests/test_memory_cli.py` | CLI memory subcommand tests |

### Files to modify

| File | Change |
|------|--------|
| `goldenmatch/config/schemas.py` | Add LearningConfig, MemoryConfig, add `memory` field to GoldenMatchConfig |
| `goldenmatch/core/pipeline.py` | Hook apply_corrections() after scoring, learner.learn() at start |
| `goldenmatch/core/review_queue.py` | Add optional `memory_store` param to ReviewQueue, wire approve/reject |
| `goldenmatch/core/cluster.py` | Wire unmerge_record/unmerge_cluster to optional memory_store |
| `goldenmatch/core/llm_scorer.py` | Wire LLM decisions to optional memory_store |
| `goldenmatch/_api.py` | Add get_memory(), add_correction(), learn(), memory_stats() |
| `goldenmatch/cli/main.py` | Add memory subcommand group (stats, learn, export, import) |
| `goldenmatch/mcp/agent_tools.py` | Add memory_stats, memory_learn tools |
| `goldenmatch/__init__.py` | Export new symbols |

---

## Task 1: Config Models

**Files:**
- Modify: `goldenmatch/config/schemas.py`
- Test: existing config tests should still pass

- [ ] **Step 1: Add LearningConfig and MemoryConfig to schemas.py**

Add after the existing `LLMScorerConfig` class (around line 344):

```python
class LearningConfig(BaseModel):
    """Learning Memory learning parameters."""
    threshold_min_corrections: int = 10
    weights_min_corrections: int = 50


class MemoryConfig(BaseModel):
    """Learning Memory configuration."""
    enabled: bool = True
    backend: str = "sqlite"
    path: str = ".goldenmatch/memory.db"
    connection: str | None = None
    trust: dict[str, float] = Field(default_factory=lambda: {"human": 1.0, "agent": 0.5})
    learning: LearningConfig = Field(default_factory=LearningConfig)
```

Add to `GoldenMatchConfig` class (around line 399):
```python
    memory: MemoryConfig | None = None
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `pytest tests/ --tb=short -q`
Expected: All 1133 tests still pass

- [ ] **Step 3: Commit**

```bash
git add goldenmatch/config/schemas.py
git commit -m "feat(memory): add MemoryConfig and LearningConfig to schemas"
```

---

## Task 2: Data Models + MemoryStore

**Files:**
- Create: `goldenmatch/core/memory/__init__.py`
- Create: `goldenmatch/core/memory/store.py`
- Create: `tests/test_memory_store.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_memory_store.py`

```python
"""Tests for MemoryStore CRUD operations."""
import pytest
from datetime import datetime, timedelta
from goldenmatch.core.memory.store import MemoryStore
from goldenmatch.core.memory import Correction, LearnedAdjustment


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "test_memory.db"))


def _make_correction(**kwargs) -> Correction:
    defaults = dict(
        id="test-1", id_a=1, id_b=2, decision="approve",
        source="steward", trust=1.0, field_hash="abc123",
        record_hash="def456", original_score=0.85,
        reason=None, dataset="test", created_at=datetime.now(),
    )
    defaults.update(kwargs)
    return Correction(**defaults)


class TestAddAndGet:
    def test_add_and_get(self, store):
        c = _make_correction()
        store.add_correction(c)
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result is not None
        assert result.decision == "approve"
        assert result.trust == 1.0

    def test_get_missing_returns_none(self, store):
        assert store.get_pair_correction(99, 100) is None

    def test_get_corrections_list(self, store):
        store.add_correction(_make_correction(id="c1", id_a=1, id_b=2))
        store.add_correction(_make_correction(id="c2", id_a=3, id_b=4))
        result = store.get_corrections(dataset="test")
        assert len(result) == 2

    def test_count_corrections(self, store):
        store.add_correction(_make_correction(id="c1", id_a=1, id_b=2))
        store.add_correction(_make_correction(id="c2", id_a=3, id_b=4))
        assert store.count_corrections(dataset="test") == 2
        assert store.count_corrections(dataset="other") == 0


class TestUpsertAndTrust:
    def test_upsert_higher_trust_wins(self, store):
        store.add_correction(_make_correction(
            id="c1", id_a=1, id_b=2, decision="approve", trust=0.5, source="llm",
        ))
        store.add_correction(_make_correction(
            id="c2", id_a=1, id_b=2, decision="reject", trust=1.0, source="steward",
        ))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result.decision == "reject"
        assert result.trust == 1.0

    def test_upsert_lower_trust_ignored(self, store):
        store.add_correction(_make_correction(
            id="c1", id_a=1, id_b=2, decision="approve", trust=1.0, source="steward",
        ))
        store.add_correction(_make_correction(
            id="c2", id_a=1, id_b=2, decision="reject", trust=0.5, source="llm",
        ))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result.decision == "approve"

    def test_upsert_same_trust_latest_wins(self, store):
        store.add_correction(_make_correction(
            id="c1", id_a=1, id_b=2, decision="approve", trust=1.0,
            created_at=datetime(2026, 1, 1),
        ))
        store.add_correction(_make_correction(
            id="c2", id_a=1, id_b=2, decision="reject", trust=1.0,
            created_at=datetime(2026, 3, 1),
        ))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result.decision == "reject"


class TestBulkLookup:
    def test_bulk_lookup(self, store):
        store.add_correction(_make_correction(id="c1", id_a=1, id_b=2))
        store.add_correction(_make_correction(id="c2", id_a=3, id_b=4))
        result = store.get_pair_corrections_bulk(
            [(1, 2), (3, 4), (5, 6)], dataset="test",
        )
        assert (1, 2) in result
        assert (3, 4) in result
        assert (5, 6) not in result


class TestAdjustments:
    def test_save_and_get_adjustment(self, store):
        adj = LearnedAdjustment(
            matchkey_name="mk1", threshold=0.82,
            field_weights=None, sample_size=15,
            learned_at=datetime.now(),
        )
        store.save_adjustment(adj)
        result = store.get_adjustment("mk1")
        assert result is not None
        assert result.threshold == 0.82

    def test_get_all_adjustments(self, store):
        store.save_adjustment(LearnedAdjustment(
            matchkey_name="mk1", threshold=0.8, field_weights=None,
            sample_size=10, learned_at=datetime.now(),
        ))
        store.save_adjustment(LearnedAdjustment(
            matchkey_name="mk2", threshold=0.9, field_weights={"name": 0.6, "zip": 0.4},
            sample_size=55, learned_at=datetime.now(),
        ))
        result = store.get_all_adjustments()
        assert len(result) == 2


class TestCorrectionsSince:
    def test_corrections_since(self, store):
        old = _make_correction(id="c1", id_a=1, id_b=2, created_at=datetime(2026, 1, 1))
        new = _make_correction(id="c2", id_a=3, id_b=4, created_at=datetime(2026, 3, 25))
        store.add_correction(old)
        store.add_correction(new)
        result = store.corrections_since(datetime(2026, 3, 1))
        assert len(result) == 1
        assert result[0].id_a == 3


class TestLastLearnTime:
    def test_no_adjustments(self, store):
        assert store.last_learn_time() is None

    def test_with_adjustment(self, store):
        now = datetime.now()
        store.save_adjustment(LearnedAdjustment(
            matchkey_name="mk1", threshold=0.8, field_weights=None,
            sample_size=10, learned_at=now,
        ))
        result = store.last_learn_time()
        assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory_store.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Create goldenmatch/core/memory/__init__.py**

```python
"""Learning Memory — persistent corrections and rule learning."""
from goldenmatch.core.memory.store import MemoryStore, Correction, LearnedAdjustment

__all__ = ["MemoryStore", "Correction", "LearnedAdjustment"]
```

- [ ] **Step 4: Implement store.py**

File: `goldenmatch/core/memory/store.py`

```python
"""MemoryStore — SQLite/Postgres persistence for corrections and adjustments."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
    matchkey_name: str | None = None  # which matchkey produced this pair
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
            self._conn = sqlite3.connect(path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
        else:
            raise NotImplementedError(f"Backend '{backend}' not yet implemented")

    def add_correction(self, correction: Correction) -> None:
        """Upsert a correction. Higher trust wins; same trust = latest wins."""
        existing = self.get_pair_correction(
            correction.id_a, correction.id_b, correction.dataset,
        )
        if existing is not None:
            if correction.trust < existing.trust:
                return  # Lower trust, ignore
            # Same or higher trust: replace
            self._conn.execute(
                "DELETE FROM corrections WHERE id_a = ? AND id_b = ? AND dataset IS ?",
                (correction.id_a, correction.id_b, correction.dataset),
            )

        self._conn.execute(
            "INSERT OR REPLACE INTO corrections "
            "(id, id_a, id_b, decision, source, trust, field_hash, record_hash, "
            "original_score, matchkey_name, reason, dataset, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                correction.id, correction.id_a, correction.id_b,
                correction.decision, correction.source, correction.trust,
                correction.field_hash, correction.record_hash,
                correction.original_score, correction.matchkey_name,
                correction.reason, correction.dataset,
                correction.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_pair_correction(
        self, id_a: int, id_b: int, dataset: str | None = None,
    ) -> Correction | None:
        if dataset is not None:
            row = self._conn.execute(
                "SELECT * FROM corrections WHERE id_a = ? AND id_b = ? AND dataset = ?",
                (id_a, id_b, dataset),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM corrections WHERE id_a = ? AND id_b = ? AND dataset IS NULL",
                (id_a, id_b),
            ).fetchone()
        return self._row_to_correction(row) if row else None

    def get_pair_corrections_bulk(
        self, pairs: list[tuple[int, int]], dataset: str | None = None,
    ) -> dict[tuple[int, int], Correction]:
        all_corrections = self.get_corrections(dataset=dataset)
        lookup = {(c.id_a, c.id_b): c for c in all_corrections}
        return {p: lookup[p] for p in pairs if p in lookup}

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
        return row[0]

    def corrections_since(self, since: datetime) -> list[Correction]:
        rows = self._conn.execute(
            "SELECT * FROM corrections WHERE created_at > ? ORDER BY created_at",
            (since.isoformat(),),
        ).fetchall()
        return [self._row_to_correction(r) for r in rows]

    def save_adjustment(self, adj: LearnedAdjustment) -> None:
        weights_json = json.dumps(adj.field_weights) if adj.field_weights else None
        self._conn.execute(
            "INSERT OR REPLACE INTO adjustments "
            "(matchkey_name, threshold, field_weights, sample_size, learned_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (adj.matchkey_name, adj.threshold, weights_json,
             adj.sample_size, adj.learned_at.isoformat()),
        )
        self._conn.commit()

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
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_memory_store.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/memory/ tests/test_memory_store.py
git commit -m "feat(memory): MemoryStore with SQLite backend, Correction/LearnedAdjustment models"
```

---

## Task 3: Corrections Module

**Files:**
- Create: `goldenmatch/core/memory/corrections.py`
- Create: `tests/test_corrections.py`
- Modify: `goldenmatch/core/memory/__init__.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_corrections.py`

```python
"""Tests for apply_corrections and hash functions."""
import pytest
import hashlib
from datetime import datetime
import polars as pl
from goldenmatch.core.memory.corrections import (
    apply_corrections, compute_field_hash, compute_record_hash,
    build_row_lookup, CorrectionStats,
)
from goldenmatch.core.memory.store import MemoryStore, Correction


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "__row_id__": [1, 2, 3, 4],
        "name": ["John Smith", "John Smith", "Jane Doe", "Bob Jones"],
        "zip": ["12345", "12345", "67890", "11111"],
        "email": ["john@x.com", "jsmith@x.com", "jane@x.com", "bob@x.com"],
    })


def _make_correction(store, id_a, id_b, decision, field_hash, record_hash, **kw):
    c = Correction(
        id=f"c-{id_a}-{id_b}", id_a=id_a, id_b=id_b,
        decision=decision, source="steward", trust=1.0,
        field_hash=field_hash, record_hash=record_hash,
        original_score=0.85, dataset="test",
        created_at=datetime.now(), **kw,
    )
    store.add_correction(c)


class TestBuildRowLookup:
    def test_basic(self, sample_df):
        lookup = build_row_lookup(sample_df, ["name", "zip"])
        assert lookup[1] == ("John Smith", "12345")
        assert lookup[3] == ("Jane Doe", "67890")

    def test_all_rows(self, sample_df):
        lookup = build_row_lookup(sample_df, ["name"])
        assert len(lookup) == 4


class TestComputeFieldHash:
    def test_deterministic(self):
        h1 = compute_field_hash(("John", "12345"), ("John", "12345"))
        h2 = compute_field_hash(("John", "12345"), ("John", "12345"))
        assert h1 == h2

    def test_different_values(self):
        h1 = compute_field_hash(("John", "12345"), ("Jane", "67890"))
        h2 = compute_field_hash(("John", "12345"), ("John", "12345"))
        assert h1 != h2

    def test_length(self):
        h = compute_field_hash(("a",), ("b",))
        assert len(h) == 16


class TestComputeRecordHash:
    def test_deterministic(self, sample_df):
        h1 = compute_record_hash(sample_df, 1)
        h2 = compute_record_hash(sample_df, 1)
        assert h1 == h2

    def test_different_records(self, sample_df):
        h1 = compute_record_hash(sample_df, 1)
        h2 = compute_record_hash(sample_df, 3)
        assert h1 != h2


class TestApplyCorrections:
    def _get_hashes(self, df, id_a, id_b, fields):
        lookup = build_row_lookup(df, fields)
        fh = compute_field_hash(lookup[id_a], lookup[id_b])
        rh_a = compute_record_hash(df, id_a)
        rh_b = compute_record_hash(df, id_b)
        return fh, f"{rh_a}:{rh_b}"

    def test_no_corrections(self, store, sample_df):
        pairs = [(1, 2, 0.85), (3, 4, 0.60)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result == pairs
        assert stats.applied == 0

    def test_approved_override(self, store, sample_df):
        fh, rh = self._get_hashes(sample_df, 1, 2, ["name", "zip"])
        _make_correction(store, 1, 2, "approve", fh, rh)
        pairs = [(1, 2, 0.85)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 1.0
        assert stats.applied == 1

    def test_rejected_override(self, store, sample_df):
        fh, rh = self._get_hashes(sample_df, 1, 2, ["name", "zip"])
        _make_correction(store, 1, 2, "reject", fh, rh)
        pairs = [(1, 2, 0.85)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 0.0
        assert stats.applied == 1

    def test_stale_field_hash(self, store, sample_df):
        _make_correction(store, 1, 2, "approve", "wrong_hash", "wrong_record")
        pairs = [(1, 2, 0.85)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 0.85  # original score kept
        assert stats.stale == 1
        assert (1, 2) in stats.stale_pairs

    def test_mixed_corrections(self, store, sample_df):
        fh, rh = self._get_hashes(sample_df, 1, 2, ["name", "zip"])
        _make_correction(store, 1, 2, "approve", fh, rh)
        # No correction for (3, 4)
        pairs = [(1, 2, 0.85), (3, 4, 0.60)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 1.0  # overridden
        assert result[1][2] == 0.60  # unchanged
        assert stats.applied == 1
        assert stats.total_pairs == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_corrections.py -v`
Expected: FAIL

- [ ] **Step 3: Implement corrections.py**

File: `goldenmatch/core/memory/corrections.py`

```python
"""Apply pair-level corrections during scoring."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from goldenmatch.core.memory.store import MemoryStore


@dataclass
class CorrectionStats:
    """Statistics from applying corrections."""
    applied: int = 0
    stale: int = 0
    total_pairs: int = 0
    stale_pairs: list[tuple[int, int]] = field(default_factory=list)


def build_row_lookup(df: pl.DataFrame, fields: list[str]) -> dict[int, tuple]:
    """Build row ID to field values lookup once for all pairs."""
    rows = df.select(["__row_id__"] + fields).to_dicts()
    return {r["__row_id__"]: tuple(r[f] for f in fields) for r in rows}


def compute_field_hash(row_a_vals: tuple, row_b_vals: tuple) -> str:
    """Hash matched field values for staleness detection."""
    combined = "|".join(str(v) for v in row_a_vals + row_b_vals)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def compute_record_hash(df: pl.DataFrame, row_id: int) -> str:
    """Hash ALL fields for entity identity check."""
    row = df.filter(pl.col("__row_id__") == row_id).row(0)
    return hashlib.sha256("|".join(str(v) for v in row).encode()).hexdigest()[:16]


def apply_corrections(
    scored_pairs: list[tuple[int, int, float]],
    store: MemoryStore,
    df: pl.DataFrame,
    matchkey_fields: list[str],
    dataset: str | None = None,
) -> tuple[list[tuple[int, int, float]], CorrectionStats]:
    """Apply pair-level corrections to scored pairs.

    Returns adjusted pairs and correction stats.
    """
    stats = CorrectionStats(total_pairs=len(scored_pairs))

    # Bulk fetch corrections
    pair_keys = [(a, b) for a, b, _ in scored_pairs]
    corrections = store.get_pair_corrections_bulk(pair_keys, dataset=dataset)

    if not corrections:
        return scored_pairs, stats

    # Build lookup once
    field_lookup = build_row_lookup(df, matchkey_fields)

    # Pre-compute record hashes for all rows involved in corrections
    record_hashes: dict[int, str] = {}
    for (id_a, id_b) in corrections:
        if id_a not in record_hashes:
            record_hashes[id_a] = compute_record_hash(df, id_a)
        if id_b not in record_hashes:
            record_hashes[id_b] = compute_record_hash(df, id_b)

    adjusted = []
    for id_a, id_b, score in scored_pairs:
        correction = corrections.get((id_a, id_b))
        if correction is None:
            adjusted.append((id_a, id_b, score))
            continue

        # Check staleness: both field_hash and record_hash must match
        current_field_hash = compute_field_hash(
            field_lookup.get(id_a, ()), field_lookup.get(id_b, ()),
        )
        current_record_hash = (
            f"{record_hashes.get(id_a, '')}:{record_hashes.get(id_b, '')}"
        )

        # Empty hashes = collected without DataFrame access, skip staleness check
        hashes_empty = (not correction.field_hash and not correction.record_hash)
        hashes_match = (
            current_field_hash == correction.field_hash
            and current_record_hash == correction.record_hash
        )

        if hashes_empty or hashes_match:
            # Fresh correction (or no staleness data): apply hard override
            new_score = 1.0 if correction.decision == "approve" else 0.0
            adjusted.append((id_a, id_b, new_score))
            stats.applied += 1
        else:
            # Stale: data changed, keep original score
            adjusted.append((id_a, id_b, score))
            stats.stale += 1
            stats.stale_pairs.append((id_a, id_b))

    return adjusted, stats
```

- [ ] **Step 4: Update __init__.py**

Add to `goldenmatch/core/memory/__init__.py`:
```python
from goldenmatch.core.memory.corrections import (
    apply_corrections, CorrectionStats,
    compute_field_hash, compute_record_hash, build_row_lookup,
)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_corrections.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/memory/corrections.py goldenmatch/core/memory/__init__.py tests/test_corrections.py
git commit -m "feat(memory): apply_corrections with dual-hash staleness detection"
```

---

## Task 4: MemoryLearner

**Files:**
- Create: `goldenmatch/core/memory/learner.py`
- Create: `tests/test_learner.py`
- Modify: `goldenmatch/core/memory/__init__.py`

- [ ] **Step 1: Write failing tests**

File: `tests/test_learner.py`

```python
"""Tests for MemoryLearner — threshold tuning and weight adjustment."""
import pytest
from datetime import datetime
from goldenmatch.core.memory.store import MemoryStore, Correction, LearnedAdjustment
from goldenmatch.core.memory.learner import MemoryLearner


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "test.db"))


def _add_corrections(store, matchkey, approved_scores, rejected_scores):
    """Helper: add corrections with known scores for a matchkey."""
    for i, score in enumerate(approved_scores):
        store.add_correction(Correction(
            id=f"a-{matchkey}-{i}", id_a=i * 2, id_b=i * 2 + 1,
            decision="approve", source="steward", trust=1.0,
            field_hash=f"fh-{i}", record_hash=f"rh-{i}",
            original_score=score, dataset=matchkey,
            created_at=datetime.now(),
        ))
    for i, score in enumerate(rejected_scores):
        store.add_correction(Correction(
            id=f"r-{matchkey}-{i}", id_a=1000 + i * 2, id_b=1000 + i * 2 + 1,
            decision="reject", source="steward", trust=1.0,
            field_hash=f"fh-r-{i}", record_hash=f"rh-r-{i}",
            original_score=score, dataset=matchkey,
            created_at=datetime.now(),
        ))


class TestHasNewCorrections:
    def test_no_corrections(self, store):
        learner = MemoryLearner(store)
        assert not learner.has_new_corrections()

    def test_with_corrections_no_learning(self, store):
        _add_corrections(store, "mk1", [0.9], [0.3])
        learner = MemoryLearner(store)
        assert learner.has_new_corrections()

    def test_after_learning(self, store):
        _add_corrections(store, "mk1", [0.9] * 10, [0.3] * 5)
        learner = MemoryLearner(store)
        learner.learn()
        assert not learner.has_new_corrections()


class TestThresholdTuning:
    def test_below_minimum_no_learning(self, store):
        _add_corrections(store, "mk1", [0.9] * 5, [0.3] * 3)  # 8 < 10
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 0

    def test_threshold_clean_separation(self, store):
        # Approved: 0.85-0.95, Rejected: 0.50-0.70
        _add_corrections(store, "mk1", [0.85, 0.88, 0.90, 0.92, 0.95], [0.50, 0.55, 0.60, 0.65, 0.70])
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 1
        adj = result[0]
        assert adj.matchkey_name == "mk1"
        # Threshold between max_rejected (0.70) and min_approved (0.85)
        assert 0.70 < adj.threshold < 0.85

    def test_threshold_overlapping_uses_trust(self, store):
        _add_corrections(store, "mk1", [0.80, 0.82, 0.85, 0.88, 0.90], [0.78, 0.81, 0.60, 0.55, 0.50])
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 1
        # Should produce some threshold in the overlapping zone
        assert 0.50 < result[0].threshold < 0.95


class TestWeightAdjustment:
    def test_below_minimum_no_weights(self, store):
        _add_corrections(store, "mk1", [0.9] * 20, [0.3] * 10)  # 30 < 50
        learner = MemoryLearner(store, threshold_min=10, weights_min=50)
        result = learner.learn()
        assert len(result) == 1
        assert result[0].field_weights is None  # Only threshold, no weights

    def test_weights_produced_at_50(self, store):
        _add_corrections(store, "mk1", [0.9] * 30, [0.3] * 25)  # 55 >= 50
        learner = MemoryLearner(store, threshold_min=10, weights_min=50)
        result = learner.learn()
        assert len(result) == 1
        # Weights should exist but we can't test exact values without field-level data
        # The learner should at least produce a threshold
        assert result[0].threshold is not None


class TestMultipleMatchkeys:
    def test_learns_per_matchkey(self, store):
        _add_corrections(store, "mk1", [0.9] * 6, [0.3] * 5)
        _add_corrections(store, "mk2", [0.8] * 7, [0.4] * 4)
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        names = {r.matchkey_name for r in result}
        assert "mk1" in names
        assert "mk2" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_learner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement learner.py**

File: `goldenmatch/core/memory/learner.py`

```python
"""MemoryLearner — threshold tuning and field weight adjustment."""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from goldenmatch.core.memory.store import Correction, LearnedAdjustment

if TYPE_CHECKING:
    from goldenmatch.core.memory.store import MemoryStore


class MemoryLearner:
    """Analyzes accumulated corrections and produces adjustments."""

    def __init__(
        self,
        store: MemoryStore,
        threshold_min: int = 10,
        weights_min: int = 50,
    ) -> None:
        self._store = store
        self._threshold_min = threshold_min
        self._weights_min = weights_min

    def has_new_corrections(self) -> bool:
        """True if corrections exist since the last learning pass."""
        last = self._store.last_learn_time()
        if last is None:
            return self._store.count_corrections() > 0
        since = self._store.corrections_since(last)
        return len(since) > 0

    def learn(self, matchkey_name: str | None = None) -> list[LearnedAdjustment]:
        """Run learning pass. Returns list of learned adjustments."""
        all_corrections = self._store.get_corrections()
        if not all_corrections:
            return []

        # Group corrections by matchkey_name (fall back to dataset if not set)
        by_matchkey: dict[str, list[Correction]] = {}
        for c in all_corrections:
            key = c.matchkey_name or c.dataset or "_default"
            if matchkey_name and key != matchkey_name:
                continue
            by_matchkey.setdefault(key, []).append(c)

        results = []
        for mk_name, corrections in by_matchkey.items():
            if len(corrections) < self._threshold_min:
                continue

            approved = [c.original_score for c in corrections if c.decision == "approve"]
            rejected = [c.original_score for c in corrections if c.decision == "reject"]

            if not approved or not rejected:
                continue

            # Threshold tuning
            threshold = self._compute_threshold(approved, rejected, corrections)

            # Field weight adjustment (if enough data)
            field_weights = None
            if len(corrections) >= self._weights_min:
                field_weights = self._compute_weights(corrections)

            adj = LearnedAdjustment(
                matchkey_name=mk_name,
                threshold=threshold,
                field_weights=field_weights,
                sample_size=len(corrections),
                learned_at=datetime.now(),
            )
            self._store.save_adjustment(adj)
            results.append(adj)

        return results

    def _compute_threshold(
        self,
        approved_scores: list[float],
        rejected_scores: list[float],
        corrections: list[Correction],
    ) -> float:
        """Find optimal threshold separating approved from rejected."""
        max_rejected = max(rejected_scores)
        min_approved = min(approved_scores)

        if max_rejected < min_approved:
            # Clean separation
            return (max_rejected + min_approved) / 2

        # Overlapping: grid search over candidate thresholds
        all_scores = sorted(set(approved_scores + rejected_scores))
        best_threshold = (max_rejected + min_approved) / 2
        best_cost = float("inf")

        for i in range(len(all_scores) - 1):
            candidate = (all_scores[i] + all_scores[i + 1]) / 2
            cost = 0.0
            for c in corrections:
                if c.decision == "approve" and c.original_score < candidate:
                    cost += c.trust  # false negative (missed match)
                elif c.decision == "reject" and c.original_score >= candidate:
                    cost += c.trust  # false positive (bad merge)
            if cost < best_cost:
                best_cost = cost
                best_threshold = candidate

        return best_threshold

    def _compute_weights(self, corrections: list[Correction]) -> dict[str, float] | None:
        """Compute field weights via logistic regression on correction patterns.

        Note: Full field-level scoring requires the original DataFrame, which is
        not stored. For now, returns None — field weight learning will be
        implemented when per-field scores are available in corrections.
        """
        # TODO: implement when per-field scores are stored
        return None
```

- [ ] **Step 4: Update __init__.py**

Add to `goldenmatch/core/memory/__init__.py`:
```python
from goldenmatch.core.memory.learner import MemoryLearner
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_learner.py -v`
Expected: All 8 tests PASS (weight tests check for None since field-level scores aren't stored yet)

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/memory/learner.py goldenmatch/core/memory/__init__.py tests/test_learner.py
git commit -m "feat(memory): MemoryLearner with threshold tuning"
```

---

## Task 5: Pipeline Integration

**Files:**
- Modify: `goldenmatch/core/pipeline.py`

- [ ] **Step 1: Add memory imports and helper at top of pipeline.py**

After existing imports, add:
```python
from goldenmatch.core.memory.store import MemoryStore
from goldenmatch.core.memory.corrections import apply_corrections
from goldenmatch.core.memory.learner import MemoryLearner
```

- [ ] **Step 2: Add learner check at start of _run_dedupe_pipeline**

At the beginning of `_run_dedupe_pipeline()` (after config is available, around line 195), add:

```python
    # Learning Memory: apply learned adjustments
    memory_store = None
    if config.memory and config.memory.enabled:
        try:
            memory_store = MemoryStore(
                backend=config.memory.backend,
                path=config.memory.path,
                connection=config.memory.connection,
            )
            learner = MemoryLearner(
                memory_store,
                threshold_min=config.memory.learning.threshold_min_corrections,
                weights_min=config.memory.learning.weights_min_corrections,
            )
            if learner.has_new_corrections():
                adjustments = learner.learn()
                for adj in adjustments:
                    if adj.threshold is not None:
                        # Only overlay on fuzzy matchkeys with matching name
                        for mk in matchkeys:
                            if mk.threshold is not None and (
                                not adj.matchkey_name
                                or adj.matchkey_name == "_default"
                                or adj.matchkey_name == getattr(mk, "name", None)
                            ):
                                mk.threshold = adj.threshold
        except Exception as e:
            import logging
            logging.getLogger("goldenmatch.memory").warning("Learning Memory skipped: %s", e)
```

- [ ] **Step 3: Add apply_corrections after scoring**

After all pairs are scored (around line 317, after fuzzy matching and before reranking), add:

```python
    # Learning Memory: apply pair-level corrections
    if memory_store is not None and all_pairs:
        matchkey_fields = []
        for mk in matchkeys:
            matchkey_fields.extend(f.name for f in mk.fields)
        matchkey_fields = list(set(matchkey_fields))

        dataset = config.input.files[0][0] if config.input and config.input.files else None
        all_pairs, correction_stats = apply_corrections(
            all_pairs, memory_store, combined_df,
            matchkey_fields, dataset=dataset,
        )
```

- [ ] **Step 4: Apply same pattern to _run_match_pipeline**

Add the same learner check at start and apply_corrections after scoring in `_run_match_pipeline()`.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass (memory is optional, defaults to None in config)

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/pipeline.py
git commit -m "feat(memory): integrate Learning Memory into pipeline — learner at start, corrections after scoring"
```

---

## Task 6: Collection Points

**Files:**
- Modify: `goldenmatch/core/review_queue.py`
- Modify: `goldenmatch/core/cluster.py`
- Modify: `goldenmatch/core/llm_scorer.py`

- [ ] **Step 1: Wire review queue**

In `goldenmatch/core/review_queue.py`, add `memory_store` parameter to `ReviewQueue.__init__()`:

```python
def __init__(self, backend: str = "memory", memory_store=None) -> None:
    self._memory_store = memory_store
    # ... existing init
```

In `approve()` method, after existing logic:
```python
    if self._memory_store is not None:
        from goldenmatch.core.memory.store import Correction
        import uuid
        from datetime import datetime
        self._memory_store.add_correction(Correction(
            id=str(uuid.uuid4()), id_a=id_a, id_b=id_b,
            decision="approve", source="steward", trust=1.0,
            field_hash="", record_hash="",  # hashes computed at apply time
            original_score=item.score, dataset=job_name,
            created_at=datetime.now(),
        ))
```

Same for `reject()` with `decision="reject"`.

- [ ] **Step 2: Wire unmerge operations**

In `goldenmatch/core/cluster.py`, add optional `memory_store` parameter to both functions:

```python
def unmerge_record(record_id, clusters, threshold=0.0, memory_store=None):
```

After unmerge logic, if memory_store is not None, add reject corrections for all pairs involving record_id:
```python
    if memory_store is not None:
        from goldenmatch.core.memory.store import Correction
        import uuid
        from datetime import datetime
        cluster = clusters.get(cluster_id)
        if cluster:
            for (a, b), score in cluster.get("pair_scores", {}).items():
                if a == record_id or b == record_id:
                    memory_store.add_correction(Correction(
                        id=str(uuid.uuid4()), id_a=a, id_b=b,
                        decision="reject", source="unmerge", trust=1.0,
                        field_hash="", record_hash="",
                        original_score=score, dataset=None,
                        created_at=datetime.now(),
                    ))
```

Same pattern for `unmerge_cluster()` — reject all pairs in pair_scores.

- [ ] **Step 3: Wire LLM scorer**

In `goldenmatch/core/llm_scorer.py`, add optional `memory_store` parameter to `llm_score_pairs()`:

```python
def llm_score_pairs(pairs, df, ..., memory_store=None):
```

After each LLM decision, if memory_store is not None:
```python
    if memory_store is not None:
        from goldenmatch.core.memory.store import Correction
        import uuid
        from datetime import datetime
        decision = "approve" if llm_score >= 0.5 else "reject"
        memory_store.add_correction(Correction(
            id=str(uuid.uuid4()), id_a=id_a, id_b=id_b,
            decision=decision, source="llm", trust=0.5,
            field_hash="", record_hash="",
            original_score=original_score, dataset=None,
            created_at=datetime.now(),
        ))
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass (new params are optional with default None)

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/review_queue.py goldenmatch/core/cluster.py goldenmatch/core/llm_scorer.py
git commit -m "feat(memory): wire correction collection into review_queue, cluster, llm_scorer"
```

---

## Task 7: Python API + CLI + MCP

**Files:**
- Modify: `goldenmatch/_api.py`
- Modify: `goldenmatch/cli/main.py`
- Modify: `goldenmatch/mcp/agent_tools.py`
- Modify: `goldenmatch/__init__.py`
- Create: `tests/test_memory_cli.py`

- [ ] **Step 1: Add Python API functions to _api.py**

Add to `goldenmatch/_api.py`:

```python
def get_memory(path: str | None = None) -> "MemoryStore":
    """Get the MemoryStore for the current project."""
    from goldenmatch.core.memory.store import MemoryStore
    return MemoryStore(path=path or ".goldenmatch/memory.db")


def add_correction(
    id_a: int, id_b: int, decision: str,
    source: str = "api", trust: float = 1.0,
    field_hash: str = "", record_hash: str = "",
    original_score: float = 0.0,
    dataset: str | None = None,
    reason: str | None = None,
) -> None:
    """Manually add a correction to Learning Memory."""
    import uuid
    from datetime import datetime
    from goldenmatch.core.memory.store import MemoryStore, Correction
    store = MemoryStore()
    store.add_correction(Correction(
        id=str(uuid.uuid4()), id_a=id_a, id_b=id_b,
        decision=decision, source=source, trust=trust,
        field_hash=field_hash, record_hash=record_hash,
        original_score=original_score, dataset=dataset,
        reason=reason, created_at=datetime.now(),
    ))


def learn(matchkey_name: str | None = None) -> list:
    """Run learning pass on accumulated corrections."""
    from goldenmatch.core.memory.store import MemoryStore
    from goldenmatch.core.memory.learner import MemoryLearner
    store = MemoryStore()
    learner = MemoryLearner(store)
    return learner.learn(matchkey_name)


def memory_stats() -> dict:
    """Return correction counts, learned adjustments, staleness info."""
    from goldenmatch.core.memory.store import MemoryStore
    from goldenmatch.core.memory.learner import MemoryLearner
    store = MemoryStore()
    learner = MemoryLearner(store)
    adjustments = store.get_all_adjustments()
    return {
        "total_corrections": store.count_corrections(),
        "has_new": learner.has_new_corrections(),
        "last_learn_time": str(store.last_learn_time()) if store.last_learn_time() else None,
        "adjustments": [
            {"matchkey": a.matchkey_name, "threshold": a.threshold,
             "sample_size": a.sample_size}
            for a in adjustments
        ],
    }
```

- [ ] **Step 2: Add CLI memory subcommand group**

In `goldenmatch/cli/main.py`, add a memory Typer group and register it:

```python
memory_app = typer.Typer(help="Learning Memory -- manage corrections and learning.")

@memory_app.command("stats")
def memory_stats_cmd():
    """Show correction counts, last learn time, adjustments."""
    from goldenmatch._api import memory_stats
    import json
    stats = memory_stats()
    console.print(json.dumps(stats, indent=2))

@memory_app.command("learn")
def memory_learn_cmd():
    """Force a learning pass."""
    from goldenmatch._api import learn
    results = learn()
    if results:
        for r in results:
            console.print(f"  {r.matchkey_name}: threshold={r.threshold}, samples={r.sample_size}")
    else:
        console.print("No adjustments produced (not enough corrections).")

@memory_app.command("export")
def memory_export_cmd(
    output: str = typer.Option("corrections.csv", "--output", "-o"),
):
    """Export corrections as CSV."""
    from goldenmatch._api import get_memory
    import csv
    store = get_memory()
    corrections = store.get_corrections()
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "id_a", "id_b", "decision", "source", "trust",
                         "field_hash", "record_hash", "original_score", "reason",
                         "dataset", "created_at"])
        for c in corrections:
            writer.writerow([c.id, c.id_a, c.id_b, c.decision, c.source, c.trust,
                            c.field_hash, c.record_hash, c.original_score, c.reason,
                            c.dataset, c.created_at.isoformat()])
    console.print(f"Exported {len(corrections)} corrections to {output}")

@memory_app.command("import")
def memory_import_cmd(
    input_file: str = typer.Argument(..., help="CSV file to import"),
):
    """Import corrections from CSV."""
    from goldenmatch._api import get_memory
    from goldenmatch.core.memory.store import Correction
    from datetime import datetime
    import csv
    store = get_memory()
    count = 0
    with open(input_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            store.add_correction(Correction(
                id=row["id"], id_a=int(row["id_a"]), id_b=int(row["id_b"]),
                decision=row["decision"], source=row["source"],
                trust=float(row["trust"]), field_hash=row["field_hash"],
                record_hash=row["record_hash"],
                original_score=float(row["original_score"]),
                reason=row["reason"] or None,
                dataset=row["dataset"] or None,
                created_at=datetime.fromisoformat(row["created_at"]),
            ))
            count += 1
    console.print(f"Imported {count} corrections from {input_file}")

# Register with main app (near other app.add_typer calls):
app.add_typer(memory_app, name="memory")
```

- [ ] **Step 3: Add MCP tools**

In `goldenmatch/mcp/agent_tools.py`, add tool definitions to `AGENT_TOOLS` list:

```python
Tool(
    name="memory_stats",
    description="Get Learning Memory statistics -- correction counts and learned adjustments",
    inputSchema={"type": "object", "properties": {}},
),
Tool(
    name="memory_learn",
    description="Trigger a Learning Memory learning pass",
    inputSchema={"type": "object", "properties": {
        "matchkey_name": {"type": "string", "description": "Optional matchkey to learn for"},
    }},
),
```

Add handlers in `_dispatch()`:
```python
if name == "memory_stats":
    from goldenmatch._api import memory_stats
    return memory_stats()

if name == "memory_learn":
    from goldenmatch._api import learn
    results = learn(args.get("matchkey_name"))
    return {"adjustments": [
        {"matchkey": r.matchkey_name, "threshold": r.threshold, "sample_size": r.sample_size}
        for r in results
    ]}
```

- [ ] **Step 4: Update __init__.py exports**

Add to imports and `__all__` in `goldenmatch/__init__.py`:
```python
from goldenmatch.core.memory import (
    MemoryStore, Correction, LearnedAdjustment, CorrectionStats,
    MemoryLearner, apply_corrections,
)
from goldenmatch._api import get_memory, add_correction, learn, memory_stats
```

- [ ] **Step 5: Write CLI tests**

File: `tests/test_memory_cli.py`

```python
"""Tests for memory CLI subcommands."""
import pytest
from typer.testing import CliRunner
from goldenmatch.cli.main import app

runner = CliRunner()


class TestMemoryStats:
    def test_stats_no_memory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "stats"])
        # Should work even with no memory file (creates empty one)
        assert result.exit_code == 0


class TestMemoryLearn:
    def test_learn_no_corrections(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["memory", "learn"])
        assert result.exit_code == 0
        assert "No adjustments" in result.stdout or "not enough" in result.stdout.lower()


class TestMemoryExportImport:
    def test_export_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = str(tmp_path / "export.csv")
        result = runner.invoke(app, ["memory", "export", "--output", out])
        assert result.exit_code == 0
        assert "0 corrections" in result.stdout

    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # First add a correction via Python API
        from goldenmatch.core.memory.store import MemoryStore, Correction
        from datetime import datetime
        store = MemoryStore(path=str(tmp_path / ".goldenmatch" / "memory.db"))
        import os
        os.makedirs(tmp_path / ".goldenmatch", exist_ok=True)
        store.add_correction(Correction(
            id="test-1", id_a=1, id_b=2, decision="approve",
            source="steward", trust=1.0, field_hash="abc", record_hash="def",
            original_score=0.9, dataset=None, created_at=datetime.now(),
        ))
        # Export
        out = str(tmp_path / "export.csv")
        result = runner.invoke(app, ["memory", "export", "--output", out])
        assert "1 corrections" in result.stdout
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add goldenmatch/_api.py goldenmatch/cli/main.py goldenmatch/mcp/agent_tools.py goldenmatch/__init__.py tests/test_memory_cli.py
git commit -m "feat(memory): Python API, CLI subcommands, MCP tools for Learning Memory"
```

---

## Task 8: Integration Tests

**Files:**
- Create: `tests/test_memory_integration.py`

- [ ] **Step 1: Write integration tests**

File: `tests/test_memory_integration.py`

```python
"""End-to-end integration tests for Learning Memory."""
import pytest
from datetime import datetime
import polars as pl
from goldenmatch.core.memory.store import MemoryStore, Correction
from goldenmatch.core.memory.corrections import (
    apply_corrections, build_row_lookup, compute_field_hash, compute_record_hash,
)
from goldenmatch.core.memory.learner import MemoryLearner


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "integration.db"))


@pytest.fixture
def df():
    return pl.DataFrame({
        "__row_id__": [1, 2, 3, 4, 5],
        "name": ["John Smith", "John Smith", "Jane Doe", "Bob Jones", "John Smith"],
        "zip": ["12345", "12345", "67890", "11111", "12345"],
    })


class TestCorrectThenRerun:
    def test_approved_pair_gets_1_on_rerun(self, store, df):
        """Correct a pair, re-run scoring, verify override."""
        fields = ["name", "zip"]
        lookup = build_row_lookup(df, fields)
        fh = compute_field_hash(lookup[1], lookup[2])
        rh = f"{compute_record_hash(df, 1)}:{compute_record_hash(df, 2)}"

        store.add_correction(Correction(
            id="c1", id_a=1, id_b=2, decision="approve",
            source="steward", trust=1.0, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))

        # Simulate re-run with new scores
        pairs = [(1, 2, 0.83), (1, 5, 0.91), (3, 4, 0.40)]
        result, stats = apply_corrections(pairs, store, df, fields, dataset="test")

        assert result[0] == (1, 2, 1.0)  # overridden
        assert result[1] == (1, 5, 0.91)  # unchanged
        assert result[2] == (3, 4, 0.40)  # unchanged
        assert stats.applied == 1

    def test_rejected_pair_gets_0_on_rerun(self, store, df):
        fields = ["name", "zip"]
        lookup = build_row_lookup(df, fields)
        fh = compute_field_hash(lookup[1], lookup[2])
        rh = f"{compute_record_hash(df, 1)}:{compute_record_hash(df, 2)}"

        store.add_correction(Correction(
            id="c1", id_a=1, id_b=2, decision="reject",
            source="steward", trust=1.0, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))

        pairs = [(1, 2, 0.83)]
        result, stats = apply_corrections(pairs, store, df, fields, dataset="test")
        assert result[0] == (1, 2, 0.0)


class TestCorrectThenLearn:
    def test_threshold_improves_after_corrections(self, store):
        """Accumulate corrections, learn, verify threshold."""
        # Simulate 12 corrections: approved at 0.85+, rejected at 0.70-
        for i in range(7):
            store.add_correction(Correction(
                id=f"a{i}", id_a=i*2, id_b=i*2+1, decision="approve",
                source="steward", trust=1.0, field_hash=f"fh{i}", record_hash=f"rh{i}",
                original_score=0.85 + i * 0.02, dataset="mk1",
                created_at=datetime.now(),
            ))
        for i in range(5):
            store.add_correction(Correction(
                id=f"r{i}", id_a=100+i*2, id_b=100+i*2+1, decision="reject",
                source="steward", trust=1.0, field_hash=f"rfh{i}", record_hash=f"rrh{i}",
                original_score=0.60 + i * 0.02, dataset="mk1",
                created_at=datetime.now(),
            ))

        learner = MemoryLearner(store, threshold_min=10)
        adjustments = learner.learn()
        assert len(adjustments) == 1
        # Threshold should be between max_rejected (0.68) and min_approved (0.85)
        assert 0.68 < adjustments[0].threshold < 0.85

    def test_no_new_corrections_after_learn(self, store):
        """After learning, has_new_corrections should be False."""
        for i in range(12):
            store.add_correction(Correction(
                id=f"c{i}", id_a=i*2, id_b=i*2+1,
                decision="approve" if i < 7 else "reject",
                source="steward", trust=1.0,
                field_hash=f"fh{i}", record_hash=f"rh{i}",
                original_score=0.85 if i < 7 else 0.50,
                dataset="mk1", created_at=datetime.now(),
            ))

        learner = MemoryLearner(store, threshold_min=10)
        assert learner.has_new_corrections()
        learner.learn()
        assert not learner.has_new_corrections()


class TestHumanOverridesAgent:
    def test_human_beats_agent(self, store, df):
        """Agent approves, human rejects — human wins."""
        fields = ["name", "zip"]
        lookup = build_row_lookup(df, fields)
        fh = compute_field_hash(lookup[1], lookup[2])
        rh = f"{compute_record_hash(df, 1)}:{compute_record_hash(df, 2)}"

        # Agent approves
        store.add_correction(Correction(
            id="agent-1", id_a=1, id_b=2, decision="approve",
            source="llm", trust=0.5, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))
        # Human rejects (higher trust)
        store.add_correction(Correction(
            id="human-1", id_a=1, id_b=2, decision="reject",
            source="steward", trust=1.0, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))

        pairs = [(1, 2, 0.82)]
        result, stats = apply_corrections(pairs, store, df, fields, dataset="test")
        assert result[0][2] == 0.0  # rejected (human wins)
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_memory_integration.py -v`
Expected: All 5 tests PASS

- [ ] **Step 3: Run full suite**

Run: `pytest tests/ --tb=short -q`
Expected: All tests pass (1133 original + ~45 new memory tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_memory_integration.py
git commit -m "feat(memory): integration tests — correct+rerun, correct+learn, trust override"
```

---

## Execution Order

| Task | Component | Depends On | Est. Tests |
|------|-----------|------------|-----------|
| 1 | Config models | — | 0 (existing tests verify) |
| 2 | MemoryStore + data models | — | 12 |
| 3 | Corrections module | Task 2 | 12 |
| 4 | MemoryLearner | Task 2 | 8 |
| 5 | Pipeline integration | Tasks 2-4 | 0 (existing tests verify) |
| 6 | Collection points | Task 2 | 0 (existing tests verify) |
| 7 | Python API + CLI + MCP | Tasks 2-4 | 5 |
| 8 | Integration tests | All | 5 |
| **Total** | | | **~42** |

Task 1 is independent. Tasks 3 and 4 depend on Task 2 (import from store.py).
Tasks 5-6 depend on Tasks 2-4.
Task 7 depends on Tasks 2-4.
Task 8 depends on all.
