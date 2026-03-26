# GoldenMatch v2.0 — Learning Memory Design

**Date:** 2026-03-26
**Status:** Approved
**Author:** Ben Severn + Claude
**Scope:** Learning Memory feature only (Zero-Config Database Discovery is a separate spec)

## Overview

Learning Memory transforms GoldenMatch from a stateless ER tool into an adaptive system that improves with every correction. Human and agent corrections persist across runs, with pair-level overrides for deterministic results and rule-level learning for generalized improvement on new data.

**No other ER tool does this.** Splink, Dedupe.io, and Zingg all require manual tuning. Learning Memory makes GoldenMatch the first ER system that converges on optimal matching over time.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| What to remember | Pair-level + rule-level (layered) | Pair corrections for deterministic re-runs; rules for new data |
| Storage | SQLite default, Postgres optional | Matches existing review queue pattern; portable + team-friendly |
| Trust model | Binary: human=1.0, agent=0.5 | Simple, covers 95% of cases; human always overrides agent |
| Pair application | Hard override with staleness check | Decisive corrections that auto-expire when data changes |
| Rule learning | Threshold tuning (10+), field weights (50+) | Graduated: quick wins first, deep learning when data supports it |
| When to learn | Automatic at pipeline start | Zero-config; corrections accumulate, next run is better |
| Architecture | Layered module set (store/corrections/learner) | Clean separation, each module under 200 lines, matches codebase patterns |

## Data Model

### Correction

A single pair decision stored in memory.

```python
@dataclass
class Correction:
    id: str                      # UUID
    id_a: int                    # row ID of record A
    id_b: int                    # row ID of record B
    decision: str                # "approve" | "reject"
    source: str                  # "steward" | "boost" | "unmerge" | "agent" | "llm"
    trust: float                 # 1.0 (human) or 0.5 (agent)
    field_hash: str              # hash of matched fields for staleness detection
    original_score: float        # score at time of correction
    reason: str | None = None    # optional explanation
    dataset: str | None = None   # dataset identifier for scoping
    created_at: datetime         # when correction was made
```

**`field_hash`:** SHA-256 (truncated to 16 chars) of the concatenated matched field values for both records. For matchkey fields `[name, zip]`: `sha256(f"{name_a}|{zip_a}|{name_b}|{zip_b}")[:16]`. If either record's matched fields change, the hash won't match and the correction is stale.

### LearnedAdjustment

Output of the rule learner, recomputed on each learning pass.

```python
@dataclass
class LearnedAdjustment:
    matchkey_name: str           # which matchkey this applies to
    threshold: float | None      # learned optimal threshold (from 10+ corrections)
    field_weights: dict[str, float] | None  # learned field weights (from 50+ corrections)
    sample_size: int             # number of corrections this was derived from
    learned_at: datetime
```

## Module Architecture

### MemoryStore (`goldenmatch/core/memory/store.py`)

Persistence layer. CRUD for corrections and adjustments.

**Interface:**
```python
class MemoryStore:
    def __init__(self, backend: str = "sqlite", path: str = ".goldenmatch/memory.db", connection: str | None = None)

    # Corrections
    def add_correction(self, correction: Correction) -> None
    def get_corrections(self, dataset: str | None = None) -> list[Correction]
    def get_pair_correction(self, id_a: int, id_b: int, dataset: str | None = None) -> Correction | None
    def count_corrections(self, dataset: str | None = None) -> int
    def corrections_since(self, since: datetime) -> list[Correction]

    # Learned adjustments
    def save_adjustment(self, adj: LearnedAdjustment) -> None
    def get_adjustment(self, matchkey_name: str) -> LearnedAdjustment | None
    def get_all_adjustments(self) -> list[LearnedAdjustment]

    # Metadata
    def last_learn_time(self) -> datetime | None
```

**Storage backends:**
- **SQLite** (default): `.goldenmatch/memory.db`. Auto-creates on first use.
- **Postgres**: Reuses existing `goldenmatch/db/connector.py` connection. Same tables in `goldenmatch` schema.

**Schema:**
```sql
CREATE TABLE corrections (
    id TEXT PRIMARY KEY,
    id_a INTEGER, id_b INTEGER,
    decision TEXT, source TEXT, trust REAL,
    field_hash TEXT, original_score REAL,
    reason TEXT, dataset TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_corrections_pair ON corrections(id_a, id_b, dataset);

CREATE TABLE adjustments (
    matchkey_name TEXT PRIMARY KEY,
    threshold REAL, field_weights TEXT,  -- JSON
    sample_size INTEGER,
    learned_at TIMESTAMP
);
```

### Corrections (`goldenmatch/core/memory/corrections.py`)

Applies pair-level corrections during scoring.

**Key function:**
```python
def apply_corrections(
    scored_pairs: list[tuple[int, int, float]],
    store: MemoryStore,
    df: pl.DataFrame,
    matchkey_fields: list[str],
    dataset: str | None = None,
) -> tuple[list[tuple[int, int, float]], CorrectionStats]
```

**Logic per scored pair:**
1. Look up `store.get_pair_correction(id_a, id_b, dataset)`
2. No correction exists: keep original score
3. Correction exists, hash matches (data unchanged): hard override (approved=1.0, rejected=0.0)
4. Correction exists, hash differs (data changed): stale, keep original score, flag for re-review

**CorrectionStats:**
```python
@dataclass
class CorrectionStats:
    applied: int          # corrections applied (hash matched)
    stale: int            # corrections skipped (data changed)
    total_pairs: int      # total pairs processed
    stale_pairs: list[tuple[int, int]]  # pairs with stale corrections
```

**Hash computation:**
```python
def compute_field_hash(df: pl.DataFrame, id_a: int, id_b: int, fields: list[str]) -> str:
    row_a = df.filter(pl.col("__row_id__") == id_a).select(fields).row(0)
    row_b = df.filter(pl.col("__row_id__") == id_b).select(fields).row(0)
    combined = "|".join(str(v) for v in row_a + row_b)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
```

**Hook point:** Called in `pipeline.py` after scoring, before clustering. Stale pairs auto-added to review queue.

### MemoryLearner (`goldenmatch/core/memory/learner.py`)

Analyzes accumulated corrections and produces adjustments.

```python
class MemoryLearner:
    def __init__(self, store: MemoryStore)

    def learn(self, matchkey_name: str | None = None) -> list[LearnedAdjustment]
    def has_new_corrections(self) -> bool
```

**Threshold tuning** (10+ corrections):
- Collect approved (positive) and rejected (negative) pairs for a matchkey
- Find threshold that maximizes separation: `(max_rejected_score + min_approved_score) / 2`
- If overlapping scores: minimize misclassification weighted by trust

**Field weight adjustment** (50+ corrections):
- Compute per-field discrimination power: how often does a field's score differ between approved and rejected pairs
- Logistic regression on field-level scores, coefficients become weights
- Normalize weights to sum to 1.0

**Pipeline integration:** At the start of `_run_dedupe_pipeline()` and `_run_match_pipeline()`, check `learner.has_new_corrections()`. If true, run `learner.learn()` and overlay adjustments on the in-memory config. Original YAML config is never modified.

## Correction Collection Points

All collection points are additive -- they call `store.add_correction()` alongside existing behavior. Nothing changes about how existing features work.

| Surface | Source | Trust | Trigger |
|---------|--------|-------|---------|
| Review Queue | `"steward"` | 1.0 | `ReviewQueue.approve()` / `.reject()` |
| Boost Tab | `"boost"` | 1.0 | User presses y/n during labeling |
| Unmerge | `"unmerge"` | 1.0 | `unmerge_record()` / `unmerge_cluster()` |
| LLM Scorer | `"llm"` | 0.5 | LLM returns match/non-match decision |
| Agent Tools | `"agent"` | 0.5 | `agent_approve_reject` MCP tool |
| REST API | `"steward"` | 1.0 | `POST /reviews/decide` |

`MemoryStore` is passed as an optional parameter to each surface. If no store provided, corrections are not persisted (backward compatible).

## Configuration

All optional. Defaults work out of the box.

```yaml
memory:
  enabled: true                    # default: true if .goldenmatch/memory.db exists
  backend: sqlite                  # "sqlite" | "postgres"
  path: .goldenmatch/memory.db     # SQLite path
  connection: postgresql://...     # Postgres connection (if backend: postgres)
  trust:
    human: 1.0                     # steward, boost, unmerge
    agent: 0.5                     # llm, agent tools
  learning:
    threshold_min_corrections: 10
    weights_min_corrections: 50
```

## Interface Additions

**Python API** (in `_api.py`):
- `get_memory(path=None) -> MemoryStore`
- `add_correction(id_a, id_b, decision, source="api", ...) -> None`
- `learn(matchkey_name=None) -> list[LearnedAdjustment]`
- `memory_stats() -> dict`

**CLI:**
- `goldenmatch memory stats` -- correction counts, last learn time, adjustments
- `goldenmatch memory learn` -- force learning pass
- `goldenmatch memory export` -- dump corrections as CSV
- `goldenmatch memory import` -- load corrections from CSV

**MCP tools:**
- `memory_stats` -- correction counts and adjustments
- `memory_learn` -- trigger learning pass

## Project Structure

**New files:**
```
goldenmatch/core/memory/
├── __init__.py          # re-exports
├── store.py             # MemoryStore (SQLite/Postgres)
├── corrections.py       # apply_corrections(), compute_field_hash()
└── learner.py           # MemoryLearner (threshold + weight learning)
```

**Modified files:** `core/review_queue.py`, `core/cluster.py`, `core/llm_scorer.py`, `tui/tabs/boost_tab.py`, `mcp/agent_tools.py`, `api/server.py`, `pipeline.py`, `_api.py`, `cli/main.py`, `mcp/server.py`, `config/schemas.py`, `__init__.py`

## Testing

**Target:** ~45 tests

| File | Scope | Count |
|------|-------|-------|
| `tests/test_memory_store.py` | CRUD, SQLite, corrections/adjustments | ~12 |
| `tests/test_corrections.py` | apply_corrections, staleness, field_hash | ~12 |
| `tests/test_learner.py` | threshold tuning, weight adjustment, graduation | ~10 |
| `tests/test_memory_integration.py` | end-to-end: correct, learn, re-run, better scores | ~6 |
| `tests/test_memory_cli.py` | CLI subcommands | ~5 |

**Key scenarios:**
- Correction applied: approved pair -> score 1.0
- Correction stale: data changed -> original score kept, pair flagged
- Threshold tuning: 10 corrections -> optimal threshold
- Weight adjustment: 50 corrections -> field weights derived
- Trust: human override beats agent on same pair
- Pipeline: corrections applied before clustering, learner at start
- Export/import CSV round-trip
