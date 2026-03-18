# Scoring Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve GoldenMatch F1 scores across all Leipzig benchmark datasets by adding record-level embedding scoring, direct-pair ANN blocking, and threshold auto-tuning.

**Architecture:** Three independent features layered on the existing scorer/blocker pipeline. Feature 1 adds a new scorer type (`record_embedding`) in `scorer.py`. Feature 2 adds a new blocking strategy (`ann_pairs`) in `blocker.py` + `ann_blocker.py`. Feature 3 adds `threshold.py` for Otsu's method with pipeline integration. Each feature is independently useful and testable.

**Tech Stack:** Python 3.12, Polars, sentence-transformers, FAISS, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-scoring-improvements-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `goldenmatch/config/schemas.py` | Modify | Add `record_embedding` to VALID_SCORERS, `columns` to MatchkeyField, `ann_pairs` to strategy, `auto_threshold` to MatchkeyConfig |
| `goldenmatch/core/scorer.py` | Modify | Add `_record_embedding_score_matrix()`, handle `pre_scored_pairs` in `find_fuzzy_matches()` |
| `goldenmatch/core/ann_blocker.py` | Modify | Add `query_with_scores()` method |
| `goldenmatch/core/blocker.py` | Modify | Add `pre_scored_pairs` to BlockResult, add `_build_ann_pair_blocks()` |
| `goldenmatch/core/threshold.py` | Create | Otsu's method for auto-threshold |
| `goldenmatch/core/pipeline.py` | Modify | Update `_get_required_columns()` for `columns` field, add auto-threshold sampling |
| `goldenmatch/tui/engine.py` | Modify | Add auto-threshold sampling (mirrors pipeline.py) |
| `tests/test_record_embedding_scorer.py` | Create | Record-level scorer tests |
| `tests/test_ann_pairs.py` | Create | Direct-pair ANN tests |
| `tests/test_threshold.py` | Create | Otsu's method tests |
| `tests/benchmarks/run_leipzig.py` | Modify | Add record_embedding + ann_pairs strategies |

---

## Task 1: Record-Level Embedding Scorer — Schema

**Files:**
- Modify: `goldenmatch/config/schemas.py:12-75`
- Test: `tests/test_record_embedding_scorer.py`

- [ ] **Step 1: Write failing tests for schema validation**

Create `tests/test_record_embedding_scorer.py`:

```python
"""Tests for record-level embedding scorer."""

from __future__ import annotations

import pytest
from goldenmatch.config.schemas import MatchkeyField, MatchkeyConfig


class TestRecordEmbeddingSchema:
    def test_record_embedding_with_columns(self):
        f = MatchkeyField(
            scorer="record_embedding",
            columns=["title", "manufacturer"],
            weight=1.0,
            model="all-MiniLM-L6-v2",
        )
        assert f.field == "__record__"
        assert f.columns == ["title", "manufacturer"]

    def test_record_embedding_requires_columns(self):
        with pytest.raises(ValueError, match="columns"):
            MatchkeyField(scorer="record_embedding", weight=1.0)

    def test_record_embedding_empty_columns_rejected(self):
        with pytest.raises(ValueError, match="columns"):
            MatchkeyField(scorer="record_embedding", columns=[], weight=1.0)

    def test_regular_scorer_still_requires_field(self):
        with pytest.raises(ValueError, match="field"):
            MatchkeyField(scorer="jaro_winkler", weight=1.0)

    def test_regular_scorer_with_field_still_works(self):
        f = MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)
        assert f.field == "name"

    def test_record_embedding_in_weighted_matchkey(self):
        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.80,
            fields=[
                MatchkeyField(
                    scorer="record_embedding",
                    columns=["title", "manufacturer"],
                    weight=0.7,
                    model="all-MiniLM-L6-v2",
                ),
                MatchkeyField(field="brand", scorer="exact", weight=0.3),
            ],
        )
        assert len(mk.fields) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_record_embedding_scorer.py -v --tb=short`
Expected: FAIL — `record_embedding` not in VALID_SCORERS

- [ ] **Step 3: Update schemas.py**

In `goldenmatch/config/schemas.py`:

1. Add `"record_embedding"` to `VALID_SCORERS` (line 18-19):
```python
VALID_SCORERS = frozenset({
    "exact", "jaro_winkler", "levenshtein", "token_sort", "soundex_match",
    "embedding", "record_embedding",
})
```

2. Add `columns` field to `MatchkeyField` (after line 60):
```python
class MatchkeyField(BaseModel):
    field: str | None = None
    column: str | None = None
    transforms: list[str] = Field(default_factory=list)
    scorer: str | None = None
    weight: float | None = None
    model: str | None = None
    columns: list[str] | None = None  # for record_embedding scorer
```

3. Update `_resolve_field_column` validator (lines 62-75):
```python
@model_validator(mode="after")
def _resolve_field_column(self) -> "MatchkeyField":
    # record_embedding uses columns (plural), not field
    if self.scorer == "record_embedding":
        if not self.columns:
            raise ValueError(
                "record_embedding scorer requires 'columns' (list of column names)."
            )
        self.field = "__record__"
        return self
    # Allow 'column' as alias for 'field'
    if self.field is None and self.column is not None:
        self.field = self.column
    elif self.field is None and self.column is None:
        raise ValueError("MatchkeyField requires 'field' or 'column'.")
    for t in self.transforms:
        FieldTransform(transform=t)
    if self.scorer is not None and self.scorer not in VALID_SCORERS:
        raise ValueError(
            f"Invalid scorer '{self.scorer}'. Must be one of {sorted(VALID_SCORERS)}."
        )
    return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_record_embedding_scorer.py -v --tb=short`
Expected: 6 PASSED

- [ ] **Step 5: Run full test suite for regressions**

Run: `pytest --tb=short -q`
Expected: All tests pass (470+)

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/config/schemas.py tests/test_record_embedding_scorer.py
git commit -m "feat: add record_embedding scorer type to schema"
```

---

## Task 2: Record-Level Embedding Scorer — Implementation

**Files:**
- Modify: `goldenmatch/core/scorer.py:133-157, 196-198`
- Modify: `goldenmatch/core/pipeline.py:93-103`
- Test: `tests/test_record_embedding_scorer.py` (append)

- [ ] **Step 1: Write failing tests for scorer logic**

Append to `tests/test_record_embedding_scorer.py`:

```python
from unittest.mock import patch
import numpy as np
import polars as pl


def _make_fake_embedder():
    """Embedder with deterministic fake model."""
    from goldenmatch.core.embedder import Embedder
    e = Embedder("fake-model")

    class FakeModel:
        def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
            rng = np.random.default_rng(42)
            vecs = rng.random((len(texts), 8))
            seen: dict[str, np.ndarray] = {}
            for i, t in enumerate(texts):
                if t in seen:
                    vecs[i] = seen[t]
                else:
                    seen[t] = vecs[i]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return vecs / norms

    e._model = FakeModel()
    return e


class TestRecordEmbeddingScorer:
    def test_record_embedding_score_matrix(self):
        from goldenmatch.core.scorer import _record_embedding_score_matrix

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1, 2],
                "title": ["Sony Turntable", "Sony Turntable", "Samsung TV"],
                "manufacturer": ["Sony", "Sony", "Samsung"],
            })
            matrix = _record_embedding_score_matrix(
                df, ["title", "manufacturer"], "fake-model"
            )
            assert matrix.shape == (3, 3)
            # Identical records should have similarity ~1.0
            assert matrix[0, 1] == pytest.approx(1.0, abs=0.01)

    def test_record_embedding_null_handling(self):
        from goldenmatch.core.scorer import _record_embedding_score_matrix

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1],
                "title": ["Sony", "Sony"],
                "manufacturer": [None, "Sony"],
            })
            matrix = _record_embedding_score_matrix(
                df, ["title", "manufacturer"], "fake-model"
            )
            assert matrix.shape == (2, 2)
            # Should not crash; null manufacturer is skipped

    def test_record_embedding_in_find_fuzzy(self):
        from goldenmatch.core.scorer import find_fuzzy_matches

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1, 2],
                "title": ["Sony Turntable", "Sony Turntable", "Samsung TV"],
                "brand": ["Sony", "Sony", "Samsung"],
            })
            mk = MatchkeyConfig(
                name="rec_emb",
                type="weighted",
                threshold=0.5,
                fields=[
                    MatchkeyField(
                        scorer="record_embedding",
                        columns=["title", "brand"],
                        weight=0.7,
                        model="fake-model",
                    ),
                    MatchkeyField(field="brand", scorer="exact", weight=0.3),
                ],
            )
            results = find_fuzzy_matches(df, mk)
            pair_ids = {(r[0], r[1]) for r in results}
            assert (0, 1) in pair_ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_record_embedding_scorer.py::TestRecordEmbeddingScorer -v --tb=short`
Expected: FAIL — `_record_embedding_score_matrix` not defined

- [ ] **Step 3: Implement `_record_embedding_score_matrix` in scorer.py**

Add after `_fuzzy_score_matrix` (after line 157) in `goldenmatch/core/scorer.py`:

```python
def _record_embedding_score_matrix(
    block_df: pl.DataFrame, columns: list[str], model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """NxN score matrix from record-level embeddings.

    Concatenates columns into a single text string per record,
    embeds the full string, and computes cosine similarity.
    """
    from goldenmatch.core.embedder import get_embedder

    concat_values = []
    for row in block_df.iter_rows(named=True):
        parts = []
        for col in columns:
            val = row.get(col)
            if val is not None:
                parts.append(f"{col}: {val}")
        concat_values.append(" | ".join(parts) if parts else "")

    row_ids = block_df["__row_id__"].to_list()
    cache_key = f"_rec_emb_{hash(tuple(columns))}_{hash(tuple(row_ids))}"

    embedder = get_embedder(model_name)
    embeddings = embedder.embed_column(concat_values, cache_key=cache_key)
    sim = embedder.cosine_similarity_matrix(embeddings)
    return np.asarray(sim, dtype=np.float64)
```

- [ ] **Step 4: Route `record_embedding` in `find_fuzzy_matches`**

In `find_fuzzy_matches()`, update the field classification (around lines 196-198). Change:

```python
exact_fields = [f for f in mk.fields if f.scorer == "exact" or f.scorer == "soundex_match"]
fuzzy_fields = [f for f in mk.fields if f.scorer not in ("exact", "soundex_match")]
```

To:

```python
exact_fields = [f for f in mk.fields if f.scorer == "exact" or f.scorer == "soundex_match"]
record_emb_fields = [f for f in mk.fields if f.scorer == "record_embedding"]
fuzzy_fields = [f for f in mk.fields if f.scorer not in ("exact", "soundex_match", "record_embedding")]
```

**Critical:** Update the `if not fuzzy_fields` branch (around line 228) to also check `record_emb_fields`. Change:
```python
if not fuzzy_fields:
```
To:
```python
if not fuzzy_fields and not record_emb_fields:
```

Then in the `else` branch (Phase 3, fuzzy scoring loop, around line 250), after scoring fuzzy fields, add scoring for record_embedding fields:

```python
for f in record_emb_fields:
    scores = _record_embedding_score_matrix(
        block_df, f.columns, model_name=f.model or "all-MiniLM-L6-v2"
    )
    # record_embedding has no null mask — nulls handled inside concatenation
    fuzzy_numerator += scores * f.weight
    fuzzy_denominator += f.weight
```

- [ ] **Step 5: Update `_get_required_columns` in pipeline.py**

In `goldenmatch/core/pipeline.py`, modify `_get_required_columns` (lines 93-103):

```python
def _get_required_columns(config: GoldenMatchConfig) -> list[str]:
    """Extract all column names referenced in matchkeys and blocking config."""
    cols = set()
    for mk in config.get_matchkeys():
        for f in mk.fields:
            if f.columns:
                cols.update(f.columns)
            elif f.field and f.field != "__record__":
                cols.add(f.field)
    if config.blocking:
        for key_config in config.blocking.keys:
            for field_name in key_config.fields:
                cols.add(field_name)
    return sorted(cols)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_record_embedding_scorer.py -v --tb=short`
Expected: All 9 tests PASSED

- [ ] **Step 7: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add goldenmatch/core/scorer.py goldenmatch/core/pipeline.py tests/test_record_embedding_scorer.py
git commit -m "feat: implement record_embedding scorer with cross-field context"
```

---

## Task 3: Direct-Pair ANN Scoring — `query_with_scores`

**Files:**
- Modify: `goldenmatch/core/ann_blocker.py`
- Test: `tests/test_ann_pairs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ann_pairs.py`:

```python
"""Tests for direct-pair ANN scoring."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestQueryWithScores:
    def test_returns_scores(self):
        from goldenmatch.core.ann_blocker import ANNBlocker

        rng = np.random.default_rng(42)
        embeddings = rng.random((10, 8)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        blocker = ANNBlocker(top_k=3)
        blocker.build_index(embeddings)
        results = blocker.query_with_scores(embeddings)

        assert len(results) > 0
        # Each result is (idx_a, idx_b, score)
        assert len(results[0]) == 3
        for a, b, score in results:
            assert a < b  # ordered
            assert 0.0 <= score <= 1.0

    def test_backward_compat_query(self):
        from goldenmatch.core.ann_blocker import ANNBlocker

        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 4)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        blocker = ANNBlocker(top_k=3)
        blocker.build_index(embeddings)
        results = blocker.query(embeddings)

        assert len(results) > 0
        # Original query returns (idx_a, idx_b) — no score
        assert len(results[0]) == 2

    def test_no_self_pairs_with_scores(self):
        from goldenmatch.core.ann_blocker import ANNBlocker

        embeddings = np.eye(5, dtype=np.float32)
        blocker = ANNBlocker(top_k=5)
        blocker.build_index(embeddings)
        results = blocker.query_with_scores(embeddings)

        for a, b, score in results:
            assert a != b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ann_pairs.py -v --tb=short`
Expected: FAIL — `ANNBlocker` has no attribute `query_with_scores`

- [ ] **Step 3: Add `query_with_scores` to ANNBlocker**

In `goldenmatch/core/ann_blocker.py`, add after the existing `query` method:

```python
def query_with_scores(self, query_embeddings: np.ndarray) -> list[tuple[int, int, float]]:
    """Find top-K neighbors with similarity scores.

    Returns (idx_a, idx_b, cosine_similarity) tuples, ordered so idx_a < idx_b.
    """
    scores_matrix, indices = self._index.search(
        query_embeddings.astype(np.float32), self.top_k,
    )
    pairs: dict[tuple[int, int], float] = {}
    for i in range(len(query_embeddings)):
        for j_idx in range(self.top_k):
            neighbor = int(indices[i][j_idx])
            if neighbor != i and neighbor >= 0:
                pair = (min(i, neighbor), max(i, neighbor))
                score = float(scores_matrix[i][j_idx])
                # Keep the max score if pair seen from both directions
                if pair not in pairs or score > pairs[pair]:
                    pairs[pair] = score
    return [(a, b, s) for (a, b), s in pairs.items()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ann_pairs.py -v --tb=short`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/ann_blocker.py tests/test_ann_pairs.py
git commit -m "feat: add query_with_scores to ANNBlocker"
```

---

## Task 4: Direct-Pair ANN Scoring — Blocker + Scorer Integration

**Files:**
- Modify: `goldenmatch/core/blocker.py:16-25, 305+`
- Modify: `goldenmatch/core/scorer.py:172-192`
- Modify: `goldenmatch/config/schemas.py:133`
- Test: `tests/test_ann_pairs.py` (append)

- [ ] **Step 1: Write failing tests for ann_pairs strategy**

Append to `tests/test_ann_pairs.py`:

```python
import polars as pl
from goldenmatch.config.schemas import (
    BlockingConfig, BlockingKeyConfig, MatchkeyConfig, MatchkeyField,
)


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestAnnPairsBlocking:
    def _make_embedder_and_patch(self):
        """Return a fake embedder for patching."""
        from goldenmatch.core.embedder import Embedder
        e = Embedder("fake-model")

        class FakeModel:
            def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
                rng = np.random.default_rng(42)
                vecs = rng.random((len(texts), 8)).astype(np.float32)
                seen = {}
                for i, t in enumerate(texts):
                    if t in seen:
                        vecs[i] = seen[t]
                    else:
                        seen[t] = vecs[i]
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                return (vecs / norms).astype(np.float32)

        e._model = FakeModel()
        return e

    def test_ann_pairs_returns_pre_scored(self):
        from unittest.mock import patch
        from goldenmatch.core.blocker import build_blocks

        fake = self._make_embedder_and_patch()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": list(range(5)),
                "name": ["sony tv", "sony tv", "samsung phone", "samsung phone", "lg monitor"],
            }).lazy()

            config = BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"])],
                strategy="ann_pairs",
                ann_column="name",
                ann_model="fake-model",
                ann_top_k=3,
            )
            blocks = build_blocks(df, config)
            assert len(blocks) == 1
            assert blocks[0].pre_scored_pairs is not None
            assert len(blocks[0].pre_scored_pairs) > 0
            for a, b, s in blocks[0].pre_scored_pairs:
                assert isinstance(s, float)

    def test_ann_pairs_no_union_find(self):
        """ann_pairs should NOT merge pairs transitively."""
        from unittest.mock import patch
        from goldenmatch.core.blocker import build_blocks

        fake = self._make_embedder_and_patch()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": list(range(10)),
                "name": [f"item_{i}" for i in range(10)],
            }).lazy()

            config = BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"])],
                strategy="ann_pairs",
                ann_column="name",
                ann_model="fake-model",
                ann_top_k=3,
            )
            blocks = build_blocks(df, config)
            # Should be a single block with pre_scored pairs, NOT multiple Union-Find clusters
            assert len(blocks) == 1
            assert blocks[0].strategy == "ann_pairs"


class TestPreScoredPairsInScorer:
    def test_pre_scored_pairs_bypass_nxn(self):
        """When pre_scored_pairs is set, scorer should use them directly."""
        from goldenmatch.core.scorer import find_fuzzy_matches
        from goldenmatch.core.blocker import BlockResult

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "name": ["a", "b", "c", "d"],
        })

        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.5,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)],
        )

        # Pass pre_scored_pairs — scorer should return these directly (filtered by threshold)
        pre_scored = [(0, 1, 0.95), (2, 3, 0.30)]
        results = find_fuzzy_matches(df, mk, pre_scored_pairs=pre_scored)

        # Only (0, 1) is above threshold 0.5
        assert len(results) == 1
        assert results[0] == (0, 1, 0.95)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ann_pairs.py::TestAnnPairsBlocking -v --tb=short`
Expected: FAIL — `ann_pairs` not a valid strategy

- [ ] **Step 3: Add `ann_pairs` to schema**

In `goldenmatch/config/schemas.py`, update the strategy literal (line 133):

```python
strategy: Literal["static", "adaptive", "sorted_neighborhood", "multi_pass", "ann", "canopy", "ann_pairs"] = "static"
```

- [ ] **Step 4: Add `pre_scored_pairs` to BlockResult**

In `goldenmatch/core/blocker.py`, update BlockResult (lines 16-24):

```python
@dataclass
class BlockResult:
    """Result of blocking: a block key and its associated LazyFrame."""

    block_key: str
    df: pl.LazyFrame
    strategy: str = "static"
    depth: int = 0
    parent_key: str | None = None
    pre_scored_pairs: list[tuple[int, int, float]] | None = None
```

- [ ] **Step 5: Implement `_build_ann_pair_blocks`**

In `goldenmatch/core/blocker.py`, add before `build_blocks`:

```python
def _build_ann_pair_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    """Build direct-pair ANN blocks without Union-Find.

    Returns a single BlockResult with pre_scored_pairs set.
    FAISS similarity scores are propagated directly.
    """
    from goldenmatch.core.ann_blocker import ANNBlocker
    from goldenmatch.core.embedder import get_embedder

    if not config.ann_column:
        raise ValueError("ann_pairs blocking requires 'ann_column' to be set.")

    df = lf.collect()
    values = df[config.ann_column].to_list()

    embedder = get_embedder(config.ann_model)
    embeddings = embedder.embed_column(values, cache_key=f"ann_{config.ann_column}")

    blocker = ANNBlocker(top_k=config.ann_top_k)
    blocker.build_index(embeddings)
    scored_pairs = blocker.query_with_scores(embeddings)

    # Map positional indices to __row_id__ values
    row_ids = df["__row_id__"].to_list()
    mapped_pairs = [
        (int(row_ids[a]), int(row_ids[b]), score)
        for a, b, score in scored_pairs
    ]

    return [BlockResult(
        block_key="ann_pairs",
        df=df.lazy(),
        strategy="ann_pairs",
        pre_scored_pairs=mapped_pairs,
    )]
```

Wire it into `build_blocks()`:

```python
if config.strategy == "ann_pairs":
    return _build_ann_pair_blocks(lf, config)
```

- [ ] **Step 6: Handle `pre_scored_pairs` in `find_fuzzy_matches`**

In `goldenmatch/core/scorer.py`, update `find_fuzzy_matches` signature and add early return (at the top of the function, after line 192):

Change signature to:
```python
def find_fuzzy_matches(
    block_df: pl.DataFrame,
    mk: MatchkeyConfig,
    exclude_pairs: set[tuple[int, int]] | None = None,
    pre_scored_pairs: list[tuple[int, int, float]] | None = None,
) -> list[tuple[int, int, float]]:
```

Add at the top (after the `n < 2` check):
```python
# Fast path: pre-scored pairs from ANN (skip NxN scoring)
if pre_scored_pairs is not None:
    results = []
    for a, b, score in pre_scored_pairs:
        if score >= mk.threshold:
            pair_key = (min(a, b), max(a, b))
            if exclude_pairs and pair_key in exclude_pairs:
                continue
            results.append((pair_key[0], pair_key[1], score))
    return results
```

Note: Use `pair_key[0], pair_key[1]` (not `a, b`) to ensure consistent ordering — `row_ids[a] < row_ids[b]` is not guaranteed even though positional `a < b`.

- [ ] **Step 7: Update pipeline to pass pre_scored_pairs**

In `goldenmatch/core/pipeline.py`, update ALL `find_fuzzy_matches` call sites to pass `pre_scored_pairs`:

1. In `run_dedupe()` — both the `across_files_only` branch and the normal branch
2. In `run_match()` — the block scoring loop

Each call should become:
```python
pairs = find_fuzzy_matches(
    bdf, mk,
    exclude_pairs=matched_pairs,
    pre_scored_pairs=block.pre_scored_pairs,
)
```

Do the same in `goldenmatch/tui/engine.py` where it calls `find_fuzzy_matches` in the block loop.

**Verify:** Search for all `find_fuzzy_matches` call sites: `grep -rn "find_fuzzy_matches" goldenmatch/` and update every one.

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_ann_pairs.py -v --tb=short`
Expected: 6 PASSED

- [ ] **Step 9: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

- [ ] **Step 10: Commit**

```bash
git add goldenmatch/config/schemas.py goldenmatch/core/blocker.py goldenmatch/core/scorer.py goldenmatch/core/pipeline.py goldenmatch/tui/engine.py tests/test_ann_pairs.py
git commit -m "feat: add ann_pairs strategy with direct-pair scoring"
```

---

## Task 5: Threshold Auto-Tuning

**Files:**
- Create: `goldenmatch/core/threshold.py`
- Modify: `goldenmatch/config/schemas.py:81-107`
- Test: `tests/test_threshold.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_threshold.py`:

```python
"""Tests for threshold auto-tuning via Otsu's method."""

from __future__ import annotations

import numpy as np
import pytest


class TestSuggestThreshold:
    def test_bimodal_distribution(self):
        from goldenmatch.core.threshold import suggest_threshold

        rng = np.random.default_rng(42)
        # Non-matches around 0.3, matches around 0.9
        non_matches = rng.normal(0.3, 0.05, 500).clip(0, 1).tolist()
        matches = rng.normal(0.9, 0.05, 100).clip(0, 1).tolist()
        scores = non_matches + matches

        result = suggest_threshold(scores)
        assert result is not None
        assert 0.5 < result < 0.8  # should find the gap

    def test_unimodal_returns_none(self):
        from goldenmatch.core.threshold import suggest_threshold

        rng = np.random.default_rng(42)
        scores = rng.normal(0.5, 0.05, 500).clip(0, 1).tolist()

        result = suggest_threshold(scores)
        assert result is None

    def test_empty_scores(self):
        from goldenmatch.core.threshold import suggest_threshold

        result = suggest_threshold([])
        assert result is None

    def test_single_score(self):
        from goldenmatch.core.threshold import suggest_threshold

        result = suggest_threshold([0.5])
        assert result is None

    def test_all_identical_scores(self):
        from goldenmatch.core.threshold import suggest_threshold

        result = suggest_threshold([0.8] * 100)
        assert result is None

    def test_two_distinct_clusters(self):
        from goldenmatch.core.threshold import suggest_threshold

        scores = [0.1] * 50 + [0.9] * 50
        result = suggest_threshold(scores)
        assert result is not None
        assert 0.3 < result < 0.7

    def test_result_within_valid_range(self):
        from goldenmatch.core.threshold import suggest_threshold

        rng = np.random.default_rng(42)
        non_matches = rng.normal(0.2, 0.1, 300).clip(0, 1).tolist()
        matches = rng.normal(0.85, 0.05, 100).clip(0, 1).tolist()

        result = suggest_threshold(non_matches + matches)
        if result is not None:
            assert 0.0 < result < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_threshold.py -v --tb=short`
Expected: FAIL — `goldenmatch.core.threshold` not found

- [ ] **Step 3: Implement `threshold.py`**

Create `goldenmatch/core/threshold.py`:

```python
"""Threshold auto-tuning for GoldenMatch using Otsu's method."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def suggest_threshold(
    scores: list[float], n_bins: int = 100,
) -> float | None:
    """Find optimal threshold using Otsu's method.

    Finds the threshold that best separates a bimodal score distribution
    into "match" and "non-match" populations by minimizing intra-class
    variance.

    Returns None if the distribution is unimodal (no clear separation)
    or if there are too few scores.
    """
    if len(scores) < 2:
        return None

    arr = np.array(scores, dtype=np.float64)

    # Check for degenerate case (all identical)
    if arr.std() < 1e-10:
        return None

    # Histogram over [0, 1]
    counts, bin_edges = np.histogram(arr, bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()

    # Otsu's method: find threshold minimizing weighted intra-class variance
    best_threshold = None
    best_variance = float("inf")

    cum_count = 0
    cum_sum = 0.0
    total_sum = sum(counts[i] * (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(n_bins))

    for i in range(n_bins):
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        cum_count += counts[i]
        cum_sum += counts[i] * bin_center

        if cum_count == 0 or cum_count == total:
            continue

        w0 = cum_count / total
        w1 = 1.0 - w0
        mean0 = cum_sum / cum_count
        mean1 = (total_sum - cum_sum) / (total - cum_count)

        # Between-class variance (maximizing this = minimizing intra-class)
        between_variance = w0 * w1 * (mean0 - mean1) ** 2

        # We want max between-class variance = min intra-class variance
        if between_variance > 0:
            intra = -between_variance  # negate to find minimum
            if intra < best_variance:
                best_variance = intra
                best_threshold = bin_edges[i + 1]

    if best_threshold is None:
        return None

    # Check if distribution is truly bimodal:
    # Compute variance ratio = between-class / total variance
    total_variance = arr.var()
    if total_variance < 1e-10:
        return None

    max_between = -best_variance
    variance_ratio = max_between / total_variance

    if variance_ratio < 0.15:
        # Distribution is essentially unimodal — Otsu is unreliable
        logger.info(
            "Score distribution is unimodal (variance ratio %.2f). "
            "Using configured threshold as fallback.",
            variance_ratio,
        )
        return None

    logger.info(
        "Auto-threshold: %.3f (variance ratio: %.2f, %d scores)",
        best_threshold,
        variance_ratio,
        len(scores),
    )
    return float(best_threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_threshold.py -v --tb=short`
Expected: 7 PASSED

- [ ] **Step 5: Add `auto_threshold` to MatchkeyConfig schema**

In `goldenmatch/config/schemas.py`, add to `MatchkeyConfig` (after `threshold` on line 86):

```python
class MatchkeyConfig(BaseModel):
    name: str
    type: Literal["exact", "weighted"] | None = None
    comparison: str | None = None
    fields: list[MatchkeyField]
    threshold: float | None = None
    auto_threshold: bool = False
```

- [ ] **Step 6: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add goldenmatch/core/threshold.py goldenmatch/config/schemas.py tests/test_threshold.py
git commit -m "feat: add threshold auto-tuning via Otsu's method"
```

---

## Task 6: Pipeline Integration for Auto-Threshold

**Files:**
- Modify: `goldenmatch/core/pipeline.py`
- Modify: `goldenmatch/tui/engine.py`

- [ ] **Step 1: Integrate auto-threshold into pipeline.py**

In `goldenmatch/core/pipeline.py`, in the fuzzy matching phase of `run_dedupe()`, add auto-threshold logic. After building blocks and before the main scoring loop:

```python
from goldenmatch.core.threshold import suggest_threshold

# Auto-threshold: sample scores to determine optimal threshold
if mk.auto_threshold:
    import random
    sample_scores = []
    sample_blocks = list(blocks)
    random.shuffle(sample_blocks)
    for block in sample_blocks:
        if len(sample_scores) >= 10000:
            break
        bdf = block.df.collect()
        if block.pre_scored_pairs:
            sample_scores.extend(s for _, _, s in block.pre_scored_pairs)
        elif bdf.height >= 2:
            pairs = find_fuzzy_matches(bdf, mk)
            sample_scores.extend(s for _, _, s in pairs)
    if len(sample_scores) >= 100:
        suggested = suggest_threshold(sample_scores)
        if suggested is not None:
            logger.info("Auto-threshold: %.3f (from %d scores)", suggested, len(sample_scores))
            mk = mk.model_copy(update={"threshold": suggested})
        else:
            logger.info("Auto-threshold: skipped (unimodal), using %.3f", mk.threshold)
```

Note: `mk.model_copy()` creates a new MatchkeyConfig without mutating the original. This is important because the config may be reused.

- [ ] **Step 2: Mirror in engine.py**

Apply the same auto-threshold logic in `goldenmatch/tui/engine.py` at the equivalent point in `_run_pipeline()`.

- [ ] **Step 3: Run full test suite**

Run: `pytest --tb=short -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add goldenmatch/core/pipeline.py goldenmatch/tui/engine.py
git commit -m "feat: wire auto-threshold into pipeline and TUI engine"
```

---

## Task 7: Benchmark Validation

**Files:**
- Modify: `tests/benchmarks/run_leipzig.py`

- [ ] **Step 1: Update `run_matching()` to support `pre_scored_pairs`**

In `tests/benchmarks/run_leipzig.py`, update the scoring loop in `run_matching()` (around line 111) to pass `pre_scored_pairs`:

```python
for block in blocks:
    bdf = block.df.collect()
    pairs = find_fuzzy_matches(bdf, mk, pre_scored_pairs=block.pre_scored_pairs)
    all_pairs.extend(pairs)
```

- [ ] **Step 2: Add record_embedding + ann_pairs strategies to benchmark**

Add new strategies to `run_abt_buy()` and `run_amazon_google()` in `tests/benchmarks/run_leipzig.py`:

```python
# Record-level embedding + ann_pairs
try:
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="rec_emb",
            fields=[
                MatchkeyField(
                    scorer="record_embedding",
                    columns=["name"],  # or ["title", "manufacturer"] for Amazon-Google
                    weight=1.0,
                    model="all-MiniLM-L6-v2",
                ),
            ],
            comparison="weighted",
            threshold=0.80,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
        strategy="ann_pairs",
        ann_column="name",
        ann_model="all-MiniLM-L6-v2",
        ann_top_k=20,
    ), standardization={"name": ["strip", "trim_whitespace"]})
    r = evaluate("Abt-Buy", "rec_emb+ann_pairs(0.80)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)
except ImportError as e:
    print(f"\n  [SKIPPED: {e}]")
```

Add similar for Amazon-Google using `columns=["title", "manufacturer"]`.

- [ ] **Step 3: Run benchmarks**

Run: `python tests/benchmarks/run_leipzig.py`

Record the results. Target:
- Abt-Buy: > 55% F1 (was 43.1%)
- Amazon-Google: > 50% F1 (was 40.5%)
- DBLP-ACM: > 97% F1 (regression check)
- DBLP-Scholar: > 74% F1 (regression check)

- [ ] **Step 4: Update README with new results**

Update the Leipzig Benchmark Results table in `README.md` with the new best results.

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/run_leipzig.py README.md
git commit -m "docs: update benchmarks with record_embedding + ann_pairs results"
```

- [ ] **Step 6: Run full test suite one final time**

Run: `pytest --tb=short -q`
Expected: All tests pass
