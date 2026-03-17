# Adaptive Blocking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace static blocking with adaptive blocking that auto-suggests keys, recursively sub-blocks oversized groups, and offers sorted neighborhood as a fallback — enabling fuzzy matching at 1M+ scale.

**Architecture:** Three layers: (1) block_analyzer.py for auto-suggesting optimal keys via cardinality + recall analysis, (2) enhanced blocker.py with adaptive sub-blocking and sorted neighborhood strategies, (3) CLI/pipeline integration. All backwards-compatible — `strategy: "static"` preserves current behavior.

**Tech Stack:** Python 3.11+, Polars, rapidfuzz (cdist for recall sampling), existing blocker/scorer/pipeline modules

**Spec:** `docs/superpowers/specs/2026-03-17-adaptive-blocking-design.md`

---

## Chunk 1: Schema Changes + BlockResult Metadata

### Task 1: Update BlockingConfig Schema

**Files:**
- Modify: `goldenmatch/config/schemas.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for new schema fields**

Add to `tests/test_config.py`:

```python
class TestBlockingConfigAdaptive:
    def test_static_strategy_default(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"], transforms=["strip"])])
        assert cfg.strategy == "static"

    def test_adaptive_strategy(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=["strip"])],
            strategy="adaptive",
            sub_block_keys=[BlockingKeyConfig(fields=["first_name"], transforms=["lowercase"])],
        )
        assert cfg.strategy == "adaptive"
        assert len(cfg.sub_block_keys) == 1

    def test_sorted_neighborhood_strategy(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, SortKeyField
        cfg = BlockingConfig(
            keys=[],
            strategy="sorted_neighborhood",
            window_size=25,
            sort_key=[
                SortKeyField(column="last_name", transforms=["lowercase", "soundex"]),
                SortKeyField(column="zip", transforms=["substring:0:3"]),
            ],
        )
        assert cfg.window_size == 25
        assert len(cfg.sort_key) == 2

    def test_invalid_strategy_rejected(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        with pytest.raises(ValueError):
            BlockingConfig(keys=[], strategy="invalid")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_config.py::TestBlockingConfigAdaptive -v`
Expected: FAIL

- [ ] **Step 3: Implement schema changes**

Add to `goldenmatch/config/schemas.py`:

```python
class SortKeyField(BaseModel):
    column: str
    transforms: list[str] = Field(default_factory=list)
```

Modify `BlockingConfig`:
```python
class BlockingConfig(BaseModel):
    keys: list[BlockingKeyConfig]
    max_block_size: int = 5000
    skip_oversized: bool = False
    strategy: Literal["static", "adaptive", "sorted_neighborhood"] = "static"
    auto_suggest: bool = False
    sub_block_keys: list[BlockingKeyConfig] | None = None
    window_size: int = 20
    sort_key: list[SortKeyField] | None = None
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/config/schemas.py tests/test_config.py
git commit -m "feat: add adaptive blocking schema (strategy, sub_block_keys, SortKeyField)"
```

---

### Task 2: Extend BlockResult with Metadata

**Files:**
- Modify: `goldenmatch/core/blocker.py`
- Modify: `tests/test_blocker.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_blocker.py`:

```python
class TestBlockResultMetadata:
    def test_default_metadata(self):
        from goldenmatch.core.blocker import BlockResult
        br = BlockResult(block_key="test", df=pl.DataFrame({"a": [1]}).lazy())
        assert br.strategy == "static"
        assert br.depth == 0
        assert br.parent_key is None

    def test_custom_metadata(self):
        from goldenmatch.core.blocker import BlockResult
        br = BlockResult(
            block_key="sub", df=pl.DataFrame({"a": [1]}).lazy(),
            strategy="adaptive", depth=1, parent_key="parent"
        )
        assert br.strategy == "adaptive"
        assert br.depth == 1
        assert br.parent_key == "parent"
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Extend BlockResult**

```python
@dataclass
class BlockResult:
    block_key: str
    df: pl.LazyFrame
    strategy: str = "static"
    depth: int = 0
    parent_key: str | None = None
```

- [ ] **Step 4: Run all tests**

Run: `cd D:/show_case/goldenmatch && pytest --tb=short`
Expected: All PASS (no regressions — existing code only accesses block_key and df)

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/blocker.py tests/test_blocker.py
git commit -m "feat: extend BlockResult with strategy, depth, parent_key metadata"
```

---

## Chunk 2: Adaptive Sub-Blocking + Sorted Neighborhood

### Task 3: Adaptive Sub-Blocking in blocker.py

**Files:**
- Modify: `goldenmatch/core/blocker.py`
- Modify: `tests/test_blocker.py`

- [ ] **Step 1: Write failing tests**

```python
class TestAdaptiveSubBlocking:
    def test_sub_blocks_oversized(self):
        """Oversized block is split using sub_block_keys."""
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["group"], transforms=[])],
            max_block_size=3,
            strategy="adaptive",
            sub_block_keys=[BlockingKeyConfig(fields=["subgroup"], transforms=[])],
        )
        df = pl.DataFrame({
            "__row_id__": list(range(6)),
            "group": ["A", "A", "A", "A", "A", "B"],
            "subgroup": ["X", "X", "Y", "Y", "Y", "Z"],
        }).lazy()
        blocks = build_blocks(df, cfg)
        # Group A (5 records) should be sub-blocked into X(2) and Y(3)
        # Group B (1 record) should be skipped (< 2)
        sizes = sorted([b.df.collect().height for b in blocks])
        assert all(s <= 3 for s in sizes)
        assert sum(sizes) == 5  # all 5 from group A

    def test_max_depth_3(self):
        """Sub-blocking stops at depth 3."""
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["k1"], transforms=[])],
            max_block_size=2,
            strategy="adaptive",
            sub_block_keys=[
                BlockingKeyConfig(fields=["k2"], transforms=[]),
                BlockingKeyConfig(fields=["k3"], transforms=[]),
                BlockingKeyConfig(fields=["k4"], transforms=[]),
            ],
        )
        # All same values for sub-keys -> can't split further
        df = pl.DataFrame({
            "__row_id__": list(range(10)),
            "k1": ["A"] * 10,
            "k2": ["B"] * 10,
            "k3": ["C"] * 10,
            "k4": ["D"] * 10,
        }).lazy()
        blocks = build_blocks(df, cfg)
        # Should not crash, should produce blocks (fallback to sorted_neighborhood or process anyway)
        assert len(blocks) >= 1

    def test_adaptive_metadata(self):
        """Sub-blocked results have correct metadata."""
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["group"], transforms=[])],
            max_block_size=2,
            strategy="adaptive",
            sub_block_keys=[BlockingKeyConfig(fields=["sub"], transforms=[])],
        )
        df = pl.DataFrame({
            "__row_id__": list(range(4)),
            "group": ["A", "A", "A", "B"],
            "sub": ["X", "Y", "Y", "Z"],
        }).lazy()
        blocks = build_blocks(df, cfg)
        sub_blocks = [b for b in blocks if b.depth > 0]
        assert len(sub_blocks) >= 1
        assert sub_blocks[0].strategy == "adaptive"

    def test_static_strategy_unchanged(self):
        """Static strategy preserves existing behavior exactly."""
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["key"], transforms=[])],
            max_block_size=2,
            strategy="static",
            skip_oversized=True,
        )
        df = pl.DataFrame({
            "__row_id__": list(range(5)),
            "key": ["A", "A", "A", "B", "B"],
        }).lazy()
        blocks = build_blocks(df, cfg)
        # A block (3) is oversized -> skipped. B block (2) -> kept.
        assert len(blocks) == 1
        assert blocks[0].df.collect().height == 2
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement adaptive sub-blocking**

Refactor `build_blocks` in `goldenmatch/core/blocker.py`:

- Extract current logic into `_build_static_blocks(lf, config) -> list[BlockResult]`
- Add `_sub_block(block_df, config, depth, parent_key) -> list[BlockResult]` for recursive splitting
- Main `build_blocks` routes by `config.strategy`
- For "adaptive": build primary blocks, then sub-block any that exceed max_block_size
- For "static": call `_build_static_blocks` (current behavior)
- For "sorted_neighborhood": delegate to sorted neighborhood function (Task 4)

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_blocker.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `cd D:/show_case/goldenmatch && pytest --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/blocker.py tests/test_blocker.py
git commit -m "feat: adaptive sub-blocking with recursive splitting and max depth 3"
```

---

### Task 4: Sorted Neighborhood

**Files:**
- Modify: `goldenmatch/core/blocker.py`
- Modify: `tests/test_blocker.py`

- [ ] **Step 1: Write failing tests**

```python
class TestSortedNeighborhood:
    def test_basic_window(self):
        """Sorted neighborhood produces overlapping windows."""
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, SortKeyField
        cfg = BlockingConfig(
            keys=[],
            strategy="sorted_neighborhood",
            window_size=3,
            sort_key=[SortKeyField(column="name", transforms=["lowercase"])],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        }).lazy()
        blocks = build_blocks(df, cfg)
        # Window 3 on 5 records: positions 0-2, 1-3, 2-4 = 3 windows
        assert len(blocks) == 3
        assert all(b.df.collect().height == 3 for b in blocks)
        assert all(b.strategy == "sorted_neighborhood" for b in blocks)

    def test_window_dedup(self):
        """Same pair in overlapping windows should still be in multiple blocks (dedup happens at scorer level)."""
        from goldenmatch.config.schemas import BlockingConfig, SortKeyField
        cfg = BlockingConfig(
            keys=[],
            strategy="sorted_neighborhood",
            window_size=3,
            sort_key=[SortKeyField(column="val", transforms=[])],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "val": ["a", "b", "c", "d"],
        }).lazy()
        blocks = build_blocks(df, cfg)
        # Windows: [a,b,c], [b,c,d] = 2 windows
        assert len(blocks) == 2

    def test_small_dataset(self):
        """Dataset smaller than window size produces single block."""
        from goldenmatch.config.schemas import BlockingConfig, SortKeyField
        cfg = BlockingConfig(
            keys=[],
            strategy="sorted_neighborhood",
            window_size=10,
            sort_key=[SortKeyField(column="val", transforms=[])],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "val": ["a", "b", "c"],
        }).lazy()
        blocks = build_blocks(df, cfg)
        assert len(blocks) == 1
        assert blocks[0].df.collect().height == 3
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement sorted neighborhood**

Add `_build_sorted_neighborhood_blocks(lf, config) -> list[BlockResult]` to blocker.py:

1. Build sort key by transforming and concatenating sort_key fields
2. Sort DataFrame by the sort key
3. Slide window of size `config.window_size` through sorted records
4. Each window position is a BlockResult with strategy="sorted_neighborhood"

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_blocker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/blocker.py tests/test_blocker.py
git commit -m "feat: sorted neighborhood blocking with sliding window"
```

---

## Chunk 3: Block Analyzer

### Task 5: Block Analyzer Core

**Files:**
- Create: `goldenmatch/core/block_analyzer.py`
- Create: `tests/test_block_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
import polars as pl
from goldenmatch.core.block_analyzer import (
    analyze_blocking,
    BlockingSuggestion,
    detect_column_type,
    generate_candidates,
    score_candidate,
)


@pytest.fixture
def analyzer_df():
    """Synthetic data with known distribution for testing."""
    import random
    random.seed(42)
    n = 1000
    last_names = [random.choice(["Smith", "Jones", "Williams", "Brown", "Davis"]) for _ in range(n)]
    first_names = [random.choice(["John", "Jane", "Bob", "Alice", "Eve"]) for _ in range(n)]
    zips = [f"{random.randint(10000, 99999)}" for _ in range(n)]
    states = [random.choice(["PA", "NJ", "NY"]) for _ in range(n)]
    return pl.DataFrame({
        "__row_id__": list(range(n)),
        "last_name": last_names,
        "first_name": first_names,
        "zip": zips,
        "state": states,
    })


class TestDetectColumnType:
    def test_name_field(self):
        assert detect_column_type("last_name") == "name"
        assert detect_column_type("first_name") == "name"
        assert detect_column_type("fname") == "name"

    def test_zip_field(self):
        assert detect_column_type("zip") == "zip"
        assert detect_column_type("postal_code") == "zip"

    def test_email_field(self):
        assert detect_column_type("email") == "email"
        assert detect_column_type("email_address") == "email"

    def test_generic(self):
        assert detect_column_type("foobar") == "generic"


class TestGenerateCandidates:
    def test_generates_candidates(self):
        candidates = generate_candidates(["last_name", "zip"])
        assert len(candidates) > 0
        # Should have single-column and compound candidates
        descriptions = [c["description"] for c in candidates]
        assert any("last_name" in d for d in descriptions)
        assert any("zip" in d for d in descriptions)


class TestScoreCandidate:
    def test_scoring(self, analyzer_df):
        candidate = {
            "key_fields": ["last_name"],
            "transforms": ["lowercase", "substring:0:3"],
            "description": "last_name[:3]",
        }
        result = score_candidate(analyzer_df, candidate, target_block_size=200)
        assert result["group_count"] > 0
        assert result["max_group_size"] > 0
        assert result["score"] > 0


class TestAnalyzeBlocking:
    def test_returns_ranked_suggestions(self, analyzer_df):
        suggestions = analyze_blocking(
            analyzer_df,
            matchkey_columns=["last_name", "zip"],
            sample_size=500,
        )
        assert len(suggestions) > 0
        assert isinstance(suggestions[0], BlockingSuggestion)
        # Should be sorted by score descending
        scores = [s.score for s in suggestions]
        assert scores == sorted(scores, reverse=True)

    def test_coverage_check(self, analyzer_df):
        suggestions = analyze_blocking(
            analyzer_df,
            matchkey_columns=["last_name", "zip"],
            sample_size=500,
        )
        # All suggestions should reference matchkey columns
        for s in suggestions:
            # At least one key field should be in matchkey columns
            all_fields = []
            for key in s.keys:
                all_fields.extend(key.get("key_fields", []))
            assert any(f in ["last_name", "zip"] for f in all_fields)
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement block analyzer**

Create `goldenmatch/core/block_analyzer.py` with:
- `detect_column_type(column_name) -> str` — heuristic name-based type detection
- `generate_candidates(matchkey_columns) -> list[dict]` — generate single + compound candidates per column type
- `score_candidate(df, candidate, target_block_size) -> dict` — cardinality scoring
- `check_coverage(candidate, matchkey_columns) -> bool` — verify key derives from matchkey fields
- `estimate_recall(df, candidate, matchkey_columns, sample_size) -> float` — pair sampling with cdist
- `analyze_blocking(df, matchkey_columns, sample_size) -> list[BlockingSuggestion]` — full analyzer pipeline

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_block_analyzer.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `cd D:/show_case/goldenmatch && pytest --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/block_analyzer.py tests/test_block_analyzer.py
git commit -m "feat: block analyzer with cardinality scoring, coverage check, and recall sampling"
```

---

## Chunk 4: Pipeline + CLI Integration

### Task 6: Pipeline Integration

**Files:**
- Modify: `goldenmatch/core/pipeline.py`
- Modify: `goldenmatch/tui/engine.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_pipeline.py`:

```python
class TestAdaptiveBlockingPipeline:
    def test_dedupe_with_adaptive_blocking(self, sample_csv, tmp_path):
        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
            OutputConfig, GoldenRulesConfig, GoldenFieldRule,
            BlockingConfig, BlockingKeyConfig,
        )
        cfg = GoldenMatchConfig(
            matchkeys=[
                MatchkeyConfig(
                    name="fuzzy_name",
                    fields=[
                        MatchkeyField(column="last_name", transforms=["lowercase"], scorer="jaro_winkler", weight=0.6),
                        MatchkeyField(column="zip", transforms=[], scorer="exact", weight=0.4),
                    ],
                    comparison="weighted",
                    threshold=0.85,
                )
            ],
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["last_name"], transforms=["lowercase", "substring:0:3"])],
                strategy="adaptive",
                sub_block_keys=[BlockingKeyConfig(fields=["zip"], transforms=["substring:0:3"])],
                max_block_size=3,
            ),
            output=OutputConfig(format="csv", directory=str(tmp_path), run_name="adaptive_test"),
            golden_rules=GoldenRulesConfig(default=GoldenFieldRule(strategy="most_complete")),
        )
        from goldenmatch.core.pipeline import run_dedupe
        results = run_dedupe(
            files=[(sample_csv, "test")],
            config=cfg,
            output_report=True,
        )
        assert results["report"]["total_records"] == 5
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Integrate into pipeline**

In `pipeline.py`, before the BLOCK + COMPARE step:
- If `config.blocking.auto_suggest`: run `analyze_blocking`, log recommendations
- `build_blocks` already reads strategy from config — no changes needed there

Same integration in `tui/engine.py`.

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/pipeline.py goldenmatch/tui/engine.py tests/test_pipeline.py
git commit -m "feat: integrate adaptive blocking into pipeline"
```

---

### Task 7: CLI Integration

**Files:**
- Modify: `goldenmatch/cli/main.py`
- Modify: `goldenmatch/cli/dedupe.py`
- Modify: `goldenmatch/cli/match.py`

- [ ] **Step 1: Add analyze-blocking command**

Add to `goldenmatch/cli/main.py`:

```python
@app.command("analyze-blocking")
def analyze_blocking_cmd(
    files: list[str] = typer.Argument(..., help="File(s) to analyze"),
    config: str = typer.Option(..., "--config", "-c", help="Config file with matchkeys"),
):
    """Analyze data and suggest optimal blocking strategies."""
    # Load files, extract matchkey columns from config, run analyzer, print Rich table
```

- [ ] **Step 2: Add --auto-block to dedupe and match**

Add `auto_block: bool = typer.Option(False, "--auto-block", help="Auto-suggest blocking keys")` to both dedupe_cmd and match_cmd. When set, enable `config.blocking.auto_suggest = True`.

- [ ] **Step 3: Run full suite**

Run: `cd D:/show_case/goldenmatch && pytest --tb=short`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add goldenmatch/cli/main.py goldenmatch/cli/dedupe.py goldenmatch/cli/match.py
git commit -m "feat: analyze-blocking CLI command and --auto-block flag"
```
