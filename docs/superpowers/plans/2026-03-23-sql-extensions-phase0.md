# SQL Extensions Phase 0: DataFrame API Entry Points

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dedupe_df()` and `match_df()` functions to the GoldenMatch Python API that accept Polars DataFrames directly, bypassing file ingest. This is the prerequisite for the Rust SQL extensions which pass Arrow data, not file paths.

**Architecture:** Two new public functions in `_api.py` that construct a temporary in-memory file spec and inject the DataFrame directly into the pipeline. The internal pipeline already works on DataFrames after `load_file()` -- we just skip that step. Also add a standalone `score_pair_df()` for scoring two JSON records, and `explain_pair_df()` for explanation from record dicts.

**Tech Stack:** Python, Polars, existing GoldenMatch pipeline

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `goldenmatch/_api.py` | Modify | Add `dedupe_df()`, `match_df()`, `score_strings()`, `score_pair_df()`, `explain_pair_df()` |
| `goldenmatch/core/pipeline.py` | Modify | Add `run_dedupe_df()`, `run_match_df()`, extract `_run_dedupe_pipeline()`, `_run_match_pipeline()` |
| `goldenmatch/__init__.py` | Modify | Export new functions |
| `tests/test_api.py` | Modify | Add tests for DataFrame entry points |

---

### Task 1: Add `dedupe_df()` to the API

**Files:**
- Modify: `goldenmatch/_api.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`:

```python
class TestDedupeDf:
    def test_dedupe_df_exact(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "first_name": ["John", "john", "Jane", "JOHN", "Bob"],
            "email": ["john@x.com", "john@x.com", "jane@y.com", "john@x.com", "bob@z.com"],
        })
        result = gm.dedupe_df(df, exact=["email"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records == 5
        assert result.total_clusters >= 1

    def test_dedupe_df_fuzzy(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "Jane Doe", "Bob Jones"],
            "zip": ["10001", "10001", "20002", "30003"],
        })
        result = gm.dedupe_df(df, fuzzy={"name": 0.80}, blocking=["zip"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records == 4

    def test_dedupe_df_with_config_object(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "email": ["a@x.com", "a@x.com", "b@y.com"],
        })
        cfg = gm.GoldenMatchConfig(
            matchkeys=[gm.MatchkeyConfig(
                name="email",
                type="exact",
                fields=[gm.MatchkeyField(field="email", transforms=["lowercase"])],
            )],
        )
        result = gm.dedupe_df(df, config=cfg)
        assert result.total_clusters >= 1

    def test_dedupe_df_returns_scored_pairs(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "email": ["a@x.com", "a@x.com", "b@y.com"],
        })
        result = gm.dedupe_df(df, exact=["email"])
        assert isinstance(result.scored_pairs, list)

    def test_dedupe_df_empty(self):
        import goldenmatch as gm
        df = pl.DataFrame({"email": []}).cast({"email": pl.Utf8})
        result = gm.dedupe_df(df, exact=["email"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records == 0

    def test_dedupe_df_missing_column_raises(self):
        import goldenmatch as gm
        df = pl.DataFrame({"name": ["John"]})
        with pytest.raises(Exception):
            gm.dedupe_df(df, exact=["nonexistent_column"])

    def test_dedupe_df_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "dedupe_df")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py::TestDedupeDf -v`
Expected: FAIL with `AttributeError: module 'goldenmatch' has no attribute 'dedupe_df'`

- [ ] **Step 3: Implement `dedupe_df()` in `_api.py`**

Add after the existing `dedupe()` function in `goldenmatch/_api.py`:

```python
def dedupe_df(
    df: pl.DataFrame,
    *,
    config: Any | None = None,
    exact: list[str] | None = None,
    fuzzy: dict[str, float] | None = None,
    blocking: list[str] | None = None,
    threshold: float | None = None,
    llm_scorer: bool = False,
    backend: str | None = None,
    source_name: str = "dataframe",
) -> DedupeResult:
    """Deduplicate a Polars DataFrame directly (no file I/O).

    Same as dedupe() but accepts a DataFrame instead of file paths.
    Designed for programmatic use and as the entry point for SQL extensions.

    Args:
        df: Polars DataFrame to deduplicate.
        config: GoldenMatchConfig object, or None for auto-config from kwargs.
        exact: List of column names for exact matching.
        fuzzy: Dict of column name -> threshold for fuzzy matching.
        blocking: List of column names for blocking.
        threshold: Override fuzzy match threshold for all fields.
        llm_scorer: Enable LLM scoring for borderline pairs.
        backend: Processing backend: None (default), "ray".
        source_name: Source label for the DataFrame (default: "dataframe").

    Returns:
        DedupeResult with golden records, clusters, dupes, unique, and stats.
    """
    from goldenmatch.core.pipeline import run_dedupe_df

    if isinstance(config, str):
        config = load_config(config)
    elif config is None:
        config = _build_config(exact, fuzzy, blocking, threshold, llm_scorer, backend)

    if backend and hasattr(config, "backend"):
        config.backend = backend

    result = run_dedupe_df(df, config, source_name=source_name)

    return DedupeResult(
        golden=result.get("golden"),
        clusters=result.get("clusters", {}),
        dupes=result.get("dupes"),
        unique=result.get("unique"),
        stats=_extract_stats(result),
        scored_pairs=_extract_pairs(result),
        config=config,
    )
```

- [ ] **Step 4: Implement `run_dedupe_df()` in pipeline.py**

Add to `goldenmatch/core/pipeline.py` after the `run_dedupe()` function. This reuses the pipeline from Step 1.5 onward (after ingest):

```python
def run_dedupe_df(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    source_name: str = "dataframe",
    output_golden: bool = False,
    output_clusters: bool = False,
    output_dupes: bool = False,
    output_unique: bool = False,
    output_report: bool = False,
) -> dict:
    """Run dedupe pipeline on a DataFrame directly (no file I/O).

    Identical to run_dedupe() but skips the file ingest step.
    """
    matchkeys = config.get_matchkeys()

    # ── Step 1: Prepare DataFrame (skip file loading) ──
    lf = df.lazy()

    # Validate columns exist (same check as run_dedupe's ingest step)
    required = _get_required_columns(config)
    validate_columns(lf, required)

    lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
    lf = _add_row_ids(lf, offset=0)
    combined_lf = lf.collect().lazy()

    # ── From here, identical to run_dedupe ──
    return _run_dedupe_pipeline(combined_lf, config, matchkeys,
                                output_golden, output_clusters,
                                output_dupes, output_unique, output_report)
```

- [ ] **Step 5: Extract shared pipeline logic into `_run_dedupe_pipeline()`**

Refactor `run_dedupe()` in `goldenmatch/core/pipeline.py` to extract everything after the ingest step (from "Step 1.5a: AUTO-FIX + VALIDATION" onward) into a shared `_run_dedupe_pipeline()` function. Both `run_dedupe()` and `run_dedupe_df()` call this.

The function signature:

```python
def _run_dedupe_pipeline(
    combined_lf: pl.LazyFrame,
    config: GoldenMatchConfig,
    matchkeys: list,
    output_golden: bool = False,
    output_clusters: bool = False,
    output_dupes: bool = False,
    output_unique: bool = False,
    output_report: bool = False,
    across_files_only: bool = False,
    llm_retrain: bool = False,
    llm_provider: str | None = None,
    llm_max_labels: int = 500,
) -> dict:
```

The body is everything from the current `run_dedupe()` starting at "Step 1.5a: AUTO-FIX + VALIDATION" through to the return statement. The existing `run_dedupe()` should call `_run_dedupe_pipeline()` after its ingest step.

- [ ] **Step 6: Export `dedupe_df` from `__init__.py`**

Add `dedupe_df` to the import from `_api` and to `__all__` in `goldenmatch/__init__.py`:

```python
from goldenmatch._api import (
    dedupe,
    dedupe_df,
    match,
    ...
)
```

And in `__all__`:
```python
"dedupe", "dedupe_df", "match", ...
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_api.py::TestDedupeDf -v`
Expected: All 7 tests PASS

- [ ] **Step 8: Run full test suite**

Run: `pytest --tb=short`
Expected: All 935+ tests pass (existing tests unaffected by refactor)

- [ ] **Step 9: Commit**

```bash
git add goldenmatch/_api.py goldenmatch/core/pipeline.py goldenmatch/__init__.py tests/test_api.py
git commit -m "feat: add dedupe_df() DataFrame entry point for SQL extensions"
```

---

### Task 2: Add `match_df()` to the API

**Files:**
- Modify: `goldenmatch/_api.py`
- Modify: `goldenmatch/core/pipeline.py`
- Modify: `goldenmatch/__init__.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`:

```python
class TestMatchDf:
    def test_match_df_exact(self):
        import goldenmatch as gm
        target = pl.DataFrame({
            "name": ["John Smith", "Jane Doe"],
            "email": ["john@x.com", "jane@y.com"],
        })
        reference = pl.DataFrame({
            "name": ["JOHN SMITH", "Bob Jones"],
            "email": ["john@x.com", "bob@z.com"],
        })
        result = gm.match_df(target, reference, exact=["email"])
        assert isinstance(result, gm.MatchResult)

    def test_match_df_fuzzy(self):
        import goldenmatch as gm
        target = pl.DataFrame({
            "name": ["John Smith"],
            "zip": ["10001"],
        })
        reference = pl.DataFrame({
            "name": ["Jon Smyth"],
            "zip": ["10001"],
        })
        result = gm.match_df(target, reference, fuzzy={"name": 0.75}, blocking=["zip"])
        assert isinstance(result, gm.MatchResult)

    def test_match_df_no_matches(self):
        import goldenmatch as gm
        target = pl.DataFrame({"email": ["a@x.com"]})
        reference = pl.DataFrame({"email": ["b@y.com"]})
        result = gm.match_df(target, reference, exact=["email"])
        assert isinstance(result, gm.MatchResult)
        # No matches expected
        if result.matched is not None:
            assert result.matched.height == 0 or result.unmatched is not None

    def test_match_df_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "match_df")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py::TestMatchDf -v`
Expected: FAIL with `AttributeError: module 'goldenmatch' has no attribute 'match_df'`

- [ ] **Step 3: Implement `match_df()` in `_api.py`**

Add after `dedupe_df()`:

```python
def match_df(
    target: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    config: Any | None = None,
    exact: list[str] | None = None,
    fuzzy: dict[str, float] | None = None,
    blocking: list[str] | None = None,
    threshold: float | None = None,
    backend: str | None = None,
) -> MatchResult:
    """Match a target DataFrame against a reference DataFrame (no file I/O).

    Same as match() but accepts DataFrames instead of file paths.

    Args:
        target: Polars DataFrame of target records.
        reference: Polars DataFrame of reference records.
        config: GoldenMatchConfig object, or None for auto-config from kwargs.
        exact: List of column names for exact matching.
        fuzzy: Dict of column name -> threshold for fuzzy matching.
        blocking: List of column names for blocking.
        threshold: Override fuzzy match threshold.
        backend: Processing backend: None, "ray".

    Returns:
        MatchResult with matched and unmatched DataFrames.
    """
    from goldenmatch.core.pipeline import run_match_df

    if isinstance(config, str):
        config = load_config(config)
    elif config is None:
        config = _build_config(exact, fuzzy, blocking, threshold, backend=backend)

    if backend and hasattr(config, "backend"):
        config.backend = backend

    result = run_match_df(target, reference, config)

    return MatchResult(
        matched=result.get("matched"),
        unmatched=result.get("unmatched"),
        stats=_extract_stats(result),
    )
```

- [ ] **Step 4: Implement `run_match_df()` in pipeline.py**

Add to `goldenmatch/core/pipeline.py`. Follow the same pattern as `run_dedupe_df()` -- inject DataFrames with `__source__` and `__row_id__` columns, then call the existing match pipeline logic:

```python
def run_match_df(
    target_df: pl.DataFrame,
    reference_df: pl.DataFrame,
    config: GoldenMatchConfig,
    target_name: str = "target",
    reference_name: str = "reference",
) -> dict:
    """Run match pipeline on DataFrames directly (no file I/O).

    Identical to run_match() but skips the file ingest step.
    """
    matchkeys = config.get_matchkeys()

    # Validate columns exist
    required = _get_required_columns(config)
    validate_columns(target_df.lazy(), required)
    validate_columns(reference_df.lazy(), required)

    # Prepare target
    target_lf = target_df.lazy()
    target_lf = target_lf.with_columns(pl.lit(target_name).alias("__source__"))
    target_lf = _add_row_ids(target_lf, offset=0)
    target_collected = target_lf.collect()
    target_ids = set(target_collected["__row_id__"].to_list())

    # Prepare reference
    ref_lf = reference_df.lazy()
    ref_lf = ref_lf.with_columns(pl.lit(reference_name).alias("__source__"))
    ref_lf = _add_row_ids(ref_lf, offset=len(target_collected))
    ref_collected = ref_lf.collect()

    # Combine and run match pipeline
    combined_lf = pl.concat([target_collected, ref_collected]).lazy()

    return _run_match_pipeline(
        combined_lf, config, matchkeys, target_ids,
    )
```

Also extract the shared match pipeline logic from `run_match()` into `_run_match_pipeline()`. The function signature must include all state the match pipeline needs:

```python
def _run_match_pipeline(
    combined_lf: pl.LazyFrame,
    config: GoldenMatchConfig,
    matchkeys: list,
    target_ids: set[int],
    output_matched: bool = False,
    output_unmatched: bool = False,
    output_scores: bool = False,
    output_report: bool = False,
    match_mode: str = "best",
) -> dict:
```

The body is everything from `run_match()` starting at "Step 2.5a: AUTO-FIX + VALIDATION" through the return statement. Key state that flows through:
- `target_ids` (set of int) -- needed for cross-source filtering and result grouping
- `source_lookup` (dict) -- built inside the pipeline from `combined_lf`
- `matched_pairs` (set) -- accumulated during scoring
- `quarantine_df_match` -- initialized to None, set if validation rules exist

The existing `run_match()` should build `target_ids` during ingest (as it already does) and pass it to `_run_match_pipeline()`.

- [ ] **Step 5: Export `match_df` from `__init__.py`**

Add `match_df` to the import from `_api` and to `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_api.py::TestMatchDf -v`
Expected: All 3 tests PASS

- [ ] **Step 7: Run full test suite**

Run: `pytest --tb=short`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add goldenmatch/_api.py goldenmatch/core/pipeline.py goldenmatch/__init__.py tests/test_api.py
git commit -m "feat: add match_df() DataFrame entry point for SQL extensions"
```

---

### Task 3: Add `score_strings()`, `score_pair_df()`, and `explain_pair_df()` to the API

These are lightweight utility functions for the SQL `goldenmatch_score()`, `goldenmatch_score_pair()`, and `goldenmatch_explain()` functions.

**Files:**
- Modify: `goldenmatch/_api.py`
- Modify: `goldenmatch/__init__.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`:

```python
class TestScoreStrings:
    def test_score_strings_jaro_winkler(self):
        import goldenmatch as gm
        score = gm.score_strings("John Smith", "Jon Smyth", "jaro_winkler")
        assert isinstance(score, float)
        assert 0.7 < score < 1.0

    def test_score_strings_exact(self):
        import goldenmatch as gm
        assert gm.score_strings("hello", "hello", "exact") == 1.0
        assert gm.score_strings("hello", "world", "exact") == 0.0

    def test_score_strings_levenshtein(self):
        import goldenmatch as gm
        score = gm.score_strings("kitten", "sitting", "levenshtein")
        assert isinstance(score, float)
        assert 0.0 < score < 1.0

    def test_score_strings_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "score_strings")


class TestScorePairDf:
    def test_score_pair_basic(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "John Smith", "email": "j@x.com"},
            {"name": "Jon Smyth", "email": "j@x.com"},
            fuzzy={"name": 0.85},
            exact=["email"],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # should be a decent match

    def test_score_pair_with_scorer(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "John Smith"},
            {"name": "Jon Smyth"},
            fuzzy={"name": 0.85},
        )
        assert isinstance(score, float)
        assert score > 0.7  # jaro_winkler default

    def test_score_pair_no_match(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "Alice"},
            {"name": "Zebra"},
            fuzzy={"name": 0.85},
        )
        assert score < 0.5


class TestExplainPairDf:
    def test_explain_basic(self):
        import goldenmatch as gm
        explanation = gm.explain_pair_df(
            {"name": "John Smith", "email": "j@x.com"},
            {"name": "Jon Smyth", "email": "j@x.com"},
            fuzzy={"name": 0.85},
            exact=["email"],
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "explain_pair_df")
        assert hasattr(gm, "score_pair_df")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py::TestScorePairDf tests/test_api.py::TestExplainPairDf -v`
Expected: FAIL

- [ ] **Step 3: Implement all three functions in `_api.py`**

```python
def score_strings(
    value_a: str,
    value_b: str,
    scorer: str = "jaro_winkler",
) -> float:
    """Score two strings using a named similarity scorer.

    Maps to the SQL function: SELECT goldenmatch_score('John', 'Jon', 'jaro_winkler');

    Args:
        value_a: First string.
        value_b: Second string.
        scorer: Scoring algorithm: "jaro_winkler", "levenshtein", "exact",
                "token_sort", "soundex_match".

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    from goldenmatch.core.scorer import score_field
    result = score_field(value_a, value_b, scorer)
    return result if result is not None else 0.0


def score_pair_df(
    record_a: dict,
    record_b: dict,
    *,
    fuzzy: dict[str, float] | None = None,
    exact: list[str] | None = None,
    scorer: str = "jaro_winkler",
) -> float:
    """Score a pair of records.

    Args:
        record_a: First record as dict.
        record_b: Second record as dict.
        fuzzy: Dict of field -> weight for fuzzy scoring.
        exact: List of fields for exact matching.
        scorer: Default scorer for fuzzy fields.

    Returns:
        Overall match score between 0.0 and 1.0.
    """
    from goldenmatch.core.scorer import score_pair, score_field
    from goldenmatch.config.schemas import MatchkeyField

    fields = []
    if exact:
        for col in exact:
            fields.append(MatchkeyField(field=col, scorer="exact", weight=1.0,
                                        transforms=["lowercase", "strip"]))
    if fuzzy:
        for col, weight in fuzzy.items():
            fields.append(MatchkeyField(field=col, scorer=scorer, weight=weight,
                                        transforms=["lowercase", "strip"]))

    if not fields:
        # Score all common fields with equal weight
        common = set(record_a.keys()) & set(record_b.keys())
        for col in sorted(common):
            fields.append(MatchkeyField(field=col, scorer=scorer, weight=1.0,
                                        transforms=["lowercase", "strip"]))

    return score_pair(record_a, record_b, fields)


def explain_pair_df(
    record_a: dict,
    record_b: dict,
    *,
    fuzzy: dict[str, float] | None = None,
    exact: list[str] | None = None,
    scorer: str = "jaro_winkler",
) -> str:
    """Generate a natural language explanation for a record pair.

    Args:
        record_a: First record as dict.
        record_b: Second record as dict.
        fuzzy: Dict of field -> weight for fuzzy scoring.
        exact: List of fields for exact matching.
        scorer: Default scorer for fuzzy fields.

    Returns:
        Human-readable explanation string.
    """
    from goldenmatch.core.scorer import score_field
    from goldenmatch.core.explain import explain_pair_nl
    from goldenmatch.config.schemas import MatchkeyField
    from goldenmatch.utils.transforms import apply_transforms

    fields = []
    if exact:
        for col in exact:
            fields.append(MatchkeyField(field=col, scorer="exact", weight=1.0,
                                        transforms=["lowercase", "strip"]))
    if fuzzy:
        for col, weight in fuzzy.items():
            fields.append(MatchkeyField(field=col, scorer=scorer, weight=weight,
                                        transforms=["lowercase", "strip"]))

    # Compute field-level scores
    field_scores = []
    weighted_sum = 0.0
    weight_sum = 0.0
    for f in fields:
        val_a = apply_transforms(record_a.get(f.field), f.transforms)
        val_b = apply_transforms(record_b.get(f.field), f.transforms)
        fs = score_field(val_a, val_b, f.scorer)
        if fs is not None:
            field_scores.append({
                "field": f.field,
                "scorer": f.scorer,
                "score": fs,
                "value_a": str(val_a) if val_a is not None else "",
                "value_b": str(val_b) if val_b is not None else "",
            })
            weighted_sum += fs * f.weight
            weight_sum += f.weight

    overall = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    return explain_pair_nl(record_a, record_b, field_scores, overall)
```

- [ ] **Step 4: Export all three functions from `__init__.py`**

Add `score_strings`, `score_pair_df`, and `explain_pair_df` to the imports from `_api` and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_api.py::TestScorePairDf tests/test_api.py::TestExplainPairDf -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest --tb=short`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add goldenmatch/_api.py goldenmatch/__init__.py tests/test_api.py
git commit -m "feat: add score_pair_df() and explain_pair_df() for SQL extensions"
```

---

### Task 4: Version bump and final validation

**Files:**
- Modify: `goldenmatch/__init__.py`
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version to 1.1.0**

In `goldenmatch/__init__.py` change:
```python
__version__ = "1.1.0"
```

In `pyproject.toml` change:
```toml
version = "1.1.0"
```

- [ ] **Step 2: Update CHANGELOG.md**

Add after the `## [Unreleased]` line:

```markdown
## [1.1.0] - 2026-03-23

### Added
- `gm.dedupe_df()` -- deduplicate a Polars DataFrame directly (no file I/O)
- `gm.match_df()` -- match two Polars DataFrames directly (no file I/O)
- `gm.score_strings()` -- score two strings with a named similarity algorithm
- `gm.score_pair_df()` -- score a pair of record dicts
- `gm.explain_pair_df()` -- explain a pair match from record dicts
- Internal: `run_dedupe_df()` and `run_match_df()` pipeline entry points
- These functions are the prerequisite for native SQL extensions (Postgres/DuckDB)
```

- [ ] **Step 3: Update `__all__` count in docs if referenced**

Check that the `__all__` list in `__init__.py` now includes all 4 new functions (total should be ~100 exports).

- [ ] **Step 4: Run full test suite one final time**

Run: `pytest --tb=short`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/__init__.py pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to v1.1.0 for DataFrame API entry points"
```
