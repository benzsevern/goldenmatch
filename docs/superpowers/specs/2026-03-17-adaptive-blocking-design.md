# Adaptive Blocking — Design Specification

## Overview

Replace GoldenMatch's static blocking with a three-tier system: **auto-analysis** that suggests optimal blocking keys, **adaptive sub-blocking** that automatically splits oversized blocks, and **sorted neighborhood / canopy** as an advanced option. The goal is to handle millions of records in fuzzy matching without the user needing to understand blocking internals.

## Problem Statement

The current blocker uses static blocking keys configured by the user (e.g., `last_name[:3]`). Common name prefixes produce blocks of 1000-4000 records, leading to 120M+ pairwise comparisons for 100K records. This makes fuzzy matching at 1M+ scale impractical without expert blocking configuration.

## Goals

- Auto-suggest optimal blocking keys by analyzing data distribution
- Automatically handle oversized blocks via recursive sub-blocking
- Provide sorted neighborhood as an alternative for edge cases
- Maintain backwards compatibility with existing static blocking config
- Hit performance targets: 1M fuzzy in < 5 minutes with adaptive blocking

## Non-Goals

- LSH (locality-sensitive hashing) — useful but complex, defer to V2
- Learned blocking (ML-based) — out of scope
- Distributed/parallel blocking across machines

## Architecture

### New Files

```
goldenmatch/core/
├── blocker.py              # Modify: add adaptive and sorted_neighborhood strategies
├── block_analyzer.py       # New: auto-blocking analyzer
```

### Modified Files

```
goldenmatch/config/schemas.py    # Add strategy, auto_suggest, sub_block_keys, window_size
goldenmatch/core/pipeline.py     # Integrate analyzer + adaptive blocking
goldenmatch/tui/engine.py        # Same integration
goldenmatch/cli/main.py          # Add analyze-blocking command
goldenmatch/cli/dedupe.py        # Add --auto-block flag
goldenmatch/cli/match.py         # Add --auto-block flag
```

## Auto-Blocking Analyzer

### Module: `goldenmatch/core/block_analyzer.py`

`analyze_blocking(df: pl.DataFrame, matchkey_columns: list[str], sample_size: int = 5000) -> list[BlockingSuggestion]`

Returns a ranked list of blocking strategies with estimated performance.

### Step 1: Candidate Generation

For each column referenced in the matchkey fields, generate candidate blocking transforms. Column types are detected via **column name heuristics** (matching patterns like "name", "email", "zip", "phone", "state" in the column name, case-insensitive). All unrecognized columns are treated as "Generic string" — specialized transforms are only applied when the name clearly indicates the type.

| Column Type | Detection Pattern | Candidates |
|-------------|------------------|-----------|
| Name fields | `*name*`, `*fname*`, `*lname*` | `col[:3]`, `col[:4]`, `col[:5]`, `soundex(col)` |
| Zip/postal | `*zip*`, `*postal*` | `col[:3]`, `col[:5]`, `col` exact |
| State | `*state*` | `col` exact |
| Email | `*email*`, `*mail*` | `col.split('@')[1]` (domain), `col[:5]` |
| Phone | `*phone*`, `*tel*`, `*mobile*` | `col[:3]` (area code), `col[:6]` |
| Generic string | (everything else) | `col[:3]`, `col[:4]`, `col[:5]` |

Also generate compound candidates by combining two single-column keys (e.g., `last_name[:3] + state`).

### Step 2: Cardinality Scoring

For each candidate, compute on the full dataset:

- `group_count`: number of distinct blocking key values
- `max_group_size`: size of the largest block
- `mean_group_size`: average block size
- `std_group_size`: standard deviation of block sizes
- `total_comparisons`: sum of `n*(n-1)/2` across all blocks

Score formula:
```
score = (group_count / total_records) * (1 / (1 + max_group_size / target_block_size)) * (1 / (1 + std_group_size / mean_group_size))
```

Where `target_block_size` defaults to `max_block_size` from config (default 2000).

### Step 3: Coverage Check

For each candidate, verify it is derived from a matchkey field:
- If the matchkey uses `last_name + zip`, a blocking key on `email` would miss pairs where two records have different emails but same name+zip.
- Only candidates that are a coarser version of at least one matchkey field pass the coverage check.
- Compound blocking keys pass if each component covers at least one matchkey field.

Candidates that fail coverage are demoted (not eliminated — they may be useful as sub-blocking keys).

### Step 4: Pair Sampling Validation

1. Take a random sample of `sample_size` records (default 1000 — this keeps comparisons at ~500K which completes in seconds)
2. Run **simplified** unblocked fuzzy matching on the sample: use only the highest-weighted matchkey field with `rapidfuzz.process.cdist` for a single-field NxN score matrix. This is a lightweight proxy for the full multi-field evaluation — fast enough for recall estimation without running the full scorer.
3. For each pair above a loose threshold (0.7), check: would the proposed blocking key have placed both records in the same block?
4. Compute estimated recall: `pairs_in_same_block / total_proxy_pairs`

**Performance note**: 1000 records × 1000 records = 1M comparisons via cdist, which completes in < 1 second per candidate. The entire analyzer including all candidates and pair sampling should complete in < 30 seconds on 1M records.

### Output: BlockingSuggestion

```python
@dataclass
class BlockingSuggestion:
    keys: list[dict]              # blocking key config (key_fields + transforms)
    group_count: int              # number of blocks
    max_group_size: int           # largest block
    mean_group_size: float
    total_comparisons: int        # estimated pairwise comparisons
    estimated_recall: float       # from pair sampling (0.0 - 1.0)
    score: float                  # combined ranking score
    description: str              # human-readable: "last_name[:4] + state"
```

## Adaptive Sub-Blocking

### How It Works

When a primary block exceeds `max_block_size`:

1. Pick the first available sub-blocking key from `sub_block_keys` config
2. Within the oversized block, apply the sub-blocking key to split into smaller groups
3. If any sub-block still exceeds `max_block_size`, recurse with the next sub-blocking key
4. Max recursion depth: 3
5. If still oversized after 3 levels: fall back to sorted neighborhood within that block

### Auto Sub-Block Key Selection

If `sub_block_keys` is not configured, the analyzer picks sub-blocking keys automatically:
- Choose columns from the matchkey that are NOT already used in the primary blocking key
- Prefer high-cardinality columns (zip, first_name) over low-cardinality (state)

### Config

```yaml
blocking:
  max_block_size: 2000
  strategy: adaptive              # "adaptive" | "static" | "sorted_neighborhood"
  auto_suggest: true              # run analyzer before matching
  keys:
    - key_fields: [last_name]
      transforms: [lowercase, "substring:0:4"]
  sub_block_keys:                 # used when primary blocks are oversized
    - key_fields: [zip]
      transforms: ["substring:0:3"]
    - key_fields: [first_name]
      transforms: [lowercase, "substring:0:2"]
```

### Implementation in blocker.py

`build_blocks` signature stays the same — `(lf, config)`. Strategy and sub_block_keys are read from the `BlockingConfig` object (single source of truth, no parameter duplication):

```python
def build_blocks(
    lf: pl.LazyFrame,
    config: BlockingConfig,
) -> list[BlockResult]:
    # Reads config.strategy, config.sub_block_keys, config.window_size, config.sort_key
```

For `config.strategy="adaptive"`:
1. Build primary blocks as usual
2. For each block exceeding `max_block_size`:
   a. Apply first sub_block_key within the block
   b. If sub-blocks still oversized and more sub_block_keys available, recurse
   c. If max depth reached, switch to sorted_neighborhood for that block
3. Return all blocks (primary + sub-divided)

## Sorted Neighborhood

### How It Works

1. Compute a sort key from configured fields (e.g., `soundex(last_name) + zip[:3]`)
2. Sort all records by this key
3. Slide a window of size W through the sorted records
4. For each window position, the records in the window form a "block"
5. Adjacent windows overlap by W-1 records, ensuring no pair at a block boundary is missed

### Deduplication of Pairs

Since windows overlap, the same pair may appear in multiple windows. Use a set to deduplicate pairs before returning.

### Config

```yaml
blocking:
  strategy: sorted_neighborhood
  window_size: 20                # default: 20, max: 50
  sort_key:
    - column: last_name
      transforms: [lowercase, soundex]
    - column: zip
      transforms: ["substring:0:3"]
```

**Note**: Sort keys use per-field transforms (a list of column+transforms dicts), unlike blocking keys which apply transforms uniformly. This is necessary because sorted neighborhood sort keys need different transforms per field (e.g., soundex on names, substring on zips). The schema uses a new `SortKeyField` model to distinguish from `BlockingKeyConfig`.

### Performance

For N records with window size W:
- Comparisons: `N * W * (W-1) / 2` ≈ `N * W^2 / 2`
- At N=1M, W=20: ~200M comparisons (still a lot, but bounded and predictable)
- At N=1M, W=10: ~50M comparisons (more practical)

The window size should be tuned based on the dataset — the analyzer can suggest an appropriate size based on the sort key's ability to group similar records together.

## BlockResult Metadata

Extend the existing `BlockResult` dataclass with optional metadata for observability:

```python
@dataclass
class BlockResult:
    block_key: str
    df: pl.LazyFrame
    strategy: str = "static"          # "static" | "adaptive" | "sorted_neighborhood"
    depth: int = 0                    # sub-blocking recursion depth (0 = primary)
    parent_key: str | None = None     # parent block key if sub-blocked
```

This metadata is used for logging, debugging, and the reporting layer. Existing code that only accesses `block_key` and `df` is unaffected.

## Schema Changes

Add to `BlockingConfig` in `schemas.py`:

```python
class BlockingConfig(BaseModel):
    keys: list[BlockingKeyConfig]
    max_block_size: int = 5000                          # kept at 5000 for backwards compatibility
    skip_oversized: bool = False
    strategy: Literal["static", "adaptive", "sorted_neighborhood"] = "static"
    auto_suggest: bool = False
    sub_block_keys: list[BlockingKeyConfig] | None = None
    window_size: int = 20                               # for sorted_neighborhood
    sort_key: list[SortKeyField] | None = None           # for sorted_neighborhood (per-field transforms)
```

## Pipeline Integration

In `pipeline.py` `run_dedupe` and `run_match`, the blocking step becomes:

```
1. If config.blocking.auto_suggest:
   a. Run analyze_blocking on the data
   b. Log recommendations
   c. If no user-configured keys, use the top suggestion
2. Build blocks using config.blocking.strategy:
   a. "static": current behavior
   b. "adaptive": build_blocks with sub-blocking
   c. "sorted_neighborhood": sliding window
3. Score pairs within blocks (unchanged)
```

Same integration in `tui/engine.py`.

**For `run_match`**: The analyzer runs on the **concatenated** target + reference DataFrame. This is necessary because blocking key quality depends on cross-dataset distribution — a key that separates records well within one file may not work for cross-file matching. The pair sampling validation step specifically checks cross-source pair recall.

## CLI Integration

### New Command: analyze-blocking

```bash
goldenmatch analyze-blocking data.csv --config config.yaml
```

Runs the analyzer and prints a Rich table with ranked suggestions:
```
Blocking Strategy Analysis (sampled 5,000 of 1,000,000 records)

  #  Strategy                  Blocks    Max Size  Est. Comparisons  Recall
  1  last_name[:4] + state     12,450    850       2.1M              99.2%   ← recommended
  2  zip[:3]                   903       1,100     4.8M              97.8%
  3  soundex(last_name)        8,200     1,500     3.5M              99.5%
  4  last_name[:3]             243       4,300     120M              99.9%   ⚠ oversized
```

### --auto-block Flag

```bash
goldenmatch dedupe data.csv --config config.yaml --auto-block
goldenmatch match targets.csv --against refs.csv --config config.yaml --auto-block
```

Enables `auto_suggest` for this run, even if the config doesn't have it set.

## Performance Targets

| Scenario | Target |
|----------|--------|
| 1M records, exact matchkey | < 15s (already achieved) |
| 1M records, fuzzy name+zip, adaptive blocking | < 5 minutes |
| 5M records, fuzzy, adaptive blocking | < 30 minutes |
| Auto-analyzer on 1M records | < 30s |
| Sub-blocking recursion | < 1s overhead per oversized block |

## Error Handling

- **No viable blocking key found**: Analyzer warns "No blocking key achieves < max_block_size. Consider adding more fields to your matchkey." Falls back to best available option with a recall warning.
- **All blocks oversized after 3 sub-block levels**: Switch to sorted neighborhood for those blocks. Log a warning with the block key and size.
- **Sorted neighborhood on huge dataset**: Window size caps at 50. Warn if estimated time > 10 minutes.
- **Auto-suggest conflicts with user config**: User's explicit blocking config always wins. Auto-suggest only fills in gaps or logs suggestions as INFO.
- **Backwards compatibility**: `strategy: "static"` (the default) preserves current behavior exactly. Existing configs work unchanged.

## Testing Strategy

### Unit Tests
- Analyzer: cardinality scoring on synthetic data with known distributions
- Analyzer: coverage check — verify it rejects blocking keys not derived from matchkey fields
- Analyzer: pair sampling recall — verify with known duplicate pairs
- Sub-blocking: recursive splitting produces blocks under max_block_size
- Sub-blocking: max depth 3 enforced, falls back to sorted neighborhood
- Sorted neighborhood: window sliding, overlap correctness, pair deduplication
- Sorted neighborhood: pair extraction matches brute-force on small datasets

### Integration Tests
- Full pipeline with adaptive blocking on synthetic 100K data — verify recall >= 99%
- Adaptive vs. static on same data — compare results (adaptive should find same or more matches)
- analyze-blocking CLI command — verify Rich output format

### Benchmarks
- 1M records adaptive blocking — measure time and recall vs. static
- 5M records if feasible — verify < 30 minute target

## Future Considerations (Out of Scope)

- LSH (locality-sensitive hashing) — random projections for approximate blocking
- MinHash / band-based blocking — for set similarity
- Learned blocking — train a model to predict blocking keys
- Distributed blocking across multiple machines
