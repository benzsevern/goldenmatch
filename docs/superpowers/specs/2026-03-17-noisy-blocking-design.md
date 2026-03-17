# Noisy Data Blocking — Design Specification

## Overview

Improve GoldenMatch's blocking recall on noisy/messy data by introducing multi-pass blocking with diverse key strategies. Instead of relying on a single blocking key that breaks when data has typos, reordering, or abbreviations, run multiple blocking passes with different key transforms and union the candidate pairs.

## Problem Statement

Single blocking keys are fragile on noisy data. The Leipzig DBLP-Scholar benchmark exposes this clearly:

- **title[:6] blocking**: 47.8% recall — prefixes diverge on typos and reordering
- **Fuzzy title blocking**: 77.7% recall — better, but still misses ~22% of true pairs
- **Overall F1**: ~50% — blocking is the bottleneck, not the scorer

Root causes:
1. **Typos and spelling variants** — "Smith" vs "Smyth", "recieved" vs "received"
2. **Token reordering** — "Machine Learning Approach" vs "Approach to Machine Learning"
3. **Abbreviations** — "J. Smith" vs "John Smith", "Intl." vs "International"
4. **Missing/extra tokens** — "Proc. of the 3rd Conf." vs "Proceedings of the Third Conference"

A single blocking key cannot handle all of these failure modes simultaneously.

## Goals

- Achieve >= 90% blocking recall on DBLP-Scholar (up from 47.8%)
- Provide configurable multi-pass blocking with union semantics
- Add phonetic, token-sort, and q-gram blocking transforms
- Add canopy clustering as an advanced option for large noisy datasets
- Maintain backwards compatibility — single-key configs work unchanged

## Non-Goals

- Full LSH / MinHash blocking — defer to a separate spec
- ML-learned blocking keys
- Optimizing scorer performance (this spec targets blocking only)
- Distributed/parallel execution

## Architecture

### New Files

```
goldenmatch/core/
├── blocker.py              # Modify: add multi_pass strategy, new transforms
├── transforms.py           # New: phonetic, token_sort, qgram transform functions
├── canopy.py               # New: TF-IDF canopy clustering
```

### Modified Files

```
goldenmatch/config/schemas.py    # Add multi_pass strategy, union_mode, canopy config
goldenmatch/core/pipeline.py     # Wire multi-pass blocking into dedupe/match
goldenmatch/tui/engine.py        # Same integration
```

## Multi-Pass Blocking

### How It Works

1. Define N blocking passes, each with its own key fields and transforms
2. Run each pass independently, producing a set of candidate pairs
3. Union all candidate pairs across passes — a pair only needs to appear in ANY pass
4. Deduplicate the unioned pairs before sending to the scorer

This trades comparison count for recall. Each additional pass adds candidates but catches pairs the other passes miss.

### Config

```yaml
blocking:
  strategy: multi_pass
  passes:
    - key_fields: [title]
      transforms: [lowercase, "substring:0:8"]
    - key_fields: [title]
      transforms: [lowercase, token_sort, "substring:0:10"]
    - key_fields: [authors]
      transforms: [lowercase, soundex]
    - key_fields: [title]
      transforms: [lowercase, "qgram:3"]
  union_mode: true  # union blocks across passes (default for multi_pass)
  max_total_comparisons: 5_000_000  # safety cap across all passes
```

### Implementation in blocker.py

```python
def build_blocks_multi_pass(
    lf: pl.LazyFrame,
    passes: list[BlockingKeyConfig],
    max_total_comparisons: int | None = None,
) -> list[BlockResult]:
    all_pairs: set[tuple[int, int]] = set()
    for pass_config in passes:
        blocks = build_blocks_single(lf, pass_config)
        all_pairs |= extract_pairs(blocks)
    if max_total_comparisons and len(all_pairs) > max_total_comparisons:
        logger.warning("Multi-pass produced %d pairs, capping at %d", len(all_pairs), max_total_comparisons)
        # Keep pairs from earlier (higher-priority) passes first
    return pairs_to_blocks(all_pairs)
```

## New Blocking Transforms

### Module: `goldenmatch/core/transforms.py`

| Transform | Key | Example Input | Example Output |
|-----------|-----|---------------|----------------|
| `soundex` | Phonetic code | "Smith" | "S530" |
| `metaphone` | Phonetic code (better for names) | "Smith" / "Smyth" | "SM0" / "SM0" |
| `token_sort` | Sort tokens alphabetically | "Machine Learning Approach" | "approach learning machine" |
| `qgram:N` | Character n-grams (sorted, joined) | "smith" (N=3) | "ith mit smi" |
| `first_token` | First word only | "John Smith" | "john" |
| `last_token` | Last word only | "John Smith" | "smith" |

All transforms are composable via the existing transform pipeline. Example: `[lowercase, token_sort, "substring:0:12"]` lowercases, sorts tokens, then takes the first 12 characters.

### Phonetic Transforms

Use the `jellyfish` library (already a lightweight dependency) for Soundex and Metaphone. These map spelling variants to the same phonetic code:

- Smith, Smyth, Smithe -> S530 (Soundex)
- Thompson, Thomson -> T525 (Soundex)

### Token-Sort Transform

```python
def token_sort(value: str) -> str:
    tokens = value.strip().split()
    return " ".join(sorted(tokens))
```

This normalizes word order so "Machine Learning Approach" and "Approach to Machine Learning" produce similar keys after substring truncation.

### Q-Gram Blocking

Extract character n-grams from the value, sort them, and use as a composite key. Two records sharing enough q-grams land in overlapping blocks.

```python
def qgram(value: str, q: int = 3) -> str:
    padded = f"##{value}##"
    grams = sorted(set(padded[i:i+q] for i in range(len(padded) - q + 1)))
    return " ".join(grams[:5])  # use top 5 sorted grams as key
```

## Canopy Clustering

### Module: `goldenmatch/core/canopy.py`

For datasets where no single key strategy achieves high recall, canopy clustering uses cheap TF-IDF similarity to form overlapping groups before expensive pairwise comparison.

1. Build a TF-IDF matrix over the blocking field(s) using scikit-learn
2. For each record (in random order), if not already in a canopy:
   - Find all records with cosine similarity > `loose_threshold` (e.g., 0.3) — these form the canopy
   - Records with similarity > `tight_threshold` (e.g., 0.7) are removed from future canopy centers
3. Each canopy becomes a block; records can appear in multiple canopies

### Config

```yaml
blocking:
  strategy: canopy
  canopy:
    fields: [title, authors]
    loose_threshold: 0.3
    tight_threshold: 0.7
    max_canopy_size: 500
```

### Performance

TF-IDF + approximate nearest neighbor (using sklearn's NearestNeighbors with cosine metric) keeps canopy construction at O(N log N) for moderate datasets. For 100K records this should complete in under 30 seconds.

## Schema Changes

Add to `BlockingConfig` in `schemas.py`:

```python
class BlockingConfig(BaseModel):
    keys: list[BlockingKeyConfig]
    max_block_size: int = 5000
    skip_oversized: bool = False
    strategy: Literal["static", "adaptive", "sorted_neighborhood", "multi_pass", "canopy"] = "static"
    # Multi-pass fields
    passes: list[BlockingKeyConfig] | None = None
    union_mode: bool = True
    max_total_comparisons: int | None = None
    # Canopy fields
    canopy: CanopyConfig | None = None
```

## Testing Strategy

### Unit Tests

- **Token-sort**: verify word-reordered strings produce identical keys
- **Soundex/Metaphone**: verify known spelling variants map to same code
- **Q-gram**: verify overlapping n-gram extraction and key generation
- **Multi-pass union**: verify pairs from any pass are included in final set
- **Multi-pass dedup**: verify duplicate pairs across passes are eliminated
- **Canopy**: verify overlapping canopy formation on small synthetic data

### Integration Tests

- **DBLP-Scholar benchmark**: run multi-pass blocking, verify recall >= 90%
- **Multi-pass vs single-pass**: verify multi-pass finds strictly more pairs
- **Backwards compatibility**: verify `strategy: static` produces identical results to current behavior

### Benchmarks

| Scenario | Target |
|----------|--------|
| DBLP-Scholar blocking recall (multi-pass) | >= 90% (up from 47.8%) |
| DBLP-Scholar end-to-end F1 (multi-pass) | >= 75% (up from 50%) |
| 100K records, 3-pass blocking | < 30 seconds |
| 1M records, 3-pass blocking | < 3 minutes |
| Canopy construction on 100K records | < 30 seconds |

## Rollout

1. Implement new transforms in `transforms.py` (phonetic, token_sort, qgram)
2. Add multi-pass orchestrator to `blocker.py`
3. Update schema with new strategy and config fields
4. Wire into pipeline and TUI
5. Implement canopy clustering
6. Benchmark on Leipzig datasets and tune defaults
