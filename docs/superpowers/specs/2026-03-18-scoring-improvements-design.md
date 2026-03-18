# Scoring Improvements — Design Specification

## Overview

Three targeted improvements to close the F1 gap between GoldenMatch and state-of-the-art entity resolution systems. Current best results: DBLP-ACM 97.2%, DBLP-Scholar 74.7%, Abt-Buy 43.1%, Amazon-Google 40.5%. SOTA achieves 89-98% on these datasets. These changes prioritize general-purpose improvement without adding significant latency.

## Problem Statement

Three root causes limit GoldenMatch's accuracy:

1. **No cross-field context** — the scorer embeds each field independently and averages. A product with `title: "Sony PSLX350H"` and `manufacturer: "Sony"` loses the cross-field signal that "Sony" appearing in both fields strengthens the match. SOTA systems embed the entire record as one string.

2. **ANN blocking creates mega-blocks** — Union-Find transitive closure merges ANN pairs into connected components. A chain A-B, B-C, C-D becomes one block {A,B,C,D} with 6 pairs scored, but only 3 were semantically close. This tanks precision (10-33% in benchmarks).

3. **Manual threshold guessing** — users set thresholds blindly. Benchmarks showed Abt-Buy embedding F1 ranging from 18.9% (0.65) to 43.1% (0.85) depending on threshold choice. The optimal value depends on the score distribution, which varies per dataset and scorer type.

## Goals

- Add a record-level embedding scorer that captures cross-field context
- Fix ANN blocking precision by eliminating Union-Find mega-blocks
- Auto-detect optimal thresholds from score distributions
- Target improvements across all datasets, not just e-commerce
- Keep latency under 30s for 100K records on CPU

## Non-Goals

- Fine-tuning models on user data (V2)
- GPU as a requirement
- Replacing field-level scoring — record-level augments, not replaces
- Match explainability beyond scores

---

## Feature 1: Record-Level Embedding Scorer

### Design

New scorer type `record_embedding` that concatenates configured columns into a single text string per record, embeds the full string, and computes cosine similarity between record pairs.

Concatenation format uses labeled delimiters for context:
```
"title: Sony PSLX350H Turntable | manufacturer: Sony | price: 149.99"
```

### Configuration

```yaml
matchkeys:
  - name: product_match
    type: weighted
    threshold: 0.80
    fields:
      - scorer: record_embedding
        columns: [title, manufacturer, price]
        model: all-MiniLM-L6-v2
        weight: 1.0
```

Hybrid with field-level scorers:
```yaml
fields:
  - scorer: record_embedding
    columns: [title, description]
    weight: 0.7
  - field: manufacturer
    scorer: exact
    weight: 0.3
```

### Schema Changes

In `schemas.py`:
- Add `"record_embedding"` to `VALID_SCORERS`
- Add `columns: list[str] | None = None` field to `MatchkeyField`
- Update `_resolve_field_column` validator: when `scorer == "record_embedding"` and `columns` is set, skip the `field`/`column` requirement. Set `field` to `"__record__"` as a sentinel so downstream code that reads `f.field` doesn't break.
- Validation: `record_embedding` requires `columns` to be non-empty; other scorers require `field` or `column`
- Note: `columns` (plural, list) is distinct from `column` (singular, alias for `field`). The naming is intentional.

### Pipeline Compatibility

In `pipeline.py`:
- Update `_get_required_columns()` (lines 93-103) to also collect from `f.columns` when present, so record_embedding columns are included in the required columns set.
- Column ordering in the `columns` list matters — it determines concatenation order. Both sides of the match must use the same column order.

### Scorer Changes

In `scorer.py`:
- New `_record_embedding_score_matrix(block_df, columns, model_name)` function:
  1. For each row, concatenate `"{col}: {value}"` joined by `" | "` for all columns, skipping null values
  2. Call `embedder.embed_column(concat_values, cache_key)`
  3. Return `embedder.cosine_similarity_matrix(embeddings)`
  4. Cache key based on block row IDs for cross-block stability: `f"_rec_emb_{hash(tuple(columns))}_{hash(tuple(block_df['__row_id__'].to_list()))}"`
- Update `find_fuzzy_matches()` to detect `record_embedding` fields and route to the new function
- Handle nulls: skip null values in concatenation (e.g., `"title: Sony | price: 149.99"` if manufacturer is null)

### Expected Impact

+15-25pts F1 on e-commerce datasets. Cross-field context is the single biggest gap to SOTA. Published results with MiniLM on Abt-Buy and Amazon-Google show 55-65% F1 with record-level embeddings alone (no fine-tuning).

---

## Feature 2: Direct-Pair ANN Scoring

### Design

New blocking strategy `ann_pairs` that returns ANN candidate pairs directly without Union-Find grouping. FAISS already computes cosine similarity during the query — we currently discard these scores. Keep them and use as match scores directly.

### Flow

Current (broken):
```
ANN query → pairs with scores → Union-Find → mega-blocks → NxN re-scoring → pairs
```

New:
```
ANN query → pairs with scores → threshold filter → matched pairs
```

### Configuration

```yaml
blocking:
  strategy: ann_pairs
  ann_column: title
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20
```

When combined with record-level embedding:
```yaml
blocking:
  strategy: ann_pairs
  ann_column: title           # column used for ANN neighbor search
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20

matchkeys:
  - name: product_match
    type: weighted
    threshold: 0.80
    fields:
      - scorer: record_embedding
        columns: [title, manufacturer]
        model: all-MiniLM-L6-v2
        weight: 1.0
```

### Implementation

In `ann_blocker.py`:
- Add new method `query_with_scores()` that returns `list[tuple[int, int, float]]` (preserving backward compat with existing `query()` → `list[tuple[int, int]]`)
- FAISS `search()` already returns `(scores, indices)` — propagate the scores
- Update `_build_ann_blocks()` in blocker.py to unpack the unchanged `query()` return type (no breaking change)

In `blocker.py`:
- New `_build_ann_pair_blocks()` function
- Calls `query_with_scores()` to get pairs with cosine similarity scores
- Add `pre_scored_pairs: list[tuple[int, int, float]] | None = None` field to `BlockResult` dataclass
- Return a single `BlockResult` with `pre_scored_pairs` set and `df` containing the full DataFrame (needed for downstream clustering)
- No Union-Find, no transitive closure
- Set `strategy="ann_pairs"` on all results

In `scorer.py` (`find_fuzzy_matches`):
- At the top, check if the block has `pre_scored_pairs` set. If so, filter by threshold and return directly — skip NxN scoring entirely. This is the primary path for `ann_pairs`.
- Fallback: when `ann_pairs` is combined with a `record_embedding` scorer that uses different columns than `ann_column`, the micro-block approach kicks in — score only the candidate pairs using the record-level embedder, not NxN over the whole block.

### Pipeline Integration

The `ann_pairs` strategy produces pairs that flow into `build_clusters()` as normal. Since there is no transitive closure in the blocking step, clusters will be small (typically 2-3 records). This is correct behavior — clustering on pre-filtered pairs produces clean results.

In `schemas.py`:
- Add `"ann_pairs"` to the strategy literal

### Expected Impact

+10-15pts precision on embedding strategies with minimal recall loss. On Abt-Buy embedding+ANN(0.80), precision should improve from 32.7% to ~50%+ because we stop scoring unrelated records that happened to be transitively connected.

---

## Feature 3: Threshold Auto-Tuning

### Design

Analyze the score distribution from scored blocks and suggest an optimal threshold using Otsu's method — find the value that best separates the "match" and "non-match" score populations.

### Implementation

New file `goldenmatch/core/threshold.py`:

```python
def suggest_threshold(scores: list[float], n_bins: int = 100) -> float:
    """Find optimal threshold using Otsu's method on score distribution."""
```

Algorithm:
1. Histogram all pairwise scores into `n_bins` bins over [0, 1]
2. For each candidate threshold (bin edge):
   - Compute weight and variance of scores below (non-match class)
   - Compute weight and variance of scores above (match class)
   - Compute weighted intra-class variance
3. Return the threshold that minimizes intra-class variance
4. Fallback: if distribution is unimodal (variance ratio < 2.0), return `None` — the caller uses the user-provided `threshold` value as fallback. Do not guess with percentiles.

### Configuration

```yaml
matchkeys:
  - name: fuzzy_name
    type: weighted
    auto_threshold: true    # overrides threshold with auto-detected value
    threshold: 0.80         # used as fallback if auto-detection fails
    fields:
      - field: name
        scorer: embedding
        weight: 1.0
```

### Schema Changes

In `schemas.py`:
- Add `auto_threshold: bool = False` to `MatchkeyConfig`

### Pipeline Integration

In `pipeline.py` (and `engine.py`):
- Sample up to 10,000 scored pairs from randomly selected blocks (not the first N, to avoid order bias). Minimum 100 pairs for Otsu to be reliable.
- Call `suggest_threshold(scores)` to get the suggested value
- If `auto_threshold` is enabled and `suggest_threshold` returns a value (not `None`), override `mk.threshold`
- If it returns `None` (unimodal distribution), keep the user-provided `threshold` as fallback
- Log: `"Auto-threshold: 0.82 (from 10,000 scored pairs)"` or `"Auto-threshold: skipped (unimodal distribution), using configured 0.80"`

### Expected Impact

+3-8pts F1 across all datasets by removing human guesswork. Especially valuable for embedding scorers where cosine similarity distributions differ from string similarity distributions.

---

## Rollout Plan

1. **Phase 1**: Record-level embedding scorer (`record_embedding`)
   - Schema changes (MatchkeyField.columns, VALID_SCORERS)
   - `_record_embedding_score_matrix()` in scorer.py
   - Tests with mocked embedder
   - Benchmark on all 4 Leipzig datasets

2. **Phase 2**: Direct-pair ANN scoring (`ann_pairs`)
   - `ann_blocker.py` returns scores
   - `_build_ann_pair_blocks()` in blocker.py
   - Schema update (strategy literal)
   - Tests for micro-block generation
   - Re-benchmark Abt-Buy and Amazon-Google

3. **Phase 3**: Threshold auto-tuning
   - `threshold.py` with Otsu's method
   - Schema update (auto_threshold)
   - Pipeline/engine integration
   - Tests with synthetic score distributions
   - Re-benchmark all datasets with auto_threshold

4. **Phase 4**: Full benchmark run
   - Run all Leipzig datasets with best config per dataset
   - Update README with new results
   - Document recommended configs per use case

## Testing Strategy

### Unit Tests

- `test_record_embedding_scorer.py` — concatenation format, null handling, NxN matrix shape, hybrid scoring, column ordering consistency
- `test_ann_pairs.py` — pre_scored_pairs propagation, no Union-Find, duplicate FAISS neighbors, top_k > dataset size, all scores below threshold (zero matches), backward compat of existing `query()` method
- `test_threshold.py` — Otsu's method on bimodal distribution, unimodal fallback returns None, edge cases (empty list, single score, all identical scores), string vs embedding score distributions

### Integration Tests

- Record-level scorer through `find_fuzzy_matches()` pipeline
- `ann_pairs` strategy through `build_blocks()` → scorer flow
- Auto-threshold through pipeline with real blocks

### Benchmark Validation

- Target: Abt-Buy > 55% F1, Amazon-Google > 50% F1
- DBLP-ACM should remain > 97% F1
- DBLP-Scholar should remain > 74% F1
- All benchmarks under 30s per dataset (excluding model load)

## Dependencies

No new dependencies. All features use existing packages:
- `sentence-transformers` (already installed) for embeddings
- `faiss-cpu` (already installed) for ANN
- `numpy` (already installed) for Otsu's method
- `scikit-learn` (already installed, used by canopy) — not needed here
