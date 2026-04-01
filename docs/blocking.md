---
layout: default
title: Blocking Strategies
nav_order: 9
---

# Blocking Strategies

Blocking reduces the comparison space from O(N^2) to O(N*B) by grouping records that share a key. GoldenMatch supports 8 strategies.

---

## Strategy overview

| Strategy | Description | Best For |
|----------|-------------|----------|
| `static` | Group by blocking key | Clean data with reliable keys |
| `adaptive` | Static + recursive sub-blocking for oversized blocks | Default choice |
| `sorted_neighborhood` | Sliding window over sorted records | Typos in blocking key |
| `multi_pass` | Union of blocks from multiple passes | Noisy data, best recall |
| `ann` | FAISS nearest-neighbor on embeddings | Semantic matching |
| `ann_pairs` | Direct-pair ANN scoring | 50--100x faster than `ann` |
| `canopy` | TF-IDF canopy clustering | Text-heavy data |
| `learned` | Data-driven predicate selection | Auto-discovers rules |

---

## Static blocking

Group records by exact value of the blocking key.

```yaml
blocking:
  strategy: static
  keys:
    - fields: [zip]
    - fields: [last_name]
      transforms: [lowercase, soundex]
```

Multiple keys produce independent blocks that are unioned. Transforms are applied before grouping.

---

## Adaptive blocking

Static blocking with automatic sub-splitting for oversized blocks. When a block exceeds `max_block_size`, it splits on the highest-cardinality column within the block.

```yaml
blocking:
  strategy: adaptive
  max_block_size: 5000
  keys:
    - fields: [zip]
```

---

## Sorted neighborhood

Sliding window over records sorted by a key. Catches near-matches that differ by one character in the blocking key.

```yaml
blocking:
  strategy: sorted_neighborhood
  window_size: 20
  sort_key:
    - column: last_name
      transforms: [lowercase, soundex]
```

---

## Multi-pass blocking

Run multiple blocking passes and union the results. Best recall for noisy data.

```yaml
blocking:
  strategy: multi_pass
  union_mode: true
  passes:
    - fields: [zip]
    - fields: [last_name]
      transforms: [lowercase, soundex]
    - fields: [first_name]
      transforms: [lowercase, first_token]
```

---

## ANN hybrid blocking

*New in v1.2.6.* Combine multi-pass string blocking with ANN fallback for oversized blocks. When a block exceeds `max_block_size` and would normally be skipped, GoldenMatch embeds only the unique text values in that block and uses FAISS to create smaller sub-blocks.

```yaml
blocking:
  strategy: multi_pass
  passes:
    - fields: [model_desc, state]
      transforms: [lowercase, strip]
    - fields: [base_model]
      transforms: [lowercase, soundex]
  max_block_size: 1000
  skip_oversized: true
  ann_column: description_text     # enables ANN fallback
  ann_top_k: 20
```

How it works:
1. Multi-pass blocking creates string-based blocks (fast, handles most data)
2. Blocks exceeding `max_block_size` trigger ANN fallback instead of being skipped
3. ANN embeds only **unique text values** (e.g., 61K records with 187 unique texts = seconds)
4. FAISS finds nearest neighbors among unique texts, Union-Find creates sub-blocks
5. Sub-blocks still exceeding `max_block_size` (after 10x cap) are skipped

On the Bulldozer dataset (401K rows), this recovered 363 sub-blocks from 15 oversized blocks that would otherwise be skipped, matching 949 additional records.

Requires Vertex AI (`GOLDENMATCH_GPU_MODE=vertex`) or local sentence-transformers for embedding.

---

## ANN blocking

Use FAISS approximate nearest-neighbor search on sentence-transformer embeddings. Requires `pip install goldenmatch[embeddings]`.

```yaml
blocking:
  strategy: ann
  ann_column: description
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20
```

`ann_pairs` is a faster variant (50--100x) that returns direct pairs instead of block groups:

```yaml
blocking:
  strategy: ann_pairs
  ann_column: title
  ann_top_k: 20
```

---

## Canopy blocking

TF-IDF-based canopy clustering with loose and tight thresholds.

```yaml
blocking:
  strategy: canopy
  canopy:
    fields: [name, address]
    loose_threshold: 0.3
    tight_threshold: 0.7
    max_canopy_size: 500
```

---

## Learned blocking

Data-driven predicate selection via a two-pass approach: sample pairs, train predicates, apply to full data. Achieves 96.9% F1 matching hand-tuned static blocking on DBLP-ACM.

```yaml
blocking:
  strategy: learned
  learned_sample_size: 5000
  learned_min_recall: 0.95
  learned_min_reduction: 0.90
  learned_predicate_depth: 2
  learned_cache_path: .goldenmatch/learned_blocking.pkl
```

```python
import goldenmatch as gm

rules = gm.learn_blocking_rules(df, matchkey, sample_size=5000)
blocks = gm.apply_learned_blocks(df, rules)
```

Cache the learned rules to skip re-training on subsequent runs.

---

## Auto-select

Let GoldenMatch pick the best blocking key by histogram analysis:

```yaml
blocking:
  auto_select: true
  keys:
    - fields: [zip]
    - fields: [last_name]
      transforms: [lowercase, soundex]
    - fields: [city]
```

The analyzer scores each key by block count, max block size, estimated comparisons, and recall. Use the CLI to see suggestions:

```bash
goldenmatch analyze-blocking customers.csv --config config.yaml
```

---

## Performance impact

Blocking key choice dominates fuzzy matching performance. A coarse key (e.g., `state`) creates huge blocks and slow scoring. A fine key (e.g., `email`) misses near-duplicates.

| Key | Records | Blocks | Max Size | Comparisons | Time |
|-----|---------|--------|----------|-------------|------|
| `zip` | 100K | 8,200 | 340 | 1.2M | 12s |
| `state` | 100K | 50 | 12,000 | 45M | 320s |
| `last_name + soundex` | 100K | 4,100 | 180 | 0.8M | 9s |
| `learned` | 100K | 3,800 | 200 | 0.9M | 10s |

Rules of thumb:
- Target max block size under 1,000 records
- Use `multi_pass` for best recall, `adaptive` for best speed
- Use `learned` to auto-discover optimal predicates
- Use `ann_pairs` for semantic/product matching
