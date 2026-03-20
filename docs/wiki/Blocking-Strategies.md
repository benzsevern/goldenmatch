# Blocking Strategies

Blocking reduces the comparison space from O(n²) to something tractable. Instead of comparing every pair of records, blocking groups records by shared attributes and only compares within groups.

## Available Strategies

### Static (Default)

Groups records by the exact value of the blocking key.

```yaml
blocking:
  strategy: static
  keys:
    - fields: [zip]
      transforms: [strip]
```

Best for: clean data with reliable exact-match fields.

### Adaptive

Static blocking with recursive sub-blocking for oversized groups. When `sub_block_keys` are configured, uses them for recursive splitting (up to depth 3). When not configured, **auto-splits by the highest-cardinality column** in the block -- zero config needed.

```yaml
blocking:
  strategy: adaptive
  keys:
    - fields: [zip]
  sub_block_keys:                    # optional -- auto-splits if omitted
    - fields: [last_name]
      transforms: [soundex]
  max_block_size: 500
```

Best for: data with skewed distributions (e.g., common zip codes).

### Sorted Neighborhood

Sliding window over records sorted by a key.

```yaml
blocking:
  strategy: sorted_neighborhood
  sort_key:
    - column: name
      transforms: [lowercase, strip]
  window_size: 20
```

Best for: data where similar records sort near each other.

### Multi-Pass

Union of blocks from multiple passes with different keys. Records found by any pass are candidates.

```yaml
blocking:
  strategy: multi_pass
  passes:
    - fields: [name]
      transforms: [lowercase, "substring:0:5"]
    - fields: [name]
      transforms: [lowercase, soundex]
    - fields: [name]
      transforms: [lowercase, token_sort, "substring:0:8"]
```

Best for: noisy data where no single key catches all matches. Our best results on Leipzig benchmarks use multi-pass.

### ANN

Approximate nearest neighbor blocking using FAISS on sentence-transformer embeddings. Groups semantically similar records.

```yaml
blocking:
  strategy: ann
  ann_column: title
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20
```

Best for: semantic matching (product names, descriptions).

### ANN Pairs

Like ANN but returns FAISS pairs directly without Union-Find transitive closure. 50-100x faster with equal or better precision.

```yaml
blocking:
  strategy: ann_pairs
  ann_column: title
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20
```

Best for: semantic matching where speed matters.

### Canopy

TF-IDF canopy clustering with cosine similarity thresholds.

```yaml
blocking:
  strategy: canopy
  canopy:
    fields: [description]
    loose_threshold: 0.3
    tight_threshold: 0.7
```

Best for: text-heavy data without embedding dependencies.

## Auto-select Best Key

When multiple blocking keys are configured, `auto_select: true` runs histogram analysis on each key and picks the one with the smallest maximum block size (while maintaining >= 50% record coverage):

```yaml
blocking:
  strategy: static
  auto_select: true
  keys:
    - fields: [zip]
    - fields: [last_name]
      transforms: [soundex]
    - fields: [zip, last_name]
      transforms: [strip, soundex]
```

This eliminates guesswork -- GoldenMatch evaluates each key's group-size distribution and picks the one that minimizes worst-case block size.

## Choosing a Strategy

| Data Type | Recommended Strategy |
|-----------|---------------------|
| Clean with exact fields | `static` |
| Clean with skewed distribution | `adaptive` |
| Noisy names/addresses | `multi_pass` |
| Product catalogs | `ann_pairs` + embeddings |
| Large text fields | `canopy` |
| Database sync | Hybrid (SQL + ANN, automatic) |
