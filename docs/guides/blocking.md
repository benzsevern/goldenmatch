# Blocking Strategies

Blocking reduces the number of comparisons from O(N^2) to O(N * B) where B is the average block size.

## Static Blocking

User-defined blocking keys:

```yaml
blocking:
  keys:
    - fields: [zip]
    - fields: [city, state]
      transforms: [lowercase]
```

Only records sharing the same blocking key are compared.

## Auto-Suggest

Let GoldenMatch analyze your data and suggest optimal blocking keys:

```yaml
blocking:
  auto_suggest: true
```

Evaluates candidates by:
- Block count and size distribution
- Estimated recall (how many true matches fall in the same block)
- Comparison reduction ratio

## Learned Blocking

Data-driven predicate selection:

```yaml
blocking:
  strategy: learned
```

Two-pass approach:
1. Sample pairs and train a predicate model
2. Apply learned predicates to full dataset

Achieves 96.9% F1, matching hand-tuned static blocking.

## Choosing a Strategy

| Data Size | Strategy | Why |
|-----------|----------|-----|
| < 10K | None needed | N^2 is fast enough |
| 10K - 100K | Static or auto-suggest | Simple, predictable |
| 100K - 1M | Learned | Optimal without manual tuning |
| 1M+ | Learned + chunked/Ray | Must reduce comparisons aggressively |

## Performance Impact

Blocking key choice dominates fuzzy performance:

- **Too coarse** (e.g., block on `state`): Huge blocks, slow
- **Too fine** (e.g., block on `full_name`): Misses fuzzy matches
- **Sweet spot**: Block on 2-3 character prefix, zip code, or soundex

```
100K records, name+zip fuzzy matching:
  No blocking:  ~45 minutes (5B comparisons)
  zip blocking:  ~13 seconds (optimized)
```
