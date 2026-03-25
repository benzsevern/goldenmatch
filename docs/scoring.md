---
layout: default
title: Scoring
nav_order: 10
---

# Scoring

GoldenMatch provides 10+ scoring methods for comparing record pairs. Scoring runs after blocking and produces `(row_id_a, row_id_b, score)` tuples.

---

## Scorer reference

| Scorer | Description | Range | Best For |
|--------|-------------|-------|----------|
| `exact` | Binary 0/1 match | 0 or 1 | Email, phone, ID |
| `jaro_winkler` | Edit distance with prefix bonus | 0.0--1.0 | Names |
| `levenshtein` | Normalized Levenshtein distance | 0.0--1.0 | General strings |
| `token_sort` | Sort tokens, then ratio | 0.0--1.0 | Names, addresses |
| `soundex_match` | Phonetic code comparison | 0 or 1 | Names |
| `ensemble` | max(jaro_winkler, token_sort, soundex) | 0.0--1.0 | Names with reordering |
| `embedding` | Cosine similarity of sentence embeddings | 0.0--1.0 | Semantic matching |
| `record_embedding` | Multi-field concatenated embeddings | 0.0--1.0 | Cross-field semantic |
| `dice` | Dice coefficient on bloom filters | 0.0--1.0 | PPRL |
| `jaccard` | Jaccard similarity on bloom filters | 0.0--1.0 | PPRL |

---

## Fuzzy scoring

Fuzzy matching uses `rapidfuzz.process.cdist` for vectorized NxN scoring within each block. This is the core scoring engine for `weighted` matchkeys.

```python
import goldenmatch as gm

score = gm.score_strings("John Smith", "Jon Smyth", "jaro_winkler")
# 0.884
```

### Weighted matchkeys

Each field gets a scorer, weight, and optional transforms. The overall score is a weighted average:

```yaml
matchkeys:
  - name: fuzzy_person
    type: weighted
    threshold: 0.85
    fields:
      - field: first_name
        scorer: jaro_winkler
        weight: 0.4
        transforms: [lowercase, strip]
      - field: last_name
        scorer: jaro_winkler
        weight: 0.4
      - field: zip
        scorer: exact
        weight: 0.2
```

`overall_score = sum(field_score * weight) / sum(weight)`

Pairs with `overall_score >= threshold` are matched.

---

## Exact scoring

Exact matching uses Polars self-join for high performance. No threshold needed.

```yaml
matchkeys:
  - name: exact_email
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]
```

```python
pairs = gm.find_exact_matches(df, fields)
```

---

## Probabilistic scoring (Fellegi-Sunter)

EM-trained m/u probabilities with comparison vectors. Match weights are log-likelihood ratios.

```yaml
matchkeys:
  - name: fs_match
    type: probabilistic
    em_iterations: 20
    fields:
      - field: first_name
        scorer: jaro_winkler
        levels: 3              # agree / partial / disagree
        partial_threshold: 0.8
      - field: last_name
        scorer: jaro_winkler
        levels: 2              # agree / disagree
      - field: zip
        scorer: exact
        levels: 2
```

```python
import goldenmatch as gm

em_result = gm.train_em(df, matchkey, n_sample_pairs=10000, blocking_fields=["zip"])
pairs = gm.score_probabilistic(block_df, matchkey, em_result)
```

Key details:
- u-probabilities estimated from random pairs and fixed during EM (Splink approach)
- Blocking fields must be excluded from training (always agree within blocks)
- Comparison vectors apply field transforms before scoring
- Achieves 98.8% precision, 57.6% recall on DBLP-ACM

---

## LLM scoring

Send borderline pairs to GPT-4o-mini or Claude for scoring. Two modes:

### Pairwise mode

Score individual pairs. Best for small candidate sets.

```yaml
llm_scorer:
  enabled: true
  mode: pairwise
  auto_threshold: 0.95       # auto-accept above this
  candidate_lo: 0.75         # LLM scores pairs in [0.75, 0.95]
  budget:
    max_cost_usd: 0.05
```

### Cluster mode

Send entire borderline blocks to the LLM for in-context clustering. More efficient for large blocks.

```yaml
llm_scorer:
  enabled: true
  mode: cluster
  cluster_max_size: 100
  cluster_min_size: 5        # below this, fall back to pairwise
```

See [LLM Integration](llm) for full details.

---

## Cross-encoder reranking

Re-score borderline pairs with a pre-trained cross-encoder for higher precision.

```yaml
matchkeys:
  - name: fuzzy_name
    type: weighted
    threshold: 0.85
    rerank: true
    rerank_band: 0.1
    rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

Pairs within `threshold +/- rerank_band` get reranked. Requires `pip install goldenmatch[embeddings]`.

```python
reranked = gm.rerank_top_pairs(pairs, df, matchkey)
```

---

## Parallel scoring

Fuzzy blocks are scored concurrently via `ThreadPoolExecutor`. RapidFuzz's `cdist` releases the GIL, so threads provide real parallelism.

```
Block 1 ──> Thread 1 ──> pairs
Block 2 ──> Thread 2 ──> pairs    (concurrent)
Block 3 ──> Thread 3 ──> pairs
```

Implementation details:
- Blocks are independent -- frozen `exclude_pairs` snapshot avoids race conditions
- For 2 or fewer blocks, threading overhead is skipped (sequential execution)
- All call sites (pipeline, engine, chunked) use the shared `score_blocks_parallel` helper
- Ray backend (`score_blocks_ray`) distributes blocks across Ray tasks for cluster-level scaling

### Intra-field early termination

After scoring each expensive field, the scorer checks if the remaining fields can push any pair above the threshold. If not, it breaks early. This reduces 100K fuzzy matching from ~100s to ~39s (2.5x speedup).

---

## Embedding scoring

Requires `pip install goldenmatch[embeddings]`.

### Single-field embedding

```yaml
fields:
  - field: description
    scorer: embedding
    weight: 1.0
    model: all-MiniLM-L6-v2
```

### Record embedding (multi-field)

Concatenate multiple fields with optional per-field weights:

```yaml
fields:
  - columns: [title, authors, venue]
    scorer: record_embedding
    weight: 1.0
    column_weights: { title: 2.0, authors: 1.0, venue: 0.5 }
```

### Vertex AI embeddings

Use Google Cloud's managed embedding API (no GPU needed):

```python
# Set GOOGLE_APPLICATION_CREDENTIALS, then use embedding scorer
# Vertex AI text-embedding-004 supports inference only (no fine-tuning)
```

---

## Scoring a single pair

```python
import goldenmatch as gm

# Score two strings
score = gm.score_strings("John Smith", "Jon Smyth", "jaro_winkler")

# Score two records
score = gm.score_pair_df(
    {"name": "John Smith", "zip": "10001"},
    {"name": "Jon Smyth", "zip": "10001"},
    fuzzy={"name": 0.7, "zip": 0.3},
)

# Explain the score
explanation = gm.explain_pair_df(
    {"name": "John Smith", "zip": "10001"},
    {"name": "Jon Smyth", "zip": "10001"},
    fuzzy={"name": 0.7, "zip": 0.3},
)
```
