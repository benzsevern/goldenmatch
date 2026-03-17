# Embedding-Based Scorer — Design Specification

## Overview

Add transformer-embedding-based matching to GoldenMatch so it can solve semantic entity resolution problems where rule-based string similarity fundamentally fails. The Leipzig benchmark showed Abt-Buy (37% F1) and Amazon-Google (29% F1) are unsolvable with lexical methods alone — the same product has completely different names across sources. State-of-the-art ER systems (Ditto, Sudowoodo) use transformer embeddings and achieve 70-90% F1 on these datasets. GoldenMatch needs the same capability.

## Problem Statement

E-commerce product matching illustrates the core issue:

- **Source A**: `"Sony PSLX350H Turntable"`
- **Source B**: `"Sony Turntable - PSLX350H"`

String similarity scorers (Jaro-Winkler, token sort ratio, Levenshtein) treat these as moderately similar at best. The relationship is semantic — a human recognizes these as the same product because they understand that word order and punctuation are irrelevant for product identity. Traditional blocking also fails: there is no reliable shared token prefix when product names diverge significantly.

Transformer embeddings solve this by mapping text into a dense vector space where semantically equivalent strings land close together, regardless of surface form.

## Goals

- Add an `embedding` scorer type that computes cosine similarity between sentence embeddings
- Support configurable HuggingFace sentence-transformer models
- Cache embeddings so they are computed once per column, not recomputed on threshold changes
- Provide ANN (approximate nearest neighbor) blocking as an alternative to traditional blocking
- Enable hybrid scoring: combine embedding similarity with rule-based similarity in weighted formulas
- Target 60%+ F1 on Abt-Buy and Amazon-Google benchmarks out of the box

## Non-Goals

- Fine-tuning models on user data (defer to V2)
- GPU support as a requirement (CPU is the baseline; GPU is a bonus if `sentence-transformers` detects it)
- Training custom ER models — use existing pre-trained sentence transformers only
- Replacing rule-based matching — embeddings augment, not replace, existing scorers

## Architecture

### New Files

```
goldenmatch/core/
├── embedder.py           # New: model loading, embedding computation, caching
├── ann_blocker.py        # New: ANN index building + querying (FAISS or Annoy)
```

### Modified Files

```
goldenmatch/core/scorer.py       # Add 'embedding' scorer type dispatching to embedder
goldenmatch/core/blocker.py      # Add 'ann' blocking strategy dispatching to ann_blocker
goldenmatch/config/schemas.py    # Add model, ann_column, ann_top_k, strategy: ann fields
goldenmatch/core/pipeline.py     # Wire up embedding cache lifecycle
setup.cfg / pyproject.toml       # Add optional [embeddings] dependency group
```

### `embedder.py`

Responsibilities:

1. **Model loading** — lazy-load a `sentence-transformers` model by name. Cache the loaded model in a module-level dict so it is loaded once per process.
2. **Batch embedding** — given a pandas Series of strings, compute embeddings in batches (default batch size 256). Return a numpy array of shape `(n, dim)`.
3. **Embedding cache** — store computed embeddings keyed by `(column_name, model_name)`. On first call, compute and cache. On subsequent calls (e.g., threshold tuning), return cached result. Optionally persist to disk as `.npy` files for cross-run caching.
4. **Cosine similarity** — given two embedding vectors, return cosine similarity as a float in [0, 1]. For batch pairs, use vectorized numpy dot product.

```python
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None  # lazy loaded
        self.model_name = model_name
        self._cache: dict[str, np.ndarray] = {}

    def embed_column(self, series: pd.Series, column_name: str) -> np.ndarray:
        """Embed all values in a column. Returns (n, dim) array. Cached."""

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""

    def save_cache(self, path: Path) -> None:
        """Persist embedding cache to disk as .npy files."""

    def load_cache(self, path: Path) -> None:
        """Load embedding cache from disk."""
```

### `ann_blocker.py`

Responsibilities:

1. **Index building** — given an embedding array, build a FAISS `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity) or an Annoy index.
2. **Querying** — for each record in the query set, find the top-K nearest neighbors. Return candidate pairs as a list of `(idx_a, idx_b)` tuples.
3. **Integration** — expose a `block()` method compatible with the existing blocker interface so it can be used as a drop-in blocking strategy.

FAISS is preferred (faster, more mature). Annoy is the fallback if FAISS installation is problematic on the target platform.

```python
class ANNBlocker:
    def __init__(self, top_k: int = 20, backend: str = "faiss"):
        self.top_k = top_k
        self.backend = backend

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build ANN index from embeddings."""

    def query(self, query_embeddings: np.ndarray) -> list[tuple[int, int]]:
        """Return candidate pairs: (query_idx, neighbor_idx)."""

    def block(self, df_left, df_right, embedder, column, model) -> list[tuple[int, int]]:
        """Full blocking pipeline: embed -> index -> query -> return pairs."""
```

### Scorer Integration

In `scorer.py`, add an `embedding` case:

```python
if scorer_type == "embedding":
    model_name = field_config.get("model", "all-MiniLM-L6-v2")
    embedder = get_or_create_embedder(model_name)
    vec_a = embedder.embed_column(left_series, col_name)
    vec_b = embedder.embed_column(right_series, col_name)
    scores = np.array([
        embedder.cosine_similarity(vec_a[i], vec_b[j])
        for i, j in candidate_pairs
    ])
```

For batch scoring, vectorize with `(vec_a * vec_b).sum(axis=1)` on pre-normalized vectors.

## Configuration

### Embedding Scorer with Hybrid Matching

```yaml
matchkeys:
  - name: semantic_product
    fields:
      - column: title
        scorer: embedding
        model: all-MiniLM-L6-v2
        weight: 0.6
      - column: manufacturer
        scorer: exact
        weight: 0.4
    comparison: weighted
    threshold: 0.75
```

### ANN Blocking

```yaml
blocking:
  strategy: ann
  ann_column: title
  ann_model: all-MiniLM-L6-v2
  ann_top_k: 20
```

### Model Options

| Model | Size | Speed (CPU) | Quality | Use Case |
|-------|------|-------------|---------|----------|
| `all-MiniLM-L6-v2` | 80 MB | ~1000 rec/s | Good | Default, fast iteration |
| `all-mpnet-base-v2` | 420 MB | ~300 rec/s | Better | Production, accuracy-critical |
| Any HuggingFace ST model | Varies | Varies | Varies | Via config override |

## Dependencies

New optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
embeddings = [
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.0",
    "numpy>=1.24.0",
]
```

Install via: `pip install goldenmatch[embeddings]`

If the `embeddings` extras are not installed and the user configures an `embedding` scorer or `ann` blocking, raise a clear error:

```
EmbeddingDependencyError: Embedding features require additional packages.
Install with: pip install goldenmatch[embeddings]
```

## Performance Considerations

| Operation | 10K records | 100K records | 1M records |
|-----------|-------------|--------------|------------|
| Embedding (MiniLM, CPU) | ~10 sec | ~100 sec | ~17 min |
| Embedding (mpnet, CPU) | ~33 sec | ~330 sec | ~55 min |
| ANN index build (FAISS) | < 1 sec | ~2 sec | ~10 sec |
| ANN query (top-20) | < 1 sec | ~5 sec | ~60 sec |
| Total pairs generated | ~200K | ~2M | ~20M |

Key points:

- Embedding computation is the bottleneck but is done **once** and cached
- ANN querying is O(n * log(n)) vs O(n^2) for pairwise — this is the critical scaling win
- For 1M records, ANN blocking produces ~20M candidate pairs (20 per record) vs 500B for pairwise
- Threshold tuning is instant because embeddings are cached — only the comparison step re-runs

## Testing Strategy

### Unit Tests

- `test_embedder.py` — model loading, embedding shape/dtype, cosine similarity correctness, cache hit/miss
- `test_ann_blocker.py` — index build, query returns correct top-K, known-pair recall
- `test_embedding_scorer.py` — integration with scorer dispatch, weighted hybrid scoring

### Benchmark Tests

- Run Abt-Buy and Amazon-Google from the Leipzig benchmark suite
- Target: **60%+ F1** with default config (MiniLM, ANN top-20, threshold 0.75)
- Compare against baseline rule-based (37% and 29% respectively)

### Edge Cases

- Empty strings and NaN values in embedding columns
- Mixed-language text (model handles it but quality degrades)
- Very short strings (single token) vs very long strings (truncation at 256 tokens)
- Duplicate records within the same source (self-join scenario)
- Missing optional dependencies — verify clean error message

## Rollout Plan

1. **Phase 1**: `embedder.py` + `embedding` scorer type with in-memory cache
2. **Phase 2**: `ann_blocker.py` + `ann` blocking strategy
3. **Phase 3**: Disk-based embedding cache (`.npy` persistence)
4. **Phase 4**: Benchmark validation on Leipzig datasets, threshold tuning guide
