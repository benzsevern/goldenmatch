# Benchmarks

GoldenMatch is benchmarked against the [University of Leipzig entity resolution benchmark datasets](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).

## Leipzig Benchmark Results

### Best Results Per Dataset

| Dataset | Records | Best Strategy | Precision | Recall | F1 | Time |
|---------|---------|--------------|-----------|--------|-----|------|
| **DBLP-ACM** | 2.6K vs 2.3K | multi-pass + fuzzy | 96.4% | 98.0% | **97.2%** | 3.6s |
| **DBLP-Scholar** | 2.6K vs 64K | multi-pass + fuzzy | 67.2% | 84.1% | **74.7%** | 83.9s |
| **Abt-Buy** | 1K vs 1K | Domain extraction + emb + LLM | 94.8% | 58.3% | **72.2%** | $0.04 |
| **Abt-Buy** (Vertex AI) | 1K vs 1K | Vertex AI + GPT-4o-mini scorer | 88.3% | 76.1% | **81.7%** | ~$0.74 |
| **Amazon-Google** | 1.4K vs 3.2K | emb+ANN + LLM | 63.3% | 35.2% | **45.3%** | $0.02 |

**Structured data (DBLP-ACM):** RapidFuzz multi-pass fuzzy at 97.2% — zero cost, within 2pts of Ditto. **Product matching (Abt-Buy):** Vertex AI for candidate generation + GPT-4o-mini as scorer on borderline pairs achieves 81.7% — within 8pts of Ditto (89.3%) but with zero training labels and no GPU.

### Comparison with Other Tools

| Tool | DBLP-ACM | Abt-Buy | Approach | Training Required |
|------|----------|---------|----------|-------------------|
| **GoldenMatch** | **97.2%** | **72.2%** (local) / **81.7%** (Vertex) | multi-pass fuzzy + domain extraction + LLM | No |
| **Ditto** | 99.0% | 89.3% | Fine-tuned DistilBERT | Yes (1000+ labels) |
| **DeepMatcher** | 98.4% | 62.8% | Deep learning | Yes |
| **Splink** | ~95% | ~70% | Fellegi-Sunter (Spark) | Yes (labels) |
| **dedupe** | ~96% | ~75% | Active learning | Yes (200+ labels) |
| **Zingg** | ~96% | ~80% | Active learning (Spark) | Yes (labels) |

### Key Findings

- **DBLP-ACM (97.2%)**: Within 2pts of Ditto with zero training — competitive with state-of-the-art. RapidFuzz fuzzy matching beats Vertex AI embeddings on this dataset.
- **Abt-Buy (72.2% local / 81.7% Vertex)**: Domain extraction (brand, model, SKU) + embedding ANN + LLM scorer on borderline pairs. Local pipeline uses MiniLM embeddings + domain extraction for $0.04. Vertex AI pipeline achieves 81.7% at ~$0.74.
- **Amazon-Google (45.3%)**: Clean emb+ANN + LLM pipeline. Software product matching is inherently harder -- adding domain extraction or token normalization hurts F1 (more noise). SOTA is ~78% (GPT-4 few-shot, Ditto fine-tuned).
- **DBLP-Scholar (74.7%)**: Multi-pass blocking + fuzzy scoring.

See [Comparison with Other Tools](Comparison.md) for a full feature-by-feature breakdown.

## Throughput (Scale Curve)

Measured on a laptop (17GB RAM, no GPU) with exact + fuzzy matching, blocking, clustering, and golden record generation:

| Records | Time | Throughput | Pairs Found | Memory |
|---------|------|------------|-------------|--------|
| 1,000 | 0.2s | 5,500 rec/s | 210 | 101 MB |
| 10,000 | 1.4s | 7,300 rec/s | 7,000 | 123 MB |
| 100,000 | 12s | **8,200 rec/s** | 571,000 | 544 MB |

Near-linear scaling: throughput stays consistent as data grows. Memory usage scales linearly.

### Fuzzy Matching Performance

100K records with 3-field fuzzy matching (first_name + last_name jaro_winkler, zip exact):

| Threshold | Time | Fuzzy Pairs | F1 |
|-----------|------|-------------|-----|
| 0.95 | **38s** | 1,534 | 93.0% |
| 0.85 | **39s** | 1,915 | 84.5% |
| 0.75 | **42s** | 2,544 | 73.3% |

Previously ~100s before parallel block scoring and intra-field early termination (2.5x speedup through the pipeline).

**Note:** The times above are measured through `run_dedupe` which uses `score_blocks_parallel` (multi-threaded). The standalone `analyze_fuzzy.py` script calls `find_fuzzy_matches` per block sequentially, so its times (38-55s) reflect single-threaded performance with intra-field early termination only.

**Optimizations applied:**
- **Parallel block scoring** — blocks are independent and scored concurrently via `ThreadPoolExecutor`. RapidFuzz's `cdist` releases the GIL, so threads provide real parallelism on the expensive NxN scoring.
- **Intra-field early termination** — after scoring each expensive field, checks if any pair in the upper triangle can still reach threshold even with perfect scores on all remaining fields. Breaks early when impossible, skipping unnecessary `cdist` calls.
- **Cross-encoder reranking** (opt-in, `rerank: true`) — re-scores borderline pairs with a pre-trained cross-encoder for improved precision without training.
- **Histogram-based auto-select** (opt-in, `auto_select: true`) — evaluates configured blocking keys by group-size histogram and picks the one with smallest max block size.
- **Dynamic block splitting** — adaptive strategy auto-splits oversized blocks by highest-cardinality column when no `sub_block_keys` are configured.
- **Weighted multi-field embedding** (opt-in, `column_weights`) — biases record embeddings toward important fields by repeating high-weight field text in the concatenation.
- **LLM scorer** (opt-in, `llm_scorer: enabled: true`) — sends borderline pairs (score 0.75-0.95) to GPT-4o-mini/Claude for yes/no match decisions. Auto-accepts high-confidence pairs (>0.95). Achieves 81.7% F1 on Abt-Buy at ~$0.74 cost.
- **PPRL bloom filters** — `bloom_filter` transform + `dice`/`jaccard` scorers for privacy-preserving fuzzy matching on encrypted PII.

### Pipeline Bottlenecks (at 100K records)

| Stage | Time | % of Total |
|-------|------|-----------|
| Fuzzy matching | 5.6s | 45% |
| Golden records | 4.0s | 33% |
| Blocking | 2.0s | 16% |
| Clustering | 0.5s | 4% |
| Auto-fix + standardize | 0.1s | 1% |
| Matchkeys + exact match | 0.1s | 1% |

The bottleneck is fuzzy NxN scoring within blocks (RapidFuzz cdist). Coarser blocking keys = faster but lower recall. Fine blocking keys = slower but higher recall.

### 1M Record Benchmark (Exact Only)

With exact matching only (no fuzzy), 1M records process in **~7.8 seconds**:

| Stage | Time | % |
|-------|------|---|
| Ingest | 0.11s | 1% |
| Auto-fix | 1.67s | 21% |
| Standardize | 1.38s | 18% |
| Matchkeys | 0.13s | 2% |
| Exact matching | 0.14s | 2% |
| Clustering | 3.08s | 39% |
| Golden (1K sample) | 1.31s | 17% |

138,730 duplicate clusters found with 100% precision and 100% recall.

### Large Datasets (1M+)

For datasets exceeding available memory, use database sync mode:

```bash
goldenmatch sync --table customers --connection-string "$DB" --config config.yaml
```

Processes in chunks, maintains persistent ANN index, matches incrementally. Tested to 10M+ records in Postgres.

### Throughput Comparison

| Tool | Throughput | Hardware Required |
|------|-----------|-------------------|
| **GoldenMatch** | **8,200 rec/s** | Laptop (no GPU) |
| dedupe | ~500 rec/s | Laptop |
| Splink | ~50,000 rec/s | Spark cluster |
| Zingg | ~30,000 rec/s | Spark cluster |

GoldenMatch is the fastest single-machine deduplication tool. Splink and Zingg are faster but require distributed Spark clusters.

## LLM Boost Results

Simulated with ground truth labels (5% noise to approximate LLM accuracy):

| Dataset | Zero-Shot | LLM Boost (300 labels) | Improvement | Cost |
|---------|-----------|----------------------|-------------|------|
| DBLP-ACM | 94.8% | 96.6% | +1.8pts | ~$0.30 |
| Abt-Buy | 44.5% | 59.5% | +15pts | ~$0.30 |

The optimal configuration: MiniLM base model, 300 labels, 3 epochs, train on multi-pass pairs, score on ANN pairs.
