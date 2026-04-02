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

### Equipment Data Benchmark (v1.2.6)

| Dataset | Records | Strategy | Clusters | Matched | LLM Cost | Time |
|---------|---------|----------|----------|---------|----------|------|
| **Bulldozer** | 401K | Multi-pass + ANN hybrid + LLM calibration | 27,937 | 384,650 | ~$0.01 | 323s |

Using iterative LLM calibration (v1.2.6), the LLM learned the optimal threshold from just 200 sampled pairs. ANN hybrid blocking recovered 363 sub-blocks from 15 oversized blocks, matching 949 additional records. 87.7% of clusters have confidence >= 0.4. See `examples/equipment_dedup.py` for the full configuration.

### Comparison with Other Tools

| Tool | DBLP-ACM | Abt-Buy | BPID (PII) | Approach | Training Required |
|------|----------|---------|------------|----------|-------------------|
| **GoldenMatch** | **97.2%** | **81.7%** (Vertex) | **75.0%** | multi-pass fuzzy + domain extraction + embeddings | No |
| **Ditto** | 99.0% | 89.3% | 75.2% | Fine-tuned DistilBERT | Yes (1000+ labels) |
| **Sudowoodo** | -- | -- | 78.8% | Contrastive self-supervised | Yes (fine-tuned) |
| **DeepMatcher** | 98.4% | 62.8% | -- | Deep learning | Yes |
| **Splink** | ~95% | ~70% | -- | Fellegi-Sunter (Spark) | Yes (labels) |
| **dedupe** | ~96% | ~75% | -- | Active learning | Yes (200+ labels) |
| **Zingg** | ~96% | ~80% | -- | Active learning (Spark) | Yes (labels) |

### BPID — PII Deduplication (EMNLP 2024)

[BPID](https://aclanthology.org/2024.emnlp-industry.40/) (Amazon, EMNLP 2024) — first open-source adversarial PII matching benchmark. 10,000 profile pairs with intentional near-miss traps.

| Configuration | Precision | Recall | F1 | Cost |
|---------------|-----------|--------|-----|------|
| Naive weighted | 54.1% | 86.5% | 66.5% | $0 |
| Optimized (DOB parsing + phone norm) | 65.5% | 86.9% | **74.7%** | $0 |
| + Vertex AI embeddings (65/35 blend) | 67.2% | 84.9% | **75.0%** | ~$0.10 |

GoldenMatch matches Ditto (75.0% vs 75.2%) with zero training data. DOB component parsing was worth +0.08 F1. LLM boost reduced F1 by 0.013 — adversarial profiles trick language models too.

See the [full benchmark writeup](https://bensevern.dev/blog/2026-04-02-goldenmatch-bpid-benchmark).

### Key Findings

- **DBLP-ACM (97.2%)**: Within 2pts of Ditto with zero training — competitive with state-of-the-art. RapidFuzz fuzzy matching beats Vertex AI embeddings on this dataset.
- **Abt-Buy (72.2% local / 81.7% Vertex)**: Domain extraction (brand, model, SKU) + embedding ANN + LLM scorer on borderline pairs. Local pipeline uses MiniLM embeddings + domain extraction for $0.04. Vertex AI pipeline achieves 81.7% at ~$0.74.
- **Amazon-Google (45.3%)**: Clean emb+ANN + LLM pipeline. Software product matching is inherently harder -- adding domain extraction or token normalization hurts F1 (more noise). SOTA is ~78% (GPT-4 few-shot, Ditto fine-tuned).
- **DBLP-Scholar (74.7%)**: Multi-pass blocking + fuzzy scoring.
- **BPID (75.0%)**: Matches Ditto with zero training. DOB parsing is the single biggest lever for PII matching. Embeddings help marginally; LLMs hurt on adversarial data.

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

## PPRL Benchmarks

Privacy-Preserving Record Linkage benchmarked on FEBRL4 (5K vs 5K synthetic person records) and NCVR (North Carolina Voter Registration).

### Auto-Config Results

| Strategy | Precision | Recall | F1 | Privacy |
|----------|-----------|--------|-----|---------|
| Normal fuzzy (baseline) | 56.5% | 74.6% | 64.3% | None |
| **PPRL auto-config (FEBRL4)** | **99.7%** | **86.1%** | **92.4%** | Per-field HMAC |
| PPRL auto-config (NCVR) | 64.0% | 93.8% | 76.1% | Per-field HMAC |
| PPRL paranoid (FEBRL4) | 98.9% | 76.0% | 86.0% | HMAC + balanced |

PPRL with auto-configuration beats manual tuning on both datasets. `auto_configure_pprl()` profiles your data and picks optimal fields, bloom filter parameters, and threshold automatically.

### FEBRL4 (Synthetic Person Data)

| Strategy | Precision | Recall | F1 | Privacy |
|----------|-----------|--------|-----|---------|
| **PPRL auto-config** | **99.7%** | **86.1%** | **92.4%** | Per-field HMAC |
| PPRL high (t=0.80) | 90.5% | 89.1% | 89.8% | Per-field HMAC |
| PPRL paranoid (t=0.80) | 98.9% | 76.0% | 86.0% | HMAC + balanced padding |
| PPRL standard (t=0.80) | 22.2% | 93.2% | 35.8% | Basic CLK |

Auto-config improves over the best manual strategy (PPRL high) by 2.6 points F1 while achieving near-perfect precision (99.7%).

### NCVR (North Carolina Voter Registration)

| Strategy | Precision | Recall | F1 | Privacy |
|----------|-----------|--------|-----|---------|
| **PPRL auto-config** | **64.0%** | **93.8%** | **76.1%** | Per-field HMAC |

NCVR is a real-world voter registration dataset with noisier data than FEBRL4. Auto-config achieves 76.1% F1 with high recall (93.8%).

### Threshold Sweep (PPRL High, FEBRL4)

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.70 | 78.2% | 93.4% | 85.1% |
| 0.75 | 85.1% | 91.2% | 88.0% |
| **0.80** | **90.5%** | **89.1%** | **89.8%** |
| 0.85 | 94.3% | 84.7% | 89.2% |
| 0.90 | 97.1% | 78.3% | 86.7% |

### Key Findings

- **PPRL auto-config is the best overall strategy** -- 92.4% F1 on FEBRL4, outperforming both manual tuning (89.8%) and normal fuzzy (64.3%) by large margins. Zero-config: profiles data, selects fields, tunes bloom filter parameters and threshold automatically.
- **Auto-config generalizes to real-world data** -- 76.1% F1 on NCVR voter registration with no manual tuning.
- **PPRL paranoid (HMAC + balanced padding) trades recall for precision** -- 98.9% precision but 76.0% recall. Best when false positives are costly.
- **PPRL standard (basic CLK) has low precision** -- the single bloom filter concatenates all fields, losing field-level granularity. High recall (93.2%) but too many false positives (22.2% precision).
- **Normal fuzzy underperforms on FEBRL4 person data** -- Jaro-Winkler on short names and postcodes is less effective than bloom filter bigram comparison, which captures character-level similarity better on structured fields.
