---
layout: default
title: Benchmarks
nav_order: 21
---

# Benchmarks

Performance and accuracy measurements on standard entity resolution datasets.

---

## Accuracy

Tested on [Leipzig benchmark datasets](https://dbs.uni-leipzig.de/research/projects/object-matching/benchmark-datasets-for-entity-resolution) and synthetic data.

### Structured data

| Dataset | Records | Strategy | Precision | Recall | F1 | Cost |
|---------|---------|----------|-----------|--------|-----|------|
| DBLP-ACM (bibliographic) | 4,910 | Weighted fuzzy | 97.2% | 97.1% | **97.2%** | $0 |
| DBLP-ACM | Fellegi-Sunter (opt-in) | 98.8% | 57.6% | 72.8% | $0 |
| DBLP-ACM | Learned blocking | 97.6% | 96.3% | 96.9% | $0 |
| DBLP-Scholar (2.6K vs 64K) | Multi-pass + fuzzy | -- | -- | **74.7%** | $0 |

For structured data (names, addresses, bibliographic records), fuzzy matching alone achieves 97%+ F1 with zero cost and zero training labels.

### Product matching

| Dataset | Records | Strategy | Precision | Recall | F1 | Cost |
|---------|---------|----------|-----------|--------|-----|------|
| Abt-Buy (electronics) | 2,162 | Embedding + ANN | 35.5% | 59.4% | 44.5% | $0 |
| Abt-Buy | Model extraction + emb | 39.3% | 71.0% | 50.6% | $0 |
| Abt-Buy | Domain + emb + LLM | **94.8%** | **58.3%** | **72.2%** | **$0.04** |
| Abt-Buy | Vertex AI + GPT-4o-mini | **94.8%** | 71.2% | **81.7%** | $0.74 |
| Amazon-Google (software) | 4,589 | emb + ANN + LLM | 63.3% | 35.2% | **45.3%** | $0.02 |
| Amazon-Google | Vertex AI + reranking | -- | -- | **44.0%** | $0.10 |

Product matching benefits from domain extraction (electronics) and LLM scoring (borderline pairs). Adding too many candidate sources can hurt software matching -- keep the candidate set clean.

### PII deduplication (BPID)

[BPID](https://aclanthology.org/2024.emnlp-industry.40/) (EMNLP 2024, Amazon) is the first open-source benchmark for personal identity deduplication. 10,000 adversarial profile pairs with 5 attributes: name, email (list), phone (list), address (list), DOB.

| Configuration | Precision | Recall | F1 | Cost |
|---------------|-----------|--------|-----|------|
| Naive weighted scoring | 54.1% | 86.5% | 66.5% | $0 |
| Optimized (DOB parsing + phone normalization) | 65.5% | 86.9% | **74.7%** | $0 |
| Classical + Vertex AI embeddings (65/35 blend) | 67.2% | 84.9% | **75.0%** | ~$0.10 |
| + GPT-4.1-mini LLM boost | 62.3% | 90.2% | 73.7% | ~$1.42 |

**Key findings:**
- **DOB component parsing** was the single biggest lever (+0.08 F1). Extracting (year, month, day) from free-text dates and detecting birth year contradictions.
- **Vertex AI embeddings** added marginal gains (+0.003 F1) because adversarial pairs are designed to be semantically similar.
- **LLM boost hurt** (-0.013 F1). GPT-4.1-mini achieved only 60.7% accuracy on borderline pairs — adversarial profiles trick LLMs just as they trick string matchers.
- **Zero training data** — GoldenMatch matches Ditto (75.0% vs 75.2%) without any labeled examples.

See the [full benchmark writeup](https://bensevern.dev/blog/2026-04-02-goldenmatch-bpid-benchmark) and `D:/show_case/bpid_bench/` for scripts.

### Comparison with other tools

| Tool | Abt-Buy F1 | DBLP-ACM F1 | BPID F1 (PII) | Training Required | Zero-Config |
|------|-----------|-------------|---------------|-------------------|-------------|
| **GoldenMatch** | **81.7%** | **97.2%** | **75.0%** | No | Yes |
| dedupe | ~75% | ~96% | -- | Yes | No |
| Splink | ~70% | ~95% | -- | Yes | No |
| Zingg | ~80% | ~96% | -- | Yes | No |
| Ditto | 89.3% | 99.0% | 75.2% | Yes (1000+ labels) | No |
| Sudowoodo | -- | -- | 78.8% | Yes (fine-tuned BERT) | No |

GoldenMatch matches Ditto on PII deduplication (75.0% vs 75.2%) and DBLP-ACM (97.2% vs 99.0%) with zero training. The Abt-Buy gap (~8pts) is the main area where fine-tuned PLMs hold an advantage.

### Equipment data (v1.2.6)

| Dataset | Records | Strategy | Clusters | Matched | LLM Cost | Time |
|---------|---------|----------|----------|---------|----------|------|
| Bulldozer auctions | 401,125 | Multi-pass + ANN hybrid + LLM calibration | 27,937 | 384,650 | ~$0.01 | 323s |

Using iterative LLM calibration, the LLM learned threshold=0.947 from 200 sampled pairs instead of scoring 37,500 pairs. ANN hybrid blocking recovered 363 sub-blocks from 15 oversized blocks, matching 949 additional records that string blocking missed. 87.7% of clusters have confidence >= 0.4.

See [`examples/equipment_dedup.py`](https://github.com/benzsevern/goldenmatch/blob/main/examples/equipment_dedup.py) for the full configuration.

---

## PPRL accuracy

Benchmarked on FEBRL4 (5K vs 5K synthetic person records) and NCVR (North Carolina Voter Registration).

| Dataset | Strategy | Precision | Recall | F1 | Privacy |
|---------|----------|-----------|--------|-----|---------|
| FEBRL4 | Normal fuzzy (baseline) | 56.5% | 74.6% | 64.3% | None |
| FEBRL4 | PPRL manual tuning | 98.2% | 82.6% | 89.8% | HMAC |
| **FEBRL4** | **PPRL auto-config** | **99.7%** | **86.1%** | **92.4%** | Per-field HMAC |
| FEBRL4 | PPRL paranoid | 98.9% | 76.0% | 86.0% | HMAC + balanced |
| **NCVR** | **PPRL auto-config** | **64.0%** | **93.8%** | **76.1%** | Per-field HMAC |

Auto-configuration beats manual tuning on both datasets. PPRL auto-config profiles your data and picks optimal fields, bloom filter parameters, and threshold.

---

## Performance (throughput)

Measured on a laptop (Windows 11, Python 3.12, 16GB RAM) with fuzzy + exact + golden record pipeline.

| Records | Time | Throughput | Pairs Found | Memory |
|---------|------|------------|-------------|--------|
| 1,000 | 0.15s | 6,667 rec/s | 210 | 101 MB |
| 10,000 | 1.67s | 5,975 rec/s | 7,000 | 123 MB |
| 100,000 | 12.78s | **7,823 rec/s** | 571,000 | 546 MB |
| 1,000,000 | 7.8s | 128,205 rec/s | -- | -- |

The 1M benchmark is exact-only (Polars self-join). Fuzzy matching at 1M requires chunked processing or the DuckDB backend.

### Fuzzy matching speedup

Parallel block scoring + intra-field early termination reduced 100K fuzzy matching from ~100s to **~39s** (2.5x speedup):

| Optimization | 100K Time |
|-------------|-----------|
| Baseline (sequential, no early termination) | ~100s |
| + Parallel block scoring (ThreadPoolExecutor) | ~55s |
| + Intra-field early termination | ~39s |
| Pipeline overhead (ingest, standardize, cluster, golden) | ~12.8s total |

---

## Scaling guidance

| Records | Recommended Approach |
|---------|---------------------|
| < 100K | Default (in-memory Polars) |
| 100K -- 500K | Default with blocking tuning |
| 500K -- 1M | Chunked processing (`--chunked`) |
| 1M -- 10M | DuckDB backend or database sync |
| 10M+ | Ray backend (`--backend ray`) or database sync + ANN |

---

## How to run benchmarks

### Leipzig benchmarks

```bash
python tests/benchmarks/run_leipzig.py
```

Runs DBLP-ACM, DBLP-Scholar with multiple strategies and reports F1.

### v0.3.0 quick benchmarks

```bash
python tests/benchmarks/run_v030_quick.py
```

Tests Fellegi-Sunter, learned blocking, and LLM budget features.

### Domain extraction

```bash
python tests/benchmarks/run_domain_bench.py          # Abt-Buy
python tests/benchmarks/run_amazon_google_bench.py    # Amazon-Google
```

### LLM + embedding

```bash
OPENAI_API_KEY=... python tests/benchmarks/run_llm_budget_bench.py
```

Requires an OpenAI API key.

### BPID (PII deduplication)

Download the dataset from [Zenodo](https://zenodo.org/records/13932202) (Apache 2.0), then:

```bash
# From the bpid_bench directory
python run_bpid_optimized.py     # Classical scoring (0.747 F1)
python run_bpid_ann.py           # + Vertex AI embeddings (0.750 F1)
python run_bpid_llm_v2.py        # + LLM boost (requires OPENAI_API_KEY)
```

### Throughput benchmark

```bash
python tests/bench_1m.py
```

Generates synthetic data at multiple scales and measures throughput.

### Analyze results

```bash
python tests/analyze_results.py
```

---

## Key findings

1. **Structured data does not need LLMs or embeddings.** Fuzzy matching achieves 97%+ F1 on bibliographic and person records.

2. **Product matching needs domain extraction + LLM.** Domain extraction gets 393/1081 model matches for free on electronics. LLM scoring handles the borderline pairs.

3. **More candidates can hurt.** Adding candidate sources (domain extraction, token normalization, manufacturer blocking) helps electronics but hurts software matching. Keep the candidate set clean for domains without precise identifiers.

4. **Blocking key choice dominates performance.** A coarse blocking key (state) makes 100K fuzzy matching 30x slower than a fine key (zip + soundex).

5. **PPRL auto-config beats manual tuning.** 92.4% F1 vs 89.8% on FEBRL4, with zero manual configuration.

6. **4 PPRL fields beats 6.** Fewer, higher-quality fields reduce noise in bloom filter comparison.

7. **DOB parsing outperforms LLM reasoning on adversarial PII data.** On BPID, parsing dates into (year, month, day) components added +0.08 F1. LLM boost reduced F1 by 0.013 — adversarial profiles are designed to trick language models just as effectively as string matchers.

8. **GoldenMatch matches Ditto on PII without training.** 75.0% vs 75.2% F1 on BPID with zero labeled pairs, zero fine-tuning, zero GPU.
