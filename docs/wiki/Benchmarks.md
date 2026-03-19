# Benchmarks

GoldenMatch is benchmarked against the [University of Leipzig entity resolution benchmark datasets](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).

## Leipzig Benchmark Results

### Best Results Per Dataset

| Dataset | Records | Best Strategy | Precision | Recall | F1 | Time |
|---------|---------|--------------|-----------|--------|-----|------|
| **DBLP-ACM** | 2.6K vs 2.3K | Vertex AI embeddings | 97.5% | 97.3% | **97.4%** | 119s |
| **DBLP-Scholar** | 2.6K vs 64K | multi-pass + fuzzy | 67.2% | 84.1% | **74.7%** | 83.9s |
| **Abt-Buy** | 1K vs 1K | Vertex AI embeddings | 85.5% | 83.9% | **84.7%** | 53s |
| **Amazon-Google** | 1.4K vs 3.2K | Vertex AI embeddings | 60.6% | 56.8% | **58.6%** | 110s |

**Previous bests (without Vertex AI):** DBLP-ACM 97.2% (multi-pass), Abt-Buy 59.5% (LLM boost), Amazon-Google 40.5% (rec_emb). Vertex AI's `text-embedding-004` provides dramatically better embeddings for product matching.

### Comparison with Other Tools

| Tool | DBLP-ACM | Abt-Buy | Approach | Training Required |
|------|----------|---------|----------|-------------------|
| **GoldenMatch** | **97.4%** | **84.7%** | Vertex AI embeddings (zero-config) | No |
| **Ditto** | 99.0% | 89.3% | Fine-tuned DistilBERT | Yes (1000+ labels) |
| **DeepMatcher** | 98.4% | 62.8% | Deep learning | Yes |
| **Splink** | ~95% | ~70% | Fellegi-Sunter (Spark) | Yes (labels) |
| **dedupe** | ~96% | ~75% | Active learning | Yes (200+ labels) |
| **Zingg** | ~96% | ~80% | Active learning (Spark) | Yes (labels) |

### Key Findings

- **DBLP-ACM (97.4%)**: Within 1.6pts of Ditto with zero training — competitive with state-of-the-art.
- **Abt-Buy (84.7%)**: Vertex AI's `text-embedding-004` closed most of the gap with Ditto (89.3%). Previously 59.5% with LLM boost on local MiniLM.
- **Amazon-Google (58.6%)**: 45% relative improvement over previous best (40.5%). Product matching with very different naming conventions remains hard across all tools.
- **DBLP-Scholar (74.7%)**: Multi-pass blocking + fuzzy scoring. Not yet tested with Vertex AI.

See [Comparison with Other Tools](Comparison.md) for a full feature-by-feature breakdown.

## 1M Record Benchmark

GoldenMatch processes 1 million records in **~15 seconds** on a laptop (exact matching, full pipeline):

| Stage | Time | % |
|-------|------|---|
| Ingest | 0.30s | 2% |
| Auto-fix | 2.29s | 15% |
| Standardize | 2.34s | 15% |
| Matchkeys | 0.17s | 1% |
| Matching | 0.29s | 2% |
| Clustering | 7.25s | 48% |
| Golden records | 2.51s | 17% |
| **Total** | **15.15s** | |

138,730 duplicate clusters found with 100% precision and 100% recall.

## LLM Boost Results

Simulated with ground truth labels (5% noise to approximate LLM accuracy):

| Dataset | Zero-Shot | LLM Boost (300 labels) | Improvement | Cost |
|---------|-----------|----------------------|-------------|------|
| DBLP-ACM | 94.8% | 96.6% | +1.8pts | ~$0.30 |
| Abt-Buy | 44.5% | 59.5% | +15pts | ~$0.30 |

The optimal configuration: MiniLM base model, 300 labels, 3 epochs, train on multi-pass pairs, score on ANN pairs.
