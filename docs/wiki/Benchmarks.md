# Benchmarks

GoldenMatch is benchmarked against the [University of Leipzig entity resolution benchmark datasets](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).

## Leipzig Benchmark Results

### Best Results Per Dataset

| Dataset | Records | Best Strategy | Precision | Recall | F1 | Time |
|---------|---------|--------------|-----------|--------|-----|------|
| **DBLP-ACM** | 2.6K vs 2.3K | multi-pass + fuzzy | 96.4% | 98.0% | **97.2%** | 2.7s |
| **DBLP-Scholar** | 2.6K vs 64K | multi-pass + fuzzy | 67.2% | 84.1% | **74.7%** | 83.9s |
| **Abt-Buy** | 1K vs 1K | LLM boost (optimal) | 50.8% | 71.7% | **59.5%** | 7 min |
| **Abt-Buy** | 1K vs 1K | rec_emb + ann_pairs | 35.5% | 59.4% | **44.5%** | 0.1s |
| **Amazon-Google** | 1.4K vs 3.2K | rec_emb + ann_pairs | 40.2% | 40.9% | **40.5%** | 0.3s |

### Comparison with Other Tools

| Tool | DBLP-ACM | Abt-Buy | Approach | Training Required |
|------|----------|---------|----------|-------------------|
| **GoldenMatch** | 97.2% | 59.5% | Hybrid (zero-shot + optional LLM boost) | No (optional) |
| **Ditto** | 99.0% | 89.3% | Fine-tuned DistilBERT | Yes (1000+ labels) |
| **DeepMatcher** | 98.4% | 62.8% | Deep learning | Yes |
| **dedupe** | ~96% | ~75% | Active learning | Yes (200+ labels) |

### Key Findings

- **DBLP-ACM (97.2%)**: Competitive with state-of-the-art. Multi-pass blocking with fuzzy scoring is within 2pts of Ditto.
- **DBLP-Scholar (74.7%)**: Multi-pass blocking improved F1 from 50.1% to 74.7% — a 49% relative gain.
- **Abt-Buy (59.5%)**: LLM boost with optimal train/score split significantly outperforms zero-shot (44.5%). The key insight: train on multi-pass pairs (clean), score on ANN pairs (high recall).
- **E-commerce gap**: Abt-Buy and Amazon-Google remain challenging because products have completely different names across sources. SOTA uses fine-tuned models with 1000+ labels.

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
