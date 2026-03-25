# Benchmarks

## Performance

| Dataset | Records | Mode | Time | Throughput |
|---------|---------|------|------|------------|
| Synthetic | 1M | Exact dedupe | 7.8s | 128K rec/s |
| Synthetic | 100K | Fuzzy (name+zip) | 12.8s | 7,823 rec/s |
| DBLP-ACM | 4,910 | Fuzzy + F-S | 2.1s | 97.2% F1 |
| Abt-Buy | 2,162 | Domain + LLM | 4.2s | 72.2% F1 |

## Accuracy

### Structured Data (Names, Addresses)

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Fuzzy matching alone | 99.1% | 95.4% | **97.2%** |
| + Fellegi-Sunter | 98.8% | 57.6% | 72.8% |
| + Learned blocking | 96.3% | 97.5% | **96.9%** |

### Product Matching (Electronics)

| Method | Precision | Recall | F1 | Cost |
|--------|-----------|--------|-----|------|
| Domain extraction only | 94.8% | 36.4% | 52.6% | $0 |
| + Embedding + ANN | 89.2% | 48.1% | 62.5% | $0 |
| + LLM scorer | 94.8% | 57.6% | **72.2%** | $0.04 |

### PPRL (Privacy-Preserving)

| Dataset | Method | Precision | Recall | F1 |
|---------|--------|-----------|--------|-----|
| FEBRL4 | Manual config | 98.2% | 82.6% | 89.8% |
| FEBRL4 | Auto-config | 97.1% | 88.2% | **92.4%** |
| NCVR | Manual config | 89.3% | 51.2% | 65.8% |
| NCVR | Auto-config | 85.7% | 68.4% | **76.1%** |

## Running Benchmarks

```bash
# Leipzig academic benchmarks
python tests/benchmarks/run_leipzig.py

# PPRL benchmarks
python tests/benchmarks/run_pprl_benchmarks.py

# Full v0.3.0 benchmark suite
python tests/benchmarks/run_v030_quick.py
```
