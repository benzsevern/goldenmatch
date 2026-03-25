---
layout: default
title: Evaluation
nav_order: 19
---

# Evaluation

Measure matching accuracy against ground truth and enforce quality gates in CI/CD pipelines.

---

## Quick start

```python
import goldenmatch as gm

metrics = gm.evaluate("data.csv", config="config.yaml", ground_truth="gt.csv")
print(f"F1: {metrics['f1']:.1%}, Precision: {metrics['precision']:.1%}, Recall: {metrics['recall']:.1%}")
```

```bash
goldenmatch evaluate data.csv --config config.yaml --gt ground_truth.csv
```

---

## Ground truth format

A CSV file with two columns identifying matched pairs:

```csv
id_a,id_b
1,42
1,108
5,200
5,201
5,203
```

Each row represents a known true match. Column names default to `id_a` and `id_b` but are configurable.

IDs correspond to GoldenMatch's `__row_id__` (int64). Ground truth CSVs may have string IDs -- `load_ground_truth_csv` attempts int conversion automatically.

```python
gt_pairs = gm.load_ground_truth_csv("gt.csv", col_a="id_a", col_b="id_b")
# Returns set of (int, int) tuples
```

---

## CI/CD quality gates

Exit with code 1 if accuracy falls below thresholds:

```bash
goldenmatch evaluate data.csv \
    --config config.yaml \
    --gt ground_truth.csv \
    --min-f1 0.90 \
    --min-precision 0.80 \
    --min-recall 0.70
```

Use in GitHub Actions:

```yaml
# .github/workflows/quality.yml
jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install goldenmatch
      - run: |
          goldenmatch evaluate data.csv \
            --config config.yaml \
            --gt ground_truth.csv \
            --min-f1 0.90 --min-precision 0.80
```

---

## EvalResult

```python
@dataclass
class EvalResult:
    precision: float    # TP / (TP + FP)
    recall: float       # TP / (TP + FN)
    f1: float           # 2 * P * R / (P + R)
    tp: int             # True positives (correct matches)
    fp: int             # False positives (incorrect matches)
    fn: int             # False negatives (missed matches)

    def summary(self) -> dict
```

---

## Evaluate pairs directly

```python
import goldenmatch as gm

predicted = {(1, 42), (1, 108), (5, 200), (7, 300)}
ground_truth = {(1, 42), (1, 108), (5, 200), (5, 201)}

result = gm.evaluate_pairs(predicted, ground_truth)
print(f"Precision: {result.precision:.1%}")  # 3/4 = 75%
print(f"Recall: {result.recall:.1%}")        # 3/4 = 75%
print(f"F1: {result.f1:.1%}")                # 75%
```

---

## Evaluate clusters

Evaluate a cluster dict (as returned by `build_clusters`). Expands cluster members into pairs for comparison.

```python
import goldenmatch as gm

result = gm.evaluate_clusters(clusters, ground_truth_pairs)
print(result.f1)
```

Note: `run_dedupe()` does not return `scored_pairs` -- use the `clusters` dict instead.

---

## Build ground truth with label command

Interactively label record pairs to create a ground truth CSV:

```bash
goldenmatch label customers.csv --config config.yaml --gt ground_truth.csv
```

The label command shows pairs and prompts for your judgment:

| Key | Meaning |
|-----|---------|
| `y` | Match (add to ground truth) |
| `n` | No match (skip) |
| `s` | Skip (unsure) |

Pairs are selected from actual pipeline output, focusing on borderline cases near the threshold.

---

## Evaluation workflow

1. **Build ground truth**: Use `goldenmatch label` or create a CSV manually
2. **Run evaluation**: `goldenmatch evaluate --gt gt.csv`
3. **Iterate**: Adjust config (thresholds, scorers, blocking) and re-evaluate
4. **Gate CI**: Add `--min-f1` threshold to your CI pipeline

```
label pairs --> ground_truth.csv --> evaluate --> adjust config --> repeat
                                         |
                                    CI/CD gate (--min-f1 0.90)
```

---

## Metrics explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Of the pairs GoldenMatch found, how many are correct? |
| **Recall** | TP / (TP + FN) | Of the true matches, how many did GoldenMatch find? |
| **F1** | 2*P*R / (P+R) | Harmonic mean of precision and recall |

For entity resolution:
- **High precision** means few false merges (records incorrectly combined)
- **High recall** means few missed duplicates
- Most production systems prioritize precision (false merges are harder to fix than missed dupes)

---

## Benchmark evaluation tips

- Always use threshold-based pair generation, NOT top-1-per-record (argmax)
- Leipzig benchmark CSVs have invalid UTF-8 -- use `pl.read_csv(encoding="utf8-lossy", ignore_errors=True)`
- Run benchmarks: `python tests/benchmarks/run_leipzig.py`
