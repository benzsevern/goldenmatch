#!/usr/bin/env python
"""Evaluate accuracy and tune thresholds.

Shows the workflow: run dedupe -> evaluate against ground truth ->
adjust threshold -> re-evaluate. No LLM needed.
"""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import goldenmatch as gm
import polars as pl
import tempfile
from pathlib import Path

# Sample data
data = pl.DataFrame({
    "id": [1, 2, 3, 4, 5, 6],
    "name": ["John Smith", "john smith", "Jane Doe", "J. Smith", "Bob Jones", "bobby jones"],
    "email": ["john@x.com", "john@x.com", "jane@y.com", "john@x.com", "bob@z.com", "bob@z.com"],
})

# Ground truth: which records are actually the same entity
# (row_id pairs, 0-indexed)
ground_truth = pl.DataFrame({
    "id_a": [0, 0, 0, 4],
    "id_b": [1, 3, 1, 5],
})

tmp = Path(tempfile.mkdtemp())
data_path = tmp / "data.csv"
gt_path = tmp / "ground_truth.csv"
config_path = tmp / "config.yaml"
data.write_csv(data_path)
ground_truth.write_csv(gt_path)

# Write config
config_path.write_text("""
matchkeys:
  - name: exact_email
    type: exact
    fields:
      - field: email
        transforms: [lowercase, strip]
""")

# Evaluate
metrics = gm.evaluate(str(data_path), config=str(config_path), ground_truth=str(gt_path))

print("Evaluation results:")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.1%}")
    else:
        print(f"  {k}: {v}")

# Programmatic evaluation with different pairs
print("\n--- Programmatic evaluation ---")
predicted = [(0, 1, 1.0), (0, 3, 1.0), (4, 5, 1.0)]  # our predictions
gt_set = {(0, 1), (0, 3), (1, 3), (4, 5)}  # true matches

result = gm.evaluate_pairs(predicted, gt_set)
print(f"Precision: {result.precision:.1%}")
print(f"Recall: {result.recall:.1%}")
print(f"F1: {result.f1:.1%}")
print(f"Missing: {result.fn} pairs not found")
