"""Evaluation engine -- precision, recall, F1 from ground truth pairs."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass
class EvalResult:
    """Evaluation metrics container."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "predicted_pairs": self.tp + self.fp,
            "ground_truth_pairs": self.tp + self.fn,
        }


def evaluate_pairs(
    predicted: list[tuple[int, int, float]],
    ground_truth: set[tuple],
) -> EvalResult:
    """Evaluate predicted pairs against ground truth.

    Ground truth pairs are matched symmetrically: (a,b) matches (b,a).
    """
    # Normalize ground truth to canonical form (min, max)
    gt_canonical = set()
    for pair in ground_truth:
        a, b = pair[0], pair[1]
        gt_canonical.add((min(a, b), max(a, b)))

    tp = fp = 0
    seen = set()
    for a, b, _score in predicted:
        canon = (min(a, b), max(a, b))
        if canon in seen:
            continue
        seen.add(canon)
        if canon in gt_canonical:
            tp += 1
        else:
            fp += 1
    fn = len(gt_canonical) - tp
    return EvalResult(tp=tp, fp=fp, fn=fn)


def evaluate_clusters(
    clusters: dict[int, dict],
    ground_truth: set[tuple],
) -> EvalResult:
    """Evaluate clusters by expanding to pairwise comparisons."""
    predicted = []
    for cid, info in clusters.items():
        members = info.get("members", [])
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            predicted.append((a, b, 1.0))
    return evaluate_pairs(predicted, ground_truth)


def load_ground_truth_csv(path: str, col_a: str = "id_a", col_b: str = "id_b") -> set[tuple]:
    """Load ground truth pairs from CSV.

    Supports both ID-based (integer) and string-based pair columns.
    """
    import polars as pl
    df = pl.read_csv(path)
    if col_a not in df.columns or col_b not in df.columns:
        # Try common alternative column names
        for alt_a, alt_b in [("idA", "idB"), ("id1", "id2"), ("left_id", "right_id")]:
            if alt_a in df.columns and alt_b in df.columns:
                col_a, col_b = alt_a, alt_b
                break
        else:
            raise ValueError(
                f"Ground truth CSV must have columns '{col_a}' and '{col_b}'. "
                f"Found: {df.columns}"
            )
    pairs = set()
    for row in df.select(col_a, col_b).to_dicts():
        a, b = row[col_a], row[col_b]
        # Try integer conversion (row IDs are ints in GoldenMatch)
        try:
            a = int(a)
        except (ValueError, TypeError):
            a = str(a).strip()
        try:
            b = int(b)
        except (ValueError, TypeError):
            b = str(b).strip()
        pairs.add((a, b))
    return pairs
