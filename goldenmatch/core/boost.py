"""LLM boost engine — feature extraction, classifier training, re-scoring.

Uses JSON serialization for model persistence (no pickle/joblib).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_sort_ratio

from goldenmatch.core.llm_labeler import (
    detect_context,
    detect_provider,
    estimate_cost,
    get_default_model,
    label_pairs,
)

logger = logging.getLogger(__name__)

MODEL_FILE = ".goldenmatch_model.json"


# ── Feature extraction ────────────────────────────────────────────────────

def _compute_pair_features(val_a: str, val_b: str) -> list[float]:
    """Compute features for a single pair of string values."""
    a = val_a or ""
    b = val_b or ""
    jw = JaroWinkler.similarity(a, b) if a and b else 0.0
    ts = token_sort_ratio(a, b) / 100.0 if a and b else 0.0
    lev = Levenshtein.normalized_similarity(a, b) if a and b else 0.0
    exact = 1.0 if a == b else 0.0
    len_a, len_b = len(a), len(b)
    length_ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0
    return [jw, ts, lev, exact, length_ratio]


def extract_feature_matrix(
    pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Build (n_pairs, n_features) matrix for all candidate pairs."""
    row_ids = df["__row_id__"].to_list()
    id_to_idx = {rid: i for i, rid in enumerate(row_ids)}
    rows = df.to_dicts()

    features = []
    for id_a, id_b, _score in pairs:
        idx_a = id_to_idx.get(id_a)
        idx_b = id_to_idx.get(id_b)
        if idx_a is None or idx_b is None:
            features.append([0.0] * (5 * len(columns)))
            continue

        row_a = rows[idx_a]
        row_b = rows[idx_b]

        pair_feats = []
        for col in columns:
            val_a = str(row_a.get(col, "") or "")
            val_b = str(row_b.get(col, "") or "")
            pair_feats.extend(_compute_pair_features(val_a, val_b))

        features.append(pair_feats)

    return np.array(features, dtype=np.float64)


# ── Pair sampling ─────────────────────────────────────────────────────────

def _sample_initial_pairs(
    pairs: list[tuple[int, int, float]], n: int = 100,
) -> list[int]:
    """Sample pair indices: 30% high-score, 30% low-score, 40% middle."""
    if len(pairs) <= n:
        return list(range(len(pairs)))

    scores = np.array([s for _, _, s in pairs])
    sorted_idx = np.argsort(scores)

    n_high = n * 30 // 100
    n_low = n * 30 // 100
    n_mid = n - n_high - n_low

    high = sorted_idx[-n_high:].tolist()
    low = sorted_idx[:n_low].tolist()

    mid_mask = (scores >= 0.4) & (scores <= 0.7)
    mid_candidates = np.where(mid_mask)[0]
    rng = np.random.default_rng(42)

    if len(mid_candidates) >= n_mid:
        mid = rng.choice(mid_candidates, n_mid, replace=False).tolist()
    else:
        used = set(high) | set(low) | set(mid_candidates.tolist())
        remaining = [i for i in range(len(pairs)) if i not in used]
        extra_n = min(n_mid - len(mid_candidates), len(remaining))
        extra = rng.choice(remaining, extra_n, replace=False).tolist() if extra_n > 0 else []
        mid = mid_candidates.tolist() + extra

    return list(set(high + low + mid))


def _sample_uncertain_pairs(
    probs: np.ndarray, labeled_indices: set[int], n: int = 100,
) -> list[int]:
    """Sample the most uncertain unlabeled pairs."""
    uncertainty = np.abs(probs - 0.5)
    for idx in labeled_indices:
        uncertainty[idx] = 999.0
    sorted_idx = np.argsort(uncertainty)
    return sorted_idx[:n].tolist()


# ── Model persistence (JSON — no serialization security risks) ────────────

def _column_hash(columns: list[str]) -> str:
    return hashlib.md5("|".join(sorted(columns)).encode()).hexdigest()[:12]


def save_model(
    model, columns: list[str], directory: Path | None = None,
) -> None:
    """Save logistic regression coefficients as JSON."""
    d = directory or Path.cwd()
    path = d / MODEL_FILE
    data = {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
        "columns": columns,
        "column_hash": _column_hash(columns),
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved boost model to %s", path)


def load_model(columns: list[str], directory: Path | None = None):
    """Load saved model if it exists and columns match. Returns model or None."""
    d = directory or Path.cwd()
    path = d / MODEL_FILE
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    if data.get("column_hash") != _column_hash(columns):
        logger.warning("Saved model columns don't match current data. Ignoring saved model.")
        return None

    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.coef_ = np.array(data["coef"])
        model.intercept_ = np.array(data["intercept"])
        model.classes_ = np.array(data["classes"])
        logger.info("Loaded saved boost model from %s", path)
        return model
    except Exception as e:
        logger.warning("Failed to load saved model: %s", e)
        return None


# ── Main boost function ───────────────────────────────────────────────────

def boost_accuracy(
    candidate_pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    columns: list[str],
    provider: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    max_labels: int = 500,
    retrain: bool = False,
    progress_callback=None,
) -> list[tuple[int, int, float]]:
    """Re-score candidate pairs using LLM-trained classifier.

    If a saved model exists and matches columns, uses it directly (no LLM calls).
    Otherwise, labels pairs with LLM and trains a new classifier.
    """
    if not candidate_pairs:
        return candidate_pairs

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        raise ImportError(
            "LLM boost requires scikit-learn. "
            "Install with: pip install goldenmatch[llm]"
        )

    matchable_columns = [c for c in columns if not c.startswith("__")]

    # Try loading saved model
    if not retrain:
        saved_model = load_model(matchable_columns)
        if saved_model is not None:
            logger.info("Using saved boost model (no LLM calls needed)")
            features = extract_feature_matrix(candidate_pairs, df, matchable_columns)
            probs = saved_model.predict_proba(features)[:, 1]
            return [
                (a, b, float(prob))
                for (a, b, _), prob in zip(candidate_pairs, probs)
            ]

    # Detect provider
    if not provider or not api_key:
        detected = detect_provider()
        if detected is None:
            logger.warning("No LLM API key found. Skipping boost.")
            return candidate_pairs
        provider, api_key = detected

    if not model_name:
        model_name = get_default_model(provider)

    context = detect_context({c: c for c in matchable_columns})

    # Extract features
    logger.info("Extracting features for %d candidate pairs...", len(candidate_pairs))
    all_features = extract_feature_matrix(candidate_pairs, df, matchable_columns)

    # Build row lookup
    rows = df.to_dicts()
    row_ids = df["__row_id__"].to_list()
    id_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    # Adaptive labeling loop
    labeled_indices: set[int] = set()
    all_labels: list[bool] = []
    all_labeled_features: list[np.ndarray] = []

    sample_indices = _sample_initial_pairs(candidate_pairs, n=100)
    total_labeled = 0

    while total_labeled < max_labels:
        pairs_to_label = []
        new_indices = []
        for idx in sample_indices:
            if idx in labeled_indices:
                continue
            id_a, id_b, _ = candidate_pairs[idx]
            idx_a = id_to_idx.get(id_a)
            idx_b = id_to_idx.get(id_b)
            if idx_a is not None and idx_b is not None:
                pairs_to_label.append((rows[idx_a], rows[idx_b]))
                new_indices.append(idx)
                labeled_indices.add(idx)

        if not pairs_to_label:
            break

        cost = estimate_cost(len(pairs_to_label), provider)
        logger.info("Labeling %d pairs with %s (~$%.2f)...", len(pairs_to_label), provider, cost)

        labels = label_pairs(
            pairs_to_label, matchable_columns, context,
            provider, api_key, model_name, progress_callback,
        )

        for idx, label in zip(new_indices, labels):
            all_labeled_features.append(all_features[idx])
            all_labels.append(label)

        total_labeled += len(labels)

        X = np.array(all_labeled_features)
        y = np.array(all_labels, dtype=int)

        if len(set(y)) < 2:
            logger.warning("All labels same class. Sampling more diverse pairs.")
            rng = np.random.default_rng(42)
            sample_indices = rng.permutation(len(candidate_pairs))[:100].tolist()
            continue

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        cv_folds = min(5, min(sum(y), sum(1 - y), len(y)))
        if cv_folds < 2:
            logger.info("Too few samples for cross-val. Continuing labeling.")
            sample_indices = _sample_uncertain_pairs(
                np.full(len(candidate_pairs), 0.5), labeled_indices, n=100,
            )
            continue

        cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="f1")
        mean_f1 = cv_scores.mean()

        logger.info("Classifier: %.1f%% cross-val F1 (%d labels)", mean_f1 * 100, total_labeled)

        if mean_f1 >= 0.75 or total_labeled >= max_labels:
            break

        clf.fit(X, y)
        probs = clf.predict_proba(all_features)[:, 1]
        sample_indices = _sample_uncertain_pairs(probs, labeled_indices, n=100)

    # Final training
    X = np.array(all_labeled_features)
    y = np.array(all_labels, dtype=int)

    if len(set(y)) < 2:
        logger.warning("Cannot train classifier — all labels identical. Returning original pairs.")
        return candidate_pairs

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)

    save_model(clf, matchable_columns)

    probs = clf.predict_proba(all_features)[:, 1]
    logger.info("Re-scored %d pairs with trained classifier.", len(candidate_pairs))

    return [
        (a, b, float(prob))
        for (a, b, _), prob in zip(candidate_pairs, probs)
    ]
