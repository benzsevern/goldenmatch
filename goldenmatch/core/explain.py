"""Natural language explanations for match decisions.

Template-based (not LLM) -- produces human-readable explanations of why
pairs matched or didn't match, at zero cost.
"""
from __future__ import annotations

import logging

from goldenmatch.config.schemas import MatchkeyConfig
from goldenmatch.core.explainer import FieldExplanation, explain_pair

logger = logging.getLogger(__name__)

# ── Score descriptors ─────────────────────────────────────────────────────

_SCORE_DESCRIPTORS = [
    (0.95, "identical"),
    (0.85, "very similar"),
    (0.70, "similar"),
    (0.50, "somewhat similar"),
    (0.30, "weakly similar"),
    (0.0, "different"),
]

_SCORER_NAMES = {
    "jaro_winkler": "string similarity",
    "levenshtein": "edit distance",
    "token_sort": "token similarity",
    "soundex_match": "phonetic match",
    "exact": "exact match",
    "ensemble": "best-of-multiple",
    "dice": "Dice coefficient",
    "jaccard": "Jaccard similarity",
    "embedding": "semantic similarity",
    "record_embedding": "record similarity",
}


def _describe_score(score: float) -> str:
    for threshold, desc in _SCORE_DESCRIPTORS:
        if score >= threshold:
            return desc
    return "different"


def _describe_scorer(scorer: str) -> str:
    return _SCORER_NAMES.get(scorer, scorer)


# ── Pair Explanation ──────────────────────────────────────────────────────


def explain_pair_nl(
    row_a: dict,
    row_b: dict,
    field_scores: list[dict],
    overall_score: float,
) -> str:
    """Generate a natural language explanation for a pair match.

    Args:
        row_a: First record dict.
        row_b: Second record dict.
        field_scores: List of field score dicts from lineage (field, scorer, value_a, value_b, score, weight).
        overall_score: Overall pair score.

    Returns:
        Human-readable explanation string.
    """
    desc = _describe_score(overall_score)
    parts = [f"Match ({desc}, score {overall_score:.2f}):"]

    # Sort fields by contribution (score * weight, descending)
    scored = sorted(
        field_scores,
        key=lambda f: f.get("score", 0) * f.get("weight", 1),
        reverse=True,
    )

    field_descs = []
    weakest = None
    weakest_score = 1.0

    for f in scored:
        field_name = f.get("field", "?")
        scorer = f.get("scorer", "?")
        val_a = f.get("value_a", "")
        val_b = f.get("value_b", "")
        score = f.get("score", 0)
        diff_type = f.get("diff_type", "")

        if score < weakest_score:
            weakest_score = score
            weakest = field_name

        scorer_desc = _describe_scorer(scorer)

        if diff_type == "identical" or score >= 0.99:
            field_descs.append(f"{field_name} match exactly ({_fmt_val(val_a)})")
        elif score >= 0.80:
            field_descs.append(
                f"{field_name} are {_describe_score(score)} "
                f"({_fmt_val(val_a)} ~ {_fmt_val(val_b)}, "
                f"{scorer_desc} {score:.2f})"
            )
        elif score > 0:
            field_descs.append(
                f"{field_name} differ "
                f"({_fmt_val(val_a)} vs {_fmt_val(val_b)}, "
                f"{scorer_desc} {score:.2f})"
            )
        else:
            field_descs.append(
                f"{field_name} do not match "
                f"({_fmt_val(val_a)} vs {_fmt_val(val_b)})"
            )

    parts.append("; ".join(field_descs))

    if weakest and weakest_score < 0.80:
        parts.append(f"Weakest signal: {weakest}.")

    return " ".join(parts)


def _fmt_val(val) -> str:
    """Format a value for display."""
    if val is None:
        return "[null]"
    s = str(val).strip()
    if len(s) > 40:
        return s[:37] + "..."
    return s


# ── Cluster Explanation ───────────────────────────────────────────────────


def explain_cluster_nl(
    cluster: dict,
    df,
    matchkeys: list[MatchkeyConfig],
) -> str:
    """Generate a template-based cluster summary.

    Args:
        cluster: Cluster info dict from build_clusters.
        df: DataFrame with record data.
        matchkeys: Matchkey configs used for scoring.

    Returns:
        Human-readable cluster summary.
    """
    members = cluster.get("members", [])
    size = cluster.get("size", len(members))
    confidence = cluster.get("confidence", 0)
    bottleneck = cluster.get("bottleneck_pair")
    pair_scores = cluster.get("pair_scores", {})

    if size <= 1:
        return f"Singleton cluster with 1 record."

    # Score statistics
    scores = list(pair_scores.values()) if pair_scores else []
    min_score = min(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0

    parts = [
        f"Cluster of {size} records "
        f"(confidence {confidence:.2f}, "
        f"scores {min_score:.2f}-{max_score:.2f}, "
        f"avg {avg_score:.2f})."
    ]

    if bottleneck:
        a, b = bottleneck
        bp_score = pair_scores.get((a, b), pair_scores.get((b, a), 0))
        parts.append(
            f"Weakest link: records {a} and {b} (score {bp_score:.2f})."
        )

    if cluster.get("oversized"):
        parts.append("WARNING: cluster exceeds max size limit.")

    return " ".join(parts)
