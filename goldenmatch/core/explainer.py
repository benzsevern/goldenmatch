"""Match explainer — shows why two records matched with per-field breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import token_sort_ratio
import jellyfish

from goldenmatch.config.schemas import MatchkeyField
from goldenmatch.utils.transforms import apply_transforms


@dataclass
class FieldExplanation:
    """Explanation for a single field comparison."""
    field_name: str
    scorer: str
    value_a: str | None
    value_b: str | None
    transformed_a: str | None
    transformed_b: str | None
    score: float
    weight: float
    contribution: float  # score * weight
    is_match: bool  # score >= some threshold
    diff_type: str  # "identical", "similar", "different", "missing"


@dataclass
class MatchExplanation:
    """Full explanation of why two records matched (or didn't)."""
    record_a: dict
    record_b: dict
    fields: list[FieldExplanation]
    total_score: float
    threshold: float
    is_match: bool
    top_contributor: str  # field name that contributed most
    weakest_field: str  # field name that scored lowest


def explain_pair(
    record_a: dict,
    record_b: dict,
    matchkey_fields: list[MatchkeyField],
    threshold: float = 0.80,
) -> MatchExplanation:
    """Explain why two records match or don't match.

    Computes per-field scores with full breakdown of transforms,
    raw vs transformed values, and contribution to overall score.
    """
    explanations = []
    weighted_sum = 0.0
    weight_sum = 0.0

    for f in matchkey_fields:
        if f.scorer == "record_embedding":
            # Can't explain embedding scores per-field
            explanations.append(FieldExplanation(
                field_name="(record embedding)",
                scorer="record_embedding",
                value_a="[concatenated]",
                value_b="[concatenated]",
                transformed_a=None,
                transformed_b=None,
                score=0.0,
                weight=f.weight or 1.0,
                contribution=0.0,
                is_match=False,
                diff_type="unknown",
            ))
            continue

        field_name = f.field or ""
        raw_a = record_a.get(field_name)
        raw_b = record_b.get(field_name)
        val_a = str(raw_a) if raw_a is not None else None
        val_b = str(raw_b) if raw_b is not None else None

        # Apply transforms
        trans_a = apply_transforms(val_a, f.transforms) if val_a is not None else None
        trans_b = apply_transforms(val_b, f.transforms) if val_b is not None else None

        # Score
        score = _score_field(trans_a, trans_b, f.scorer or "exact")
        weight = f.weight or 1.0

        # Determine diff type
        if trans_a is None or trans_b is None:
            diff_type = "missing"
        elif trans_a == trans_b:
            diff_type = "identical"
        elif score >= 0.8:
            diff_type = "similar"
        else:
            diff_type = "different"

        contribution = score * weight if score is not None else 0.0
        if score is not None:
            weighted_sum += contribution
            weight_sum += weight

        explanations.append(FieldExplanation(
            field_name=field_name,
            scorer=f.scorer or "exact",
            value_a=str(val_a) if val_a is not None else None,
            value_b=str(val_b) if val_b is not None else None,
            transformed_a=str(trans_a) if trans_a is not None else None,
            transformed_b=str(trans_b) if trans_b is not None else None,
            score=score if score is not None else 0.0,
            weight=weight,
            contribution=contribution,
            is_match=score >= threshold if score is not None else False,
            diff_type=diff_type,
        ))

    total_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    # Find top contributor and weakest field
    scored_fields = [e for e in explanations if e.diff_type != "missing"]
    top = max(scored_fields, key=lambda e: e.contribution) if scored_fields else explanations[0]
    weakest = min(scored_fields, key=lambda e: e.score) if scored_fields else explanations[0]

    return MatchExplanation(
        record_a=record_a,
        record_b=record_b,
        fields=explanations,
        total_score=total_score,
        threshold=threshold,
        is_match=total_score >= threshold,
        top_contributor=top.field_name,
        weakest_field=weakest.field_name,
    )


def _score_field(val_a: str | None, val_b: str | None, scorer: str) -> float | None:
    """Score two field values."""
    if val_a is None or val_b is None:
        return None

    if scorer == "exact":
        return 1.0 if val_a == val_b else 0.0
    elif scorer == "jaro_winkler":
        return JaroWinkler.similarity(val_a, val_b)
    elif scorer == "levenshtein":
        return Levenshtein.normalized_similarity(val_a, val_b)
    elif scorer == "token_sort":
        return token_sort_ratio(val_a, val_b) / 100.0
    elif scorer == "soundex_match":
        return 1.0 if jellyfish.soundex(val_a) == jellyfish.soundex(val_b) else 0.0
    elif scorer == "ensemble":
        jw = JaroWinkler.similarity(val_a, val_b)
        ts = token_sort_ratio(val_a, val_b) / 100.0
        sx = 0.8 if jellyfish.soundex(val_a) == jellyfish.soundex(val_b) else 0.0
        return max(jw, ts, sx)
    else:
        return 0.0


def format_explanation_text(exp: MatchExplanation) -> str:
    """Format explanation as readable text."""
    lines = []
    status = "[green]MATCH[/]" if exp.is_match else "[red]NO MATCH[/]"
    lines.append(f"Score: {exp.total_score:.3f} (threshold: {exp.threshold}) — {status}")
    lines.append("")
    lines.append(f"{'Field':<15} {'Scorer':<14} {'Value A':<20} {'Value B':<20} {'Score':>6} {'Weight':>6} {'Contrib':>7}")
    lines.append("─" * 100)

    for f in exp.fields:
        va = (f.value_a or "—")[:18]
        vb = (f.value_b or "—")[:18]

        # Color code by diff type
        if f.diff_type == "identical":
            score_str = f"[green]{f.score:.3f}[/]"
        elif f.diff_type == "similar":
            score_str = f"[yellow]{f.score:.3f}[/]"
        elif f.diff_type == "missing":
            score_str = "[dim]  —  [/]"
        else:
            score_str = f"[red]{f.score:.3f}[/]"

        lines.append(
            f"{f.field_name:<15} {f.scorer:<14} {va:<20} {vb:<20} {score_str:>6} {f.weight:>6.1f} {f.contribution:>7.3f}"
        )

    lines.append("")
    lines.append(f"Top contributor: [bold]{exp.top_contributor}[/]")
    lines.append(f"Weakest field:   [bold]{exp.weakest_field}[/]")

    return "\n".join(lines)
