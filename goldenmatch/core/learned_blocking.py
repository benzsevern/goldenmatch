"""Learned blocking -- data-driven blocking predicate selection.

Replaces manual blocking key choice with automatic predicate learning:
1. Sample run with conservative static blocking generates training pairs
2. Predicate library generates candidate blocking rules
3. Rules evaluated by recall (% true matches in same block) vs reduction ratio
4. Best rules selected and applied

Usage in config:
    blocking:
      strategy: learned
      learned:
        sample_size: 5000
        min_recall: 0.95
        min_reduction: 0.90
        predicate_depth: 2
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import polars as pl

from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig

logger = logging.getLogger(__name__)


@dataclass
class BlockingPredicate:
    """A single blocking predicate: transform applied to a field."""

    field: str
    transform: str  # "exact", "first_3", "first_5", "soundex", "first_token", "digits_only"

    def key(self) -> str:
        return f"{self.field}:{self.transform}"


@dataclass
class BlockingRule:
    """A conjunction of predicates forming a blocking rule."""

    predicates: list[BlockingPredicate]
    recall: float = 0.0
    reduction_ratio: float = 0.0
    n_blocks: int = 0

    def key(self) -> str:
        return " AND ".join(p.key() for p in sorted(self.predicates, key=lambda p: p.key()))


# ── Predicate Library ─────────────────────────────────────────────────────

_TRANSFORM_MAP = {
    "exact": lambda v: str(v).strip().lower() if v else "",
    "first_3": lambda v: str(v).strip().lower()[:3] if v else "",
    "first_5": lambda v: str(v).strip().lower()[:5] if v else "",
    "soundex": lambda v: _safe_soundex(str(v)) if v else "",
    "first_token": lambda v: str(v).strip().lower().split()[0] if v and str(v).strip() else "",
    "digits_only": lambda v: "".join(c for c in str(v) if c.isdigit()) if v else "",
}


def _safe_soundex(val: str) -> str:
    try:
        import jellyfish
        return jellyfish.soundex(val)
    except Exception:
        return val[:4].upper()


def generate_predicates(columns: list[str]) -> list[BlockingPredicate]:
    """Generate candidate predicates for all columns."""
    predicates = []
    for col in columns:
        for transform_name in _TRANSFORM_MAP:
            predicates.append(BlockingPredicate(field=col, transform=transform_name))
    return predicates


def _apply_predicate(value, predicate: BlockingPredicate) -> str:
    """Apply a predicate transform to a value."""
    fn = _TRANSFORM_MAP.get(predicate.transform, _TRANSFORM_MAP["exact"])
    return fn(value)


def _compute_block_key(row: dict, predicates: list[BlockingPredicate]) -> str:
    """Compute a block key from multiple predicates (conjunction)."""
    parts = []
    for p in predicates:
        val = row.get(p.field)
        parts.append(_apply_predicate(val, p))
    return "||".join(parts)


# ── Evaluation ────────────────────────────────────────────────────────────


def evaluate_rule(
    df: pl.DataFrame,
    rule: BlockingRule,
    true_pairs: set[tuple[int, int]],
) -> tuple[float, float, int]:
    """Evaluate a blocking rule by recall and reduction ratio.

    Returns (recall, reduction_ratio, n_blocks).
    """
    if not true_pairs:
        return 0.0, 1.0, 0

    # Assign block keys
    rows = df.select(["__row_id__"] + [p.field for p in rule.predicates]).to_dicts()
    blocks: dict[str, list[int]] = {}
    for row in rows:
        key = _compute_block_key(row, rule.predicates)
        if key:
            blocks.setdefault(key, []).append(row["__row_id__"])

    # Compute pairs within blocks
    blocked_pairs: set[tuple[int, int]] = set()
    for members in blocks.values():
        if len(members) > 1:
            for a, b in combinations(members, 2):
                blocked_pairs.add((min(a, b), max(a, b)))

    # Total possible pairs
    n = df.height
    total_pairs = n * (n - 1) // 2

    # Recall: fraction of true pairs that land in the same block
    if true_pairs:
        recall = len(true_pairs & blocked_pairs) / len(true_pairs)
    else:
        recall = 0.0

    # Reduction ratio: fraction of pairs eliminated
    if total_pairs > 0:
        reduction = 1 - (len(blocked_pairs) / total_pairs)
    else:
        reduction = 1.0

    return recall, reduction, len(blocks)


# ── Rule Learning ─────────────────────────────────────────────────────────


def learn_blocking_rules(
    df: pl.DataFrame,
    scored_pairs: list[tuple[int, int, float]],
    columns: list[str] | None = None,
    min_recall: float = 0.95,
    min_reduction: float = 0.90,
    predicate_depth: int = 2,
    threshold: float = 0.7,
) -> list[BlockingRule]:
    """Learn blocking rules from scored pairs.

    Args:
        df: DataFrame with __row_id__ and data columns.
        scored_pairs: Pairs from a sample run (row_id_a, row_id_b, score).
        columns: Columns to consider for predicates. Defaults to all non-internal.
        min_recall: Minimum recall requirement for selected rules.
        min_reduction: Minimum reduction ratio requirement.
        predicate_depth: Max predicates per rule (conjunction depth).
        threshold: Score threshold for true positive pairs.

    Returns:
        List of blocking rules meeting the constraints, best first.
    """
    if columns is None:
        columns = [c for c in df.columns if not c.startswith("__")]

    # True pairs from scored pairs above threshold
    true_pairs = {
        (min(a, b), max(a, b))
        for a, b, s in scored_pairs
        if s >= threshold
    }

    if not true_pairs:
        logger.warning("No true pairs found above threshold %.2f. Using first column as fallback.", threshold)
        return [BlockingRule(
            predicates=[BlockingPredicate(field=columns[0], transform="first_5")],
            recall=0.0, reduction_ratio=0.0,
        )]

    logger.info("Learning blocking rules from %d true pairs, %d columns", len(true_pairs), len(columns))

    # Generate single predicates
    all_predicates = generate_predicates(columns)

    # Evaluate single predicates
    single_rules: list[BlockingRule] = []
    for pred in all_predicates:
        rule = BlockingRule(predicates=[pred])
        recall, reduction, n_blocks = evaluate_rule(df, rule, true_pairs)
        rule.recall = recall
        rule.reduction_ratio = reduction
        rule.n_blocks = n_blocks
        single_rules.append(rule)

    # Filter to predicates with reasonable recall
    good_singles = [r for r in single_rules if r.recall >= min_recall * 0.5]  # relaxed for combination
    good_singles.sort(key=lambda r: r.recall, reverse=True)
    good_singles = good_singles[:20]  # limit for combinatorial explosion

    # Check if any single predicate meets both constraints
    passing_rules = [
        r for r in single_rules
        if r.recall >= min_recall and r.reduction_ratio >= min_reduction
    ]

    # Try depth-2 combinations if no single predicate is sufficient
    if not passing_rules and predicate_depth >= 2 and len(good_singles) >= 2:
        for r1, r2 in combinations(good_singles, 2):
            p1 = r1.predicates[0]
            p2 = r2.predicates[0]
            # Skip if same field+transform
            if p1.key() == p2.key():
                continue
            combo = BlockingRule(predicates=[p1, p2])
            recall, reduction, n_blocks = evaluate_rule(df, combo, true_pairs)
            combo.recall = recall
            combo.reduction_ratio = reduction
            combo.n_blocks = n_blocks
            if recall >= min_recall and reduction >= min_reduction:
                passing_rules.append(combo)

    # Sort by recall (highest first), then reduction ratio
    passing_rules.sort(key=lambda r: (r.recall, r.reduction_ratio), reverse=True)

    if not passing_rules:
        # Fallback: pick the single rule with best recall
        best = max(single_rules, key=lambda r: r.recall) if single_rules else None
        if best:
            logger.warning(
                "No rule meets constraints (min_recall=%.2f, min_reduction=%.2f). "
                "Best: recall=%.2f, reduction=%.2f",
                min_recall, min_reduction, best.recall, best.reduction_ratio,
            )
            passing_rules = [best]

    logger.info(
        "Learned %d blocking rules. Best: recall=%.3f, reduction=%.3f",
        len(passing_rules),
        passing_rules[0].recall if passing_rules else 0,
        passing_rules[0].reduction_ratio if passing_rules else 0,
    )

    return passing_rules


# ── Apply Learned Blocks ──────────────────────────────────────────────────


def apply_learned_blocks(
    lf: pl.LazyFrame,
    rules: list[BlockingRule],
    max_block_size: int = 5000,
) -> list:
    """Apply learned blocking rules to produce BlockResult list.

    Uses union of all rules (multi-pass style) for maximum recall.
    """
    from goldenmatch.core.blocker import BlockResult

    df = lf.collect()
    all_blocks: list = []

    for rule in rules[:3]:  # limit to top 3 rules
        rows = df.select(
            ["__row_id__"] + list({p.field for p in rule.predicates})
        ).to_dicts()

        blocks: dict[str, list[int]] = {}
        for row in rows:
            key = _compute_block_key(row, rule.predicates)
            if key:
                blocks.setdefault(key, []).append(row["__row_id__"])

        for block_key, member_ids in blocks.items():
            if len(member_ids) < 2:
                continue
            if len(member_ids) > max_block_size:
                continue
            block_lf = df.filter(pl.col("__row_id__").is_in(member_ids)).lazy()
            all_blocks.append(BlockResult(
                block_key=f"learned:{rule.key()}:{block_key}",
                df=block_lf,
                strategy="learned",
            ))

    # Deduplicate blocks by member set
    seen: set[frozenset[int]] = set()
    deduped: list = []
    for block in all_blocks:
        block_df = block.df.collect()
        members = frozenset(block_df["__row_id__"].to_list())
        if members not in seen:
            seen.add(members)
            deduped.append(block)

    logger.info("Learned blocking produced %d blocks from %d rules", len(deduped), min(len(rules), 3))
    return deduped


# ── Cache ─────────────────────────────────────────────────────────────────


def save_learned_rules(rules: list[BlockingRule], path: str | Path) -> None:
    """Save learned rules to JSON for reuse."""
    data = [
        {
            "predicates": [{"field": p.field, "transform": p.transform} for p in r.predicates],
            "recall": r.recall,
            "reduction_ratio": r.reduction_ratio,
            "n_blocks": r.n_blocks,
        }
        for r in rules
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_learned_rules(path: str | Path) -> list[BlockingRule] | None:
    """Load cached learned rules. Returns None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return [
        BlockingRule(
            predicates=[BlockingPredicate(**pred) for pred in item["predicates"]],
            recall=item["recall"],
            reduction_ratio=item["reduction_ratio"],
            n_blocks=item.get("n_blocks", 0),
        )
        for item in data
    ]
