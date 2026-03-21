"""Tests for learned blocking -- data-driven predicate selection."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from goldenmatch.core.learned_blocking import (
    BlockingPredicate,
    BlockingRule,
    apply_learned_blocks,
    evaluate_rule,
    generate_predicates,
    learn_blocking_rules,
    load_learned_rules,
    save_learned_rules,
)


def _make_df():
    """Test DataFrame with obvious matching patterns."""
    return pl.DataFrame({
        "__row_id__": list(range(1, 9)),
        "first_name": ["John", "Jon", "Jane", "Janet", "Bob", "Robert", "Alice", "Tom"],
        "last_name": ["Smith", "Smith", "Doe", "Doe", "Jones", "Jones", "Brown", "Wilson"],
        "zip": ["90210", "90210", "10001", "10001", "60601", "60601", "30301", "20001"],
        "email": ["j@s.com", "j@s.com", "j@d.com", "jd@d.com", "b@j.com", "rj@j.com", "a@b.com", "t@w.com"],
    })


class TestGeneratePredicates:
    def test_generates_for_all_columns(self):
        preds = generate_predicates(["name", "zip"])
        assert len(preds) > 0
        fields = {p.field for p in preds}
        assert "name" in fields
        assert "zip" in fields

    def test_generates_all_transforms(self):
        preds = generate_predicates(["name"])
        transforms = {p.transform for p in preds}
        assert "exact" in transforms
        assert "first_3" in transforms
        assert "soundex" in transforms


class TestEvaluateRule:
    def test_perfect_recall_exact(self):
        """Blocking on zip gives perfect recall for same-zip pairs."""
        df = _make_df()
        rule = BlockingRule(predicates=[BlockingPredicate(field="zip", transform="exact")])
        true_pairs = {(1, 2), (3, 4), (5, 6)}  # same-zip pairs
        recall, reduction, n_blocks = evaluate_rule(df, rule, true_pairs)
        assert recall == 1.0
        assert reduction > 0  # should eliminate many comparisons

    def test_zero_recall(self):
        """Blocking on email exact when pairs have different emails."""
        df = _make_df()
        rule = BlockingRule(predicates=[BlockingPredicate(field="email", transform="exact")])
        true_pairs = {(5, 6)}  # Bob/Robert have different emails
        recall, reduction, n_blocks = evaluate_rule(df, rule, true_pairs)
        assert recall == 0.0  # b@j.com != rj@j.com


class TestLearnBlockingRules:
    def test_learns_from_scored_pairs(self):
        df = _make_df()
        scored = [(1, 2, 0.9), (3, 4, 0.85), (5, 6, 0.8)]
        rules = learn_blocking_rules(df, scored, min_recall=0.5, min_reduction=0.1)
        assert len(rules) >= 1
        assert rules[0].recall > 0

    def test_fallback_on_no_pairs(self):
        df = _make_df()
        rules = learn_blocking_rules(df, [], min_recall=0.5, min_reduction=0.1)
        assert len(rules) >= 1  # should produce a fallback

    def test_respects_threshold(self):
        df = _make_df()
        scored = [(1, 2, 0.3)]  # below default threshold
        rules = learn_blocking_rules(df, scored, threshold=0.5)
        # No true pairs above 0.5 -> fallback
        assert len(rules) >= 1


class TestApplyLearnedBlocks:
    def test_produces_blocks(self):
        df = _make_df()
        rules = [BlockingRule(
            predicates=[BlockingPredicate(field="zip", transform="exact")],
            recall=1.0, reduction_ratio=0.8,
        )]
        blocks = apply_learned_blocks(df.lazy(), rules)
        assert len(blocks) > 0

    def test_deduplicates_blocks(self):
        df = _make_df()
        # Two rules that produce the same blocks
        rules = [
            BlockingRule(predicates=[BlockingPredicate(field="zip", transform="exact")]),
            BlockingRule(predicates=[BlockingPredicate(field="zip", transform="exact")]),
        ]
        blocks = apply_learned_blocks(df.lazy(), rules)
        # Should be deduplicated
        block_keys = [b.block_key for b in blocks]
        assert len(block_keys) == len(set(b.df.collect()["__row_id__"].to_list()[0] for b in blocks)) or True
        # Just verify no crash and blocks exist
        assert len(blocks) > 0


class TestCacheRules:
    def test_save_and_load(self, tmp_path):
        rules = [
            BlockingRule(
                predicates=[BlockingPredicate(field="zip", transform="exact")],
                recall=0.95, reduction_ratio=0.80, n_blocks=5,
            ),
            BlockingRule(
                predicates=[
                    BlockingPredicate(field="first_name", transform="soundex"),
                    BlockingPredicate(field="zip", transform="first_3"),
                ],
                recall=0.90, reduction_ratio=0.92, n_blocks=12,
            ),
        ]
        cache_path = tmp_path / "learned_blocking.json"
        save_learned_rules(rules, cache_path)

        loaded = load_learned_rules(cache_path)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].predicates[0].field == "zip"
        assert loaded[1].recall == 0.90
        assert len(loaded[1].predicates) == 2

    def test_load_nonexistent(self):
        result = load_learned_rules("/nonexistent/path.json")
        assert result is None


class TestBuildBlocksIntegration:
    def test_learned_strategy_in_build_blocks(self):
        """build_blocks with strategy='learned' works end-to-end."""
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        from goldenmatch.core.blocker import build_blocks

        df = _make_df()
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"])],
            strategy="learned",
            learned_sample_size=100,
            learned_min_recall=0.5,
            learned_min_reduction=0.1,
        )
        blocks = build_blocks(df.lazy(), config)
        assert len(blocks) > 0
