"""Tests for cross-encoder and data augmentation."""

from __future__ import annotations

import pytest

from goldenmatch.core.cross_encoder import (
    _column_drop,
    _span_delete,
    _span_shuffle,
    augment_pair,
    augment_training_data,
    merge_scores,
    serialize_record,
)
import random


class TestSerializeRecord:
    def test_basic(self):
        row = {"name": "Sony Turntable", "price": "149.00"}
        result = serialize_record(row, ["name", "price"])
        assert result == "name: Sony Turntable | price: 149.00"

    def test_null_skipped(self):
        row = {"name": "Sony", "price": None, "brand": "Sony"}
        result = serialize_record(row, ["name", "price", "brand"])
        assert result == "name: Sony | brand: Sony"

    def test_empty(self):
        result = serialize_record({}, ["name"])
        assert result == ""

    def test_column_order(self):
        row = {"b": "2", "a": "1"}
        assert serialize_record(row, ["a", "b"]) == "a: 1 | b: 2"
        assert serialize_record(row, ["b", "a"]) == "b: 2 | a: 1"


class TestSpanDelete:
    def test_removes_tokens(self):
        rng = random.Random(42)
        result = _span_delete("one two three four five", rng)
        tokens = result.split()
        assert len(tokens) < 5
        assert len(tokens) >= 2

    def test_short_text_unchanged(self):
        rng = random.Random(42)
        result = _span_delete("ab", rng)
        assert result == "ab"


class TestSpanShuffle:
    def test_reorders_tokens(self):
        rng = random.Random(42)
        text = "one two three four five"
        result = _span_shuffle(text, rng)
        # Same tokens, possibly different order
        assert set(result.split()) == set(text.split())
        assert len(result.split()) == 5

    def test_short_text_unchanged(self):
        rng = random.Random(42)
        result = _span_shuffle("ab", rng)
        assert result == "ab"


class TestColumnDrop:
    def test_drops_column(self):
        rng = random.Random(42)
        text = "name: Sony | price: 149 | brand: Sony"
        result = _column_drop(text, rng)
        parts = [p.strip() for p in result.split("|")]
        assert len(parts) == 2

    def test_single_column_unchanged(self):
        rng = random.Random(42)
        result = _column_drop("name: Sony", rng)
        assert result == "name: Sony"


class TestAugmentPair:
    def test_produces_n_augments(self):
        result = augment_pair("text a here", "text b here", True, n_augments=3)
        assert len(result) == 3

    def test_labels_preserved(self):
        result = augment_pair("a b c", "d e f", False, n_augments=5)
        assert all(label is False for _, _, label in result)

    def test_text_modified(self):
        result = augment_pair(
            "name: Sony PSLX350H | price: 149",
            "name: Sony Turntable | price: 150",
            True, n_augments=10,
        )
        # At least some augmentations should differ from originals
        originals = {"name: Sony PSLX350H | price: 149", "name: Sony Turntable | price: 150"}
        modified = sum(1 for a, b, _ in result if a not in originals or b not in originals)
        assert modified > 0


class TestAugmentTrainingData:
    def test_expands_dataset(self):
        pairs = [
            ("text a", "text b", True),
            ("text c", "text d", False),
        ]
        result = augment_training_data(pairs, n_augments=3)
        # 2 original + 2*3 augmented = 8
        assert len(result) == 8

    def test_preserves_originals(self):
        pairs = [("a b c", "d e f", True)]
        result = augment_training_data(pairs, n_augments=2)
        assert result[0] == ("a b c", "d e f", True)


class TestMergeScores:
    def test_uses_cross_encoder_for_uncertain(self):
        bi_pairs = [(0, 1, 0.5), (2, 3, 0.9), (4, 5, 0.1)]
        cross_scores = {(0, 1): 0.85}

        result = merge_scores(bi_pairs, cross_scores)
        scores = {(a, b): s for a, b, s in result}

        assert scores[(0, 1)] == 0.85  # cross-encoder replaced
        assert scores[(2, 3)] == 0.9   # kept (high confidence)
        assert scores[(4, 5)] == 0.1   # kept (low confidence)

    def test_empty_cross_scores(self):
        bi_pairs = [(0, 1, 0.7)]
        result = merge_scores(bi_pairs, {})
        assert result == [(0, 1, 0.7)]

    def test_no_pairs(self):
        result = merge_scores([], {})
        assert result == []
