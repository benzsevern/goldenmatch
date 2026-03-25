"""Tests for goldenmatch scorer."""

import polars as pl
import pytest

from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
from goldenmatch.core.scorer import (
    score_field,
    score_pair,
    find_exact_matches,
    find_fuzzy_matches,
)


class TestScoreField:
    """Tests for score_field."""

    def test_exact_match(self):
        assert score_field("John", "John", "exact") == 1.0

    def test_exact_no_match(self):
        assert score_field("John", "Jane", "exact") == 0.0

    def test_jaro_winkler_identical(self):
        assert score_field("John", "John", "jaro_winkler") == 1.0

    def test_jaro_winkler_similar(self):
        score = score_field("John", "Jon", "jaro_winkler")
        assert 0.0 < score < 1.0

    def test_levenshtein_identical(self):
        assert score_field("John", "John", "levenshtein") == 1.0

    def test_levenshtein_similar(self):
        score = score_field("John", "Jon", "levenshtein")
        assert 0.0 < score < 1.0

    def test_token_sort_identical(self):
        assert score_field("John Smith", "John Smith", "token_sort") == 1.0

    def test_token_sort_reordered(self):
        score = score_field("John Smith", "Smith John", "token_sort")
        assert score == 1.0

    def test_soundex_match(self):
        # Robert and Rupert have same soundex (R163)
        assert score_field("Robert", "Rupert", "soundex_match") == 1.0

    def test_soundex_no_match(self):
        assert score_field("John", "Mary", "soundex_match") == 0.0

    def test_none_first(self):
        assert score_field(None, "John", "exact") is None

    def test_none_second(self):
        assert score_field("John", None, "exact") is None

    def test_both_none(self):
        assert score_field(None, None, "exact") is None


class TestScorePair:
    """Tests for score_pair."""

    def test_weighted_scoring(self):
        """Weighted average of field scores."""
        fields = [
            MatchkeyField(field="first_name", transforms=[], scorer="exact", weight=2.0),
            MatchkeyField(field="last_name", transforms=[], scorer="exact", weight=1.0),
        ]
        row_a = {"first_name": "John", "last_name": "Smith"}
        row_b = {"first_name": "John", "last_name": "Jones"}
        # first_name exact match = 1.0 * 2.0, last_name no match = 0.0 * 1.0
        # score = 2.0 / 3.0
        score = score_pair(row_a, row_b, fields)
        assert abs(score - 2.0 / 3.0) < 1e-9

    def test_partial_match(self):
        """Partial fuzzy match with weights."""
        fields = [
            MatchkeyField(field="first_name", transforms=["lowercase"], scorer="jaro_winkler", weight=1.0),
            MatchkeyField(field="last_name", transforms=["lowercase"], scorer="exact", weight=1.0),
        ]
        row_a = {"first_name": "John", "last_name": "Smith"}
        row_b = {"first_name": "Jon", "last_name": "Smith"}
        score = score_pair(row_a, row_b, fields)
        # last_name exact = 1.0, first_name jaro_winkler > 0
        assert 0.5 < score <= 1.0

    def test_null_excluded_from_score(self):
        """Null fields are excluded from weighted average."""
        fields = [
            MatchkeyField(field="first_name", transforms=[], scorer="exact", weight=1.0),
            MatchkeyField(field="last_name", transforms=[], scorer="exact", weight=1.0),
        ]
        row_a = {"first_name": "John", "last_name": None}
        row_b = {"first_name": "John", "last_name": "Smith"}
        # Only first_name contributes: 1.0 * 1.0 / 1.0 = 1.0
        score = score_pair(row_a, row_b, fields)
        assert score == 1.0

    def test_all_null_returns_zero(self):
        """All null fields return 0.0."""
        fields = [
            MatchkeyField(field="first_name", transforms=[], scorer="exact", weight=1.0),
        ]
        row_a = {"first_name": None}
        row_b = {"first_name": None}
        score = score_pair(row_a, row_b, fields)
        assert score == 0.0


class TestFindExactMatches:
    """Tests for find_exact_matches."""

    def test_finds_exact_duplicates(self):
        """Groups by matchkey column and returns pairs with score 1.0."""
        mk = MatchkeyConfig(name="name", type="exact", fields=[
            MatchkeyField(field="first_name"),
        ])
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "first_name": ["John", "Jane", "John", "John"],
            "__mk_name__": ["john", "jane", "john", "john"],
        })
        lf = df.lazy()
        matches = find_exact_matches(lf, mk)
        # Row IDs 0, 2, 3 share the same matchkey "john" -> 3 pairs
        assert len(matches) == 3
        # All scores should be 1.0
        assert all(score == 1.0 for _, _, score in matches)
        # Check pairs are present (order of IDs in tuple: smaller first)
        pair_ids = {(min(a, b), max(a, b)) for a, b, _ in matches}
        assert pair_ids == {(0, 2), (0, 3), (2, 3)}

    def test_no_duplicates(self):
        """No matches when all matchkeys are unique."""
        mk = MatchkeyConfig(name="name", type="exact", fields=[
            MatchkeyField(field="first_name"),
        ])
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "first_name": ["John", "Jane", "Bob"],
            "__mk_name__": ["john", "jane", "bob"],
        })
        lf = df.lazy()
        matches = find_exact_matches(lf, mk)
        assert matches == []


class TestFindFuzzyMatches:
    """Tests for find_fuzzy_matches."""

    def test_finds_fuzzy_matches_above_threshold(self):
        """Pairs above the threshold are returned."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="first_name", transforms=["lowercase"], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "first_name": ["John", "Jon"],
        })
        matches = find_fuzzy_matches(df, mk)
        assert len(matches) == 1
        a, b, score = matches[0]
        assert {a, b} == {0, 1}
        assert score >= 0.5

    def test_filters_below_threshold(self):
        """Pairs below the threshold are excluded."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.99,
            fields=[
                MatchkeyField(field="first_name", transforms=[], scorer="exact", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "first_name": ["John", "Jane"],
        })
        matches = find_fuzzy_matches(df, mk)
        assert matches == []

    def test_empty_block(self):
        """Empty DataFrame returns no matches."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="first_name", transforms=[], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": pl.Series([], dtype=pl.Int64),
            "first_name": pl.Series([], dtype=pl.Utf8),
        })
        matches = find_fuzzy_matches(df, mk)
        assert matches == []

    def test_single_record_block(self):
        """Single-record block returns no matches."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="first_name", transforms=[], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0],
            "first_name": ["John"],
        })
        matches = find_fuzzy_matches(df, mk)
        assert matches == []

    def test_exclude_pairs(self):
        """Excluded pairs are not returned."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="first_name", transforms=[], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "first_name": ["John", "Jon", "John"],
        })
        all_matches = find_fuzzy_matches(df, mk)
        excluded = find_fuzzy_matches(df, mk, exclude_pairs={(0, 1)})
        assert len(excluded) < len(all_matches) or len(all_matches) == 0

    def test_pre_scored_pairs(self):
        """Pre-scored pairs bypass NxN scoring."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.7,
            fields=[
                MatchkeyField(field="first_name", transforms=[], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "first_name": ["John", "Jon", "Jane"],
        })
        pre_scored = [(0, 1, 0.9), (0, 2, 0.5), (1, 2, 0.3)]
        matches = find_fuzzy_matches(df, mk, pre_scored_pairs=pre_scored)
        # Only (0,1) with score 0.9 should pass the 0.7 threshold
        assert len(matches) == 1
        assert matches[0][2] == 0.9

    def test_pre_scored_pairs_with_exclude(self):
        """Pre-scored pairs respect exclude_pairs."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="first_name", transforms=[], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "first_name": ["John", "Jon"],
        })
        pre_scored = [(0, 1, 0.9)]
        matches = find_fuzzy_matches(df, mk, pre_scored_pairs=pre_scored, exclude_pairs={(0, 1)})
        assert matches == []

    def test_zero_total_weight(self):
        """Fields with zero weight return no matches."""
        mk = MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="first_name", transforms=[], scorer="jaro_winkler", weight=0.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "first_name": ["John", "John"],
        })
        matches = find_fuzzy_matches(df, mk)
        assert matches == []

    def test_mixed_exact_and_fuzzy_fields(self):
        """Early termination with mixed exact and fuzzy fields."""
        mk = MatchkeyConfig(
            name="mixed",
            type="weighted",
            threshold=0.8,
            fields=[
                MatchkeyField(field="zip", transforms=[], scorer="exact", weight=1.0),
                MatchkeyField(field="name", transforms=["lowercase"], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "zip": ["10001", "10001"],
            "name": ["John Smith", "Jon Smyth"],
        })
        matches = find_fuzzy_matches(df, mk)
        # zip exact=1.0, name fuzzy~0.85 -> avg~0.93 -> above 0.8
        assert len(matches) == 1
        assert matches[0][2] > 0.8

    def test_early_termination_impossible_pairs(self):
        """Pairs where exact fields disagree and fuzzy can't compensate."""
        mk = MatchkeyConfig(
            name="term",
            type="weighted",
            threshold=0.95,
            fields=[
                MatchkeyField(field="zip", transforms=[], scorer="exact", weight=3.0),
                MatchkeyField(field="name", transforms=[], scorer="jaro_winkler", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "zip": ["10001", "99999"],
            "name": ["John", "Jon"],
        })
        # zip disagree (0.0*3) + name max (1.0*1) = 1/4 = 0.25 < 0.95
        matches = find_fuzzy_matches(df, mk)
        assert matches == []

    def test_only_exact_fields_no_fuzzy(self):
        """Fuzzy matching with only exact-type fields."""
        mk = MatchkeyConfig(
            name="exact_only",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="email", transforms=["lowercase"], scorer="exact", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "email": ["a@x.com", "A@x.com", "b@y.com"],
        })
        matches = find_fuzzy_matches(df, mk)
        assert len(matches) == 1
        assert matches[0][2] == 1.0

    def test_soundex_field_in_fuzzy(self):
        """Soundex scorer works in fuzzy matching."""
        mk = MatchkeyConfig(
            name="soundex",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="name", transforms=[], scorer="soundex_match", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["Robert", "Rupert", "Alice"],
        })
        matches = find_fuzzy_matches(df, mk)
        # Robert/Rupert same soundex -> 1.0
        pair_ids = {(min(a, b), max(a, b)) for a, b, _ in matches}
        assert (0, 1) in pair_ids

    def test_levenshtein_fuzzy_scoring(self):
        """Levenshtein scorer in fuzzy matching."""
        mk = MatchkeyConfig(
            name="lev",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="name", transforms=[], scorer="levenshtein", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["kitten", "sitting"],
        })
        matches = find_fuzzy_matches(df, mk)
        # levenshtein similarity > 0.5 for kitten/sitting
        assert len(matches) == 1

    def test_token_sort_fuzzy_scoring(self):
        """Token sort scorer in fuzzy matching."""
        mk = MatchkeyConfig(
            name="ts",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="name", transforms=[], scorer="token_sort", weight=1.0),
            ],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John Smith", "Smith John"],
        })
        matches = find_fuzzy_matches(df, mk)
        assert len(matches) == 1
        assert matches[0][2] == 1.0


# ---------------------------------------------------------------------------
# Dice / Jaccard score_field tests
# ---------------------------------------------------------------------------

class TestDiceJaccardScoreField:
    """Tests for score_field with dice and jaccard scorers (bloom filter hex)."""

    def _make_bloom_hex(self, val: int, size: int = 4) -> str:
        """Create a hex string from an integer for testing."""
        return val.to_bytes(size, byteorder="big").hex()

    def test_dice_identical(self):
        bf = self._make_bloom_hex(0xFF)
        score = score_field(bf, bf, "dice")
        assert score == 1.0

    def test_dice_disjoint(self):
        a = self._make_bloom_hex(0xF0)
        b = self._make_bloom_hex(0x0F)
        score = score_field(a, b, "dice")
        assert score == 0.0

    def test_dice_partial_overlap(self):
        a = self._make_bloom_hex(0xFF)
        b = self._make_bloom_hex(0x0F)
        score = score_field(a, b, "dice")
        assert 0.0 < score < 1.0

    def test_jaccard_identical(self):
        bf = self._make_bloom_hex(0xFF)
        score = score_field(bf, bf, "jaccard")
        assert score == 1.0

    def test_jaccard_disjoint(self):
        a = self._make_bloom_hex(0xF0)
        b = self._make_bloom_hex(0x0F)
        score = score_field(a, b, "jaccard")
        assert score == 0.0

    def test_jaccard_partial_overlap(self):
        a = self._make_bloom_hex(0xFF)
        b = self._make_bloom_hex(0x0F)
        score = score_field(a, b, "jaccard")
        assert 0.0 < score < 1.0

    def test_dice_none_input(self):
        bf = "ff"
        assert score_field(None, bf, "dice") is None
        assert score_field(bf, None, "dice") is None

    def test_jaccard_none_input(self):
        bf = "ff"
        assert score_field(None, bf, "jaccard") is None
        assert score_field(bf, None, "jaccard") is None

    def test_unknown_scorer_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            score_field("a", "b", "nonexistent_scorer_12345")


# ---------------------------------------------------------------------------
# Internal matrix helpers
# ---------------------------------------------------------------------------

import numpy as np
from goldenmatch.core.scorer import (
    _exact_score_matrix,
    _soundex_score_matrix,
    _dice_score_matrix,
    _jaccard_score_matrix,
    _build_null_mask,
    _fuzzy_score_matrix,
)


class TestExactScoreMatrix:
    def test_basic(self):
        values = ["a", "b", "a", "c"]
        mat = _exact_score_matrix(values)
        assert mat.shape == (4, 4)
        assert mat[0, 2] == 1.0
        assert mat[2, 0] == 1.0
        assert mat[0, 1] == 0.0

    def test_with_nones(self):
        values = ["a", None, "a"]
        mat = _exact_score_matrix(values)
        assert mat[0, 2] == 1.0
        assert mat[0, 1] == 0.0  # None doesn't match anything


class TestSoundexScoreMatrix:
    def test_same_soundex(self):
        values = ["Robert", "Rupert", "Alice"]
        mat = _soundex_score_matrix(values)
        assert mat[0, 1] == 1.0  # same soundex
        assert mat[0, 2] == 0.0  # different soundex


class TestDiceScoreMatrix:
    def test_identity(self):
        bf = bytes([0xFF]).hex()
        mat = _dice_score_matrix([bf, bf])
        assert mat[0, 1] == pytest.approx(1.0)

    def test_with_none(self):
        bf = bytes([0xFF]).hex()
        mat = _dice_score_matrix([bf, None])
        assert mat.shape == (2, 2)

    def test_empty_list(self):
        mat = _dice_score_matrix([])
        assert mat.shape == (0, 0)


class TestJaccardScoreMatrix:
    def test_identity(self):
        bf = bytes([0xFF]).hex()
        mat = _jaccard_score_matrix([bf, bf])
        assert mat[0, 1] == pytest.approx(1.0)


class TestBuildNullMask:
    def test_no_nulls(self):
        mask = _build_null_mask(["a", "b"])
        assert not mask.any()

    def test_with_null(self):
        mask = _build_null_mask(["a", None])
        assert mask[0, 1] == True
        assert mask[1, 0] == True
        assert mask[0, 0] == False
        assert mask[1, 1] == True


class TestFuzzyScoreMatrix:
    def test_jaro_winkler_matrix(self):
        mat = _fuzzy_score_matrix(["John", "Jon", "Jane"], "jaro_winkler")
        assert mat.shape == (3, 3)
        assert mat[0, 0] == pytest.approx(1.0)
        assert mat[0, 1] > 0.5

    def test_levenshtein_matrix(self):
        mat = _fuzzy_score_matrix(["kitten", "sitting"], "levenshtein")
        assert mat.shape == (2, 2)
        assert 0 < mat[0, 1] < 1

    def test_token_sort_matrix(self):
        mat = _fuzzy_score_matrix(["John Smith", "Smith John"], "token_sort")
        assert mat[0, 1] == pytest.approx(1.0)

    def test_dice_matrix_via_fuzzy(self):
        bf = bytes([0xFF]).hex()
        mat = _fuzzy_score_matrix([bf, bf], "dice")
        assert mat[0, 1] == pytest.approx(1.0)

    def test_jaccard_matrix_via_fuzzy(self):
        bf = bytes([0xFF]).hex()
        mat = _fuzzy_score_matrix([bf, bf], "jaccard")
        assert mat[0, 1] == pytest.approx(1.0)

    def test_unknown_scorer_raises(self):
        with pytest.raises(ValueError, match="Unknown fuzzy scorer"):
            _fuzzy_score_matrix(["a", "b"], "nonexistent_scorer_999")

    def test_with_none_values(self):
        mat = _fuzzy_score_matrix(["John", None, "Jon"], "jaro_winkler")
        assert mat.shape == (3, 3)
        # None replaced with "" -> low score vs real values
        assert mat[0, 2] > mat[0, 1]


# ---------------------------------------------------------------------------
# score_blocks_parallel tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
from goldenmatch.core.scorer import score_blocks_parallel
from goldenmatch.core.blocker import BlockResult


def _make_block(row_ids, names, block_key="k1"):
    """Helper to create a BlockResult for testing."""
    df = pl.DataFrame({
        "__row_id__": row_ids,
        "first_name": names,
    }).lazy()
    return BlockResult(block_key=block_key, df=df)


class TestScoreBlocksParallel:
    def _mk(self):
        return MatchkeyConfig(
            name="name",
            type="weighted",
            threshold=0.5,
            fields=[
                MatchkeyField(field="first_name", transforms=["lowercase"], scorer="jaro_winkler", weight=1.0),
            ],
        )

    def test_empty_blocks(self):
        pairs = score_blocks_parallel([], self._mk(), set())
        assert pairs == []

    def test_single_block_no_threading(self):
        """<=2 blocks skips thread overhead."""
        block = _make_block([0, 1], ["John", "Jon"])
        pairs = score_blocks_parallel([block], self._mk(), set())
        assert len(pairs) >= 1
        assert pairs[0][2] >= 0.5

    def test_two_blocks_no_threading(self):
        """Two blocks still skip thread overhead."""
        b1 = _make_block([0, 1], ["John", "Jon"], "k1")
        b2 = _make_block([2, 3], ["Jane", "Janet"], "k2")
        pairs = score_blocks_parallel([b1, b2], self._mk(), set())
        assert len(pairs) >= 2

    def test_three_blocks_uses_threading(self):
        """Three or more blocks uses thread pool."""
        b1 = _make_block([0, 1], ["John", "Jon"], "k1")
        b2 = _make_block([2, 3], ["Jane", "Janet"], "k2")
        b3 = _make_block([4, 5], ["Bob", "Bobby"], "k3")
        pairs = score_blocks_parallel([b1, b2, b3], self._mk(), set())
        assert len(pairs) >= 2

    def test_matched_pairs_updated(self):
        """matched_pairs set is updated with found pairs."""
        block = _make_block([0, 1], ["John", "Jon"])
        matched = set()
        score_blocks_parallel([block], self._mk(), matched)
        assert len(matched) >= 1

    def test_target_ids_filter(self):
        """target_ids filters to cross-source pairs."""
        block = _make_block([0, 1, 2], ["John", "Jon", "John"])
        # Only pairs where exactly one is in target_ids
        pairs = score_blocks_parallel(
            [block], self._mk(), set(), target_ids={0},
        )
        for a, b, s in pairs:
            assert (a in {0}) != (b in {0})

    def test_target_ids_filter_threaded(self):
        """target_ids works with threaded path (>2 blocks)."""
        b1 = _make_block([0, 1], ["John", "Jon"], "k1")
        b2 = _make_block([2, 3], ["Jane", "Janet"], "k2")
        b3 = _make_block([4, 5], ["Bob", "Bobby"], "k3")
        pairs = score_blocks_parallel(
            [b1, b2, b3], self._mk(), set(), target_ids={0, 2, 4},
        )
        for a, b, s in pairs:
            assert (a in {0, 2, 4}) != (b in {0, 2, 4})


# ---------------------------------------------------------------------------
# rerank_top_pairs tests
# ---------------------------------------------------------------------------

from goldenmatch.core.scorer import rerank_top_pairs


class TestRerankTopPairs:
    def _mk(self, rerank=False, threshold=0.8):
        return MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=threshold,
            rerank=rerank,
            rerank_band=0.1,
            fields=[
                MatchkeyField(field="name", transforms=[], scorer="jaro_winkler", weight=1.0),
            ],
        )

    def test_no_rerank_returns_unchanged(self):
        """When rerank=False, returns pairs unchanged."""
        mk = self._mk(rerank=False)
        pairs = [(0, 1, 0.85)]
        result = rerank_top_pairs(pairs, pl.DataFrame(), mk)
        assert result == pairs

    def test_empty_pairs(self):
        mk = self._mk(rerank=True)
        result = rerank_top_pairs([], pl.DataFrame(), mk)
        assert result == []

    def test_no_threshold(self):
        """When threshold is None, returns unchanged."""
        mk = MatchkeyConfig(
            name="test",
            type="exact",
            rerank=True,
            fields=[MatchkeyField(field="name", transforms=[])],
        )
        pairs = [(0, 1, 0.85)]
        result = rerank_top_pairs(pairs, pl.DataFrame(), mk)
        assert result == pairs

    def test_no_borderline_pairs(self):
        """When no pairs in the rerank band, returns unchanged."""
        mk = self._mk(rerank=True, threshold=0.8)
        # All scores far from threshold (0.8 +/- 0.1 = [0.7, 0.9])
        pairs = [(0, 1, 0.99), (2, 3, 0.50)]
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "name": ["John", "Jon", "Alice", "Bob"],
        })
        result = rerank_top_pairs(pairs, df, mk)
        assert result == pairs
