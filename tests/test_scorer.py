"""Tests for goldenmatch scorer."""

import polars as pl

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
