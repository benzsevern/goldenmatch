"""Tests for noisy-data blocking transforms: token_sort, qgram, first_token, last_token."""

import pytest

from goldenmatch.utils.transforms import apply_transform, apply_transforms


class TestTokenSort:
    def test_basic(self):
        assert apply_transform("Machine Learning Approach", "token_sort") == "Approach Learning Machine"

    def test_already_sorted(self):
        assert apply_transform("alpha beta gamma", "token_sort") == "alpha beta gamma"

    def test_single_token(self):
        assert apply_transform("hello", "token_sort") == "hello"

    def test_null_passthrough(self):
        assert apply_transform(None, "token_sort") is None

    def test_extra_whitespace(self):
        assert apply_transform("  foo   bar  ", "token_sort") == "bar foo"


class TestQgram:
    def test_qgram_3(self):
        result = apply_transform("smith", "qgram:3")
        # padded = "##smith##", 3-grams: ##s, #sm, smi, mit, ith, th#, h##
        # sorted unique: ##s, #sm, h##, ith, mit, smi, th#
        # first 5: ##s, #sm, h##, ith, mit
        assert result == "##s #sm h## ith mit"

    def test_qgram_2(self):
        result = apply_transform("ab", "qgram:2")
        # padded = "##ab##", 2-grams: ##, #a, ab, b#, ##
        # sorted unique: ##, #a, ab, b#
        # first 5 (only 4): ##, #a, ab, b#
        assert result == "## #a ab b#"

    def test_null_passthrough(self):
        assert apply_transform(None, "qgram:3") is None

    def test_short_string(self):
        result = apply_transform("a", "qgram:3")
        # padded = "##a##", 3-grams: ##a, #a#, a##
        # sorted: ##a, #a#, a##
        assert result == "##a #a# a##"


class TestFirstToken:
    def test_basic(self):
        assert apply_transform("John Smith", "first_token") == "John"

    def test_single_word(self):
        assert apply_transform("John", "first_token") == "John"

    def test_null_passthrough(self):
        assert apply_transform(None, "first_token") is None

    def test_extra_whitespace(self):
        assert apply_transform("  John   Smith  ", "first_token") == "John"


class TestLastToken:
    def test_basic(self):
        assert apply_transform("John Smith", "last_token") == "Smith"

    def test_single_word(self):
        assert apply_transform("John", "last_token") == "John"

    def test_null_passthrough(self):
        assert apply_transform(None, "last_token") is None

    def test_extra_whitespace(self):
        assert apply_transform("  John   Smith  ", "last_token") == "Smith"


class TestChainingWithNewTransforms:
    def test_token_sort_then_substring(self):
        result = apply_transforms("Machine Learning Approach", ["token_sort", "substring:0:10"])
        assert result == "Approach L"

    def test_lowercase_then_first_token(self):
        result = apply_transforms("John Smith", ["lowercase", "first_token"])
        assert result == "john"

    def test_lowercase_then_last_token_then_soundex(self):
        result = apply_transforms("John Smith", ["lowercase", "last_token", "soundex"])
        assert isinstance(result, str)
        assert len(result) == 4

    def test_lowercase_then_qgram(self):
        result = apply_transforms("Smith", ["lowercase", "qgram:3"])
        # lowercase -> "smith", then qgram:3 on "smith"
        assert result == "##s #sm h## ith mit"
