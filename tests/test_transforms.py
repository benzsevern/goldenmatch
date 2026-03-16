"""Tests for goldenmatch field transforms."""

import pytest

from goldenmatch.utils.transforms import apply_transform, apply_transforms


class TestApplyTransform:
    """Tests for the apply_transform function."""

    def test_none_passthrough(self):
        assert apply_transform(None, "lowercase") is None

    def test_lowercase(self):
        assert apply_transform("HELLO", "lowercase") == "hello"

    def test_uppercase(self):
        assert apply_transform("hello", "uppercase") == "HELLO"

    def test_strip(self):
        assert apply_transform("  hello  ", "strip") == "hello"

    def test_strip_all(self):
        assert apply_transform("  h e l l o  ", "strip_all") == "hello"

    def test_substring(self):
        assert apply_transform("hello world", "substring:0:5") == "hello"

    def test_substring_from_middle(self):
        assert apply_transform("hello world", "substring:6:11") == "world"

    def test_substring_open_end(self):
        # substring:3: should work like [3:]
        assert apply_transform("hello", "substring:3:5") == "lo"

    def test_soundex(self):
        result = apply_transform("Robert", "soundex")
        assert isinstance(result, str)
        assert len(result) == 4  # soundex codes are 4 chars
        assert result == "R163"

    def test_metaphone(self):
        result = apply_transform("Robert", "metaphone")
        assert isinstance(result, str)
        assert result == "RBRT"

    def test_digits_only(self):
        assert apply_transform("abc123def456", "digits_only") == "123456"

    def test_digits_only_no_digits(self):
        assert apply_transform("abcdef", "digits_only") == ""

    def test_alpha_only(self):
        assert apply_transform("abc123def456", "alpha_only") == "abcdef"

    def test_alpha_only_no_alpha(self):
        assert apply_transform("123456", "alpha_only") == ""

    def test_normalize_whitespace(self):
        assert apply_transform("hello   world   foo", "normalize_whitespace") == "hello world foo"

    def test_normalize_whitespace_tabs_and_newlines(self):
        assert apply_transform("hello\t\t world\n\nfoo", "normalize_whitespace") == "hello world foo"

    def test_invalid_transform(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_transform("hello", "nonexistent_transform")


class TestApplyTransforms:
    """Tests for the apply_transforms function."""

    def test_none_passthrough(self):
        assert apply_transforms(None, ["lowercase", "strip"]) is None

    def test_empty_list(self):
        assert apply_transforms("Hello", []) == "Hello"

    def test_single_transform(self):
        assert apply_transforms("HELLO", ["lowercase"]) == "hello"

    def test_chaining(self):
        result = apply_transforms("  HELLO WORLD  ", ["strip", "lowercase"])
        assert result == "hello world"

    def test_chaining_complex(self):
        result = apply_transforms("  John Smith  ", ["strip", "lowercase", "substring:0:4"])
        assert result == "john"

    def test_chaining_with_digits(self):
        result = apply_transforms("Phone: (267) 555-1234", ["digits_only"])
        assert result == "2675551234"

    def test_invalid_in_chain(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            apply_transforms("hello", ["lowercase", "bad_transform"])
