"""Tests for the config wizard heuristic suggestions."""

from __future__ import annotations


from goldenmatch.config.wizard import suggest_transforms, suggest_scorer


class TestSuggestTransforms:
    def test_name_field(self):
        result = suggest_transforms("first_name")
        assert "lowercase" in result
        assert "strip" in result

    def test_last_name(self):
        result = suggest_transforms("last_name")
        assert "lowercase" in result
        assert "normalize_whitespace" in result

    def test_email_field(self):
        result = suggest_transforms("email")
        assert "lowercase" in result
        assert "strip" in result
        assert "normalize_whitespace" not in result

    def test_phone_field(self):
        result = suggest_transforms("phone")
        assert "digits_only" in result

    def test_mobile(self):
        result = suggest_transforms("mobile_number")
        assert "digits_only" in result

    def test_zip_field(self):
        result = suggest_transforms("zip")
        assert "strip" in result
        assert any("substring" in t for t in result)

    def test_zip_code(self):
        result = suggest_transforms("zip_code")
        assert "strip" in result

    def test_postal_code(self):
        result = suggest_transforms("postal_code")
        assert "strip" in result
        assert any("substring" in t for t in result)

    def test_address_field(self):
        result = suggest_transforms("address")
        assert "lowercase" in result
        assert "normalize_whitespace" in result

    def test_street(self):
        result = suggest_transforms("street_address")
        assert "lowercase" in result

    def test_unknown_field(self):
        result = suggest_transforms("random_field_xyz")
        assert "strip" in result

    def test_case_insensitive(self):
        result = suggest_transforms("EMAIL_ADDRESS")
        assert "lowercase" in result


class TestSuggestScorer:
    def test_name_field(self):
        assert suggest_scorer("first_name") == "jaro_winkler"

    def test_last_name(self):
        assert suggest_scorer("last_name") == "jaro_winkler"

    def test_email_field(self):
        assert suggest_scorer("email") == "levenshtein"

    def test_phone_field(self):
        assert suggest_scorer("phone") == "exact"

    def test_zip_field(self):
        assert suggest_scorer("zip") == "exact"

    def test_zip_code(self):
        assert suggest_scorer("zip_code") == "exact"

    def test_address_field(self):
        assert suggest_scorer("address") == "token_sort"

    def test_street(self):
        assert suggest_scorer("street") == "token_sort"

    def test_unknown_field(self):
        assert suggest_scorer("random_field_xyz") == "jaro_winkler"

    def test_case_insensitive(self):
        assert suggest_scorer("PHONE_NUMBER") == "exact"
