"""Tests for PPRL auto-configuration."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.pprl.autoconfig import (
    auto_configure_pprl,
    profile_for_pprl,
    PPRLAutoConfigResult,
)


@pytest.fixture
def person_df():
    return pl.DataFrame({
        "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
        "last_name": ["Smith", "Doe", "Jones", "Brown", "Wilson"],
        "middle_name": ["A", "B", "", "C", "D"],
        "zip_code": ["10001", "20002", "30003", "10001", "50005"],
        "birth_year": ["1990", "1985", "1970", "1995", "1980"],
        "gender_code": ["M", "F", "M", "F", "M"],
        "email": ["john@x.com", "jane@y.com", "bob@z.com", "alice@w.com", "charlie@v.com"],
        "res_street_address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St", "654 Maple Dr"],
    })


class TestProfileForPPRL:
    def test_detects_person_fields(self, person_df):
        profiles = profile_for_pprl(person_df)
        types = {p.column: p.field_type for p in profiles}
        assert types["first_name"] == "first_name"
        assert types["last_name"] == "last_name"
        assert types["zip_code"] == "zip"
        assert types["birth_year"] == "birth_year"
        assert types["gender_code"] == "gender"

    def test_name_fields_have_high_usefulness(self, person_df):
        profiles = profile_for_pprl(person_df)
        scores = {p.column: p.usefulness_score for p in profiles}
        # Names should be among the most useful
        assert scores["first_name"] > 0.5
        assert scores["last_name"] > 0.5

    def test_skips_internal_columns(self):
        df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "__source__": ["a", "b", "c"],
            "name": ["John", "Jane", "Bob"],
        })
        profiles = profile_for_pprl(df)
        cols = [p.column for p in profiles]
        assert "__row_id__" not in cols
        assert "__source__" not in cols
        assert "name" in cols


class TestAutoConfigurePPRL:
    def test_selects_person_fields(self, person_df):
        result = auto_configure_pprl(person_df)
        # Should select names + at least one identifier
        assert "first_name" in result.recommended_fields
        assert "last_name" in result.recommended_fields
        assert len(result.recommended_fields) >= 3

    def test_returns_valid_config(self, person_df):
        result = auto_configure_pprl(person_df)
        config = result.recommended_config
        assert config.fields == result.recommended_fields
        assert 0.70 <= config.threshold <= 0.95
        assert config.bloom_filter_size in (512, 1024, 2048)
        assert config.hash_functions >= 20
        assert config.ngram_size in (2, 3)

    def test_security_level_affects_params(self, person_df):
        standard = auto_configure_pprl(person_df, security_level="standard")
        paranoid = auto_configure_pprl(person_df, security_level="paranoid")
        assert paranoid.recommended_config.bloom_filter_size >= standard.recommended_config.bloom_filter_size

    def test_has_explanation(self, person_df):
        result = auto_configure_pprl(person_df)
        assert len(result.explanation) > 20
        assert "field" in result.explanation.lower() or "selected" in result.explanation.lower()

    def test_max_fields_respected(self, person_df):
        result = auto_configure_pprl(person_df, max_fields=2)
        assert len(result.recommended_fields) <= 2

    def test_handles_non_person_data(self):
        df = pl.DataFrame({
            "product_name": ["Widget A", "Gadget B", "Gizmo C"],
            "price": ["9.99", "19.99", "29.99"],
            "category": ["electronics", "tools", "toys"],
        })
        result = auto_configure_pprl(df)
        # Should still work, just with lower usefulness scores
        assert len(result.recommended_fields) >= 1

    def test_no_string_columns_raises(self):
        """DataFrame with no string columns raises ValueError."""
        df = pl.DataFrame({"age": [25, 30, 35], "score": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="No string columns"):
            auto_configure_pprl(df)

    def test_all_low_usefulness_falls_back(self):
        """When all fields have very low usefulness, falls back to top 3."""
        df = pl.DataFrame({
            "rec_id": [f"id_{i}" for i in range(50)],
            "code_id": [f"code_{i}" for i in range(50)],
        })
        result = auto_configure_pprl(df)
        # Should still produce fields even with low-usefulness columns
        assert len(result.recommended_fields) >= 1

    def test_long_fields_get_larger_bloom_filter(self):
        """Fields with long average length get larger BF parameters."""
        df = pl.DataFrame({
            "address": [f"{'x' * 30} Street {i}" for i in range(20)],
            "description": [f"Long description text {'y' * 30} number {i}" for i in range(20)],
        })
        result = auto_configure_pprl(df, security_level="standard")
        # Long fields should push toward larger BF
        assert result.recommended_config.bloom_filter_size >= 512


class TestProfileForPPRLEdgeCases:
    def test_all_null_column(self):
        """Column with all nulls is skipped."""
        df = pl.DataFrame({
            "name": ["John", "Jane", "Bob"],
            "empty": [None, None, None],
        }).cast({"empty": pl.Utf8})
        profiles = profile_for_pprl(df)
        cols = [p.column for p in profiles]
        assert "empty" not in cols
        assert "name" in cols

    def test_near_unique_column_penalized(self):
        """Near-unique columns (like IDs) get low usefulness."""
        n = 100
        df = pl.DataFrame({
            "person_id": [f"id_{i}" for i in range(n)],
            "first_name": [f"Name_{i % 20}" for i in range(n)],
        })
        profiles = profile_for_pprl(df)
        scores = {p.column: p.usefulness_score for p in profiles}
        # person_id is near-unique AND has _id penalty
        assert scores["person_id"] < scores["first_name"]

    def test_high_null_rate_penalized(self):
        """Fields with >50% nulls are penalized."""
        df = pl.DataFrame({
            "name": ["John", "Jane", "Bob", None, None, None, None, None, None, None],
            "zip": ["10001", "20002", "30003", "40004", "50005",
                     "60006", "70007", "80008", "90009", "00000"],
        })
        profiles = profile_for_pprl(df)
        scores = {p.column: p.usefulness_score for p in profiles}
        assert scores["name"] < scores["zip"]

    def test_id_suffix_penalty(self):
        """Columns with _id, reg_num, ncid, rec_id get penalized."""
        df = pl.DataFrame({
            "customer_id": ["a", "b", "c", "d", "e"],
            "name": ["John", "Jane", "Bob", "Alice", "Charlie"],
        })
        profiles = profile_for_pprl(df)
        scores = {p.column: p.usefulness_score for p in profiles}
        assert scores["customer_id"] < scores["name"]

    def test_skips_non_string_columns(self):
        """Non-Utf8 columns are skipped."""
        df = pl.DataFrame({
            "name": ["John", "Jane"],
            "age": [25, 30],
        })
        profiles = profile_for_pprl(df)
        cols = [p.column for p in profiles]
        assert "name" in cols
        assert "age" not in cols

    def test_email_field_detection(self, person_df):
        profiles = profile_for_pprl(person_df)
        types = {p.column: p.field_type for p in profiles}
        assert types["email"] == "email"

    def test_address_field_detection(self, person_df):
        profiles = profile_for_pprl(person_df)
        types = {p.column: p.field_type for p in profiles}
        assert types["res_street_address"] == "address"


class TestEstimateThreshold:
    def test_basic_threshold_in_range(self, person_df):
        from goldenmatch.pprl.autoconfig import _estimate_threshold
        t = _estimate_threshold(person_df, ["first_name", "last_name"], 2, 30, 1024)
        assert 0.85 <= t <= 0.95

    def test_threshold_with_small_sample(self):
        """Works with very small datasets."""
        from goldenmatch.pprl.autoconfig import _estimate_threshold
        df = pl.DataFrame({
            "name": ["John", "Jane"],
        })
        t = _estimate_threshold(df, ["name"], 2, 20, 512, sample_size=2)
        assert 0.85 <= t <= 0.95


class TestAutoConfigurePPRLLlm:
    def test_no_api_key_falls_back_to_baseline(self, person_df):
        """Without API key, returns baseline result."""
        from goldenmatch.pprl.autoconfig import auto_configure_pprl_llm
        import os
        # Ensure no key is in env
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            result = auto_configure_pprl_llm(person_df, api_key=None)
            assert isinstance(result, PPRLAutoConfigResult)
            assert len(result.recommended_fields) >= 1
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_llm_call_mocked(self, person_df):
        """Mock the LLM call and verify JSON parsing."""
        from goldenmatch.pprl.autoconfig import auto_configure_pprl_llm
        from unittest.mock import patch

        mock_response = (
            '{"fields": ["first_name", "last_name", "zip_code"], '
            '"threshold": 0.88, "reasoning": "Name fields are best for linkage"}',
            100, 50,
        )
        with patch("goldenmatch.core.llm_scorer._call_openai", return_value=mock_response):
            result = auto_configure_pprl_llm(person_df, api_key="fake-key")
        assert result.recommended_fields == ["first_name", "last_name", "zip_code"]
        assert result.recommended_config.threshold == 0.88
        assert "LLM" in result.explanation

    def test_llm_call_returns_json_in_markdown(self, person_df):
        """LLM response wrapped in markdown code block."""
        from goldenmatch.pprl.autoconfig import auto_configure_pprl_llm
        from unittest.mock import patch

        mock_response = (
            '```json\n{"fields": ["first_name", "last_name"], '
            '"threshold": 0.90, "reasoning": "test"}\n```',
            80, 40,
        )
        with patch("goldenmatch.core.llm_scorer._call_openai", return_value=mock_response):
            result = auto_configure_pprl_llm(person_df, api_key="fake-key")
        assert result.recommended_fields == ["first_name", "last_name"]

    def test_llm_call_fails_gracefully(self, person_df):
        """LLM failure falls back to baseline."""
        from goldenmatch.pprl.autoconfig import auto_configure_pprl_llm
        from unittest.mock import patch

        with patch("goldenmatch.core.llm_scorer._call_openai", side_effect=Exception("API error")):
            result = auto_configure_pprl_llm(person_df, api_key="fake-key")
        assert isinstance(result, PPRLAutoConfigResult)
        assert len(result.recommended_fields) >= 1

    def test_llm_returns_invalid_fields(self, person_df):
        """LLM returns field names that don't exist in the DataFrame."""
        from goldenmatch.pprl.autoconfig import auto_configure_pprl_llm
        from unittest.mock import patch

        mock_response = (
            '{"fields": ["nonexistent_field_1", "nonexistent_field_2"], '
            '"threshold": 0.88, "reasoning": "bad fields"}',
            100, 50,
        )
        with patch("goldenmatch.core.llm_scorer._call_openai", return_value=mock_response):
            result = auto_configure_pprl_llm(person_df, api_key="fake-key")
        # Invalid fields filtered out -> falls back to baseline
        assert isinstance(result, PPRLAutoConfigResult)
        assert len(result.recommended_fields) >= 1
