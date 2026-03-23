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
