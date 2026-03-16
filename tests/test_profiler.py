"""Tests for the data quality profiler."""

import polars as pl
import pytest

from goldenmatch.core.profiler import profile_column, profile_dataframe


@pytest.fixture
def messy_df():
    return pl.DataFrame({
        "email": ["john@test.com", "JANE@TEST.COM", "not-an-email", None, "", "bob@example.com", "john@test.com", None, "  ", "alice@test.co.uk"],
        "phone": ["267-555-1234", "(267) 555-1234", "abc", None, "555.123.4567", "12675551234", "", "267 555 1234", None, "310-555-0000"],
        "name": ["John Smith", "JANE DOE", "bob", None, "  ", "Alice Wonder-Land", "john smith", "J", None, "Mary-Jane Watson"],
        "zip": ["19382", "19382-1234", "abc", "1234", None, "90210", "10001", "193", "", "30301-5555"],
        "junk_col": [None, None, None, None, None, None, None, None, None, None],
        "low_card": ["A", "B", "A", "A", "B", "A", "A", "B", "A", "A"],
    })


class TestProfileColumn:
    def test_email_detection(self, messy_df):
        result = profile_column(messy_df["email"])
        assert result["suspected_type"] == "email"
        assert result["null_count"] == 2
        assert result["empty_string_count"] >= 2  # "" and "  "

    def test_phone_detection(self, messy_df):
        result = profile_column(messy_df["phone"])
        assert result["suspected_type"] == "phone"

    def test_all_null_column(self, messy_df):
        result = profile_column(messy_df["junk_col"])
        assert result["null_rate"] == 1.0

    def test_low_cardinality(self, messy_df):
        result = profile_column(messy_df["low_card"])
        assert result["unique_count"] == 2


class TestProfileDataframe:
    def test_issues_detected(self, messy_df):
        profile = profile_dataframe(messy_df)
        assert profile["total_rows"] == 10
        assert profile["total_columns"] == 6
        # Should detect junk_col as error (100% null)
        error_issues = [i for i in profile["issues"] if i["severity"] == "error"]
        assert any("junk_col" in i.get("column", "") for i in error_issues)

    def test_duplicate_detection(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profile = profile_dataframe(df)
        assert profile["duplicate_row_count"] == 1

    def test_empty_row_detection(self):
        df = pl.DataFrame({"a": ["hello", None, "world"], "b": ["x", None, "z"]})
        profile = profile_dataframe(df)
        assert profile["empty_row_count"] == 1

    def test_clean_data_no_errors(self):
        df = pl.DataFrame({
            "email": ["a@b.com", "c@d.com"],
            "name": ["John", "Jane"],
        })
        profile = profile_dataframe(df)
        errors = [i for i in profile["issues"] if i["severity"] == "error"]
        assert len(errors) == 0
