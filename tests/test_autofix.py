"""Tests for the auto-fix module."""

from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.core.autofix import auto_fix_dataframe


class TestBOMRemoval:
    def test_strips_bom_from_first_cell(self):
        df = pl.DataFrame({"name": ["\ufeffAlice", "Bob"], "age": ["30", "25"]})
        fixed, fixes = auto_fix_dataframe(df)
        assert fixed["name"][0] == "Alice"
        assert any(f["fix"] == "strip_bom" for f in fixes)

    def test_strips_bom_from_other_cells(self):
        df = pl.DataFrame({"a": ["ok", "\ufeffweird"], "b": ["\ufeffx", "y"]})
        fixed, _ = auto_fix_dataframe(df)
        assert "\ufeff" not in fixed["a"][1]
        assert "\ufeff" not in fixed["b"][0]


class TestDropEmptyRows:
    def test_drops_all_null_row(self):
        df = pl.DataFrame({"a": ["x", None, "z"], "b": ["1", None, "3"]})
        fixed, fixes = auto_fix_dataframe(df)
        assert fixed.height == 2
        assert any(f["fix"] == "drop_empty_rows" for f in fixes)

    def test_drops_whitespace_only_row(self):
        df = pl.DataFrame({"a": ["x", "  ", "z"], "b": ["1", "  ", "3"]})
        fixed, fixes = auto_fix_dataframe(df)
        assert fixed.height == 2

    def test_keeps_row_with_partial_data(self):
        df = pl.DataFrame({"a": ["x", None, "z"], "b": ["1", "data", "3"]})
        fixed, _ = auto_fix_dataframe(df)
        assert fixed.height == 3


class TestDropNullColumns:
    def test_drops_fully_null_column(self):
        df = pl.DataFrame({
            "good": ["a", "b", "c"],
            "bad": [None, None, None],
        })
        fixed, fixes = auto_fix_dataframe(df)
        assert "bad" not in fixed.columns
        assert any(f["fix"] == "drop_null_columns" for f in fixes)

    def test_keeps_partially_null_column(self):
        df = pl.DataFrame({
            "good": ["a", "b", "c"],
            "partial": [None, "x", None],
        })
        fixed, _ = auto_fix_dataframe(df)
        assert "partial" in fixed.columns

    def test_uses_profile_info(self):
        df = pl.DataFrame({
            "good": ["a", "b"],
            "bad": [None, None],
        })
        profile = {
            "columns": [
                {"name": "good", "null_rate": 0.0},
                {"name": "bad", "null_rate": 1.0},
            ]
        }
        fixed, fixes = auto_fix_dataframe(df, profile=profile)
        assert "bad" not in fixed.columns


class TestTrimWhitespace:
    def test_strips_leading_trailing(self):
        df = pl.DataFrame({"name": ["  Alice  ", "Bob  "]})
        fixed, fixes = auto_fix_dataframe(df)
        assert fixed["name"][0] == "Alice"
        assert fixed["name"][1] == "Bob"
        assert any(f["fix"] == "trim_whitespace" for f in fixes)


class TestNullNormalization:
    def test_converts_null_strings(self):
        null_variants = ["NULL", "N/A", "NA", "n/a", "None", "none", "", "-", ".", "nan", "NaN"]
        # Add a second column with real data so empty-string row is not dropped as empty
        df = pl.DataFrame({
            "val": null_variants + ["real_data"],
            "keep": ["x"] * (len(null_variants) + 1),
        })
        fixed, fixes = auto_fix_dataframe(df)
        # All null variants should become actual null
        for i in range(len(null_variants)):
            assert fixed["val"][i] is None, f"Expected null at index {i}, got {fixed['val'][i]}"
        assert fixed["val"][-1] == "real_data"
        assert any(f["fix"] == "normalize_nulls" for f in fixes)


class TestCollapseWhitespace:
    def test_collapses_multiple_spaces(self):
        df = pl.DataFrame({"name": ["John   Smith", "Jane  Doe"]})
        fixed, fixes = auto_fix_dataframe(df)
        assert fixed["name"][0] == "John Smith"
        assert fixed["name"][1] == "Jane Doe"
        assert any(f["fix"] == "collapse_whitespace" for f in fixes)


class TestNonPrintableChars:
    def test_removes_control_chars(self):
        df = pl.DataFrame({"name": ["Al\x00ice", "B\x07ob"]})
        fixed, fixes = auto_fix_dataframe(df)
        assert fixed["name"][0] == "Alice"
        assert fixed["name"][1] == "Bob"
        assert any(f["fix"] == "remove_non_printable" for f in fixes)

    def test_preserves_tabs_and_newlines(self):
        df = pl.DataFrame({"note": ["line1\nline2", "col1\tcol2"]})
        fixed, _ = auto_fix_dataframe(df)
        assert "\n" in fixed["note"][0]
        assert "\t" in fixed["note"][1]


class TestIdempotency:
    def test_running_twice_same_result(self):
        df = pl.DataFrame({
            "name": ["\ufeff  John   Smith\x00  ", "N/A", "  ", None, "Jane"],
            "val": ["1", "2", None, None, "5"],
        })
        fixed1, _ = auto_fix_dataframe(df)
        fixed2, fixes2 = auto_fix_dataframe(fixed1)
        assert fixed1.equals(fixed2)
        # Second run should report 0 rows affected for most fixes
        total_affected = sum(f["rows_affected"] for f in fixes2)
        assert total_affected == 0


class TestFixOrder:
    def test_all_fixes_applied(self):
        df = pl.DataFrame({
            "name": ["\ufeff  John   Smith\x00  ", "N/A", "  real  data  "],
            "empty_col": [None, None, None],
        })
        fixed, fixes = auto_fix_dataframe(df)
        fix_names = [f["fix"] for f in fixes]
        # All fix types should appear
        assert "strip_bom" in fix_names
        assert "drop_null_columns" in fix_names
        assert "trim_whitespace" in fix_names
        assert "normalize_nulls" in fix_names
        assert "collapse_whitespace" in fix_names
        assert "remove_non_printable" in fix_names
