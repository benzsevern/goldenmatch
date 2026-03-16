"""Tests for goldenmatch file ingest."""

from pathlib import Path

import polars as pl
import pytest

from goldenmatch.core.ingest import load_file, load_files, validate_columns


class TestLoadFile:
    """Tests for the load_file function."""

    def test_load_csv(self, sample_csv):
        lf = load_file(sample_csv)
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 5
        assert "first_name" in df.columns

    def test_load_csv_with_delimiter(self, tmp_path):
        path = tmp_path / "tab.csv"
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        df.write_csv(path, separator="\t")
        lf = load_file(path, delimiter="\t")
        result = lf.collect()
        assert len(result) == 2
        assert result.columns == ["a", "b"]

    def test_load_parquet(self, sample_parquet):
        lf = load_file(sample_parquet)
        assert isinstance(lf, pl.LazyFrame)
        df = lf.collect()
        assert len(df) == 3
        assert "first_name" in df.columns

    def test_load_excel(self, tmp_path):
        path = tmp_path / "test.xlsx"
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        df.write_excel(path)
        lf = load_file(path)
        assert isinstance(lf, pl.LazyFrame)
        result = lf.collect()
        assert len(result) == 3
        assert "name" in result.columns

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_file(Path("/nonexistent/file.csv"))

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.json"
        path.write_text('{"a": 1}')
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(path)


class TestLoadFiles:
    """Tests for the load_files function."""

    def test_multi_file_loading(self, sample_csv, sample_csv_b):
        specs = [(sample_csv, "source_a"), (sample_csv_b, "source_b")]
        frames = load_files(specs)
        assert len(frames) == 2

        df_a = frames[0].collect()
        df_b = frames[1].collect()
        assert "__source__" in df_a.columns
        assert "__source__" in df_b.columns
        assert df_a["__source__"].unique().to_list() == ["source_a"]
        assert df_b["__source__"].unique().to_list() == ["source_b"]

    def test_single_file_loading(self, sample_csv):
        specs = [(sample_csv, "only_source")]
        frames = load_files(specs)
        assert len(frames) == 1
        df = frames[0].collect()
        assert df["__source__"].unique().to_list() == ["only_source"]


class TestValidateColumns:
    """Tests for the validate_columns function."""

    def test_valid_columns(self, sample_csv):
        lf = load_file(sample_csv)
        # Should not raise
        validate_columns(lf, ["id", "first_name", "last_name"])

    def test_missing_columns(self, sample_csv):
        lf = load_file(sample_csv)
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(lf, ["id", "nonexistent_column", "another_missing"])

    def test_missing_columns_lists_available(self, sample_csv):
        lf = load_file(sample_csv)
        with pytest.raises(ValueError, match="Available columns"):
            validate_columns(lf, ["nonexistent"])

    def test_empty_required(self, sample_csv):
        lf = load_file(sample_csv)
        # Should not raise with empty list
        validate_columns(lf, [])
