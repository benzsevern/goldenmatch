"""Tests for column mapping feature."""

import polars as pl
import pytest
from pathlib import Path

from goldenmatch.core.ingest import apply_column_map, suggest_column_mapping, load_file


class TestApplyColumnMap:
    def test_renames_columns(self, tmp_path):
        path = tmp_path / "test.csv"
        pl.DataFrame({"LNAME": ["Smith"], "FNAME": ["John"], "ZIP5": ["19382"]}).write_csv(path)
        lf = load_file(path)
        mapped = apply_column_map(lf, {"LNAME": "last_name", "FNAME": "first_name", "ZIP5": "zip"})
        result = mapped.collect()
        assert "last_name" in result.columns
        assert "first_name" in result.columns
        assert "zip" in result.columns
        assert "LNAME" not in result.columns

    def test_partial_rename(self, tmp_path):
        path = tmp_path / "test.csv"
        pl.DataFrame({"LNAME": ["Smith"], "email": ["a@b.com"]}).write_csv(path)
        lf = load_file(path)
        mapped = apply_column_map(lf, {"LNAME": "last_name"})
        result = mapped.collect()
        assert "last_name" in result.columns
        assert "email" in result.columns  # untouched

    def test_missing_source_column_raises(self, tmp_path):
        path = tmp_path / "test.csv"
        pl.DataFrame({"a": [1]}).write_csv(path)
        lf = load_file(path)
        with pytest.raises(ValueError, match="not in file"):
            apply_column_map(lf, {"nonexistent": "target"})

    def test_empty_map_is_noop(self, tmp_path):
        path = tmp_path / "test.csv"
        pl.DataFrame({"a": [1], "b": [2]}).write_csv(path)
        lf = load_file(path)
        mapped = apply_column_map(lf, {})
        assert mapped.collect_schema().names() == ["a", "b"]


class TestSuggestColumnMapping:
    def test_case_insensitive_exact(self):
        suggestions = suggest_column_mapping(
            file_columns=["LastName", "FirstName", "Email"],
            target_columns=["last_name", "first_name", "email"],
        )
        # "Email" -> "email" (case-insensitive exact, different case)
        assert suggestions.get("Email") == "email"

    def test_fuzzy_match(self):
        suggestions = suggest_column_mapping(
            file_columns=["surname", "given_name", "postal_code"],
            target_columns=["last_name", "first_name", "zip"],
        )
        # At least one fuzzy match should be found
        assert len(suggestions) > 0

    def test_no_match_below_threshold(self):
        suggestions = suggest_column_mapping(
            file_columns=["xyz_totally_different"],
            target_columns=["last_name"],
            threshold=0.9,
        )
        assert len(suggestions) == 0

    def test_exact_same_name_not_included(self):
        # If names are already identical, no mapping needed
        suggestions = suggest_column_mapping(
            file_columns=["last_name", "email"],
            target_columns=["last_name", "email"],
        )
        assert len(suggestions) == 0  # no renames needed


class TestColumnMapInPipeline:
    def test_dedupe_with_column_map(self, tmp_path):
        # File with non-standard column names
        path = tmp_path / "weird_cols.csv"
        pl.DataFrame({
            "LNAME": ["Smith", "Smith", "Jones"],
            "MAIL": ["john@test.com", "john@test.com", "bob@test.com"],
            "ZIP5": ["19382", "19382", "90210"],
        }).write_csv(path)

        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
            OutputConfig, GoldenRulesConfig, GoldenFieldRule,
        )
        from goldenmatch.core.pipeline import run_dedupe

        cfg = GoldenMatchConfig(
            matchkeys=[
                MatchkeyConfig(
                    name="email_key",
                    fields=[MatchkeyField(column="email", transforms=["lowercase"])],
                    comparison="exact",
                )
            ],
            output=OutputConfig(format="csv", directory=str(tmp_path), run_name="colmap_test"),
            golden_rules=GoldenRulesConfig(
                default=GoldenFieldRule(strategy="most_complete"),
            ),
        )

        col_map = {"LNAME": "last_name", "MAIL": "email", "ZIP5": "zip"}
        results = run_dedupe(
            files=[(path, "test_source", col_map)],
            config=cfg,
            output_report=True,
        )
        assert results["report"]["total_records"] == 3
        # john@test.com appears twice -> at least 1 cluster
        multi = [c for c in results["clusters"].values() if c["size"] > 1]
        assert len(multi) >= 1

    def test_match_with_column_map(self, tmp_path):
        # Target with standard names
        target = tmp_path / "target.csv"
        pl.DataFrame({
            "email": ["john@test.com", "jane@test.com"],
        }).write_csv(target)

        # Reference with different names
        ref = tmp_path / "reference.csv"
        pl.DataFrame({
            "MAIL_ADDR": ["john@test.com", "bob@test.com"],
        }).write_csv(ref)

        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField, OutputConfig,
        )
        from goldenmatch.core.pipeline import run_match

        cfg = GoldenMatchConfig(
            matchkeys=[
                MatchkeyConfig(
                    name="email_key",
                    fields=[MatchkeyField(column="email", transforms=["lowercase"])],
                    comparison="exact",
                )
            ],
            output=OutputConfig(format="csv", directory=str(tmp_path), run_name="match_colmap"),
        )

        results = run_match(
            target_file=(target, "targets", None),
            reference_files=[(ref, "refs", {"MAIL_ADDR": "email"})],
            config=cfg,
            output_report=True,
        )
        assert results["report"]["matched"] == 1  # john@test.com
        assert results["report"]["unmatched"] == 1  # jane@test.com
