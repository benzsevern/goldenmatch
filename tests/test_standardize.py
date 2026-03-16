"""Tests for data standardization module."""

import polars as pl
import pytest
import yaml
from pathlib import Path

from goldenmatch.core.standardize import (
    std_email,
    std_name_proper,
    std_name_upper,
    std_name_lower,
    std_phone,
    std_zip5,
    std_address,
    std_state,
    std_strip,
    std_trim_whitespace,
    apply_standardization,
    get_standardizer,
)


# ── Individual standardizer tests ────────────────────────────────────────────


class TestStdEmail:
    def test_lowercase_and_strip(self):
        assert std_email("  JOHN@EXAMPLE.COM  ") == "john@example.com"

    def test_invalid_no_at(self):
        assert std_email("notanemail") is None

    def test_invalid_no_dot(self):
        assert std_email("user@localhost") is None

    def test_valid_email(self):
        assert std_email("user@test.co.uk") == "user@test.co.uk"

    def test_none(self):
        assert std_email(None) is None

    def test_empty(self):
        assert std_email("") is None


class TestStdNameProper:
    def test_basic(self):
        assert std_name_proper("john smith") == "John Smith"

    def test_already_proper(self):
        assert std_name_proper("John Smith") == "John Smith"

    def test_all_upper(self):
        assert std_name_proper("JOHN SMITH") == "John Smith"

    def test_hyphenated(self):
        assert std_name_proper("mary-jane watson") == "Mary-Jane Watson"

    def test_none(self):
        assert std_name_proper(None) is None

    def test_empty(self):
        assert std_name_proper("  ") is None


class TestStdNameUpper:
    def test_basic(self):
        assert std_name_upper("john smith") == "JOHN SMITH"

    def test_none(self):
        assert std_name_upper(None) is None


class TestStdNameLower:
    def test_basic(self):
        assert std_name_lower("JOHN SMITH") == "john smith"

    def test_none(self):
        assert std_name_lower(None) is None


class TestStdPhone:
    def test_strips_formatting(self):
        assert std_phone("(267) 555-1234") == "2675551234"

    def test_strips_country_code(self):
        assert std_phone("1-267-555-1234") == "2675551234"

    def test_keeps_international(self):
        # 12 digits, doesn't start with 1-then-10
        assert std_phone("+44 20 7946 0958") == "442079460958"

    def test_too_short(self):
        assert std_phone("123") is None

    def test_none(self):
        assert std_phone(None) is None

    def test_empty(self):
        assert std_phone("") is None


class TestStdZip5:
    def test_basic(self):
        assert std_zip5("19382") == "19382"

    def test_zip_plus_4(self):
        assert std_zip5("19382-1234") == "19382"

    def test_pads_short(self):
        assert std_zip5("1234") == "01234"

    def test_long_zip(self):
        assert std_zip5("193821234") == "19382"

    def test_none(self):
        assert std_zip5(None) is None


class TestStdAddress:
    def test_abbreviations(self):
        result = std_address("123 Main Street")
        assert result == "123 Main St"

    def test_avenue(self):
        result = std_address("456 Oak Avenue")
        assert result == "456 Oak Ave"

    def test_directional(self):
        result = std_address("789 North Elm Boulevard")
        assert result == "789 N Elm Blvd"

    def test_po_box(self):
        result = std_address("P.O. Box 123")
        assert result == "PO Box 123"

    def test_whitespace_normalization(self):
        result = std_address("123  Main    Street")
        assert result == "123 Main St"

    def test_none(self):
        assert std_address(None) is None


class TestStdState:
    def test_uppercase(self):
        assert std_state("pa") == "PA"

    def test_already_upper(self):
        assert std_state("NY") == "NY"

    def test_none(self):
        assert std_state(None) is None


class TestStdStrip:
    def test_basic(self):
        assert std_strip("  hello  ") == "hello"

    def test_empty_to_none(self):
        assert std_strip("   ") is None

    def test_none(self):
        assert std_strip(None) is None


class TestStdTrimWhitespace:
    def test_collapses_spaces(self):
        assert std_trim_whitespace("John   Smith") == "John Smith"

    def test_strips_and_collapses(self):
        assert std_trim_whitespace("  John   Smith  ") == "John Smith"


# ── Registry tests ───────────────────────────────────────────────────────────


class TestGetStandardizer:
    def test_valid(self):
        fn = get_standardizer("email")
        assert callable(fn)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown standardizer"):
            get_standardizer("nonexistent")


# ── DataFrame integration ────────────────────────────────────────────────────


class TestApplyStandardization:
    def test_email_standardization(self):
        lf = pl.DataFrame({
            "email": ["  JOHN@TEST.COM  ", "not-an-email", "jane@test.com"],
        }).lazy()
        result = apply_standardization(lf, {"email": ["email"]}).collect()
        assert result["email"].to_list() == ["john@test.com", None, "jane@test.com"]

    def test_chained_standardizers(self):
        lf = pl.DataFrame({
            "name": ["  john smith  ", "JANE DOE", None],
        }).lazy()
        result = apply_standardization(lf, {"name": ["strip", "name_proper"]}).collect()
        assert result["name"].to_list() == ["John Smith", "Jane Doe", None]

    def test_multiple_columns(self):
        lf = pl.DataFrame({
            "email": ["JOHN@TEST.COM"],
            "phone": ["(267) 555-1234"],
            "zip": ["19382-1234"],
        }).lazy()
        rules = {
            "email": ["email"],
            "phone": ["phone"],
            "zip": ["zip5"],
        }
        result = apply_standardization(lf, rules).collect()
        assert result["email"][0] == "john@test.com"
        assert result["phone"][0] == "2675551234"
        assert result["zip"][0] == "19382"

    def test_missing_column_skipped(self, caplog):
        import logging
        lf = pl.DataFrame({"a": [1]}).lazy()
        with caplog.at_level(logging.WARNING):
            result = apply_standardization(lf, {"nonexistent": ["strip"]}).collect()
        assert any("not found" in r.message for r in caplog.records)
        assert result.columns == ["a"]  # unchanged

    def test_empty_rules(self):
        lf = pl.DataFrame({"a": ["hello"]}).lazy()
        result = apply_standardization(lf, {}).collect()
        assert result["a"][0] == "hello"


# ── Config schema tests ──────────────────────────────────────────────────────


class TestStandardizationConfig:
    def test_valid_config(self):
        from goldenmatch.config.schemas import StandardizationConfig
        cfg = StandardizationConfig(rules={
            "email": ["email"],
            "name": ["strip", "name_proper"],
        })
        assert cfg.rules["email"] == ["email"]

    def test_invalid_standardizer_rejected(self):
        from goldenmatch.config.schemas import StandardizationConfig
        with pytest.raises(ValueError, match="Invalid standardizer"):
            StandardizationConfig(rules={"col": ["not_real"]})


# ── YAML loading tests ──────────────────────────────────────────────────────


class TestStandardizationYaml:
    def test_flat_format(self, tmp_path):
        cfg_path = tmp_path / "goldenmatch.yaml"
        cfg_path.write_text(yaml.dump({
            "matchkeys": [{
                "name": "email_key",
                "fields": [{"column": "email", "transforms": ["lowercase"]}],
                "comparison": "exact",
            }],
            "standardization": {
                "email": ["email"],
                "phone": ["phone"],
            },
        }))
        from goldenmatch.config.loader import load_config
        cfg = load_config(cfg_path)
        assert cfg.standardization is not None
        assert cfg.standardization.rules["email"] == ["email"]

    def test_explicit_rules_format(self, tmp_path):
        cfg_path = tmp_path / "goldenmatch.yaml"
        cfg_path.write_text(yaml.dump({
            "matchkeys": [{
                "name": "email_key",
                "fields": [{"column": "email", "transforms": ["lowercase"]}],
                "comparison": "exact",
            }],
            "standardization": {
                "rules": {
                    "email": ["email"],
                },
            },
        }))
        from goldenmatch.config.loader import load_config
        cfg = load_config(cfg_path)
        assert cfg.standardization.rules["email"] == ["email"]


# ── Pipeline integration ─────────────────────────────────────────────────────


class TestStandardizationInPipeline:
    def test_dedupe_with_standardization(self, tmp_path):
        path = tmp_path / "messy.csv"
        pl.DataFrame({
            "email": ["  JOHN@TEST.COM  ", "john@test.com", "JANE@test.COM"],
            "phone": ["(267) 555-1234", "267-555-1234", "(212) 555-9999"],
            "name": ["john smith", "JOHN SMITH", "Jane Doe"],
        }).write_csv(path)

        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
            OutputConfig, GoldenRulesConfig, GoldenFieldRule,
            StandardizationConfig,
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
            standardization=StandardizationConfig(rules={
                "email": ["email"],
                "phone": ["phone"],
                "name": ["strip", "name_proper"],
            }),
            output=OutputConfig(format="csv", directory=str(tmp_path), run_name="std_test"),
            golden_rules=GoldenRulesConfig(
                default=GoldenFieldRule(strategy="most_complete"),
            ),
        )

        results = run_dedupe(
            files=[(path, "test")],
            config=cfg,
            output_report=True,
            output_golden=True,
        )
        # Standardization should clean emails so john@test.com matches
        assert results["report"]["total_records"] == 3
        multi = [c for c in results["clusters"].values() if c["size"] > 1]
        assert len(multi) >= 1

        # Verify standardized data
        golden = results["golden"]
        if golden is not None and len(golden) > 0:
            row = golden.to_dicts()[0]
            # Phone should be digits only
            if "phone" in row and row["phone"]:
                assert row["phone"].isdigit()
