"""Tests for the validation rules module."""

from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.core.validate import ValidationRule, validate_dataframe


class TestRegexValidation:
    def test_regex_match(self):
        df = pl.DataFrame({"code": ["ABC-123", "XYZ-999", "bad", "DEF-001"]})
        rules = [ValidationRule(column="code", rule_type="regex", params={"pattern": r"^[A-Z]{3}-\d{3}$"}, action="flag")]
        valid, quarantine, report = validate_dataframe(df, rules)
        assert valid.height == 4
        assert valid["__vf_code_regex__"][0] is True
        assert valid["__vf_code_regex__"][2] is False

    def test_regex_quarantine(self):
        df = pl.DataFrame({"code": ["ABC-123", "bad"]})
        rules = [ValidationRule(column="code", rule_type="regex", params={"pattern": r"^[A-Z]{3}-\d{3}$"}, action="quarantine")]
        valid, quarantine, _ = validate_dataframe(df, rules)
        assert valid.height == 1
        assert quarantine.height == 1
        assert "__quarantine_reason__" in quarantine.columns


class TestMinLength:
    def test_min_length_flag(self):
        df = pl.DataFrame({"name": ["Al", "Alice", "Bo", "Charlie"]})
        rules = [ValidationRule(column="name", rule_type="min_length", params={"length": 3}, action="flag")]
        valid, _, report = validate_dataframe(df, rules)
        assert valid["__vf_name_min_length__"][0] is False
        assert valid["__vf_name_min_length__"][1] is True
        assert report[0]["failed"] == 2


class TestMaxLength:
    def test_max_length_flag(self):
        df = pl.DataFrame({"code": ["AB", "ABCDE", "ABCDEFGH"]})
        rules = [ValidationRule(column="code", rule_type="max_length", params={"length": 5}, action="flag")]
        valid, _, report = validate_dataframe(df, rules)
        assert valid["__vf_code_max_length__"][0] is True
        assert valid["__vf_code_max_length__"][2] is False


class TestNotNull:
    def test_not_null_flag(self):
        df = pl.DataFrame({"email": ["a@b.com", None, "c@d.com", None]})
        rules = [ValidationRule(column="email", rule_type="not_null", action="flag")]
        valid, _, report = validate_dataframe(df, rules)
        assert valid["__vf_email_not_null__"][0] is True
        assert valid["__vf_email_not_null__"][1] is False
        assert report[0]["failed"] == 2

    def test_not_null_quarantine(self):
        df = pl.DataFrame({"email": ["a@b.com", None, "c@d.com"]})
        rules = [ValidationRule(column="email", rule_type="not_null", action="quarantine")]
        valid, quarantine, _ = validate_dataframe(df, rules)
        assert valid.height == 2
        assert quarantine.height == 1


class TestInSet:
    def test_in_set_flag(self):
        df = pl.DataFrame({"status": ["active", "inactive", "deleted", "active"]})
        rules = [ValidationRule(column="status", rule_type="in_set", params={"values": ["active", "inactive"]}, action="flag")]
        valid, _, report = validate_dataframe(df, rules)
        assert valid["__vf_status_in_set__"][2] is False
        assert report[0]["failed"] == 1


class TestFormatValidation:
    def test_email_format(self):
        df = pl.DataFrame({"email": ["good@test.com", "bad-email", "also@good.org"]})
        rules = [ValidationRule(column="email", rule_type="format", params={"type": "email"}, action="flag")]
        valid, _, report = validate_dataframe(df, rules)
        assert valid["__vf_email_format__"][0] is True
        assert valid["__vf_email_format__"][1] is False

    def test_phone_format(self):
        df = pl.DataFrame({"phone": ["267-555-1234", "abc", "1234567"]})
        rules = [ValidationRule(column="phone", rule_type="format", params={"type": "phone"}, action="flag")]
        valid, _, report = validate_dataframe(df, rules)
        assert valid["__vf_phone_format__"][0] is True
        assert valid["__vf_phone_format__"][1] is False

    def test_zip5_format(self):
        df = pl.DataFrame({"zip": ["19382", "abc", "90210-1234"]})
        rules = [ValidationRule(column="zip", rule_type="format", params={"type": "zip5"}, action="flag")]
        valid, _, report = validate_dataframe(df, rules)
        assert valid["__vf_zip_format__"][0] is True
        assert valid["__vf_zip_format__"][1] is False


class TestNullAction:
    def test_null_action_sets_null(self):
        df = pl.DataFrame({"email": ["good@test.com", "bad-email", "ok@here.com"]})
        rules = [ValidationRule(column="email", rule_type="format", params={"type": "email"}, action="null")]
        valid, quarantine, _ = validate_dataframe(df, rules)
        assert valid["email"][0] == "good@test.com"
        assert valid["email"][1] is None
        assert valid["email"][2] == "ok@here.com"
        assert quarantine.height == 0


class TestMultipleRulesSameColumn:
    def test_two_rules_on_same_column(self):
        df = pl.DataFrame({"name": ["Al", "Alice", None, "X"]})
        rules = [
            ValidationRule(column="name", rule_type="not_null", action="flag"),
            ValidationRule(column="name", rule_type="min_length", params={"length": 2}, action="flag"),
        ]
        valid, _, report = validate_dataframe(df, rules)
        assert "__vf_name_not_null__" in valid.columns
        assert "__vf_name_min_length__" in valid.columns
        assert len(report) == 2


class TestValidationReport:
    def test_report_accuracy(self):
        df = pl.DataFrame({"code": ["ABC", "XY", "DEFG", "A"]})
        rules = [ValidationRule(column="code", rule_type="min_length", params={"length": 3}, action="flag")]
        _, _, report = validate_dataframe(df, rules)
        r = report[0]
        assert r["column"] == "code"
        assert r["total_checked"] == 4
        assert r["passed"] == 2
        assert r["failed"] == 2
        assert r["fail_rate"] == pytest.approx(0.5)


class TestQuarantineReason:
    def test_quarantine_has_reason_column(self):
        df = pl.DataFrame({"val": ["ok", "bad"]})
        rules = [ValidationRule(column="val", rule_type="regex", params={"pattern": "^ok$"}, action="quarantine")]
        _, quarantine, _ = validate_dataframe(df, rules)
        assert "__quarantine_reason__" in quarantine.columns
        assert "regex" in quarantine["__quarantine_reason__"][0]


class TestMixedActions:
    def test_flag_and_quarantine_together(self):
        df = pl.DataFrame({
            "name": ["Alice", "B", "Charlie", None],
            "email": ["a@b.com", "bad", "c@d.com", "e@f.com"],
        })
        rules = [
            ValidationRule(column="name", rule_type="min_length", params={"length": 2}, action="flag"),
            ValidationRule(column="email", rule_type="format", params={"type": "email"}, action="quarantine"),
        ]
        valid, quarantine, report = validate_dataframe(df, rules)
        # "bad" email row quarantined
        assert quarantine.height == 1
        # remaining rows should have the flag column
        assert "__vf_name_min_length__" in valid.columns
        assert len(report) == 2
