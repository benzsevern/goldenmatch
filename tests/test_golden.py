"""Tests for goldenmatch golden record builder."""

import polars as pl
import pytest

from goldenmatch.config.schemas import GoldenFieldRule, GoldenRulesConfig
from goldenmatch.core.golden import merge_field, build_golden_record, build_golden_record_with_provenance, GoldenRecordResult


class TestMergeFieldMostComplete:
    """Tests for most_complete strategy."""

    def test_longest_string_wins(self):
        rule = GoldenFieldRule(strategy="most_complete")
        val, conf, _idx = merge_field(["Jo", "John", "Jon"], rule)
        assert val == "John"
        assert conf == 1.0

    def test_tied_length(self):
        rule = GoldenFieldRule(strategy="most_complete")
        val, conf, _idx = merge_field(["abc", "xyz"], rule)
        assert val in ("abc", "xyz")
        assert conf == 0.7


class TestMergeFieldMajorityVote:
    """Tests for majority_vote strategy."""

    def test_majority_wins(self):
        rule = GoldenFieldRule(strategy="majority_vote")
        val, conf, _idx = merge_field(["A", "B", "A", "A"], rule)
        assert val == "A"
        assert conf == pytest.approx(3 / 4)

    def test_tie_picks_one(self):
        rule = GoldenFieldRule(strategy="majority_vote")
        val, conf, _idx = merge_field(["A", "B"], rule)
        assert val in ("A", "B")
        assert conf == pytest.approx(0.5)


class TestMergeFieldSourcePriority:
    """Tests for source_priority strategy."""

    def test_first_source_match(self):
        rule = GoldenFieldRule(
            strategy="source_priority",
            source_priority=["src_a", "src_b"],
        )
        val, conf, _idx = merge_field(
            ["v_a", "v_b"],
            rule,
            sources=["src_a", "src_b"],
        )
        assert val == "v_a"
        assert conf == pytest.approx(1.0)

    def test_second_source_match(self):
        rule = GoldenFieldRule(
            strategy="source_priority",
            source_priority=["src_a", "src_b", "src_c"],
        )
        val, conf, _idx = merge_field(
            [None, "v_b", "v_c"],
            rule,
            sources=["src_a", "src_b", "src_c"],
        )
        assert val == "v_b"
        assert conf == pytest.approx(0.9)

    def test_confidence_floor(self):
        """Confidence never drops below 0.1."""
        rule = GoldenFieldRule(
            strategy="source_priority",
            source_priority=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"],
        )
        vals = [None] * 11 + ["found", "other"]
        sources = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"]
        val, conf, _idx = merge_field(vals, rule, sources=sources)
        assert val == "found"
        assert conf == pytest.approx(0.1)


class TestMergeFieldFirstNonNull:
    """Tests for first_non_null strategy."""

    def test_first_value(self):
        rule = GoldenFieldRule(strategy="first_non_null")
        val, conf, _idx = merge_field([None, "hello", "world"], rule)
        assert val == "hello"
        assert conf == 0.6


class TestMergeFieldAllAgree:
    """When all non-null values are identical, confidence is 1.0."""

    def test_all_identical(self):
        rule = GoldenFieldRule(strategy="majority_vote")
        val, conf, _idx = merge_field(["same", "same", "same"], rule)
        assert val == "same"
        assert conf == 1.0


class TestMergeFieldAllNull:
    """All null returns (None, 0.0)."""

    def test_all_none(self):
        rule = GoldenFieldRule(strategy="most_complete")
        val, conf, _idx = merge_field([None, None], rule)
        assert val is None
        assert conf == 0.0

    def test_empty_list(self):
        rule = GoldenFieldRule(strategy="most_complete")
        val, conf, _idx = merge_field([], rule)
        assert val is None
        assert conf == 0.0


class TestBuildGoldenRecord:
    """Integration test for build_golden_record."""

    def test_build_golden_record(self):
        df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "__source__": ["a", "b", "c"],
            "__block_key__": ["k", "k", "k"],
            "__mk_exact": ["x", "x", "x"],
            "name": ["John", "John", "Jonathan"],
            "email": ["j@a.com", "john@b.com", "j@a.com"],
        })
        rules = GoldenRulesConfig(
            default_strategy="most_complete",
            field_rules={
                "email": GoldenFieldRule(strategy="majority_vote"),
            },
        )
        result = build_golden_record(df, rules)
        # Internal columns should be absent
        assert "__row_id__" not in result
        assert "__source__" not in result
        assert "__block_key__" not in result
        assert "__mk_exact" not in result
        # name: most_complete -> "Jonathan" (longest)
        assert result["name"]["value"] == "Jonathan"
        # email: majority_vote -> "j@a.com" (2 vs 1)
        assert result["email"]["value"] == "j@a.com"
        # golden confidence is mean of individual confidences
        assert "__golden_confidence__" in result
        assert 0.0 < result["__golden_confidence__"] <= 1.0


def test_provenance_structure():
    """build_golden_record_with_provenance returns GoldenRecordResult with provenance."""
    df = pl.DataFrame({
        "__row_id__": [1, 2],
        "__cluster_id__": [1, 1],
        "name": ["Alice", "Alice"],
        "email": ["a@test.com", "alice@test.com"],
    })
    rules = GoldenRulesConfig(default_strategy="most_complete")
    clusters = {1: {"members": [1, 2], "size": 2, "cluster_quality": "strong", "confidence": 0.9}}
    result = build_golden_record_with_provenance(df, rules, clusters)
    assert isinstance(result, GoldenRecordResult)
    assert isinstance(result.df, pl.DataFrame)
    assert len(result.provenance) == 1
    prov = result.provenance[0]
    assert prov.cluster_id == 1
    assert prov.cluster_quality == "strong"
    assert "name" in prov.fields
    assert "email" in prov.fields
    assert prov.fields["name"].strategy == "most_complete"
    assert prov.fields["name"].source_row_id in [1, 2]
    assert len(prov.fields["name"].candidates) == 2


def test_provenance_df_matches_golden_record():
    """GoldenRecordResult.df matches build_golden_record output."""
    df = pl.DataFrame({
        "__row_id__": [1, 2],
        "name": ["Alice", "Bob"],
    })
    rules = GoldenRulesConfig(default_strategy="most_complete")
    clusters = {1: {"members": [1, 2], "size": 2, "cluster_quality": "strong", "confidence": 0.9}}
    old = build_golden_record(df, rules)
    result = build_golden_record_with_provenance(df, rules, clusters)
    assert result.df["name"][0] == old["name"]["value"]


# --- Quality-weighted strategy tests ---


def test_most_complete_quality_tiebreak():
    """Quality breaks ties in most_complete strategy."""
    df = pl.DataFrame({"__row_id__": [1, 2], "name": ["Alice", "Bobby"]})  # Same length
    rules = GoldenRulesConfig(default_strategy="most_complete")
    quality = {(1, "name"): 0.5, (2, "name"): 0.9}
    result = build_golden_record(df, rules, quality_scores=quality)
    assert result["name"]["value"] == "Bobby"


def test_majority_vote_quality_weighted():
    """Quality weights affect majority vote outcome."""
    df = pl.DataFrame({"__row_id__": [1, 2, 3], "status": ["active", "active", "inactive"]})
    rules = GoldenRulesConfig(default_strategy="majority_vote")
    quality = {(1, "status"): 0.1, (2, "status"): 0.1, (3, "status"): 0.9}
    result = build_golden_record(df, rules, quality_scores=quality)
    assert result["status"]["value"] == "inactive"


def test_first_non_null_quality_ordering():
    """first_non_null picks highest quality source first."""
    df = pl.DataFrame({"__row_id__": [1, 2, 3], "phone": [None, "+15551234567", "+15559876543"]})
    rules = GoldenRulesConfig(default_strategy="first_non_null")
    quality = {(2, "phone"): 0.3, (3, "phone"): 0.9}
    result = build_golden_record(df, rules, quality_scores=quality)
    assert result["phone"]["value"] == "+15559876543"


def test_backward_compat_no_quality_scores():
    """quality_scores=None produces identical output to omitting it."""
    df = pl.DataFrame({"__row_id__": [1, 2], "name": ["Alice", "Bob"]})
    rules = GoldenRulesConfig(default_strategy="most_complete")
    result_default = build_golden_record(df, rules)
    result_none = build_golden_record(df, rules, quality_scores=None)
    assert result_default["name"]["value"] == result_none["name"]["value"]
    assert abs(result_default["name"]["confidence"] - result_none["name"]["confidence"]) < 0.0001
