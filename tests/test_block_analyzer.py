"""Tests for goldenmatch block analyzer."""

import polars as pl
import pytest

from goldenmatch.core.block_analyzer import (
    BlockingSuggestion,
    analyze_blocking,
    check_coverage,
    detect_column_type,
    generate_candidates,
    score_candidate,
)


# ── detect_column_type ──────────────────────────────────────────────────────


class TestDetectColumnType:
    def test_name_fields(self):
        assert detect_column_type("first_name") == "name"
        assert detect_column_type("LAST_NAME") == "name"
        assert detect_column_type("fname") == "name"
        assert detect_column_type("lname") == "name"

    def test_zip_fields(self):
        assert detect_column_type("zip") == "zip"
        assert detect_column_type("zip_code") == "zip"
        assert detect_column_type("postal_code") == "zip"

    def test_email_fields(self):
        assert detect_column_type("email") == "email"
        assert detect_column_type("email_address") == "email"
        assert detect_column_type("mail") == "email"

    def test_phone_fields(self):
        assert detect_column_type("phone") == "phone"
        assert detect_column_type("telephone") == "phone"
        assert detect_column_type("mobile_number") == "phone"

    def test_state_fields(self):
        assert detect_column_type("state") == "state"
        assert detect_column_type("STATE") == "state"

    def test_generic_fields(self):
        assert detect_column_type("city") == "generic"
        assert detect_column_type("address") == "generic"
        assert detect_column_type("id") == "generic"


# ── generate_candidates ─────────────────────────────────────────────────────


class TestGenerateCandidates:
    def test_single_name_column(self):
        candidates = generate_candidates(["first_name"])
        assert len(candidates) > 0
        # Should have substring and soundex candidates
        descriptions = [c["description"] for c in candidates]
        assert any("first_name[:3]" in d for d in descriptions)
        assert any("soundex" in d for d in descriptions)

    def test_zip_column(self):
        candidates = generate_candidates(["zip"])
        descriptions = [c["description"] for c in candidates]
        assert any("zip[:3]" in d for d in descriptions)
        assert any("zip[:5]" in d for d in descriptions)
        assert any("zip" == d for d in descriptions)

    def test_state_column(self):
        candidates = generate_candidates(["state"])
        descriptions = [c["description"] for c in candidates]
        assert any("state" == d for d in descriptions)

    def test_compound_candidates_generated(self):
        candidates = generate_candidates(["last_name", "state"])
        # Should have both single and compound candidates
        compound = [c for c in candidates if len(c["key_fields"]) == 2]
        assert len(compound) > 0

    def test_candidate_structure(self):
        candidates = generate_candidates(["first_name"])
        for c in candidates:
            assert "key_fields" in c
            assert "transforms" in c
            assert "description" in c
            assert isinstance(c["key_fields"], list)
            assert isinstance(c["transforms"], list)


# ── score_candidate ──────────────────────────────────────────────────────────


class TestScoreCandidate:
    def test_returns_valid_metrics(self):
        df = pl.DataFrame({
            "zip": ["10001"] * 50 + ["90210"] * 50 + ["19382"] * 50,
        })
        candidate = {
            "key_fields": ["zip"],
            "transforms": [],
            "description": "zip",
        }
        result = score_candidate(df, candidate, target_block_size=5000)
        assert "group_count" in result
        assert "max_group_size" in result
        assert "mean_group_size" in result
        assert "std_group_size" in result
        assert "total_comparisons" in result
        assert "score" in result
        assert result["group_count"] == 3
        assert result["max_group_size"] == 50
        assert result["score"] > 0

    def test_total_comparisons_correct(self):
        df = pl.DataFrame({
            "state": ["NY"] * 10 + ["CA"] * 20,
        })
        candidate = {
            "key_fields": ["state"],
            "transforms": [],
            "description": "state",
        }
        result = score_candidate(df, candidate)
        # NY: 10*9/2 = 45, CA: 20*19/2 = 190 => 235
        assert result["total_comparisons"] == 235

    def test_with_transforms(self):
        df = pl.DataFrame({
            "last_name": ["Smith", "Smyth", "Johnson", "Jones", "SMITH"],
        })
        candidate = {
            "key_fields": ["last_name"],
            "transforms": ["lowercase", "substring:0:3"],
            "description": "last_name[:3]",
        }
        result = score_candidate(df, candidate)
        assert result["group_count"] >= 1
        assert result["score"] > 0

    def test_all_null_column(self):
        df = pl.DataFrame({
            "col": [None, None, None],
        })
        candidate = {
            "key_fields": ["col"],
            "transforms": [],
            "description": "col",
        }
        result = score_candidate(df, candidate)
        assert result["group_count"] == 0
        assert result["score"] == 0.0

    def test_single_value_column(self):
        df = pl.DataFrame({
            "col": ["A"] * 100,
        })
        candidate = {
            "key_fields": ["col"],
            "transforms": [],
            "description": "col",
        }
        result = score_candidate(df, candidate)
        assert result["group_count"] == 1
        assert result["max_group_size"] == 100


# ── check_coverage ──────────────────────────────────────────────────────────


class TestCheckCoverage:
    def test_covers_single_field(self):
        candidate = {"key_fields": ["last_name"], "transforms": [], "description": ""}
        assert check_coverage(candidate, ["last_name", "zip"]) is True

    def test_rejects_missing_field(self):
        candidate = {"key_fields": ["email"], "transforms": [], "description": ""}
        assert check_coverage(candidate, ["last_name", "zip"]) is False

    def test_compound_all_covered(self):
        candidate = {"key_fields": ["last_name", "zip"], "transforms": [], "description": ""}
        assert check_coverage(candidate, ["last_name", "zip", "state"]) is True

    def test_compound_partial_coverage(self):
        candidate = {"key_fields": ["last_name", "email"], "transforms": [], "description": ""}
        assert check_coverage(candidate, ["last_name", "zip"]) is False


# ── analyze_blocking ─────────────────────────────────────────────────────────


class TestAnalyzeBlocking:
    def test_returns_ranked_suggestions(self):
        # Build a dataset large enough for meaningful analysis
        import random
        random.seed(42)
        n = 500
        df = pl.DataFrame({
            "last_name": [random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones",
                                          "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"])
                         for _ in range(n)],
            "state": [random.choice(["NY", "CA", "TX", "FL", "PA"]) for _ in range(n)],
            "zip": [random.choice(["10001", "90210", "75001", "33101", "19382",
                                    "60601", "30301", "98101", "02101", "85001"])
                   for _ in range(n)],
        })
        suggestions = analyze_blocking(df, ["last_name", "state", "zip"], sample_size=200)
        assert len(suggestions) > 0
        assert all(isinstance(s, BlockingSuggestion) for s in suggestions)
        # Should be sorted by score (descending)
        for i in range(len(suggestions) - 1):
            assert suggestions[i].score >= suggestions[i + 1].score

    def test_suggestion_fields(self):
        import random
        random.seed(123)
        n = 300
        df = pl.DataFrame({
            "first_name": [random.choice(["Alice", "Bob", "Carol", "Dave"]) for _ in range(n)],
            "zip": [random.choice(["10001", "90210", "75001"]) for _ in range(n)],
        })
        suggestions = analyze_blocking(df, ["first_name", "zip"], sample_size=100)
        if suggestions:
            s = suggestions[0]
            assert isinstance(s.keys, list)
            assert isinstance(s.group_count, int)
            assert isinstance(s.max_group_size, int)
            assert isinstance(s.mean_group_size, float)
            assert isinstance(s.total_comparisons, int)
            assert isinstance(s.estimated_recall, float)
            assert isinstance(s.score, float)
            assert isinstance(s.description, str)
            assert 0.0 <= s.estimated_recall <= 1.0
