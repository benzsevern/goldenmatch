"""Tests for goldenmatch.core.agent -- intelligence layer."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, MagicMock

import polars as pl
import pytest

from goldenmatch.core.agent import (
    AgentSession,
    DataProfile,
    FieldProfile,
    StrategyDecision,
    build_alternatives,
    profile_for_agent,
    select_strategy,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def names_df():
    """DataFrame with name/address columns -- typical fuzzy scenario."""
    return pl.DataFrame({
        "name": ["Alice Smith", "Bob Jones", "alice smith", "Charlie Brown", "Bob Jones"],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA"],
        "zip": ["10001", "90001", "10001", "60601", "90001"],
    })


@pytest.fixture
def id_df():
    """DataFrame with high-uniqueness ID column -- exact scenario."""
    return pl.DataFrame({
        "email": [f"user{i}@example.com" for i in range(100)],
        "age": list(range(100)),
    })


@pytest.fixture
def mixed_df():
    """DataFrame with both strong IDs and fuzzy candidates."""
    return pl.DataFrame({
        "email": [f"u{i}@x.com" for i in range(100)],
        "name": [f"Person {i % 30}" for i in range(100)],
        "city": [f"City {i % 10}" for i in range(100)],
    })


@pytest.fixture
def sensitive_df():
    """DataFrame with sensitive PII columns."""
    return pl.DataFrame({
        "name": ["Alice", "Bob"],
        "ssn": ["123-45-6789", "987-65-4321"],
        "dob": ["1990-01-01", "1985-06-15"],
    })


@pytest.fixture
def tmp_csv(names_df):
    """Write names_df to a temp CSV and return its path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        names_df.write_csv(f.name)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_csv_b():
    """A second temp CSV for match_sources tests."""
    df = pl.DataFrame({
        "name": ["Alice Smith", "David Lee"],
        "city": ["NYC", "Boston"],
        "zip": ["10001", "02101"],
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.write_csv(f.name)
        path = f.name
    yield path
    os.unlink(path)


# ── DataProfile tests ────────────────────────────────────────────────────────


class TestProfileForAgent:
    def test_basic_profile(self, names_df):
        profile = profile_for_agent(names_df)
        assert profile.row_count == 5
        assert len(profile.fields) == 3
        assert profile.has_sensitive is False

    def test_field_types(self, names_df):
        profile = profile_for_agent(names_df)
        types = {f.name: f.type for f in profile.fields}
        assert types["name"] == "string"
        assert types["city"] == "string"
        assert types["zip"] == "string"

    def test_numeric_type(self, id_df):
        profile = profile_for_agent(id_df)
        types = {f.name: f.type for f in profile.fields}
        assert types["age"] == "numeric"

    def test_high_uniqueness(self, id_df):
        profile = profile_for_agent(id_df)
        email_field = next(f for f in profile.fields if f.name == "email")
        assert email_field.uniqueness == 1.0

    def test_low_uniqueness(self, names_df):
        profile = profile_for_agent(names_df)
        city_field = next(f for f in profile.fields if f.name == "city")
        # 3 unique cities out of 5 rows
        assert city_field.uniqueness == pytest.approx(3 / 5)

    def test_null_rate(self):
        df = pl.DataFrame({"a": ["x", None, "y", None, "z"]})
        profile = profile_for_agent(df)
        assert profile.fields[0].null_rate == pytest.approx(0.4)

    def test_avg_length(self, names_df):
        profile = profile_for_agent(names_df)
        name_field = next(f for f in profile.fields if f.name == "name")
        assert name_field.avg_length > 0

    def test_numeric_avg_length_zero(self, id_df):
        profile = profile_for_agent(id_df)
        age_field = next(f for f in profile.fields if f.name == "age")
        assert age_field.avg_length == 0.0

    def test_sensitive_detection(self, sensitive_df):
        profile = profile_for_agent(sensitive_df)
        assert profile.has_sensitive is True

    def test_sensitive_dob(self):
        df = pl.DataFrame({"first_name": ["Alice"], "date_of_birth": ["1990-01-01"]})
        profile = profile_for_agent(df)
        assert profile.has_sensitive is True

    def test_no_sensitive(self, names_df):
        profile = profile_for_agent(names_df)
        assert profile.has_sensitive is False

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": pl.Series([], dtype=pl.Utf8)})
        profile = profile_for_agent(df)
        assert profile.row_count == 0
        assert len(profile.fields) == 1


# ── Strategy selection tests ─────────────────────────────────────────────────


class TestSelectStrategy:
    def test_pprl_for_sensitive(self, sensitive_df):
        profile = profile_for_agent(sensitive_df)
        decision = select_strategy(profile)
        assert decision.strategy == "pprl"
        assert decision.auto_execute is False

    def test_exact_only(self, id_df):
        profile = profile_for_agent(id_df)
        decision = select_strategy(profile)
        assert decision.strategy == "exact_only"
        assert "email" in decision.strong_ids

    def test_exact_then_fuzzy(self, mixed_df):
        profile = profile_for_agent(mixed_df)
        decision = select_strategy(profile)
        assert decision.strategy == "exact_then_fuzzy"
        assert len(decision.strong_ids) > 0
        assert len(decision.fuzzy_fields) > 0

    def test_fuzzy(self, names_df):
        profile = profile_for_agent(names_df)
        decision = select_strategy(profile)
        assert decision.strategy == "fuzzy"
        assert len(decision.fuzzy_fields) > 0

    def test_backend_large(self):
        """Row count > 500K should suggest ray backend."""
        profile = DataProfile(
            row_count=600_000,
            fields=[FieldProfile("name", "string", 0.5, 0.0, 10.0)],
            has_sensitive=False,
        )
        decision = select_strategy(profile)
        assert decision.backend == "ray"

    def test_backend_small(self, names_df):
        profile = profile_for_agent(names_df)
        decision = select_strategy(profile)
        assert decision.backend is None

    def test_strategy_has_why(self, names_df):
        profile = profile_for_agent(names_df)
        decision = select_strategy(profile)
        assert len(decision.why) > 0

    def test_domain_detection(self):
        """If domain_registry detects a domain, it shows up in the decision."""
        profile = DataProfile(
            row_count=100,
            fields=[
                FieldProfile("brand", "string", 0.3, 0.0, 8.0),
                FieldProfile("model", "string", 0.5, 0.0, 10.0),
                FieldProfile("sku", "string", 0.8, 0.0, 12.0),
            ],
            has_sensitive=False,
        )
        decision = select_strategy(profile)
        # Domain may or may not be detected depending on built-in packs;
        # just check the function doesn't crash
        assert decision.strategy in {"fuzzy", "exact_only", "exact_then_fuzzy", "domain_extraction"}


# ── build_alternatives tests ─────────────────────────────────────────────────


class TestBuildAlternatives:
    def test_always_includes_pprl(self, names_df):
        profile = profile_for_agent(names_df)
        decision = select_strategy(profile)
        alts = build_alternatives(decision, profile)
        strategies = {a["strategy"] for a in alts}
        assert "pprl" in strategies

    def test_always_includes_fellegi_sunter(self, names_df):
        profile = profile_for_agent(names_df)
        decision = select_strategy(profile)
        alts = build_alternatives(decision, profile)
        strategies = {a["strategy"] for a in alts}
        assert "fellegi_sunter" in strategies

    def test_pprl_not_duplicated(self, sensitive_df):
        """When strategy is already pprl, it should not appear in alternatives."""
        profile = profile_for_agent(sensitive_df)
        decision = select_strategy(profile)
        assert decision.strategy == "pprl"
        alts = build_alternatives(decision, profile)
        strategies = {a["strategy"] for a in alts}
        assert "pprl" not in strategies
        assert "fellegi_sunter" in strategies

    def test_alternatives_have_why_not(self, names_df):
        profile = profile_for_agent(names_df)
        decision = select_strategy(profile)
        alts = build_alternatives(decision, profile)
        for alt in alts:
            assert "why_not" in alt
            assert len(alt["why_not"]) > 0


# ── AgentSession tests ───────────────────────────────────────────────────────


class TestAgentSession:
    def test_init(self):
        session = AgentSession()
        assert session.data is None
        assert session.config is None
        assert session.result is None
        assert session.review_queue is not None
        assert session.reasoning == {}

    def test_analyze(self, tmp_csv):
        session = AgentSession()
        result = session.analyze(tmp_csv)
        assert "strategy" in result
        assert "why" in result
        assert "profile" in result
        assert result["profile"]["row_count"] > 0
        assert len(result["profile"]["fields"]) > 0
        assert "alternatives" in result
        assert session.data is not None

    def test_analyze_strategy_info(self, tmp_csv):
        session = AgentSession()
        result = session.analyze(tmp_csv)
        assert result["strategy"] in {
            "exact_only", "exact_then_fuzzy", "fuzzy", "pprl", "domain_extraction",
        }
        assert isinstance(result["auto_execute"], bool)

    def test_deduplicate(self, tmp_csv):
        session = AgentSession()
        result = session.deduplicate(tmp_csv)
        assert "results" in result
        assert "reasoning" in result
        assert "confidence_distribution" in result
        assert "storage" in result
        assert result["storage"] == "memory"

    def test_deduplicate_confidence_distribution(self, tmp_csv):
        session = AgentSession()
        result = session.deduplicate(tmp_csv)
        cd = result["confidence_distribution"]
        assert "auto_merged" in cd
        assert "review" in cd
        assert "auto_rejected" in cd
        assert "total_pairs" in cd
        assert cd["total_pairs"] == cd["auto_merged"] + cd["review"] + cd["auto_rejected"]

    def test_deduplicate_with_custom_config(self, tmp_csv):
        from goldenmatch.config.schemas import (
            GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
        )
        cfg = GoldenMatchConfig(matchkeys=[
            MatchkeyConfig(
                name="exact_name",
                type="exact",
                fields=[MatchkeyField(field="name", transforms=["lowercase", "strip"])],
            ),
        ])
        session = AgentSession()
        result = session.deduplicate(tmp_csv, config=cfg)
        assert result["results"] is not None

    def test_match_sources(self, tmp_csv, tmp_csv_b):
        session = AgentSession()
        result = session.match_sources(tmp_csv, tmp_csv_b)
        assert "results" in result
        assert "reasoning" in result

    def test_compare_strategies(self, tmp_csv):
        session = AgentSession()
        result = session.compare_strategies(tmp_csv)
        assert "recommended" in result
        assert "strategies" in result
        assert len(result["strategies"]) >= 1
        # Each strategy entry should have metrics
        for name, metrics in result["strategies"].items():
            if "error" not in metrics:
                assert "clusters" in metrics or "match_rate" in metrics
