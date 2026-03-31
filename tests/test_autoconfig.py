"""Tests for auto-configuration engine."""

from __future__ import annotations

import random
from unittest.mock import patch, MagicMock

import polars as pl
import pytest

from goldenmatch.core.autoconfig import (
    ColumnProfile,
    _classify_by_data,
    _classify_by_name,
    auto_configure,
    build_blocking,
    build_matchkeys,
    profile_columns,
    select_model,
)


class TestClassifyByName:
    def test_email(self):
        assert _classify_by_name("email") == "email"
        assert _classify_by_name("Email_Address") == "email"
        assert _classify_by_name("e_mail") == "email"

    def test_name(self):
        assert _classify_by_name("name") == "name"
        assert _classify_by_name("first_name") == "name"
        assert _classify_by_name("LastName") == "name"
        assert _classify_by_name("full_name") == "name"

    def test_phone(self):
        assert _classify_by_name("phone") == "phone"
        assert _classify_by_name("mobile") == "phone"
        assert _classify_by_name("tel") == "phone"

    def test_zip(self):
        assert _classify_by_name("zip") == "zip"
        assert _classify_by_name("postal_code") == "zip"
        assert _classify_by_name("postcode") == "zip"

    def test_address(self):
        assert _classify_by_name("address") == "address"
        assert _classify_by_name("street") == "address"

    def test_geo(self):
        assert _classify_by_name("city") == "geo"
        assert _classify_by_name("state") == "geo"
        assert _classify_by_name("country") == "geo"

    def test_identifier(self):
        assert _classify_by_name("id") == "identifier"
        assert _classify_by_name("customer_id") == "identifier"
        assert _classify_by_name("sku") == "identifier"

    def test_unknown(self):
        assert _classify_by_name("foobar") is None
        assert _classify_by_name("amount") is None


class TestClassifyByData:
    def test_email_data(self):
        values = ["john@example.com", "jane@test.org", "bob@mail.com"] * 10
        col_type, conf = _classify_by_data(values)
        assert col_type == "email"

    def test_phone_data(self):
        values = ["555-123-4567", "(555) 987-6543", "5551234567"] * 10
        col_type, conf = _classify_by_data(values)
        assert col_type == "phone"

    def test_zip_data(self):
        values = ["10001", "90210", "30301", "60601", "02134"] * 10
        col_type, conf = _classify_by_data(values)
        assert col_type == "zip"

    def test_name_data(self):
        values = ["John Smith", "Jane Doe", "Bob Johnson", "Alice Brown"] * 10
        col_type, conf = _classify_by_data(values)
        assert col_type == "name"

    def test_description_data(self):
        values = [
            "This is a very long product description that goes on and on about features and specifications",
        ] * 10
        col_type, conf = _classify_by_data(values)
        assert col_type == "description"

    def test_empty_values(self):
        col_type, conf = _classify_by_data([])
        assert col_type == "string"


class TestProfileColumns:
    def test_profiles_basic_csv(self):
        df = pl.DataFrame({
            "name": ["John Smith", "Jane Doe", "Bob Johnson"],
            "email": ["john@test.com", "jane@test.com", "bob@test.com"],
            "zip": ["10001", "90210", "30301"],
        })
        profiles = profile_columns(df)
        types = {p.name: p.col_type for p in profiles}
        assert types["name"] == "name"
        assert types["email"] == "email"
        assert types["zip"] == "zip"

    def test_skips_internal_columns(self):
        df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "name": ["a", "b", "c"],
        })
        profiles = profile_columns(df)
        assert all(p.name != "__row_id__" for p in profiles)


class TestBuildMatchkeys:
    def test_exact_fields(self):
        profiles = [
            ColumnProfile("email", "Utf8", "email", 0.9),
            ColumnProfile("phone", "Utf8", "phone", 0.9),
        ]
        mks = build_matchkeys(profiles)
        assert len(mks) >= 2
        assert all(mk.type == "exact" for mk in mks)

    def test_fuzzy_fields(self):
        profiles = [
            ColumnProfile("name", "Utf8", "name", 0.9),
            ColumnProfile("address", "Utf8", "address", 0.8),
        ]
        mks = build_matchkeys(profiles)
        weighted = [mk for mk in mks if mk.type == "weighted"]
        assert len(weighted) == 1
        assert weighted[0].threshold == 0.80  # mix of fuzzy

    def test_description_uses_record_embedding(self):
        profiles = [
            ColumnProfile("title", "Utf8", "description", 0.7),
        ]
        mks = build_matchkeys(profiles)
        weighted = [mk for mk in mks if mk.type == "weighted"]
        assert len(weighted) == 1
        rec_emb = [f for f in weighted[0].fields if f.scorer == "record_embedding"]
        assert len(rec_emb) == 1

    def test_fallback_for_unknown_columns(self):
        profiles = [
            ColumnProfile("field_1", "Utf8", "string", 0.3),
            ColumnProfile("field_2", "Utf8", "string", 0.3),
        ]
        mks = build_matchkeys(profiles)
        assert len(mks) >= 1

    def test_numeric_columns_skipped(self):
        profiles = [
            ColumnProfile("amount", "Float64", "numeric", 0.9),
            ColumnProfile("name", "Utf8", "name", 0.9),
        ]
        mks = build_matchkeys(profiles)
        # amount should not appear in any matchkey
        for mk in mks:
            for f in mk.fields:
                assert f.field != "amount"


class TestBuildBlocking:
    def test_blocks_on_exact_column(self):
        profiles = [
            ColumnProfile("email", "Utf8", "email", 0.9),
            ColumnProfile("name", "Utf8", "name", 0.9),
        ]
        df = pl.DataFrame({
            "email": ["a@b.com", "c@d.com", "e@f.com"],
            "name": ["John", "Jane", "Bob"],
        })
        blocking = build_blocking(profiles, df)
        assert blocking.strategy == "static"
        assert blocking.keys[0].fields == ["email"]

    def test_name_columns_use_multi_pass(self):
        profiles = [
            ColumnProfile("name", "Utf8", "name", 0.9),
        ]
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"]})
        blocking = build_blocking(profiles, df)
        assert blocking.strategy == "multi_pass"

    def test_description_only_uses_canopy(self):
        profiles = [
            ColumnProfile("desc", "Utf8", "description", 0.7),
        ]
        df = pl.DataFrame({"desc": ["long text here"] * 3})
        blocking = build_blocking(profiles, df)
        assert blocking.strategy == "canopy"


class TestSelectModel:
    def test_no_embeddings(self):
        assert select_model(1000, False) is None

    def test_small_dataset(self):
        result = select_model(10000, True)
        assert result == "gte-base-en-v1.5"

    def test_large_dataset(self):
        result = select_model(100000, True)
        assert result == "all-MiniLM-L6-v2"


class TestAdaptiveThreshold:
    def test_all_exact(self):
        from goldenmatch.core.autoconfig import _adaptive_threshold
        from goldenmatch.config.schemas import MatchkeyField
        fields = [MatchkeyField(field="email", scorer="exact", weight=1.0)]
        assert _adaptive_threshold(fields) == 0.95

    def test_all_fuzzy(self):
        from goldenmatch.core.autoconfig import _adaptive_threshold
        from goldenmatch.config.schemas import MatchkeyField
        fields = [MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)]
        assert _adaptive_threshold(fields) == 0.85  # single field

    def test_embedding(self):
        from goldenmatch.core.autoconfig import _adaptive_threshold
        from goldenmatch.config.schemas import MatchkeyField
        fields = [
            MatchkeyField(scorer="record_embedding", columns=["title"], weight=1.0),
        ]
        assert _adaptive_threshold(fields) == 0.70


class TestAutoConfigureIntegration:
    def test_auto_configure_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "name,email,zip\n"
            "John Smith,john@test.com,10001\n"
            "Jane Doe,jane@test.com,90210\n"
            "Bob Johnson,bob@test.com,30301\n"
        )
        config = auto_configure([(str(csv_path), "test")])
        assert config.golden_rules is not None
        assert len(config.get_matchkeys()) >= 1

    def test_auto_configure_unknown_columns(self, tmp_path):
        csv_path = tmp_path / "mystery.csv"
        csv_path.write_text(
            "field_a,field_b\n"
            "hello world,foo bar\n"
            "test value,baz qux\n"
        )
        config = auto_configure([(str(csv_path), "mystery")])
        assert len(config.get_matchkeys()) >= 1


# ── Compound blocking tests ───────────────────────────────────────────────


class TestCompoundBlocking:
    """Tests for compound blocking key generation."""

    def test_compound_keys_when_single_columns_oversized(self):
        """When all single columns produce blocks > max_safe_block, build_blocking
        should generate compound BlockingKeyConfig(fields=[col_a, col_b])."""
        random.seed(42)
        n = 10000
        models = [f"Model{i}" for i in range(5)]     # 5 unique -> avg 2000/block
        states = [f"State{i}" for i in range(4)]      # 4 unique -> avg 2500/block
        df = pl.DataFrame({
            "model": [random.choice(models) for _ in range(n)],
            "state": [random.choice(states) for _ in range(n)],
            "price": [str(random.randint(1000, 9999)) for _ in range(n)],
        })

        profiles = [
            ColumnProfile(name="model", dtype="String", col_type="name", confidence=0.8),
            ColumnProfile(name="state", dtype="String", col_type="geo", confidence=0.8),
            ColumnProfile(name="price", dtype="String", col_type="numeric", confidence=0.8),
        ]

        config = build_blocking(profiles, df)

        # Should produce multi_pass with compound keys
        assert config.strategy == "multi_pass"
        assert config.skip_oversized is True
        assert config.max_block_size == 1000
        # At least one pass should have 2 fields (compound key)
        compound_passes = [p for p in (config.passes or []) if len(p.fields) == 2]
        assert len(compound_passes) >= 1, f"Expected compound passes, got: {config.passes}"

    def test_compound_fallback_when_no_pair_works(self):
        """When no compound pair brings blocks below threshold, fall through to
        name-based fallback with skip_oversized=True."""
        n = 5000
        df = pl.DataFrame({
            "col_a": ["A"] * n,
            "col_b": ["B"] * n,
        })
        profiles = [
            ColumnProfile(name="col_a", dtype="String", col_type="name", confidence=0.8),
            ColumnProfile(name="col_b", dtype="String", col_type="string", confidence=0.8),
        ]
        config = build_blocking(profiles, df)
        assert config is not None
        assert config.skip_oversized is True

    def test_candidate_pool_excludes_numeric_date_identifier(self):
        """Compound candidate pool should exclude numeric, date, and identifier columns."""
        random.seed(42)
        n = 10000
        df = pl.DataFrame({
            "name": [f"Name{random.randint(1, 5)}" for _ in range(n)],
            "state": [f"State{random.randint(1, 4)}" for _ in range(n)],
            "year": [str(random.randint(2000, 2025)) for _ in range(n)],
            "id": [str(i) for i in range(n)],
            "amount": [str(random.randint(1, 999)) for _ in range(n)],
        })
        profiles = [
            ColumnProfile(name="name", dtype="String", col_type="name", confidence=0.8),
            ColumnProfile(name="state", dtype="String", col_type="geo", confidence=0.8),
            ColumnProfile(name="year", dtype="String", col_type="date", confidence=0.8),
            ColumnProfile(name="id", dtype="String", col_type="identifier", confidence=0.8),
            ColumnProfile(name="amount", dtype="String", col_type="numeric", confidence=0.8),
        ]
        config = build_blocking(profiles, df)
        # Compound keys should only use name + geo, not date/identifier/numeric
        if config.strategy == "multi_pass" and config.passes:
            all_fields = set()
            for p in config.passes:
                all_fields.update(p.fields)
            assert "id" not in all_fields
            assert "amount" not in all_fields


class TestLLMBlockingKeySuggestion:
    """Tests for LLM-assisted blocking key selection."""

    def _make_df_and_profiles(self):
        random.seed(42)
        n = 10000
        models = [f"Model{i}" for i in range(5)]
        states = [f"State{i}" for i in range(4)]
        df = pl.DataFrame({
            "model": [random.choice(models) for _ in range(n)],
            "state": [random.choice(states) for _ in range(n)],
        })
        profiles = [
            ColumnProfile(name="model", dtype="String", col_type="name", confidence=0.8),
            ColumnProfile(name="state", dtype="String", col_type="geo", confidence=0.8),
        ]
        return df, profiles

    @patch("goldenmatch.core.autoconfig._call_llm_for_blocking")
    def test_valid_llm_suggestion_used(self, mock_llm):
        mock_llm.return_value = '{"passes": [{"fields": ["model", "state"], "reason": "test"}]}'
        df, profiles = self._make_df_and_profiles()
        config = build_blocking(profiles, df, llm_provider="openai")
        assert config.strategy == "multi_pass"
        compound_passes = [p for p in (config.passes or []) if len(p.fields) == 2]
        assert len(compound_passes) >= 1
        mock_llm.assert_called_once()

    @patch("goldenmatch.core.autoconfig._call_llm_for_blocking")
    def test_llm_bad_column_name_rejected(self, mock_llm):
        mock_llm.return_value = '{"passes": [{"fields": ["nonexistent", "state"], "reason": "bad"}]}'
        df, profiles = self._make_df_and_profiles()
        config = build_blocking(profiles, df, llm_provider="openai")
        # Should still produce a valid config via greedy fallback
        assert config is not None
        assert config.strategy == "multi_pass"

    @patch("goldenmatch.core.autoconfig._call_llm_for_blocking")
    def test_llm_failure_falls_back_to_greedy(self, mock_llm):
        mock_llm.side_effect = Exception("API timeout")
        df, profiles = self._make_df_and_profiles()
        config = build_blocking(profiles, df, llm_provider="openai")
        assert config is not None
        assert config.skip_oversized is True

    @patch("goldenmatch.core.autoconfig._call_llm_for_blocking")
    def test_llm_invalid_json_falls_back(self, mock_llm):
        mock_llm.return_value = "not json at all"
        df, profiles = self._make_df_and_profiles()
        config = build_blocking(profiles, df, llm_provider="openai")
        assert config is not None

    @patch("goldenmatch.core.autoconfig._call_llm_for_blocking")
    def test_llm_oversized_suggestion_rejected(self, mock_llm):
        mock_llm.return_value = '{"passes": [{"fields": ["model"], "reason": "bad idea"}]}'
        df, profiles = self._make_df_and_profiles()
        config = build_blocking(profiles, df, llm_provider="openai")
        assert config is not None


class TestCompoundBlockingIntegration:
    """Integration test: auto_configure_df on a wide dataset with oversized single-column blocks."""

    def test_wide_dataset_produces_safe_config(self):
        random.seed(42)
        n = 10000
        models = [f"Model{random.choice('ABCDEFGHIJ')}{i}" for i in range(100)]
        states = [f"State{i}" for i in range(25)]
        types = [f"Type{i}" for i in range(10)]

        df = pl.DataFrame({
            "equipment_name": [random.choice(models) for _ in range(n)],
            "state": [random.choice(states) for _ in range(n)],
            "equipment_type": [random.choice(types) for _ in range(n)],
            "year_made": [str(random.randint(2000, 2025)) for _ in range(n)],
            "serial": [str(i) for i in range(n)],
        })

        from goldenmatch.core.autoconfig import auto_configure_df
        config = auto_configure_df(df)

        assert config.blocking is not None
        if config.blocking.strategy == "multi_pass":
            assert config.blocking.skip_oversized is True
            assert config.blocking.max_block_size <= 1000
