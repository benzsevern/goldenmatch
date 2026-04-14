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

    def test_price(self):
        assert _classify_by_name("SalePrice") == "numeric"
        assert _classify_by_name("amount") == "numeric"
        assert _classify_by_name("total_cost") == "numeric"
        assert _classify_by_name("revenue") == "numeric"

    def test_id_before_phone(self):
        """ID pattern should match before phone — SalesID is an ID, not a phone."""
        assert _classify_by_name("SalesID") == "identifier"
        assert _classify_by_name("MachineID") == "identifier"
        assert _classify_by_name("PhoneID") == "identifier"
        assert _classify_by_name("TelephoneId") == "identifier"

    def test_phone_still_works(self):
        """Phone columns without ID suffix still classified as phone."""
        assert _classify_by_name("phone") == "phone"
        assert _classify_by_name("mobile") == "phone"
        assert _classify_by_name("fax_number") == "phone"

    def test_unknown(self):
        assert _classify_by_name("foobar") is None


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


class TestIdentifierNotOverridden:
    """Identifier columns should not be overridden by phone/zip data profiling."""

    def test_id_column_with_numeric_data(self):
        """SalesID with 7-digit integers should stay 'identifier', not become 'phone'."""
        df = pl.DataFrame({
            "SalesID": [str(1139246 + i) for i in range(100)],
            "name": [f"Record {i}" for i in range(100)],
        })
        profiles = profile_columns(df)
        types = {p.name: p.col_type for p in profiles}
        assert types["SalesID"] == "identifier"

    def test_price_column_not_zip(self):
        """SalePrice with 5-digit values should be 'numeric', not 'zip'."""
        df = pl.DataFrame({
            "SalePrice": [str(v) for v in [66000, 57000, 10000, 38500, 11000] * 20],
            "name": [f"Record {i}" for i in range(100)],
        })
        profiles = profile_columns(df)
        types = {p.name: p.col_type for p in profiles}
        assert types["SalePrice"] == "numeric"


class TestUtilityRanking:
    """Fuzzy field truncation should rank by match utility, not column order."""

    def test_high_utility_field_kept(self):
        """A high-cardinality, long-string field should rank above short low-cardinality ones."""
        profiles = [
            ColumnProfile("UsageBand", "Utf8", "name", 0.7,
                          sample_values=["Low", "Medium", "High"],
                          cardinality_ratio=0.01, avg_len=4.0),
            ColumnProfile("ProductSize", "Utf8", "name", 0.7,
                          sample_values=["Small", "Large"],
                          cardinality_ratio=0.005, avg_len=5.0),
            ColumnProfile("Drive_System", "Utf8", "name", 0.7,
                          sample_values=["2WD", "4WD"],
                          cardinality_ratio=0.005, avg_len=3.0),
            ColumnProfile("Enclosure", "Utf8", "name", 0.7,
                          sample_values=["OROPS", "EROPS"],
                          cardinality_ratio=0.005, avg_len=5.0),
            ColumnProfile("Forks", "Utf8", "name", 0.7,
                          sample_values=["Yes", "No"],
                          cardinality_ratio=0.005, avg_len=3.0),
            ColumnProfile("Ride_Control", "Utf8", "name", 0.7,
                          sample_values=["Yes", "No"],
                          cardinality_ratio=0.005, avg_len=3.0),
            ColumnProfile("fiModelDesc", "Utf8", "string", 0.5,
                          sample_values=["580D LL", "310SE", "416D 4x4x4"],
                          cardinality_ratio=0.8, avg_len=12.0),
        ]
        mks = build_matchkeys(profiles)
        weighted = [mk for mk in mks if mk.type == "weighted"]
        assert len(weighted) == 1
        field_names = [f.field for f in weighted[0].fields if f.field]
        # fiModelDesc should be included despite being last in column order
        assert "fiModelDesc" in field_names


class TestBuildMatchkeys:
    def test_exact_fields(self):
        profiles = [
            ColumnProfile("email", "Utf8", "email", 0.9, cardinality_ratio=0.9),
            ColumnProfile("phone", "Utf8", "phone", 0.9, cardinality_ratio=0.5),
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

    def test_name_with_geo_compounds_blocking(self):
        """When geo columns exist, name-based blocking should compound with geo."""
        profiles = [
            ColumnProfile("facility_name", "Utf8", "name", 0.9, cardinality_ratio=0.8),
            ColumnProfile("state", "Utf8", "geo", 0.9, cardinality_ratio=0.01),
            ColumnProfile("address", "Utf8", "address", 0.7, cardinality_ratio=0.9),
        ]
        # Create data where name-only blocking produces large blocks
        # but state+name produces small blocks
        states = ["AL", "CA", "NY", "TX", "FL"]
        names = ["MEMORIAL HOSPITAL", "COMMUNITY HOSPITAL", "ST LUKES"]
        rows = []
        for s in states:
            for n in names:
                rows.append({"facility_name": n, "state": s, "address": f"123 {s} ST"})
        df = pl.DataFrame(rows)
        blocking = build_blocking(profiles, df)
        # Primary key should include both geo and name
        primary_fields = blocking.keys[0].fields
        assert "state" in primary_fields, f"Expected 'state' in blocking keys, got {primary_fields}"
        assert "facility_name" in primary_fields

    def test_name_without_geo_stays_name_only(self):
        """Without geo columns, name-based blocking should remain name-only."""
        profiles = [
            ColumnProfile("name", "Utf8", "name", 0.9),
            ColumnProfile("description", "Utf8", "description", 0.7),
        ]
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"], "description": ["a", "b", "c"]})
        blocking = build_blocking(profiles, df)
        assert blocking.strategy == "multi_pass"
        assert blocking.keys[0].fields == ["name"]

    def test_geo_compound_high_null_geo_skipped(self):
        """Geo columns with >20% nulls should not be used for compound blocking."""
        profiles = [
            ColumnProfile("name", "Utf8", "name", 0.9),
            ColumnProfile("state", "Utf8", "geo", 0.9, null_rate=0.3),
        ]
        # 40% nulls in state column
        df = pl.DataFrame({
            "name": ["John", "Jane", "Bob", "Alice", "Eve"],
            "state": ["AL", "CA", None, None, "NY"],
        })
        blocking = build_blocking(profiles, df)
        # Should fall back to name-only since geo has high null rate
        assert blocking.keys[0].fields == ["name"]


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


class TestLLMColumnClassification:
    """Test LLM-assisted column classification in profile_columns."""

    def test_happy_path_corrects_ambiguous_types(self):
        """LLM response should override ambiguous classifications."""
        from goldenmatch.core.autoconfig import _llm_classify_columns, ColumnProfile

        profiles = [
            ColumnProfile("SalesID", "Utf8", "phone", 0.7, ["1139246", "1139248"]),
            ColumnProfile("SalePrice", "Utf8", "zip", 0.7, ["66000", "57000"]),
            ColumnProfile("fiModelDesc", "Utf8", "string", 0.3, ["580D", "310SE"]),
            ColumnProfile("state", "Utf8", "geo", 0.9, ["CA", "TX"]),  # high confidence, should NOT change
        ]

        llm_response = '{"classifications": {"SalesID": "identifier", "SalePrice": "numeric", "fiModelDesc": "description"}, "match_ranking": ["fiModelDesc", "SalesID", "SalePrice"]}'

        with patch("goldenmatch.core.autoconfig._call_llm_for_blocking", return_value=llm_response):
            result = _llm_classify_columns(profiles, "openai")

        types = {p.name: p.col_type for p in result}
        assert types["SalesID"] == "identifier"
        assert types["SalePrice"] == "numeric"
        assert types["fiModelDesc"] == "description"
        assert types["state"] == "geo"  # unchanged (high confidence)

    def test_markdown_wrapped_json(self):
        """LLM response wrapped in markdown code blocks should be parsed."""
        from goldenmatch.core.autoconfig import _llm_classify_columns, ColumnProfile

        profiles = [
            ColumnProfile("col1", "Utf8", "string", 0.3, ["abc"]),
        ]

        llm_response = '```json\n{"classifications": {"col1": "name"}, "match_ranking": ["col1"]}\n```'

        with patch("goldenmatch.core.autoconfig._call_llm_for_blocking", return_value=llm_response):
            result = _llm_classify_columns(profiles, "openai")

        assert result[0].col_type == "name"

    def test_unparseable_response_returns_original(self):
        """Garbage LLM response should return profiles unchanged."""
        from goldenmatch.core.autoconfig import _llm_classify_columns, ColumnProfile

        profiles = [
            ColumnProfile("col1", "Utf8", "string", 0.3, ["abc"]),
        ]

        with patch("goldenmatch.core.autoconfig._call_llm_for_blocking", return_value="not json at all"):
            result = _llm_classify_columns(profiles, "openai")

        assert result[0].col_type == "string"  # unchanged

    def test_api_failure_returns_original(self):
        """LLM API failure should return profiles unchanged."""
        from goldenmatch.core.autoconfig import _llm_classify_columns, ColumnProfile
        import urllib.error

        profiles = [
            ColumnProfile("col1", "Utf8", "string", 0.3, ["abc"]),
        ]

        with patch("goldenmatch.core.autoconfig._call_llm_for_blocking",
                    side_effect=urllib.error.URLError("network down")):
            result = _llm_classify_columns(profiles, "openai")

        assert result[0].col_type == "string"

    def test_non_string_type_ignored(self):
        """LLM returning non-string type values should not crash."""
        from goldenmatch.core.autoconfig import _llm_classify_columns, ColumnProfile

        profiles = [
            ColumnProfile("col1", "Utf8", "string", 0.3, ["abc"]),
        ]

        llm_response = '{"classifications": {"col1": 123}, "match_ranking": []}'

        with patch("goldenmatch.core.autoconfig._call_llm_for_blocking", return_value=llm_response):
            result = _llm_classify_columns(profiles, "openai")

        assert result[0].col_type == "string"  # unchanged, 123 ignored


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


# ── Cardinality guard tests ──────────────────────────────────────────────


class TestBlockingCardinalityGuard:
    """Near-unique columns (cardinality_ratio >= 0.95) should be excluded from blocking."""

    def test_unique_id_excluded_from_blocking(self):
        """DataFrame with unique rec_id column — blocking should not use it."""
        random.seed(42)
        n = 500
        df = pl.DataFrame({
            "rec_id": [f"rec-{i}" for i in range(n)],
            "name": [random.choice(["John", "Jane", "Bob", "Alice"]) for _ in range(n)],
        })
        profiles = profile_columns(df)
        config = build_blocking(profiles, df)
        # rec_id has cardinality_ratio=1.0, so it must NOT appear in blocking keys
        all_fields = set()
        for k in config.keys:
            all_fields.update(k.fields)
        if config.passes:
            for p in config.passes:
                all_fields.update(p.fields)
        assert "rec_id" not in all_fields

    def test_unique_id_column_named_id(self):
        """DataFrame with unique 'id' column (like DBLP-ACM) — blocking should not use it."""
        random.seed(42)
        n = 500
        df = pl.DataFrame({
            "id": [str(i) for i in range(n)],
            "title": [f"Paper about topic {random.randint(1, 50)}" for _ in range(n)],
        })
        profiles = profile_columns(df)
        config = build_blocking(profiles, df)
        all_fields = set()
        for k in config.keys:
            all_fields.update(k.fields)
        if config.passes:
            for p in config.passes:
                all_fields.update(p.fields)
        assert "id" not in all_fields


class TestExactMatchkeyCardinalityFloor:
    """Low-cardinality columns should not get exact matchkeys (would match nearly everything)."""

    def test_low_cardinality_geo_excluded_from_exact(self):
        """State column with 4 values / 1000 rows = 0.004 ratio — no exact matchkey."""
        random.seed(42)
        n = 1000
        states = ["CA", "TX", "NY", "FL"]
        df = pl.DataFrame({
            "state": [random.choice(states) for _ in range(n)],
            "name": [f"Person {i}" for i in range(n)],
        })
        profiles = profile_columns(df)
        mks = build_matchkeys(profiles, df=df)
        exact_mks = [mk for mk in mks if mk.type == "exact"]
        for mk in exact_mks:
            for f in mk.fields:
                assert f.field != "state", "state should not have an exact matchkey"

    def test_single_value_column_excluded(self):
        """Column with all same value (cardinality_ratio ~ 1/N) — no exact matchkey."""
        random.seed(42)
        n = 200
        df = pl.DataFrame({
            "county_desc": ["Los Angeles"] * n,
            "name": [f"Person {i}" for i in range(n)],
        })
        profiles = profile_columns(df)
        mks = build_matchkeys(profiles, df=df)
        exact_mks = [mk for mk in mks if mk.type == "exact"]
        for mk in exact_mks:
            for f in mk.fields:
                assert f.field != "county_desc", "county_desc should not have an exact matchkey"


class TestDescriptionFuzzyFallback:
    """Description columns should also get a fuzzy scorer (token_sort), not just record_embedding."""

    def test_description_column_gets_fuzzy_scorer(self):
        """DataFrame with long title strings (avg_len > 50) — title should appear in fuzzy matchkey fields."""
        profiles = [
            ColumnProfile(
                "title", "Utf8", "description", 0.7,
                sample_values=["A very long description of a research paper about machine learning and NLP techniques"],
                avg_len=80.0,
            ),
        ]
        mks = build_matchkeys(profiles)
        weighted = [mk for mk in mks if mk.type == "weighted"]
        assert len(weighted) >= 1
        fuzzy_field_names = []
        for mk in weighted:
            for f in mk.fields:
                if f.scorer and f.scorer != "record_embedding":
                    fuzzy_field_names.append(f.field)
        assert "title" in fuzzy_field_names, (
            f"title should appear as a fuzzy field, got fields: {fuzzy_field_names}"
        )


class TestAutoConfigBenchmarkDatasets:
    """End-to-end autoconfig tests against real benchmark datasets."""

    def test_febrl_autoconfig(self):
        """Febrl3: rec_id should NOT be in blocking, state should NOT get exact matchkey."""
        pytest.importorskip("recordlinkage")
        import recordlinkage.datasets

        df_rl, _links = recordlinkage.datasets.load_febrl3(return_links=True)
        # Convert to Polars — recordlinkage returns pandas with index as rec_id
        import pandas as pd
        df_pd = df_rl.reset_index()
        df = pl.from_pandas(df_pd)

        profiles = profile_columns(df)
        blocking = build_blocking(profiles, df)

        # rec_id is unique — should NOT appear in blocking keys
        all_blocking_fields = set()
        for k in blocking.keys:
            all_blocking_fields.update(k.fields)
        if blocking.passes:
            for p in blocking.passes:
                all_blocking_fields.update(p.fields)
        assert "rec_id" not in all_blocking_fields, (
            f"rec_id should be excluded from blocking, got fields: {all_blocking_fields}"
        )

        # state has low cardinality — check that if cardinality_ratio < 0.01, it's excluded
        # Febrl3 state has ~36 unique / 5000 rows, but profile_columns samples, so
        # actual ratio may be slightly above the 0.01 floor. Verify the guard works
        # by checking that any exact matchkey columns have cardinality_ratio >= 0.01
        profile_by_name = {p.name: p for p in profiles}
        mks = build_matchkeys(profiles, df=df)
        exact_mks = [mk for mk in mks if mk.type == "exact"]
        for mk in exact_mks:
            for f in mk.fields:
                p = profile_by_name.get(f.field)
                if p:
                    assert p.cardinality_ratio >= 0.01, (
                        f"{f.field} has cardinality_ratio={p.cardinality_ratio:.4f} "
                        "but got an exact matchkey (should be excluded below 0.01)"
                    )

    def test_dblp_acm_autoconfig(self):
        """DBLP-ACM: id should NOT be in blocking, title should appear in fuzzy fields."""
        from pathlib import Path
        _DBLP_ACM_DIR = Path(__file__).parent / "benchmarks" / "datasets" / "DBLP-ACM"
        dblp_path = _DBLP_ACM_DIR / "DBLP2.csv"
        if not dblp_path.exists():
            pytest.skip("DBLP-ACM benchmark dataset not available")
        df = pl.read_csv(str(dblp_path), encoding="utf8-lossy", ignore_errors=True)

        profiles = profile_columns(df)
        blocking = build_blocking(profiles, df)

        # id is unique — should NOT appear in blocking keys
        all_blocking_fields = set()
        for k in blocking.keys:
            all_blocking_fields.update(k.fields)
        if blocking.passes:
            for p in blocking.passes:
                all_blocking_fields.update(p.fields)
        assert "id" not in all_blocking_fields, (
            f"id should be excluded from blocking, got fields: {all_blocking_fields}"
        )

        # title is a description column — should appear in fuzzy matchkey fields
        mks = build_matchkeys(profiles, df=df)
        weighted = [mk for mk in mks if mk.type == "weighted"]
        fuzzy_field_names = set()
        for mk in weighted:
            for f in mk.fields:
                if f.field:
                    fuzzy_field_names.add(f.field)
        assert "title" in fuzzy_field_names, (
            f"title should appear in fuzzy fields, got: {fuzzy_field_names}"
        )


# -- Cardinality boundary value tests ----------------------------------------


class TestCardinalityBoundaryValues:
    def test_blocking_boundary_0_95_excluded(self):
        """Column with cardinality_ratio exactly 0.95 should be excluded from blocking."""
        random.seed(42)
        # 95 unique values in 100 rows = 0.95
        df = pl.DataFrame({
            "email": [f"user{i}@test.com" for i in range(95)] + [f"user{i}@test.com" for i in range(5)],
            "name": [random.choice(["alice", "bob", "charlie"]) for _ in range(100)],
        })
        profiles = profile_columns(df)
        email_profile = next(p for p in profiles if p.name == "email")
        assert email_profile.cardinality_ratio >= 0.95
        config = build_blocking(profiles, df)
        all_fields = []
        for k in [config.keys[0]] + (config.passes or []):
            all_fields.extend(k.fields)
        assert "email" not in all_fields

    def test_blocking_boundary_0_94_included(self):
        """Column with cardinality_ratio 0.94 should be eligible for blocking."""
        random.seed(42)
        # 47 unique values in 50 rows = 0.94
        df = pl.DataFrame({
            "email": [f"user{i}@test.com" for i in range(47)] + [f"user{i % 47}@test.com" for i in range(3)],
            "name": [random.choice(["alice", "bob", "charlie"]) for _ in range(50)],
        })
        profiles = profile_columns(df)
        email_profile = next(p for p in profiles if p.name == "email")
        assert email_profile.cardinality_ratio < 0.95
        # email should now be eligible (though may or may not be selected depending on block size)

    def test_geo_never_promoted_to_exact_matchkey(self):
        """col_type='geo' is a blocking signal, not an identity claim.

        Two records sharing a city or state are not the same entity. Geo
        columns are excluded from exact matchkeys regardless of cardinality,
        and still participate in blocking via build_blocking().
        """
        random.seed(42)
        profiles = [
            ColumnProfile("state", "Utf8", "geo", 0.9, cardinality_ratio=0.9),
            ColumnProfile("name", "Utf8", "name", 0.9, cardinality_ratio=0.5),
        ]
        matchkeys = build_matchkeys(profiles)
        exact_fields = []
        for mk in matchkeys:
            if mk.type == "exact":
                exact_fields.extend(f.field for f in mk.fields)
        assert "state" not in exact_fields

    def test_zip_never_promoted_to_exact_matchkey(self):
        """col_type='zip' is a blocking signal, not an identity claim."""
        random.seed(42)
        profiles = [
            ColumnProfile("zip_code", "Utf8", "zip", 0.9, cardinality_ratio=0.9),
            ColumnProfile("name", "Utf8", "name", 0.9, cardinality_ratio=0.5),
        ]
        matchkeys = build_matchkeys(profiles)
        exact_fields = []
        for mk in matchkeys:
            if mk.type == "exact":
                exact_fields.extend(f.field for f in mk.fields)
        assert "zip_code" not in exact_fields

    def test_exact_typed_low_cardinality_excluded(self):
        """Even an exact-scoring column (e.g. phone, email) with
        cardinality_ratio < 0.5 must not back an exact matchkey.

        This is the birth_year-as-phone failure mode: a 4-digit year that
        an upstream transform reshapes into an ISO date looks phone-shaped
        to the phone classifier, but with ~80 distinct values in 5K rows
        its cardinality is too low to function as an identity claim.
        """
        random.seed(42)
        profiles = [
            ColumnProfile("year_looks_phoney", "Utf8", "phone", 0.9, cardinality_ratio=0.016),
            ColumnProfile("name", "Utf8", "name", 0.9, cardinality_ratio=0.5),
        ]
        matchkeys = build_matchkeys(profiles)
        exact_fields = []
        for mk in matchkeys:
            if mk.type == "exact":
                exact_fields.extend(f.field for f in mk.fields)
        assert "year_looks_phoney" not in exact_fields

    def test_exact_typed_high_cardinality_included(self):
        """An exact-scoring column (email) with high cardinality is still
        promoted to an exact matchkey."""
        random.seed(42)
        profiles = [
            ColumnProfile("email", "Utf8", "email", 0.9, cardinality_ratio=0.9),
            ColumnProfile("name", "Utf8", "name", 0.9, cardinality_ratio=0.5),
        ]
        matchkeys = build_matchkeys(profiles)
        exact_fields = []
        for mk in matchkeys:
            if mk.type == "exact":
                exact_fields.extend(f.field for f in mk.fields)
        assert "email" in exact_fields

    def test_matchkey_default_cardinality_not_affected(self):
        """Profiles with default cardinality_ratio=0.0 should NOT be affected
        by the guard (the guard only triggers when cardinality_ratio > 0).

        Callers that construct ColumnProfile without a cardinality_ratio
        (tests, small synthetic fixtures) should not have their exact
        matchkeys silently dropped — the guard only suppresses columns
        with real measured low cardinality.
        """
        random.seed(42)
        profiles = [
            ColumnProfile("email", "Utf8", "email", 0.9),  # cardinality_ratio defaults to 0.0
            ColumnProfile("name", "Utf8", "name", 0.9),
        ]
        matchkeys = build_matchkeys(profiles)
        exact_fields = []
        for mk in matchkeys:
            if mk.type == "exact":
                exact_fields.extend(f.field for f in mk.fields)
        assert "email" in exact_fields, "Default cardinality_ratio=0.0 should not trigger the guard"


class TestCrossSourceOverlap:
    """Tests for _check_source_overlap blocking guard."""

    def test_zero_overlap_skips_blocking_key(self):
        """Blocking key with zero overlap between sources should return 0.0."""
        from goldenmatch.core.autoconfig import _check_source_overlap

        df = pl.DataFrame({
            "venue": ["VLDB", "SIGMOD", "VLDB", "Very Large Databases", "SIGMOD Conf"],
            "__source__": ["A", "A", "A", "B", "B"],
        })
        overlap = _check_source_overlap(df, "venue")
        assert overlap == 0.0

    def test_full_overlap_passes(self):
        """Blocking key with overlap should return > 0.0."""
        from goldenmatch.core.autoconfig import _check_source_overlap

        df = pl.DataFrame({
            "year": [2020, 2021, 2022, 2020, 2021],
            "__source__": ["A", "A", "A", "B", "B"],
        })
        overlap = _check_source_overlap(df, "year")
        assert overlap > 0.0

    def test_no_source_column_returns_one(self):
        """Without __source__ column, should return 1.0 (no check)."""
        from goldenmatch.core.autoconfig import _check_source_overlap

        df = pl.DataFrame({
            "venue": ["VLDB", "SIGMOD", "VLDB"],
        })
        overlap = _check_source_overlap(df, "venue")
        assert overlap == 1.0

    def test_single_source_returns_one(self):
        """With only one source, should return 1.0."""
        from goldenmatch.core.autoconfig import _check_source_overlap

        df = pl.DataFrame({
            "venue": ["VLDB", "SIGMOD"],
            "__source__": ["A", "A"],
        })
        overlap = _check_source_overlap(df, "venue")
        assert overlap == 1.0


class TestDomainAwareAutoConfig:
    """Tests for domain-aware auto-config."""

    def test_high_confidence_domain_extracts_and_generates_matchkeys(self):
        """Electronics-like data should trigger extraction and domain matchkeys."""
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "title": [
                "Sony WH-1000XM5 Wireless Headphones Black",
                "Sony WH-1000XM5 Wireless Headphones Silver",
                "Samsung Galaxy S24 Ultra 256GB Phantom Black",
                "Apple iPhone 15 Pro Max 512GB Natural Titanium",
                "Apple iPhone 15 Pro Max 512GB Blue Titanium",
            ],
            "brand": ["Sony", "Sony", "Samsung", "Apple", "Apple"],
            "price": [348.0, 348.0, 1199.0, 1399.0, 1399.0],
        })

        config = auto_configure_df(df)
        all_fields = []
        for mk in config.get_matchkeys():
            all_fields.extend(f.field for f in mk.fields if f.field)

        domain_fields = [f for f in all_fields if f and f.startswith("__")]
        assert len(domain_fields) > 0, f"Expected domain fields in matchkeys, got: {all_fields}"
        assert any("brand" in f for f in domain_fields), f"Expected __brand__ field, got: {domain_fields}"

    def test_low_confidence_no_extraction(self):
        """Ambiguous data should not trigger domain extraction."""
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "col_a": ["foo", "bar", "baz"],
            "col_b": ["x", "y", "z"],
            "col_c": [1, 2, 3],
        })

        config = auto_configure_df(df)
        all_fields = []
        for mk in config.get_matchkeys():
            all_fields.extend(f.field for f in mk.fields if f.field)

        domain_fields = [f for f in all_fields if f and f.startswith("__")]
        assert len(domain_fields) == 0

    def test_manual_domain_config_overrides(self):
        """When domain_config is provided, auto-config skips domain detection."""
        from goldenmatch.core.autoconfig import auto_configure_df
        from goldenmatch.config.schemas import DomainConfig
        from unittest.mock import patch

        df = pl.DataFrame({
            "brand": ["Sony", "Samsung", "Apple"],
            "model": ["XM5", "S24", "iPhone15"],
        })

        with patch("goldenmatch.core.domain.detect_domain") as mock_detect:
            config = auto_configure_df(df, domain_config=DomainConfig(enabled=False))

        # detect_domain should NOT have been called
        mock_detect.assert_not_called()

    def test_unknown_domain_normal_behavior(self):
        """Unknown domain should produce normal auto-config (no extraction)."""
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "name": ["John Smith", "Jane Doe", "Bob Wilson"],
            "email": ["john@test.com", "jane@test.com", "bob@test.com"],
        })

        config = auto_configure_df(df)
        all_fields = []
        for mk in config.get_matchkeys():
            all_fields.extend(f.field for f in mk.fields if f.field)

        domain_fields = [f for f in all_fields if f and f.startswith("__")]
        assert len(domain_fields) == 0
        assert len(config.get_matchkeys()) > 0


class TestDataDrivenStrategy:
    """Tests for data-driven strategy selection in auto-config."""

    def test_learned_blocking_large_dataset(self):
        """Datasets >= 50K rows should get learned blocking.

        The threshold was raised from 5K in fix/autoconfig-regressions
        because at 5K the learner trained its predicates on 100% of the
        dataset (sample_size=5K), producing multi-minute runtimes and
        overfit predicates on small inputs. The gate is now 50K rows,
        and sample_size is capped at 25% of the dataset.
        """
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "name": [f"Person {i}" for i in range(60_000)],
            "city": [f"City {i % 100}" for i in range(60_000)],
        })

        config = auto_configure_df(df)
        assert config.blocking is not None
        assert config.blocking.strategy == "learned"
        assert config.blocking.learned_min_recall == 0.95
        assert config.blocking.learned_sample_size <= 60_000 // 4

    def test_static_blocking_small_dataset(self):
        """Datasets < 50K rows should keep static/multi_pass blocking."""
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "name": ["John Smith", "Jane Doe", "Bob Wilson"],
            "city": ["NYC", "LA", "Chicago"],
        })

        config = auto_configure_df(df)
        assert config.blocking is not None
        assert config.blocking.strategy != "learned"

    def test_medium_dataset_not_upgraded_to_learned(self):
        """5K-49K row datasets must stay on static/multi_pass blocking."""
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "name": [f"Person {i}" for i in range(5_000)],
            "city": [f"City {i % 100}" for i in range(5_000)],
        })

        config = auto_configure_df(df)
        assert config.blocking is not None
        assert config.blocking.strategy != "learned"

    def test_reranking_enabled_three_plus_fields(self):
        """Weighted matchkey with 3+ fields should enable reranking."""
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "first_name": ["John", "Jane", "Bob", "Alice"],
            "last_name": ["Smith", "Doe", "Wilson", "Brown"],
            "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm Blvd"],
            "city": ["NYC", "LA", "Chicago", "Houston"],
        })

        config = auto_configure_df(df)
        weighted_mks = [mk for mk in config.get_matchkeys() if mk.type == "weighted"]
        assert len(weighted_mks) > 0, "Expected at least one weighted matchkey"
        multi_field = [mk for mk in weighted_mks if len(mk.fields) >= 3]
        if multi_field:
            assert multi_field[0].rerank is True

    def test_threshold_lowered_high_null_rate(self):
        """High null rate across fuzzy fields should lower threshold."""
        from goldenmatch.core.autoconfig import auto_configure_df

        names = ["John Smith", "Jane Doe", None, "Bob Wilson", None,
                 "Alice Brown", None, "Eve Davis"]
        cities = ["New York City", "Los Angeles", "Chicago Metro", "Houston Area",
                  "Phoenix Valley", "Philadelphia", "Dallas Fort Worth", "Denver Metro"]
        df = pl.DataFrame({"name": names, "city": cities})

        config = auto_configure_df(df)
        weighted_mks = [mk for mk in config.get_matchkeys() if mk.type == "weighted"]
        assert len(weighted_mks) > 0, "Expected at least one weighted matchkey"
        assert weighted_mks[0].threshold <= 0.80

    def test_threshold_raised_short_strings(self):
        """Short string fields should raise threshold."""
        from goldenmatch.core.autoconfig import auto_configure_df

        df = pl.DataFrame({
            "first_name": ["Al", "Bo", "Ed", "Jo", "Mo"],
            "last_name": ["Li", "Wu", "Xu", "Ye", "Hu"],
        })

        config = auto_configure_df(df)
        weighted_mks = [mk for mk in config.get_matchkeys() if mk.type == "weighted"]
        assert len(weighted_mks) > 0, "Expected at least one weighted matchkey"
        assert weighted_mks[0].threshold >= 0.80


class TestLLMMemoryAutoEnablement:
    """Tests for LLM + memory auto-enablement."""

    def test_llm_auto_with_api_key(self):
        from goldenmatch.core.autoconfig import auto_configure_df
        from unittest.mock import patch
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"], "email": ["a@t.com", "b@t.com", "c@t.com"]})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}):
            config = auto_configure_df(df, llm_auto=True)
        assert config.llm_scorer is not None
        assert config.llm_scorer.enabled is True
        assert config.llm_scorer.budget.max_cost_usd == 0.05

    def test_llm_auto_no_key(self):
        from goldenmatch.core.autoconfig import auto_configure_df
        from unittest.mock import patch
        import os
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"], "email": ["a@t.com", "b@t.com", "c@t.com"]})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)
            config = auto_configure_df(df, llm_auto=True)
        assert config.llm_scorer is None

    def test_llm_auto_off(self):
        from goldenmatch.core.autoconfig import auto_configure_df
        from unittest.mock import patch
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"], "email": ["a@t.com", "b@t.com", "c@t.com"]})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}):
            config = auto_configure_df(df, llm_auto=False)
        assert config.llm_scorer is None

    def test_memory_with_llm_auto(self):
        from goldenmatch.core.autoconfig import auto_configure_df
        from unittest.mock import patch
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"], "email": ["a@t.com", "b@t.com", "c@t.com"]})
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-fake"}):
            config = auto_configure_df(df, llm_auto=True)
        assert config.memory is not None
        assert config.memory.enabled is True

    def test_memory_off_by_default(self):
        from goldenmatch.core.autoconfig import auto_configure_df
        df = pl.DataFrame({"name": ["John", "Jane", "Bob"], "email": ["a@t.com", "b@t.com", "c@t.com"]})
        config = auto_configure_df(df)
        assert config.memory is None


def test_classify_by_data_cardinality_guard_beats_phone():
    from goldenmatch.core.autoconfig import _classify_by_data
    values = [str(9000000 + i) for i in range(10)]
    col_type, confidence = _classify_by_data(values)
    assert col_type == "identifier", f"got {col_type!r} (confidence={confidence})"
    assert confidence == 0.9
