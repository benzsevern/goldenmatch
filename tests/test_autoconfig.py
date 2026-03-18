"""Tests for auto-configuration engine."""

from __future__ import annotations

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
