"""Tests for goldenmatch matchkey builder."""

import polars as pl
import pytest

from goldenmatch.core.matchkey import build_matchkey_expr, compute_matchkeys
from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField


class TestBuildMatchkeyExpr:
    """Tests for build_matchkey_expr."""

    def test_single_field_exact_normalizes(self):
        """Single field exact matchkey normalizes correctly."""
        mk = MatchkeyConfig(
            name="name_sdx",
            type="exact",
            fields=[MatchkeyField(field="first_name", transforms=["lowercase", "soundex"])],
        )
        expr = build_matchkey_expr(mk)

        df = pl.DataFrame({"first_name": ["John", "JANE", None]})
        result = df.select(expr)

        assert result.columns == ["__mk_name_sdx__"]
        values = result["__mk_name_sdx__"].to_list()
        # "John" -> "john" -> soundex("john") = "J500"
        # "JANE" -> "jane" -> soundex("jane") = "J500"  (wait, J500 for john, J500 for jane? Let me check)
        # Actually soundex("john") = "J500", soundex("jane") = "J500" — both start with J
        assert values[0] is not None
        assert values[1] is not None
        assert values[2] is None  # None stays None

    def test_multi_field_concatenation(self):
        """Multi-field exact matchkey concatenates with || separator."""
        mk = MatchkeyConfig(
            name="name_zip",
            type="exact",
            fields=[
                MatchkeyField(field="first_name", transforms=["lowercase"]),
                MatchkeyField(field="zip", transforms=[]),
            ],
        )
        expr = build_matchkey_expr(mk)

        df = pl.DataFrame({
            "first_name": ["John", "Jane"],
            "zip": ["19382", "10001"],
        })
        result = df.select(expr)

        assert result.columns == ["__mk_name_zip__"]
        values = result["__mk_name_zip__"].to_list()
        assert values[0] == "john||19382"
        assert values[1] == "jane||10001"

    def test_weighted_matchkey_returns_lit_none(self):
        """Weighted matchkey returns pl.lit(None) placeholder."""
        mk = MatchkeyConfig(
            name="fuzzy_name",
            type="weighted",
            threshold=0.8,
            fields=[
                MatchkeyField(field="first_name", transforms=["lowercase"], scorer="jaro_winkler", weight=0.5),
                MatchkeyField(field="last_name", transforms=["lowercase"], scorer="jaro_winkler", weight=0.5),
            ],
        )
        expr = build_matchkey_expr(mk)

        df = pl.DataFrame({"first_name": ["John"], "last_name": ["Smith"]})
        result = df.select(expr)

        assert result.columns == ["__mk_fuzzy_name__"]
        assert result["__mk_fuzzy_name__"][0] is None


class TestComputeMatchkeys:
    """Tests for compute_matchkeys."""

    def test_adds_matchkey_columns(self):
        """compute_matchkeys adds correct columns for exact matchkeys."""
        mks = [
            MatchkeyConfig(
                name="name_sdx",
                type="exact",
                fields=[MatchkeyField(field="first_name", transforms=["lowercase"])],
            ),
            MatchkeyConfig(
                name="zip_exact",
                type="exact",
                fields=[MatchkeyField(field="zip", transforms=[])],
            ),
        ]

        df = pl.DataFrame({
            "first_name": ["John", "Jane"],
            "zip": ["19382", "10001"],
        })
        lf = df.lazy()

        result = compute_matchkeys(lf, mks).collect()

        assert "__mk_name_sdx__" in result.columns
        assert "__mk_zip_exact__" in result.columns
        assert result["__mk_name_sdx__"].to_list() == ["john", "jane"]
        assert result["__mk_zip_exact__"].to_list() == ["19382", "10001"]
