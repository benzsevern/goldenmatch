"""Tests for custom domain registry."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.core.domain_registry import (
    DomainRulebook,
    discover_rulebooks,
    extract_with_rulebook,
    load_rulebook,
    match_domain,
    save_rulebook,
)


class TestDomainRulebook:
    def test_basic_extraction(self):
        rb = DomainRulebook(
            name="test",
            identifier_patterns={"model": r"\b([A-Z]{2}\d{3})\b"},
            brand_patterns=["Acme", "Globex"],
        )
        rb.compile()
        result = rb.extract("Acme Widget XY123 Premium")
        assert result["brand"] == "Acme"
        assert result["identifiers"]["model"] == "XY123"
        assert result["confidence"] > 0

    def test_name_normalization(self):
        rb = DomainRulebook(
            name="test",
            stop_words=["the", "premium", "edition"],
        )
        rb.compile()
        result = rb.extract("The Premium Widget Edition 2024")
        assert "premium" not in (result["name_normalized"] or "")
        assert "widget" in (result["name_normalized"] or "")

    def test_no_match(self):
        rb = DomainRulebook(
            name="test",
            identifier_patterns={"ndc": r"\b(\d{5}-\d{4}-\d{2})\b"},
        )
        rb.compile()
        result = rb.extract("just some random text")
        assert result["identifiers"] == {}

    def test_attribute_extraction(self):
        rb = DomainRulebook(
            name="test",
            attribute_patterns={"size_mm": r"(\d+)\s*mm"},
        )
        rb.compile()
        result = rb.extract("Catheter 12mm diameter sterile")
        assert "12mm" in result["attributes"].get("size_mm", "")


class TestSaveAndLoad:
    def test_round_trip(self, tmp_path):
        rb = DomainRulebook(
            name="medical",
            signals=["ndc", "fda", "implant"],
            identifier_patterns={"ndc": r"\b(\d{5}-\d{4}-\d{2})\b"},
            brand_patterns=["Medtronic", "Abbott"],
            attribute_patterns={"gauge": r"(\d+)\s*ga"},
            stop_words=["sterile", "disposable"],
        )

        path = save_rulebook(rb, tmp_path / "medical.yaml")
        loaded = load_rulebook(path)

        assert loaded.name == "medical"
        assert loaded.signals == ["ndc", "fda", "implant"]
        assert "ndc" in loaded.identifier_patterns
        assert "Medtronic" in loaded.brand_patterns
        assert "gauge" in loaded.attribute_patterns
        assert "sterile" in loaded.stop_words

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_rulebook("/nonexistent/path.yaml")


class TestDiscovery:
    def test_discover_builtin(self):
        rulebooks = discover_rulebooks()
        # Should find at least the built-in electronics and software rulebooks
        assert "electronics" in rulebooks or "software" in rulebooks

    def test_match_domain_electronics(self):
        rulebooks = discover_rulebooks()
        if not rulebooks:
            pytest.skip("No rulebooks found")
        result = match_domain(["brand", "model", "sku", "price"], rulebooks)
        if result:
            assert result.name in ("electronics", "software", "retail")


class TestExtractWithRulebook:
    def test_extract_df(self):
        rb = DomainRulebook(
            name="test",
            identifier_patterns={"code": r"\b([A-Z]{2}\d{3})\b"},
            brand_patterns=["Acme"],
        )
        rb.compile()

        df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "title": ["Acme Widget AB123", "Globex Gadget CD456", "Unknown thing"],
        })

        enhanced, low_conf = extract_with_rulebook(df, "title", rb, confidence_threshold=0.3)
        assert "__domain_name__" in enhanced.columns
        assert "__domain_brand__" in enhanced.columns
        assert "__domain_ids__" in enhanced.columns
        assert enhanced.height == 3
