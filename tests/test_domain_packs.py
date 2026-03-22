"""Tests for pre-built domain packs."""
from __future__ import annotations

from pathlib import Path

import pytest

from goldenmatch.core.domain_registry import discover_rulebooks, load_rulebook

DOMAINS_DIR = Path(__file__).parent.parent / "goldenmatch" / "domains"

EXPECTED_PACKS = ["electronics", "software", "healthcare", "financial", "real_estate", "people", "retail"]


class TestDomainPacksDiscovery:
    def test_all_packs_discovered(self):
        rulebooks = discover_rulebooks()
        for name in EXPECTED_PACKS:
            assert name in rulebooks, f"Missing domain pack: {name}"

    def test_all_yamls_exist(self):
        for name in EXPECTED_PACKS:
            path = DOMAINS_DIR / f"{name}.yaml"
            assert path.exists(), f"Missing YAML: {path}"


class TestHealthcarePack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "healthcare.yaml")
        rb.compile()
        return rb

    def test_ndc_extraction(self, rb):
        result = rb.extract("Medtronic Catheter NDC 12345-6789-01 sterile")
        assert result["identifiers"].get("ndc")
        assert result["brand"] == "Medtronic"

    def test_signals(self, rb):
        assert "ndc" in rb.signals or "patient" in rb.signals


class TestFinancialPack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "financial.yaml")
        rb.compile()
        return rb

    def test_cusip_extraction(self, rb):
        result = rb.extract("Goldman Sachs Bond CUSIP: 38141G104")
        assert result["identifiers"].get("cusip")

    def test_lei_extraction(self, rb):
        result = rb.extract("Entity LEI: 5493001KJTIIGC8Y1R12")
        assert result["identifiers"].get("lei")


class TestRealEstatePack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "real_estate.yaml")
        rb.compile()
        return rb

    def test_sqft_extraction(self, rb):
        result = rb.extract("3 bed 2 bath 1500 sqft ranch home")
        assert result["attributes"].get("sqft")

    def test_zip_extraction(self, rb):
        result = rb.extract("123 Main St Springfield IL 62704")
        assert result["identifiers"].get("zip")


class TestPeoplePack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "people.yaml")
        rb.compile()
        return rb

    def test_ssn_extraction(self, rb):
        result = rb.extract("John Smith SSN 123-45-6789")
        assert result["identifiers"].get("ssn")

    def test_dob_extraction(self, rb):
        result = rb.extract("Jane Doe DOB 01/15/1990")
        assert result["identifiers"].get("dob")


class TestRetailPack:
    @pytest.fixture
    def rb(self):
        rb = load_rulebook(DOMAINS_DIR / "retail.yaml")
        rb.compile()
        return rb

    def test_upc_extraction(self, rb):
        result = rb.extract("Tide Detergent UPC 037000127864 64oz")
        assert result["identifiers"].get("upc")

    def test_brand(self, rb):
        result = rb.extract("Procter & Gamble Tide Original")
        assert result["brand"] in ("Procter & Gamble", "Tide")
