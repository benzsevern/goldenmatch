"""Tests for domain detection from column profiles."""
from __future__ import annotations

import pytest

from goldenmatch.core.autoconfig import ColumnProfile
from goldenmatch.core.domain_detector import detect_domain, DomainDetectionResult


def _make_profile(name: str, col_type: str = "string", confidence: float = 0.7) -> ColumnProfile:
    return ColumnProfile(
        name=name, dtype="String", col_type=col_type, confidence=confidence,
    )


class TestDetectDomain:
    def test_electronics_detected(self):
        profiles = [
            _make_profile("brand", "name"),
            _make_profile("model_number", "string"),
            _make_profile("sku", "identifier"),
            _make_profile("upc", "string"),
            _make_profile("bluetooth", "string"),
        ]
        result = detect_domain(profiles)
        assert result.domain == "electronics"
        assert result.confidence > 0.3

    def test_people_detected(self):
        profiles = [
            _make_profile("first_name", "name"),
            _make_profile("last_name", "name"),
            _make_profile("email", "email"),
            _make_profile("phone", "phone"),
        ]
        result = detect_domain(profiles)
        assert result.domain == "people"
        assert result.confidence > 0.3

    def test_generic_fallback(self):
        profiles = [
            _make_profile("col_a", "string"),
            _make_profile("col_b", "numeric"),
            _make_profile("col_c", "string"),
        ]
        result = detect_domain(profiles)
        assert result.domain == "generic"

    def test_preset_none_for_generic(self):
        profiles = [_make_profile("x", "string")]
        result = detect_domain(profiles)
        assert result.domain == "generic"
        assert result.preset is None

    def test_returns_detection_result(self):
        profiles = [
            _make_profile("brand", "name"),
            _make_profile("model", "string"),
        ]
        result = detect_domain(profiles)
        assert isinstance(result, DomainDetectionResult)
        assert result.rulebook is not None or result.domain == "generic"
