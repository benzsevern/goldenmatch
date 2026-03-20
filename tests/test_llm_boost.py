"""Tests for LLM boost — labeler, feature extraction, classifier, persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import polars as pl
import pytest

from goldenmatch.core.llm_labeler import (
    build_prompt,
    detect_context,
    detect_provider,
    estimate_cost,
    get_default_model,
    parse_response,
)


# ── LLM Labeler Tests ─────────────────────────────────────────────────────


class TestParseResponse:
    def test_yes(self):
        assert parse_response("yes") is True
        assert parse_response("Yes") is True
        assert parse_response("YES") is True
        assert parse_response("yes.") is True

    def test_no(self):
        assert parse_response("no") is False
        assert parse_response("No") is False
        assert parse_response("NO") is False
        assert parse_response("no.") is False

    def test_ambiguous(self):
        assert parse_response("maybe") is None
        assert parse_response("I think so") is None
        assert parse_response("") is None


class TestDetectContext:
    def test_contact_list(self):
        cols = {"name": "name", "email": "email", "phone": "phone"}
        assert detect_context(cols) == "contact list"

    def test_product_catalog(self):
        cols = {"title": "title", "manufacturer": "manufacturer", "price": "price"}
        assert detect_context(cols) == "product catalog"

    def test_publication(self):
        cols = {"title": "title", "authors": "authors", "year": "year"}
        assert detect_context(cols) == "publication database"

    def test_fallback(self):
        cols = {"field_1": "field_1", "field_2": "field_2"}
        assert detect_context(cols) == "dataset"


class TestBuildPrompt:
    def test_basic_prompt(self):
        record_a = {"name": "John Smith", "email": "john@test.com"}
        record_b = {"name": "Jon Smith", "email": "jon@test.com"}
        prompt = build_prompt(record_a, record_b, ["name", "email"], "contact list")
        assert "contact list" in prompt
        assert "John Smith" in prompt
        assert "Jon Smith" in prompt
        assert "yes or no" in prompt

    def test_null_values(self):
        record_a = {"name": "John", "email": None}
        record_b = {"name": "John", "email": "john@test.com"}
        prompt = build_prompt(record_a, record_b, ["name", "email"], "dataset")
        assert "John" in prompt


class TestProviderDetection:
    def test_anthropic_env_var(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = detect_provider()
        assert result == ("anthropic", "sk-ant-test")

    def test_openai_env_var(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = detect_provider()
        assert result == ("openai", "sk-test")

    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = detect_provider()
        assert result is None

    def test_anthropic_preferred(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = detect_provider()
        assert result[0] == "anthropic"


class TestDefaultModel:
    def test_anthropic(self):
        assert "haiku" in get_default_model("anthropic")

    def test_openai(self):
        assert "mini" in get_default_model("openai")


class TestCostEstimate:
    def test_estimate(self):
        cost = estimate_cost(100, "anthropic")
        assert 0.05 < cost < 0.50


# ── Boost Engine Tests ─────────────────────────────────────────────────────


class TestFeatureExtraction:
    def test_feature_matrix_shape(self):
        from goldenmatch.core.boost import extract_feature_matrix

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["John Smith", "Jon Smith", "Jane Doe"],
            "email": ["john@t.com", "jon@t.com", "jane@t.com"],
        })
        pairs = [(0, 1, 0.9), (0, 2, 0.3)]
        matrix = extract_feature_matrix(pairs, df, ["name", "email"])

        assert matrix.shape == (2, 11)  # 2 pairs, 1 original_score + 5 features * 2 columns

    def test_identical_records_high_features(self):
        from goldenmatch.core.boost import extract_feature_matrix

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John Smith", "John Smith"],
        })
        pairs = [(0, 1, 1.0)]
        matrix = extract_feature_matrix(pairs, df, ["name"])

        # All features should be 1.0 for identical strings
        assert matrix[0, 0] == pytest.approx(1.0)  # jaro_winkler
        assert matrix[0, 3] == pytest.approx(1.0)  # exact match


class TestPairSampling:
    def test_initial_sampling(self):
        from goldenmatch.core.boost import _sample_initial_pairs

        pairs = [(i, i + 1, i / 200.0) for i in range(200)]
        indices = _sample_initial_pairs(pairs, n=100)

        assert len(indices) <= 100
        assert len(indices) > 50  # should have a good number

    def test_small_dataset(self):
        from goldenmatch.core.boost import _sample_initial_pairs

        pairs = [(0, 1, 0.5), (2, 3, 0.8)]
        indices = _sample_initial_pairs(pairs, n=100)
        assert set(indices) == {0, 1}  # all pairs returned


class TestModelPersistence:
    def test_save_and_load(self, tmp_path):
        from goldenmatch.core.boost import save_model, load_model
        from sklearn.linear_model import LogisticRegression

        X = np.array([[0.1, 0.2], [0.9, 0.8], [0.1, 0.3], [0.8, 0.9]])
        y = np.array([0, 1, 0, 1])
        model = LogisticRegression()
        model.fit(X, y)

        columns = ["name", "email"]
        save_model(model, columns, tmp_path)

        assert (tmp_path / ".goldenmatch_model.json").exists()

        loaded = load_model(columns, tmp_path)
        assert loaded is not None

        # Should produce same predictions
        orig_probs = model.predict_proba(X)
        loaded_probs = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(orig_probs, loaded_probs)

    def test_column_mismatch_returns_none(self, tmp_path):
        from goldenmatch.core.boost import save_model, load_model
        from sklearn.linear_model import LogisticRegression

        X = np.array([[0.1, 0.2], [0.9, 0.8], [0.1, 0.3], [0.8, 0.9]])
        y = np.array([0, 1, 0, 1])
        model = LogisticRegression()
        model.fit(X, y)

        save_model(model, ["name", "email"], tmp_path)
        loaded = load_model(["name", "phone"], tmp_path)
        assert loaded is None

    def test_missing_file_returns_none(self, tmp_path):
        from goldenmatch.core.boost import load_model
        assert load_model(["name"], tmp_path) is None


class TestBoostAccuracy:
    def test_boost_with_mocked_llm(self):
        """End-to-end boost with mocked LLM responses."""
        from goldenmatch.core.boost import boost_accuracy

        df = pl.DataFrame({
            "__row_id__": list(range(6)),
            "name": ["John Smith", "Jon Smith", "Jane Doe", "Janet Doe", "Bob", "Robert"],
        })

        pairs = [
            (0, 1, 0.8),  # true match
            (2, 3, 0.7),  # true match
            (0, 2, 0.3),  # non-match
            (4, 5, 0.2),  # non-match
        ]

        # Mock label_pairs to return deterministic labels
        def mock_label_pairs(pairs_to_label, columns, context, provider, api_key, model, progress_callback=None):
            labels = []
            for record_a, record_b in pairs_to_label:
                name_a = record_a.get("name", "")
                name_b = record_b.get("name", "")
                # Simple heuristic: same first letter = match
                labels.append(name_a[0] == name_b[0] if name_a and name_b else False)
            return labels

        with patch("goldenmatch.core.boost.label_pairs", side_effect=mock_label_pairs):
            with patch("goldenmatch.core.boost.detect_provider", return_value=("anthropic", "fake-key")):
                result = boost_accuracy(
                    pairs, df, ["name"],
                    retrain=True,
                    max_labels=100,
                )

        assert len(result) == len(pairs)
        # All results should have probabilities between 0 and 1
        for _, _, score in result:
            assert 0.0 <= score <= 1.0

    def test_boost_no_api_key(self):
        """Without API key, returns original pairs."""
        from goldenmatch.core.boost import boost_accuracy

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John", "Jane"],
        })
        pairs = [(0, 1, 0.5)]

        with patch("goldenmatch.core.boost.detect_provider", return_value=None):
            result = boost_accuracy(pairs, df, ["name"], retrain=True)

        assert result == pairs  # unchanged

    def test_boost_empty_pairs(self):
        from goldenmatch.core.boost import boost_accuracy

        df = pl.DataFrame({"__row_id__": [0], "name": ["John"]})
        result = boost_accuracy([], df, ["name"])
        assert result == []
