"""Tests for embedding scorer fallback to token_sort."""
from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock

import numpy as np

from goldenmatch.core.scorer import _fuzzy_score_matrix


def test_embedding_failure_falls_back_to_token_sort(caplog):
    """When embedding scorer fails, falls back to token_sort."""
    values = ["hello world", "hello world", "goodbye moon"]

    mock_embedder = MagicMock()
    mock_embedder.embed_column.side_effect = OSError("HF auth failed")

    with patch("goldenmatch.core.embedder.get_embedder", return_value=mock_embedder), \
         caplog.at_level(logging.WARNING, logger="goldenmatch.core.scorer"):
        scores = _fuzzy_score_matrix(values, "embedding")

    assert scores.shape == (3, 3)
    assert scores[0, 1] == 1.0  # identical strings
    assert scores[0, 2] < 1.0   # different strings
    assert any("falling back to token_sort" in r.message for r in caplog.records)


def test_embedding_import_error_falls_back(caplog):
    """When embedder module can't be imported, falls back to token_sort."""
    values = ["alpha beta", "alpha beta", "gamma delta"]

    with patch.dict("sys.modules", {"goldenmatch.core.embedder": None}), \
         caplog.at_level(logging.WARNING, logger="goldenmatch.core.scorer"):
        scores = _fuzzy_score_matrix(values, "embedding")

    assert scores.shape == (3, 3)
    assert scores[0, 1] == 1.0


def test_non_embedding_scorer_unaffected():
    """token_sort scorer should work normally (no fallback needed)."""
    values = ["hello world", "world hello", "goodbye"]

    scores = _fuzzy_score_matrix(values, "token_sort")
    assert scores.shape == (3, 3)
    assert scores[0, 1] > 0.9  # token_sort handles reordering
