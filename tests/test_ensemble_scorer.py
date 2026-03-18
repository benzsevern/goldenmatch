"""Tests for ensemble scorer."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField


class TestEnsembleScorer:
    def test_ensemble_score_matrix(self):
        from goldenmatch.core.scorer import _fuzzy_score_matrix

        values = ["John Smith", "Smith John", "Jane Doe"]
        matrix = _fuzzy_score_matrix(values, "ensemble")

        assert matrix.shape == (3, 3)
        # "John Smith" vs "Smith John" — token_sort should catch this
        assert matrix[0, 1] > 0.8
        # Diagonal should be 1.0
        assert matrix[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_ensemble_beats_single_scorer(self):
        from goldenmatch.core.scorer import _fuzzy_score_matrix

        values = ["John Smith", "Smith, John"]

        ensemble = _fuzzy_score_matrix(values, "ensemble")
        jw = _fuzzy_score_matrix(values, "jaro_winkler")

        # Ensemble should be >= jaro_winkler for reordered names
        assert ensemble[0, 1] >= jw[0, 1]

    def test_ensemble_null_handling(self):
        from goldenmatch.core.scorer import _fuzzy_score_matrix

        values = ["John", None, "Jane"]
        matrix = _fuzzy_score_matrix(values, "ensemble")
        assert matrix.shape == (3, 3)

    def test_ensemble_in_find_fuzzy(self):
        from goldenmatch.core.scorer import find_fuzzy_matches

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["John Smith", "Smith John", "Jane Doe"],
        })
        mk = MatchkeyConfig(
            name="ens",
            type="weighted",
            threshold=0.7,
            fields=[MatchkeyField(field="name", scorer="ensemble", weight=1.0)],
        )
        results = find_fuzzy_matches(df, mk)
        pair_ids = {(r[0], r[1]) for r in results}
        # Reordered name should match
        assert (0, 1) in pair_ids

    def test_ensemble_schema_valid(self):
        f = MatchkeyField(field="name", scorer="ensemble", weight=1.0)
        assert f.scorer == "ensemble"
