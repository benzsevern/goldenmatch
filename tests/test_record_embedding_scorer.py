"""Tests for record-level embedding scorer."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from goldenmatch.config.schemas import MatchkeyField, MatchkeyConfig


class TestRecordEmbeddingSchema:
    def test_record_embedding_with_columns(self):
        f = MatchkeyField(
            scorer="record_embedding",
            columns=["title", "manufacturer"],
            weight=1.0,
            model="all-MiniLM-L6-v2",
        )
        assert f.field == "__record__"
        assert f.columns == ["title", "manufacturer"]

    def test_record_embedding_requires_columns(self):
        with pytest.raises(ValueError, match="columns"):
            MatchkeyField(scorer="record_embedding", weight=1.0)

    def test_record_embedding_empty_columns_rejected(self):
        with pytest.raises(ValueError, match="columns"):
            MatchkeyField(scorer="record_embedding", columns=[], weight=1.0)

    def test_regular_scorer_still_requires_field(self):
        with pytest.raises(ValueError, match="field"):
            MatchkeyField(scorer="jaro_winkler", weight=1.0)

    def test_regular_scorer_with_field_still_works(self):
        f = MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)
        assert f.field == "name"

    def test_record_embedding_in_weighted_matchkey(self):
        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.80,
            fields=[
                MatchkeyField(
                    scorer="record_embedding",
                    columns=["title", "manufacturer"],
                    weight=0.7,
                    model="all-MiniLM-L6-v2",
                ),
                MatchkeyField(field="brand", scorer="exact", weight=0.3),
            ],
        )
        assert len(mk.fields) == 2


def _make_fake_embedder():
    """Embedder with deterministic fake model."""
    from goldenmatch.core.embedder import Embedder
    e = Embedder("fake-model")

    class FakeModel:
        def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
            rng = np.random.default_rng(42)
            vecs = rng.random((len(texts), 8))
            seen: dict[str, np.ndarray] = {}
            for i, t in enumerate(texts):
                if t in seen:
                    vecs[i] = seen[t]
                else:
                    seen[t] = vecs[i]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return vecs / norms

    e._model = FakeModel()
    return e


class TestRecordEmbeddingScorer:
    def test_record_embedding_score_matrix(self):
        from goldenmatch.core.scorer import _record_embedding_score_matrix

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1, 2],
                "title": ["Sony Turntable", "Sony Turntable", "Samsung TV"],
                "manufacturer": ["Sony", "Sony", "Samsung"],
            })
            matrix = _record_embedding_score_matrix(
                df, ["title", "manufacturer"], "fake-model"
            )
            assert matrix.shape == (3, 3)
            assert matrix[0, 1] == pytest.approx(1.0, abs=0.01)

    def test_record_embedding_null_handling(self):
        from goldenmatch.core.scorer import _record_embedding_score_matrix

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1],
                "title": ["Sony", "Sony"],
                "manufacturer": [None, "Sony"],
            })
            matrix = _record_embedding_score_matrix(
                df, ["title", "manufacturer"], "fake-model"
            )
            assert matrix.shape == (2, 2)

    def test_record_embedding_in_find_fuzzy(self):
        from goldenmatch.core.scorer import find_fuzzy_matches

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1, 2],
                "title": ["Sony Turntable", "Sony Turntable", "Samsung TV"],
                "brand": ["Sony", "Sony", "Samsung"],
            })
            mk = MatchkeyConfig(
                name="rec_emb",
                type="weighted",
                threshold=0.5,
                fields=[
                    MatchkeyField(
                        scorer="record_embedding",
                        columns=["title", "brand"],
                        weight=0.7,
                        model="fake-model",
                    ),
                    MatchkeyField(field="brand", scorer="exact", weight=0.3),
                ],
            )
            results = find_fuzzy_matches(df, mk)
            pair_ids = {(r[0], r[1]) for r in results}
            assert (0, 1) in pair_ids

    def test_record_embedding_only_field(self):
        """record_embedding as the only field in a matchkey."""
        from goldenmatch.core.scorer import find_fuzzy_matches

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1, 2],
                "title": ["Sony Turntable", "Sony Turntable", "Samsung TV"],
            })
            mk = MatchkeyConfig(
                name="rec_only",
                type="weighted",
                threshold=0.5,
                fields=[
                    MatchkeyField(
                        scorer="record_embedding",
                        columns=["title"],
                        weight=1.0,
                        model="fake-model",
                    ),
                ],
            )
            results = find_fuzzy_matches(df, mk)
            pair_ids = {(r[0], r[1]) for r in results}
            assert (0, 1) in pair_ids
