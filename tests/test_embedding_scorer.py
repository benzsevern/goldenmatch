"""Integration tests for embedding scorer dispatch and disk cache."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
from goldenmatch.core.embedder import Embedder


# ---------------------------------------------------------------------------
# Disk cache tests (no model needed)
# ---------------------------------------------------------------------------


class TestDiskCache:
    def test_save_and_load_cache(self, tmp_path):
        e = Embedder("test-model")
        # Manually inject cached embeddings
        fake = np.random.default_rng(0).random((3, 8)).astype(np.float32)
        e._cache["col_a"] = fake

        e.save_cache(tmp_path / "emb_cache")
        assert (tmp_path / "emb_cache" / "col_a.npy").exists()

        e2 = Embedder("test-model")
        e2.load_cache(tmp_path / "emb_cache")
        assert "col_a" in e2._cache
        np.testing.assert_array_equal(e2._cache["col_a"], fake)

    def test_load_cache_nonexistent_dir(self, tmp_path):
        e = Embedder("test-model")
        e.load_cache(tmp_path / "does_not_exist")
        assert len(e._cache) == 0

    def test_load_cache_does_not_overwrite_existing(self, tmp_path):
        original = np.ones((2, 4), dtype=np.float32)
        disk_data = np.zeros((2, 4), dtype=np.float32)

        # Save disk_data
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        np.save(cache_dir / "key1.npy", disk_data)

        e = Embedder("test-model")
        e._cache["key1"] = original
        e.load_cache(cache_dir)

        # In-memory value should be preserved
        np.testing.assert_array_equal(e._cache["key1"], original)

    def test_save_multiple_keys(self, tmp_path):
        e = Embedder("test-model")
        e._cache["a"] = np.array([[1, 2]])
        e._cache["b"] = np.array([[3, 4]])
        e.save_cache(tmp_path)

        e2 = Embedder("test-model")
        e2.load_cache(tmp_path)
        assert set(e2._cache.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Scorer integration (mocked embedder)
# ---------------------------------------------------------------------------

def _make_fake_embedder():
    """Create an embedder with a mocked model that returns deterministic embeddings."""
    e = Embedder("fake-model")

    def fake_encode(texts, show_progress_bar=False, normalize_embeddings=True):
        rng = np.random.default_rng(42)
        vecs = rng.random((len(texts), 8))
        # Make identical texts produce identical embeddings
        seen: dict[str, np.ndarray] = {}
        for i, t in enumerate(texts):
            if t in seen:
                vecs[i] = seen[t]
            else:
                seen[t] = vecs[i]
        # L2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

    class FakeModel:
        def encode(self, texts, **kwargs):
            return fake_encode(texts, **kwargs)

    e._model = FakeModel()
    return e


class TestEmbeddingScorerIntegration:
    def test_fuzzy_score_matrix_embedding(self):
        """Embedding scorer dispatches through _fuzzy_score_matrix."""
        from goldenmatch.core.scorer import _fuzzy_score_matrix

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            values = ["sony turntable", "sony turntable", "samsung tv"]
            matrix = _fuzzy_score_matrix(values, "embedding", model_name="fake-model")

            assert matrix.shape == (3, 3)
            # Identical strings should have similarity ~1.0
            assert matrix[0, 1] == pytest.approx(1.0, abs=0.01)
            # Diagonal should be 1.0
            assert matrix[0, 0] == pytest.approx(1.0, abs=0.01)

    def test_hybrid_weighted_scoring(self):
        """Embedding field combined with exact field in weighted matchkey."""
        from goldenmatch.core.scorer import find_fuzzy_matches

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1, 2],
                "title": ["sony turntable", "sony turntable", "samsung tv"],
                "brand": ["Sony", "Sony", "Samsung"],
            })

            mk = MatchkeyConfig(
                name="hybrid",
                type="weighted",
                fields=[
                    MatchkeyField(field="title", scorer="embedding", weight=0.6, model="fake-model"),
                    MatchkeyField(field="brand", scorer="exact", weight=0.4),
                ],
                threshold=0.5,
            )

            results = find_fuzzy_matches(df, mk)
            # Rows 0 and 1 are identical — should match
            pair_ids = {(r[0], r[1]) for r in results}
            assert (0, 1) in pair_ids

    def test_embedding_scorer_with_nulls(self):
        """Null values handled gracefully in embedding scorer."""
        from goldenmatch.core.scorer import find_fuzzy_matches

        fake = _make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": [0, 1, 2],
                "title": ["sony turntable", None, "sony turntable"],
            })

            mk = MatchkeyConfig(
                name="emb_null",
                type="weighted",
                fields=[
                    MatchkeyField(field="title", scorer="embedding", weight=1.0, model="fake-model"),
                ],
                threshold=0.5,
            )

            results = find_fuzzy_matches(df, mk)
            # Should not crash; row 1 (null) excluded from valid pairs
            pair_ids = {(r[0], r[1]) for r in results}
            assert (0, 2) in pair_ids

    def test_model_field_in_config(self):
        """MatchkeyField accepts model parameter for embedding scorer."""
        f = MatchkeyField(field="title", scorer="embedding", model="all-mpnet-base-v2")
        assert f.model == "all-mpnet-base-v2"

    def test_model_field_defaults_none(self):
        """Model defaults to None for non-embedding scorers."""
        f = MatchkeyField(field="name", scorer="jaro_winkler")
        assert f.model is None
