"""Tests for direct-pair ANN scoring."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from goldenmatch.config.schemas import (
    BlockingConfig, BlockingKeyConfig, MatchkeyConfig, MatchkeyField,
)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestQueryWithScores:
    def test_returns_scores(self):
        from goldenmatch.core.ann_blocker import ANNBlocker

        rng = np.random.default_rng(42)
        embeddings = rng.random((10, 8)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        blocker = ANNBlocker(top_k=3)
        blocker.build_index(embeddings)
        results = blocker.query_with_scores(embeddings)

        assert len(results) > 0
        assert len(results[0]) == 3
        for a, b, score in results:
            assert a < b
            assert 0.0 <= score <= 1.0

    def test_backward_compat_query(self):
        from goldenmatch.core.ann_blocker import ANNBlocker

        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 4)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        blocker = ANNBlocker(top_k=3)
        blocker.build_index(embeddings)
        results = blocker.query(embeddings)

        assert len(results) > 0
        assert len(results[0]) == 2

    def test_no_self_pairs_with_scores(self):
        from goldenmatch.core.ann_blocker import ANNBlocker

        embeddings = np.eye(5, dtype=np.float32)
        blocker = ANNBlocker(top_k=5)
        blocker.build_index(embeddings)
        results = blocker.query_with_scores(embeddings)

        for a, b, score in results:
            assert a != b


@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestAnnPairsBlocking:
    def _make_fake_embedder(self):
        from goldenmatch.core.embedder import Embedder
        e = Embedder("fake-model")

        class FakeModel:
            def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
                rng = np.random.default_rng(42)
                vecs = rng.random((len(texts), 8)).astype(np.float32)
                seen = {}
                for i, t in enumerate(texts):
                    if t in seen:
                        vecs[i] = seen[t]
                    else:
                        seen[t] = vecs[i]
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1
                return (vecs / norms).astype(np.float32)

        e._model = FakeModel()
        return e

    def test_ann_pairs_returns_pre_scored(self):
        from unittest.mock import patch
        from goldenmatch.core.blocker import build_blocks

        fake = self._make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": list(range(5)),
                "name": ["sony tv", "sony tv", "samsung phone", "samsung phone", "lg monitor"],
            }).lazy()

            config = BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"])],
                strategy="ann_pairs",
                ann_column="name",
                ann_model="fake-model",
                ann_top_k=3,
            )
            blocks = build_blocks(df, config)
            assert len(blocks) == 1
            assert blocks[0].pre_scored_pairs is not None
            assert len(blocks[0].pre_scored_pairs) > 0
            for a, b, s in blocks[0].pre_scored_pairs:
                assert isinstance(s, float)

    def test_ann_pairs_no_union_find(self):
        from unittest.mock import patch
        from goldenmatch.core.blocker import build_blocks

        fake = self._make_fake_embedder()
        with patch("goldenmatch.core.embedder.get_embedder", return_value=fake):
            df = pl.DataFrame({
                "__row_id__": list(range(10)),
                "name": [f"item_{i}" for i in range(10)],
            }).lazy()

            config = BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"])],
                strategy="ann_pairs",
                ann_column="name",
                ann_model="fake-model",
                ann_top_k=3,
            )
            blocks = build_blocks(df, config)
            assert len(blocks) == 1
            assert blocks[0].strategy == "ann_pairs"


class TestPreScoredPairsInScorer:
    def test_pre_scored_pairs_bypass_nxn(self):
        from goldenmatch.core.scorer import find_fuzzy_matches

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "name": ["a", "b", "c", "d"],
        })

        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.5,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)],
        )

        pre_scored = [(0, 1, 0.95), (2, 3, 0.30)]
        results = find_fuzzy_matches(df, mk, pre_scored_pairs=pre_scored)

        assert len(results) == 1
        assert results[0] == (0, 1, 0.95)

    def test_pre_scored_pairs_respects_exclude(self):
        from goldenmatch.core.scorer import find_fuzzy_matches

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "name": ["a", "b", "c", "d"],
        })

        mk = MatchkeyConfig(
            name="test",
            type="weighted",
            threshold=0.5,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)],
        )

        pre_scored = [(0, 1, 0.95), (2, 3, 0.80)]
        results = find_fuzzy_matches(
            df, mk,
            exclude_pairs={(0, 1)},
            pre_scored_pairs=pre_scored,
        )

        assert len(results) == 1
        assert results[0] == (2, 3, 0.80)
