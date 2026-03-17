"""Tests for the embedding scorer module."""

from __future__ import annotations

import pytest


class TestEmbedder:
    def test_import_error_without_deps(self):
        """If sentence-transformers not installed, clear error message."""
        try:
            from goldenmatch.core.embedder import Embedder

            e = Embedder("all-MiniLM-L6-v2")
            result = e.embed_column(["hello world", "test"], cache_key="test_import")
            assert result.shape[0] == 2
            assert result.shape[1] > 0  # embedding dim
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_cache_hit(self):
        try:
            from goldenmatch.core.embedder import Embedder

            e = Embedder("all-MiniLM-L6-v2")
            r1 = e.embed_column(["hello"], cache_key="test_cache")
            r2 = e.embed_column(["different"], cache_key="test_cache")
            # Should return cached result (same cache_key)
            assert (r1 == r2).all()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_cosine_similarity_matrix(self):
        try:
            from goldenmatch.core.embedder import Embedder

            e = Embedder("all-MiniLM-L6-v2")
            emb = e.embed_column(["cat", "dog", "automobile"], cache_key="sim_test")
            sim = e.cosine_similarity_matrix(emb)
            assert sim.shape == (3, 3)
            # Diagonal should be ~1.0
            assert all(abs(sim[i][i] - 1.0) < 0.01 for i in range(3))
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_get_embedder_caches_instance(self):
        try:
            from goldenmatch.core.embedder import get_embedder

            e1 = get_embedder("all-MiniLM-L6-v2")
            e2 = get_embedder("all-MiniLM-L6-v2")
            assert e1 is e2
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_none_and_empty_handling(self):
        try:
            from goldenmatch.core.embedder import Embedder

            e = Embedder("all-MiniLM-L6-v2")
            result = e.embed_column([None, "", "hello", "  "], cache_key="null_test")
            assert result.shape[0] == 4
        except ImportError:
            pytest.skip("sentence-transformers not installed")
