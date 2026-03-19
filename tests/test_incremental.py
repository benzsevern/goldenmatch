"""Tests for incremental matching — ANN index, hybrid blocking, progressive embedding."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ── Persistent ANN Index Tests ────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestPersistentANNIndex:
    def test_build_and_query(self, tmp_path):
        from goldenmatch.db.ann_index import PersistentANNIndex

        index = PersistentANNIndex(index_dir=tmp_path / "faiss")

        # Add some embeddings
        rng = np.random.default_rng(42)
        ids = [10, 20, 30, 40, 50]
        embeddings = rng.random((5, 8)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index.add(ids, embeddings)
        assert index.record_count == 5

        # Query
        results = index.query(embeddings[:1], top_k=3)
        assert len(results) > 0
        # First result should be the record itself
        db_ids = {r[1] for r in results}
        assert 10 in db_ids

    def test_save_and_load(self, tmp_path):
        from goldenmatch.db.ann_index import PersistentANNIndex

        index_dir = tmp_path / "faiss"

        # Build and save
        index = PersistentANNIndex(index_dir=index_dir)
        rng = np.random.default_rng(42)
        ids = [1, 2, 3]
        embeddings = rng.random((3, 4)).astype(np.float32)
        index.add(ids, embeddings)
        index.save()

        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "id_map.npy").exists()
        assert (index_dir / "index_meta.json").exists()

        # Load into new instance
        index2 = PersistentANNIndex(index_dir=index_dir)
        count = index2._load_from_disk()
        assert count == 3
        assert index2._id_map == [1, 2, 3]

    def test_incremental_add(self, tmp_path):
        from goldenmatch.db.ann_index import PersistentANNIndex

        index = PersistentANNIndex(index_dir=tmp_path / "faiss")

        rng = np.random.default_rng(42)
        emb1 = rng.random((2, 4)).astype(np.float32)
        index.add([1, 2], emb1)
        assert index.record_count == 2

        emb2 = rng.random((3, 4)).astype(np.float32)
        index.add([3, 4, 5], emb2)
        assert index.record_count == 5

    def test_is_available_with_coverage(self, tmp_path):
        from goldenmatch.db.ann_index import PersistentANNIndex

        index = PersistentANNIndex(
            index_dir=tmp_path / "faiss",
            min_coverage=0.5,  # need 50% coverage
        )
        # No connector, so is_available just checks if index has records
        assert index.is_available is False

        rng = np.random.default_rng(42)
        index.add([1], rng.random((1, 4)).astype(np.float32))
        assert index.is_available is True  # no connector = any records = available

    def test_empty_query(self, tmp_path):
        from goldenmatch.db.ann_index import PersistentANNIndex

        index = PersistentANNIndex(index_dir=tmp_path / "faiss")
        results = index.query(np.random.random((1, 4)).astype(np.float32))
        assert results == []

    def test_meta_json_content(self, tmp_path):
        from goldenmatch.db.ann_index import PersistentANNIndex

        index_dir = tmp_path / "faiss"
        index = PersistentANNIndex(index_dir=index_dir, source_table="customers")
        rng = np.random.default_rng(42)
        index.add([1, 2], rng.random((2, 8)).astype(np.float32))
        index.save()

        with open(index_dir / "index_meta.json") as f:
            meta = json.load(f)

        assert meta["record_count"] == 2
        assert meta["dim"] == 8
        assert meta["source_table"] == "customers"


# ── Hybrid Blocking Tests ─────────────────────────────────────────────────

class TestHybridBlockingUnit:
    """Unit tests for hybrid blocking logic (no DB needed)."""

    def test_embed_record(self):
        from goldenmatch.db.hybrid_blocking import _embed_record

        # This will fail gracefully if sentence-transformers not installed
        result = _embed_record(
            {"name": "test product", "price": "10.00"},
            ["name", "price"],
            "all-MiniLM-L6-v2",
        )
        # Either returns embedding or None (if ST not installed)
        if result is not None:
            assert result.shape[0] == 1
            assert result.shape[1] > 0

    def test_embed_empty_record(self):
        from goldenmatch.db.hybrid_blocking import _embed_record

        result = _embed_record({}, ["name"], "all-MiniLM-L6-v2")
        assert result is None


# ── Progressive Embedding Tests ───────────────────────────────────────────

@pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu not installed")
class TestProgressiveEmbedding:
    def test_index_grows_across_adds(self, tmp_path):
        """Simulate progressive embedding across multiple runs."""
        from goldenmatch.db.ann_index import PersistentANNIndex

        index_dir = tmp_path / "faiss"
        rng = np.random.default_rng(42)

        # Run 1: embed first batch
        index = PersistentANNIndex(index_dir=index_dir)
        index.add([1, 2, 3], rng.random((3, 4)).astype(np.float32))
        index.save()
        assert index.record_count == 3

        # Run 2: load and add more
        index2 = PersistentANNIndex(index_dir=index_dir)
        index2._load_from_disk()
        index2.add([4, 5], rng.random((2, 4)).astype(np.float32))
        index2.save()
        assert index2.record_count == 5

        # Run 3: verify persistence
        index3 = PersistentANNIndex(index_dir=index_dir)
        count = index3._load_from_disk()
        assert count == 5

    def test_query_after_progressive_add(self, tmp_path):
        """Verify queries work correctly after incremental adds."""
        from goldenmatch.db.ann_index import PersistentANNIndex

        index = PersistentANNIndex(index_dir=tmp_path / "faiss")
        rng = np.random.default_rng(42)

        # Add identical vectors for predictable results
        vec = rng.random((1, 8)).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        index.add([100], vec)
        index.add([200], vec + 0.01)  # very similar
        opposite = -vec  # maximally different
        index.add([300], opposite)

        results = index.query(vec, top_k=3)
        assert len(results) == 3
        # Records 100 and 200 should be top matches, 300 should be last
        top_ids = [r[1] for r in sorted(results, key=lambda x: -x[2])]
        assert 100 in top_ids[:2]
        assert 200 in top_ids[:2]
        assert top_ids[2] == 300  # opposite vector is worst match
