"""Tests for the ANN blocker module."""

from __future__ import annotations

import pytest


class TestANNBlocker:
    def test_build_and_query(self):
        try:
            import numpy as np
            from goldenmatch.core.ann_blocker import ANNBlocker

            # Create synthetic embeddings (10 records, 64 dims)
            np.random.seed(42)
            embeddings = np.random.randn(10, 64).astype(np.float32)
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            blocker = ANNBlocker(top_k=3)
            blocker.build_index(embeddings)
            pairs = blocker.query(embeddings)
            assert len(pairs) > 0
            assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        except ImportError:
            pytest.skip("faiss-cpu not installed")

    def test_pairs_are_ordered(self):
        try:
            import numpy as np
            from goldenmatch.core.ann_blocker import ANNBlocker

            np.random.seed(123)
            embeddings = np.random.randn(5, 32).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            blocker = ANNBlocker(top_k=2)
            blocker.build_index(embeddings)
            pairs = blocker.query(embeddings)
            # All pairs should be (min, max) ordered
            for a, b in pairs:
                assert a < b
        except ImportError:
            pytest.skip("faiss-cpu not installed")

    def test_no_self_pairs(self):
        try:
            import numpy as np
            from goldenmatch.core.ann_blocker import ANNBlocker

            np.random.seed(99)
            embeddings = np.random.randn(8, 16).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            blocker = ANNBlocker(top_k=4)
            blocker.build_index(embeddings)
            pairs = blocker.query(embeddings)
            for a, b in pairs:
                assert a != b
        except ImportError:
            pytest.skip("faiss-cpu not installed")
