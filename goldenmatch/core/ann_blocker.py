"""ANN (Approximate Nearest Neighbor) blocker for GoldenMatch using FAISS."""

from __future__ import annotations

import numpy as np


class ANNBlocker:
    """Build a FAISS inner-product index and query for top-K neighbors."""

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self._index = None

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index from L2-normalized embeddings."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "ANN blocking requires faiss-cpu. "
                "Install with: pip install goldenmatch[embeddings]"
            )
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vectors
        self._index.add(embeddings.astype(np.float32))

    def query(self, query_embeddings: np.ndarray) -> list[tuple[int, int]]:
        """Find top-K neighbors for each query. Returns (query_idx, neighbor_idx) pairs."""
        scores, indices = self._index.search(
            query_embeddings.astype(np.float32), self.top_k,
        )
        pairs: set[tuple[int, int]] = set()
        for i in range(len(query_embeddings)):
            for j_idx in range(self.top_k):
                neighbor = int(indices[i][j_idx])
                if neighbor != i and neighbor >= 0:
                    pairs.add((min(i, neighbor), max(i, neighbor)))
        return list(pairs)
