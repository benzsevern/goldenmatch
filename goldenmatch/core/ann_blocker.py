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

    @property
    def index_size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal if self._index is not None else 0

    def add_to_index(self, embedding: np.ndarray) -> int:
        """Add a single embedding vector to the FAISS index.

        Args:
            embedding: (dim,) or (1, dim) numpy array, L2-normalized.

        Returns:
            The index position of the new vector.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        vec = embedding.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        pos = self._index.ntotal
        self._index.add(vec)
        return pos

    def query_one(self, embedding: np.ndarray) -> list[tuple[int, float]]:
        """Query top-K neighbors for a single vector.

        Args:
            embedding: (dim,) or (1, dim) numpy array, L2-normalized.

        Returns:
            List of (neighbor_index, similarity_score) tuples.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        vec = embedding.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        scores, indices = self._index.search(vec, self.top_k)
        results = []
        for j in range(self.top_k):
            neighbor = int(indices[0][j])
            if neighbor >= 0:
                results.append((neighbor, float(scores[0][j])))
        return results

    def query_with_scores(self, query_embeddings: np.ndarray) -> list[tuple[int, int, float]]:
        """Find top-K neighbors with similarity scores.

        Returns (idx_a, idx_b, cosine_similarity) tuples, ordered so idx_a < idx_b.
        """
        scores_matrix, indices = self._index.search(
            query_embeddings.astype(np.float32), self.top_k,
        )
        pairs: dict[tuple[int, int], float] = {}
        for i in range(len(query_embeddings)):
            for j_idx in range(self.top_k):
                neighbor = int(indices[i][j_idx])
                if neighbor != i and neighbor >= 0:
                    pair = (min(i, neighbor), max(i, neighbor))
                    score = float(scores_matrix[i][j_idx])
                    if pair not in pairs or score > pairs[pair]:
                        pairs[pair] = score
        return [(a, b, s) for (a, b), s in pairs.items()]
