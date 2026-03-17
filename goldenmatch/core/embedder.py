"""Embedder for GoldenMatch — sentence-transformer embedding and caching."""

from __future__ import annotations

import numpy as np


class Embedder:
    """Wraps a sentence-transformer model with lazy loading and caching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self):
        """Lazy-load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Embedding features require sentence-transformers. "
                "Install with: pip install goldenmatch[embeddings]"
            )
        self._model = SentenceTransformer(self.model_name)

    def embed_column(self, values: list[str], cache_key: str) -> np.ndarray:
        """Embed a list of string values. Returns (n, dim) array. Cached by cache_key."""
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self._model is None:
            self._load_model()
        # Replace None/empty with empty string
        clean = [str(v) if v is not None and str(v).strip() else "" for v in values]
        embeddings = self._model.encode(
            clean, show_progress_bar=False, normalize_embeddings=True,
        )
        self._cache[cache_key] = embeddings
        return embeddings

    def cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """NxN cosine similarity matrix. Embeddings must be L2-normalized."""
        return embeddings @ embeddings.T


# ---------------------------------------------------------------------------
# Module-level cache for embedder instances
# ---------------------------------------------------------------------------

_embedders: dict[str, Embedder] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Return a cached Embedder instance for *model_name*."""
    if model_name not in _embedders:
        _embedders[model_name] = Embedder(model_name)
    return _embedders[model_name]
