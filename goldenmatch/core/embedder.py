"""Embedder for GoldenMatch — sentence-transformer embedding and caching."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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

    def save_cache(self, path: Path) -> None:
        """Persist embedding cache to disk as .npy files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for key, arr in self._cache.items():
            file_path = path / f"{key}.npy"
            np.save(file_path, arr)
        logger.info("Saved %d cached embeddings to %s", len(self._cache), path)

    def load_cache(self, path: Path) -> None:
        """Load embedding cache from disk (.npy files)."""
        path = Path(path)
        if not path.is_dir():
            return
        loaded = 0
        for npy_file in path.glob("*.npy"):
            key = npy_file.stem
            if key not in self._cache:
                self._cache[key] = np.load(npy_file)
                loaded += 1
        if loaded:
            logger.info("Loaded %d cached embeddings from %s", loaded, path)


# ---------------------------------------------------------------------------
# Module-level cache for embedder instances
# ---------------------------------------------------------------------------

_embedders: dict[str, Embedder] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Return a cached Embedder instance, using GPU routing when available.

    Checks GOLDENMATCH_GPU_MODE to select the right backend:
    - vertex: uses VertexEmbedder (Google Vertex AI, no local GPU needed)
    - remote: uses RemoteEmbedder (custom endpoint)
    - local/cpu_safe: uses local sentence-transformers Embedder
    """
    if model_name not in _embedders:
        try:
            from goldenmatch.core.gpu import detect_gpu_mode, GPUMode
            mode = detect_gpu_mode()
        except Exception:
            logger.warning("GPU detection failed, defaulting to local embedder.", exc_info=True)
            mode = None

        if mode is not None and mode.value == "vertex":
            try:
                from goldenmatch.core.vertex_embedder import VertexEmbedder
                logger.info("GPU mode=vertex: using VertexEmbedder (ignoring model_name=%s)", model_name)
                _embedders[model_name] = VertexEmbedder()
            except ImportError:
                logger.error(
                    "GOLDENMATCH_GPU_MODE=vertex but google-cloud-aiplatform is not installed. "
                    "Install with: pip install goldenmatch[vertex]. Falling back to local embedder."
                )
                _embedders[model_name] = Embedder(model_name)
            except Exception as e:
                logger.error(
                    "VertexEmbedder initialization failed: %s. Falling back to local embedder.", e,
                )
                _embedders[model_name] = Embedder(model_name)
        else:
            _embedders[model_name] = Embedder(model_name)
    return _embedders[model_name]
