"""GPU routing — auto-detect local GPU, route to remote endpoint, or CPU-safe fallback.

Environment variables:
    GOLDENMATCH_GPU_ENDPOINT  — URL of a remote embedding endpoint (any provider)
    GOLDENMATCH_GPU_API_KEY   — API key for the remote endpoint (optional)
    GOLDENMATCH_GPU_MODE      — Force mode: "local", "remote", "cpu_safe" (auto-detect if unset)

Remote endpoint protocol:
    POST /embed
    Body: {"texts": ["text1", "text2", ...], "model": "all-MiniLM-L6-v2"}
    Response: {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}

Any server implementing this protocol works — Colab notebook, GCP Cloud Run,
self-hosted FastAPI, etc. See docs/wiki/GPU-Routing.md for examples.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class GPUMode(Enum):
    LOCAL = "local"        # Local GPU (CUDA/MPS available)
    REMOTE = "remote"      # Remote endpoint configured
    VERTEX = "vertex"      # Google Vertex AI managed embeddings
    CPU_SAFE = "cpu_safe"  # No GPU, no endpoint — use lightweight scorers only


def detect_gpu_mode() -> GPUMode:
    """Auto-detect the best GPU routing mode.

    Priority:
    1. GOLDENMATCH_GPU_MODE env var (explicit override)
    2. GOLDENMATCH_GPU_ENDPOINT set → REMOTE
    3. GOOGLE_CLOUD_PROJECT set → VERTEX
    4. CUDA available → LOCAL
    5. MPS available (Apple Silicon) → LOCAL
    6. Fallback → CPU_SAFE
    """
    explicit = os.environ.get("GOLDENMATCH_GPU_MODE", "").lower()
    if explicit in ("local", "remote", "vertex", "cpu_safe"):
        mode = GPUMode(explicit)
        logger.info("GPU mode: %s (explicit)", mode.value)
        return mode

    if os.environ.get("GOLDENMATCH_GPU_ENDPOINT"):
        logger.info("GPU mode: remote (endpoint configured)")
        return GPUMode.REMOTE

    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        from goldenmatch.core.vertex_embedder import is_vertex_available
        if is_vertex_available():
            logger.info("GPU mode: vertex (Google Cloud credentials found)")
            return GPUMode.VERTEX

    if _has_cuda():
        logger.info("GPU mode: local (CUDA detected)")
        return GPUMode.LOCAL

    if _has_mps():
        logger.info("GPU mode: local (MPS detected)")
        return GPUMode.LOCAL

    logger.info("GPU mode: cpu_safe (no GPU detected)")
    return GPUMode.CPU_SAFE


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_mps() -> bool:
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:
        return False


def get_device_string() -> str:
    """Get the torch device string for the current environment."""
    mode = detect_gpu_mode()
    if mode == GPUMode.LOCAL:
        if _has_cuda():
            return "cuda"
        if _has_mps():
            return "mps"
    return "cpu"


# ── Remote Embedding Client ──────────────────────────────────────────────

class RemoteEmbedder:
    """Client for a remote embedding endpoint."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.endpoint = endpoint or os.environ.get("GOLDENMATCH_GPU_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("GOLDENMATCH_GPU_API_KEY", "")
        self.model_name = model_name
        self._cache: dict[str, np.ndarray] = {}

        if not self.endpoint:
            raise ValueError(
                "Remote embedding endpoint not configured. "
                "Set GOLDENMATCH_GPU_ENDPOINT environment variable."
            )

    def embed_column(self, values: list[str], cache_key: str) -> np.ndarray:
        """Embed via remote endpoint. Same interface as Embedder."""
        if cache_key in self._cache:
            return self._cache[cache_key]

        import urllib.request

        clean = [str(v) if v is not None and str(v).strip() else "" for v in values]

        # Batch in chunks of 500 to avoid request size limits
        all_embeddings = []
        for i in range(0, len(clean), 500):
            batch = clean[i:i + 500]
            payload = json.dumps({
                "texts": batch,
                "model": self.model_name,
            }).encode()

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            url = self.endpoint.rstrip("/") + "/embed"
            req = urllib.request.Request(url, data=payload, headers=headers)

            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                    batch_emb = np.array(result["embeddings"], dtype=np.float32)
                    all_embeddings.append(batch_emb)
            except Exception as e:
                logger.error("Remote embedding failed: %s", e)
                raise RuntimeError(f"Remote embedding endpoint error: {e}") from e

        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        self._cache[cache_key] = embeddings
        return embeddings


# ── Smart Embedder Factory ───────────────────────────────────────────────

_smart_embedders: dict[str, object] = {}


def get_smart_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Get the best available embedder based on GPU mode.

    Returns:
        Embedder (local) or RemoteEmbedder (remote).
        Raises RuntimeError in CPU_SAFE mode.
    """
    cache_key = f"{model_name}_{detect_gpu_mode().value}"
    if cache_key in _smart_embedders:
        return _smart_embedders[cache_key]

    mode = detect_gpu_mode()

    if mode == GPUMode.REMOTE:
        embedder = RemoteEmbedder(model_name=model_name)
        _smart_embedders[cache_key] = embedder
        logger.info("Using remote embedder: %s", embedder.endpoint)
        return embedder

    if mode == GPUMode.VERTEX:
        from goldenmatch.core.vertex_embedder import VertexEmbedder
        embedder = VertexEmbedder()
        _smart_embedders[cache_key] = embedder
        logger.info("Using Vertex AI embedder: %s/%s", embedder.project, embedder.model)
        return embedder

    if mode == GPUMode.LOCAL:
        from goldenmatch.core.embedder import Embedder
        embedder = Embedder(model_name)
        _smart_embedders[cache_key] = embedder
        logger.info("Using local GPU embedder: %s", model_name)
        return embedder

    # CPU_SAFE — don't load heavy models
    raise RuntimeError(
        "No GPU available and no remote endpoint configured. "
        "Embedding-based features are disabled in CPU-safe mode. "
        "Options:\n"
        "  1. Set GOLDENMATCH_GPU_ENDPOINT to a remote embedding server\n"
        "  2. Install CUDA/MPS GPU support\n"
        "  3. Use non-embedding scorers (jaro_winkler, exact, soundex, etc.)\n"
        "  4. Set GOLDENMATCH_GPU_MODE=local to force CPU (may be slow/crash)"
    )


def is_embedding_available() -> bool:
    """Check if embedding features are available without loading models."""
    mode = detect_gpu_mode()
    if mode in (GPUMode.REMOTE, GPUMode.VERTEX):
        return True
    if mode == GPUMode.LOCAL:
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    return False


def get_gpu_status() -> dict:
    """Get current GPU routing status for display."""
    mode = detect_gpu_mode()
    status = {
        "mode": mode.value,
        "embedding_available": is_embedding_available(),
        "device": get_device_string(),
    }

    if mode == GPUMode.REMOTE:
        status["endpoint"] = os.environ.get("GOLDENMATCH_GPU_ENDPOINT", "")

    if mode == GPUMode.VERTEX:
        status["project"] = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        status["model"] = os.environ.get("VERTEX_EMBEDDING_MODEL", "text-embedding-004")

    if mode == GPUMode.LOCAL:
        status["cuda"] = _has_cuda()
        status["mps"] = _has_mps()

    if mode == GPUMode.CPU_SAFE:
        status["safe_scorers"] = [
            "exact", "jaro_winkler", "levenshtein",
            "token_sort", "soundex_match", "ensemble",
        ]

    return status
