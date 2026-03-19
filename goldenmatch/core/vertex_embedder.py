"""Vertex AI Embeddings — Google Cloud managed embedding API.

No GPU needed, no torch, no model loading. Sends text to Vertex AI,
gets embeddings back. Uses application default credentials or service
account key.

Environment variables:
    GOOGLE_CLOUD_PROJECT         — GCP project ID
    GOOGLE_APPLICATION_CREDENTIALS — path to service account JSON key
    GOOGLE_CLOUD_LOCATION        — region (default: us-central1)
    VERTEX_EMBEDDING_MODEL       — model name (default: text-embedding-004)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VertexEmbedder:
    """Embedder using Google Vertex AI text-embedding API.

    Same interface as goldenmatch.core.embedder.Embedder so it can be
    used as a drop-in replacement.
    """

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
    ):
        self.project = project or os.environ.get(
            "GOOGLE_CLOUD_PROJECT", ""
        )
        self.location = location or os.environ.get(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )
        self.model = model or os.environ.get(
            "VERTEX_EMBEDDING_MODEL", "text-embedding-004"
        )
        self._cache: dict[str, np.ndarray] = {}
        self._token: str | None = None
        self._token_expiry: float = 0

        if not self.project:
            raise ValueError(
                "Google Cloud project required. "
                "Set GOOGLE_CLOUD_PROJECT environment variable."
            )

    def _get_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        import time

        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        # Try application default credentials
        self._token = self._refresh_adc_token()
        self._token_expiry = time.time() + 3500  # tokens last ~1hr
        return self._token

    def _refresh_adc_token(self) -> str:
        """Refresh token from application default credentials."""
        import urllib.request
        import urllib.parse

        # Check for service account key
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if sa_path and Path(sa_path).exists():
            return self._get_sa_token(sa_path)

        # Fall back to user credentials (gcloud auth)
        adc_path = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
        if not adc_path.exists():
            # Try Unix path
            adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

        if not adc_path.exists():
            raise RuntimeError(
                "No Google credentials found. Run 'gcloud auth application-default login' "
                "or set GOOGLE_APPLICATION_CREDENTIALS to a service account key."
            )

        creds = json.loads(adc_path.read_text())
        data = urllib.parse.urlencode({
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"],
            "grant_type": "refresh_token",
        }).encode()

        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())["access_token"]

    def _get_sa_token(self, key_path: str) -> str:
        """Get token from service account key using JWT."""
        import time
        import urllib.request
        import urllib.parse
        import hashlib
        import hmac
        import base64

        sa_key = json.loads(Path(key_path).read_text())

        # Build JWT
        now = int(time.time())
        header = base64.urlsafe_b64encode(json.dumps(
            {"alg": "RS256", "typ": "JWT"}
        ).encode()).rstrip(b"=").decode()

        payload = base64.urlsafe_b64encode(json.dumps({
            "iss": sa_key["client_email"],
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
        }).encode()).rstrip(b"=").decode()

        # Sign with RSA
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            private_key = serialization.load_pem_private_key(
                sa_key["private_key"].encode(), password=None,
            )
            signature = private_key.sign(
                f"{header}.{payload}".encode(),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            sig = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        except ImportError:
            # Fall back to google-auth if cryptography not available
            raise ImportError(
                "Service account auth requires 'cryptography' package. "
                "Install with: pip install cryptography"
            )

        jwt = f"{header}.{payload}.{sig}"

        data = urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt,
        }).encode()

        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read())["access_token"]

    def embed_column(self, values: list[str], cache_key: str) -> np.ndarray:
        """Embed a list of strings via Vertex AI. Returns (n, dim) array."""
        if cache_key in self._cache:
            return self._cache[cache_key]

        import urllib.request

        token = self._get_token()
        clean = [str(v).strip() if v is not None and str(v).strip() else " " for v in values]

        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/{self.location}/"
            f"publishers/google/models/{self.model}:predict"
        )

        # Vertex AI: batch in chunks with retry for rate limits
        all_embeddings = []
        batch_size = 50
        import time as _time

        for i in range(0, len(clean), batch_size):
            batch = clean[i:i + batch_size]

            body = json.dumps({
                "instances": [{"content": text} for text in batch],
            }).encode()

            for attempt in range(3):
                req = urllib.request.Request(url, data=body, headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                })

                try:
                    resp = urllib.request.urlopen(req, timeout=60)
                    result = json.loads(resp.read())
                    predictions = result.get("predictions", [])

                    for pred in predictions:
                        emb = pred["embeddings"]["values"]
                        all_embeddings.append(emb)
                    break  # success

                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 2:
                        wait = (attempt + 1) * 5
                        logger.warning("Rate limited, waiting %ds...", wait)
                        _time.sleep(wait)
                        continue
                    logger.error("Vertex AI embedding failed: %s", e)
                    raise RuntimeError(f"Vertex AI embedding error: {e}") from e
                except Exception as e:
                    if attempt < 2:
                        _time.sleep(2)
                        continue
                    logger.error("Vertex AI embedding failed: %s", e)
                    raise RuntimeError(f"Vertex AI embedding error: {e}") from e

            if (i // batch_size) % 5 == 0 and i > 0:
                logger.info("Embedded %d/%d texts...", i, len(clean))

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        self._cache[cache_key] = embeddings
        logger.info("Embedded %d texts via Vertex AI (%s)", len(values), self.model)
        return embeddings

    def cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """NxN cosine similarity matrix."""
        return embeddings @ embeddings.T


def is_vertex_available() -> bool:
    """Check if Vertex AI credentials are configured."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        return False

    # Check for any form of Google credentials
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if sa_path and Path(sa_path).exists():
        return True

    adc_path = Path.home() / "AppData" / "Roaming" / "gcloud" / "application_default_credentials.json"
    if adc_path.exists():
        return True

    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return adc_path.exists()
