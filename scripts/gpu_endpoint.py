"""Minimal GPU embedding endpoint — deploy to Colab, GCP, or any server.

Usage:
    # Local (with GPU):
    pip install sentence-transformers flask
    python gpu_endpoint.py --port 8090

    # Google Colab:
    !pip install sentence-transformers flask flask-ngrok
    # Copy this file to Colab and run it

    # GCP Cloud Run / any Docker:
    # Use as the entry point

Then set in your local environment:
    export GOLDENMATCH_GPU_ENDPOINT=http://localhost:8090
    # or the Colab/GCP URL

Protocol:
    POST /embed
    Body: {"texts": ["text1", "text2", ...], "model": "all-MiniLM-L6-v2"}
    Response: {"embeddings": [[0.1, 0.2, ...], ...], "model": "...", "count": N}

    GET /health
    Response: {"status": "ok", "device": "cuda", "models_loaded": [...]}
"""

from __future__ import annotations

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np


# Cache loaded models
_models = {}


def get_model(model_name: str):
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {device}...")
        _models[model_name] = SentenceTransformer(model_name, device=device)
        print(f"Model {model_name} loaded.")
    return _models[model_name]


class EmbedHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            import torch
            self._json({
                "status": "ok",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "models_loaded": list(_models.keys()),
            })
        else:
            self._json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path.rstrip("/") == "/embed":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                texts = body.get("texts", [])
                model_name = body.get("model", "all-MiniLM-L6-v2")

                model = get_model(model_name)
                embeddings = model.encode(
                    texts, show_progress_bar=False, normalize_embeddings=True,
                )

                self._json({
                    "embeddings": embeddings.tolist(),
                    "model": model_name,
                    "count": len(texts),
                })
            except Exception as e:
                self._json({"error": str(e)}, 500)
        else:
            self._json({"error": "Not found"}, 404)

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)).encode())

    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GoldenMatch GPU Embedding Endpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    # Pre-load default model
    get_model("all-MiniLM-L6-v2")

    server = HTTPServer((args.host, args.port), EmbedHandler)
    print(f"\nGoldenMatch GPU Endpoint running at http://{args.host}:{args.port}")
    print(f"Set: export GOLDENMATCH_GPU_ENDPOINT=http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
