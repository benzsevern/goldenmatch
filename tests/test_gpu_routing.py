"""Tests for GPU routing — mode detection, remote embedder, CPU-safe fallback.

All tests mock _has_cuda/_has_mps to avoid importing torch (which
hangs/crashes on machines without GPU resources).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

# Mock _has_cuda and _has_mps at module level BEFORE they can be called
# This prevents torch from being imported during tests
_mock_cuda = patch("goldenmatch.core.gpu._has_cuda", return_value=False)
_mock_mps = patch("goldenmatch.core.gpu._has_mps", return_value=False)
_mock_cuda.start()
_mock_mps.start()

from goldenmatch.core.gpu import (
    GPUMode,
    detect_gpu_mode,
    get_device_string,
    get_gpu_status,
    is_embedding_available,
    RemoteEmbedder,
)


def teardown_module():
    _mock_cuda.stop()
    _mock_mps.stop()


class TestGPUModeDetection:
    def test_explicit_cpu_safe(self):
        with patch.dict(os.environ, {"GOLDENMATCH_GPU_MODE": "cpu_safe"}):
            assert detect_gpu_mode() == GPUMode.CPU_SAFE

    def test_explicit_local(self):
        with patch.dict(os.environ, {"GOLDENMATCH_GPU_MODE": "local"}):
            assert detect_gpu_mode() == GPUMode.LOCAL

    def test_explicit_remote(self):
        with patch.dict(os.environ, {"GOLDENMATCH_GPU_MODE": "remote"}):
            assert detect_gpu_mode() == GPUMode.REMOTE

    def test_remote_from_endpoint_env(self):
        with patch.dict(os.environ, {"GOLDENMATCH_GPU_ENDPOINT": "http://localhost:8090"}, clear=False):
            os.environ.pop("GOLDENMATCH_GPU_MODE", None)
            assert detect_gpu_mode() == GPUMode.REMOTE

    def test_cpu_safe_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            # _has_cuda and _has_mps are mocked to False at module level
            assert detect_gpu_mode() == GPUMode.CPU_SAFE

    def test_local_when_cuda_available(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("goldenmatch.core.gpu._has_cuda", return_value=True):
                assert detect_gpu_mode() == GPUMode.LOCAL

    def test_local_when_mps_available(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("goldenmatch.core.gpu._has_cuda", return_value=False):
                with patch("goldenmatch.core.gpu._has_mps", return_value=True):
                    assert detect_gpu_mode() == GPUMode.LOCAL


class TestDeviceString:
    def test_cpu_fallback(self):
        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.CPU_SAFE):
            assert get_device_string() == "cpu"

    def test_cuda_device(self):
        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.LOCAL):
            with patch("goldenmatch.core.gpu._has_cuda", return_value=True):
                assert get_device_string() == "cuda"


class TestGPUStatus:
    def test_cpu_safe_status(self):
        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.CPU_SAFE):
            status = get_gpu_status()
            assert status["mode"] == "cpu_safe"
            assert "safe_scorers" in status
            assert "exact" in status["safe_scorers"]
            assert "jaro_winkler" in status["safe_scorers"]

    def test_remote_status(self):
        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.REMOTE):
            with patch.dict(os.environ, {"GOLDENMATCH_GPU_ENDPOINT": "http://gpu:8090"}):
                status = get_gpu_status()
                assert status["mode"] == "remote"
                assert status["endpoint"] == "http://gpu:8090"


class TestRemoteEmbedder:
    def test_requires_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOLDENMATCH_GPU_ENDPOINT", None)
            with pytest.raises(ValueError, match="endpoint not configured"):
                RemoteEmbedder(endpoint="")

    def test_cache_hit(self):
        embedder = RemoteEmbedder(endpoint="http://fake:8090")
        fake_emb = np.random.random((3, 4)).astype(np.float32)
        embedder._cache["test_key"] = fake_emb

        result = embedder.embed_column(["a", "b", "c"], cache_key="test_key")
        np.testing.assert_array_equal(result, fake_emb)


class TestEmbeddingAvailability:
    def test_remote_always_available(self):
        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.REMOTE):
            assert is_embedding_available() is True

    def test_cpu_safe_not_available(self):
        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.CPU_SAFE):
            assert is_embedding_available() is False
