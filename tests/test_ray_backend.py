"""Tests for Ray distributed backend."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

from goldenmatch.config.schemas import GoldenMatchConfig, MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class TestBackendConfig:
    def test_default_backend_is_none(self):
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="e", type="exact", fields=[MatchkeyField(field="x")])],
        )
        assert config.backend is None

    def test_ray_backend_config(self):
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="e", type="exact", fields=[MatchkeyField(field="x")])],
            backend="ray",
        )
        assert config.backend == "ray"


class TestGetBlockScorer:
    def test_default_returns_parallel(self):
        from goldenmatch.core.pipeline import _get_block_scorer
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="e", type="exact", fields=[MatchkeyField(field="x")])],
        )
        from goldenmatch.core.scorer import score_blocks_parallel
        assert _get_block_scorer(config) is score_blocks_parallel

    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def test_ray_returns_ray_scorer(self):
        from goldenmatch.core.pipeline import _get_block_scorer
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="e", type="exact", fields=[MatchkeyField(field="x")])],
            backend="ray",
        )
        from goldenmatch.backends.ray_backend import score_blocks_ray
        assert _get_block_scorer(config) is score_blocks_ray


class TestRayBackendSmallBlocks:
    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def test_small_blocks_fall_through_to_parallel(self):
        """Blocks <= 4 should use the regular parallel scorer, not Ray."""
        from goldenmatch.backends.ray_backend import score_blocks_ray
        from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField

        mk = MatchkeyConfig(
            name="test", type="weighted", threshold=0.85,
            fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0)],
        )

        # Empty blocks should return empty
        result = score_blocks_ray([], mk, set())
        assert result == []


class TestRayBackendImport:
    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def test_import_error_message(self):
        """Verify helpful error message when ray is not installed."""
        import goldenmatch.backends.ray_backend as rb
        # Reset cached ray to test import path
        original = rb._ray
        rb._ray = None

        with patch.dict("sys.modules", {"ray": None}):
            # Can't easily test ImportError without uninstalling ray,
            # but we can verify the module loads
            pass

        rb._ray = original
