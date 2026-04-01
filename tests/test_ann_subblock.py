"""Tests for ANN sub-block fallback in blocker."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from unittest.mock import patch, MagicMock

from goldenmatch.core.blocker import _ann_sub_block, _build_static_blocks
from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig


class TestANNSubBlock:
    def _make_block_df(self, n, ann_col="__text__", text_fn=None):
        """Create a block DataFrame with __row_id__ and ann_column."""
        if text_fn is None:
            text_fn = lambda i: f"model_{i % 5}"  # 5 unique texts
        return pl.DataFrame({
            "__row_id__": list(range(n)),
            ann_col: [text_fn(i) for i in range(n)],
            "name": [f"record_{i}" for i in range(n)],
        })

    def test_too_large_returns_empty(self):
        """Blocks >10x max_block_size should be skipped."""
        df = self._make_block_df(11000)
        result = _ann_sub_block(df, "__text__", 20, "model", 1000, "test_block")
        assert result == []

    def test_missing_column_returns_empty(self):
        """Missing ann_column should return empty."""
        df = self._make_block_df(2000)
        result = _ann_sub_block(df, "nonexistent_col", 20, "model", 1000, "test_block")
        assert result == []

    def test_single_unique_text_returns_empty(self):
        """<2 unique texts should return empty."""
        df = self._make_block_df(2000, text_fn=lambda i: "same_text")
        result = _ann_sub_block(df, "__text__", 20, "model", 1000, "test_block")
        assert result == []

    def test_happy_path_creates_subblocks(self):
        """ANN sub-blocking should create sub-blocks from oversized block."""
        n = 2000
        # 10 unique texts, records cycle through them
        df = self._make_block_df(n, text_fn=lambda i: f"equipment_{i % 10}")

        # Mock embedder: return 10 embeddings (one per unique text)
        mock_embedder = MagicMock()
        dim = 8
        # Each unique text gets a distinct embedding direction
        unique_embeddings = np.random.RandomState(42).randn(10, dim).astype(np.float32)
        unique_embeddings /= np.linalg.norm(unique_embeddings, axis=1, keepdims=True)
        mock_embedder.embed_column.return_value = unique_embeddings

        # Mock ANNBlocker: return pairs that cluster nearby texts
        mock_blocker_cls = MagicMock()
        mock_blocker = MagicMock()
        mock_blocker_cls.return_value = mock_blocker
        # Pair up consecutive text indices: (0,1), (2,3), (4,5), (6,7), (8,9)
        mock_blocker.query.return_value = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

        with patch("goldenmatch.core.embedder.get_embedder", return_value=mock_embedder):
            with patch("goldenmatch.core.ann_blocker.ANNBlocker", mock_blocker_cls):
                result = _ann_sub_block(df, "__text__", 20, "model", 1000, "test_block")

        assert len(result) > 0
        # All sub-blocks should have strategy "ann"
        for block in result:
            assert block.strategy == "ann"
        # Total records in sub-blocks should be <= original
        total = sum(len(b.df.collect()) for b in result)
        assert total <= n


class TestStaticBlocksANNFallback:
    def test_ann_fallback_triggered_for_oversized(self):
        """When skip_oversized=True and ann_column is set, oversized blocks use ANN fallback."""
        # Create data that produces one oversized block
        df = pl.DataFrame({
            "__row_id__": list(range(50)),
            "key": ["A"] * 50,  # all in one block
            "__text__": [f"text_{i % 5}" for i in range(50)],
        })

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["key"])],
            max_block_size=10,  # block of 50 exceeds this
            skip_oversized=True,
            ann_column="__text__",
            ann_top_k=5,
        )

        with patch("goldenmatch.core.blocker._ann_sub_block") as mock_ann:
            mock_ann.return_value = []  # just verify it's called
            _build_static_blocks(df.lazy(), config)
            mock_ann.assert_called_once()

    def test_ann_fallback_not_triggered_without_ann_column(self):
        """Without ann_column, oversized blocks are just skipped."""
        df = pl.DataFrame({
            "__row_id__": list(range(50)),
            "key": ["A"] * 50,
        })

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["key"])],
            max_block_size=10,
            skip_oversized=True,
            # no ann_column
        )

        with patch("goldenmatch.core.blocker._ann_sub_block") as mock_ann:
            result = _build_static_blocks(df.lazy(), config)
            mock_ann.assert_not_called()
            assert len(result) == 0  # block skipped

    def test_ann_fallback_error_doesnt_crash(self):
        """If ANN sub-blocking raises, the block is skipped gracefully."""
        df = pl.DataFrame({
            "__row_id__": list(range(50)),
            "key": ["A"] * 50,
            "__text__": [f"text_{i}" for i in range(50)],
        })

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["key"])],
            max_block_size=10,
            skip_oversized=True,
            ann_column="__text__",
        )

        with patch("goldenmatch.core.blocker._ann_sub_block", side_effect=RuntimeError("FAISS failed")):
            result = _build_static_blocks(df.lazy(), config)
            assert len(result) == 0  # block skipped, no crash
