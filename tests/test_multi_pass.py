"""Tests for multi-pass blocking strategy."""

import polars as pl

from goldenmatch.core.blocker import build_blocks, BlockResult
from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig


class TestMultiPassBlocking:
    def test_multi_pass_finds_more_pairs(self):
        """Two passes with different keys find pairs that neither would alone."""
        # Pass 1 blocks on zip: groups {1,2} and {3,4}
        # Pass 2 blocks on last_name: groups {1,3} and {2,4}
        # Union should include blocks from both passes
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "zip": ["19382", "19382", "10001", "10001"],
            "last_name": ["Smith", "Jones", "Smith", "Jones"],
        })
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],  # dummy, not used by multi_pass
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=["zip"], transforms=[]),
                BlockingKeyConfig(fields=["last_name"], transforms=[]),
            ],
        )
        results = build_blocks(df.lazy(), config)

        # Should have blocks from both passes (zip gives 2 blocks, last_name gives 2 blocks)
        assert len(results) >= 4
        block_keys = {r.block_key for r in results}
        assert "19382" in block_keys
        assert "10001" in block_keys
        assert "Smith" in block_keys
        assert "Jones" in block_keys

    def test_multi_pass_backwards_compat(self):
        """strategy='static' still works unchanged."""
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "zip": ["19382", "19382", "10001", "10001"],
        })
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            strategy="static",
        )
        results = build_blocks(df.lazy(), config)
        assert len(results) == 2
        for r in results:
            assert r.strategy == "static"

    def test_multi_pass_dedup(self):
        """Same block key from different passes doesn't duplicate."""
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "zip": ["19382", "19382", "10001", "10001"],
        })
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=["zip"], transforms=[]),
                BlockingKeyConfig(fields=["zip"], transforms=[]),  # same key twice
            ],
        )
        results = build_blocks(df.lazy(), config)

        # Should deduplicate — only 2 unique block keys
        block_keys = [r.block_key for r in results]
        assert len(block_keys) == len(set(block_keys))
        assert len(results) == 2

    def test_multi_pass_strategy_metadata(self):
        """Multi-pass blocks have strategy='multi_pass'."""
        df = pl.DataFrame({
            "id": [1, 2],
            "zip": ["19382", "19382"],
        })
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            strategy="multi_pass",
            passes=[
                BlockingKeyConfig(fields=["zip"], transforms=[]),
            ],
        )
        results = build_blocks(df.lazy(), config)
        assert len(results) == 1
        assert results[0].strategy == "multi_pass"
