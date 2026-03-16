"""Tests for goldenmatch blocker."""

import logging

import polars as pl
import pytest

from goldenmatch.core.blocker import build_blocks, BlockResult
from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig


class TestBuildBlocks:
    """Tests for build_blocks."""

    def test_groups_records_by_block_key(self):
        """Groups records by block key correctly."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=["strip"])],
            max_block_size=100,
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "first_name": ["John", "Jane", "Bob", "Alice"],
            "zip": ["19382", "10001", "19382", "10001"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        assert len(results) == 2
        assert all(isinstance(r, BlockResult) for r in results)

        # Collect all block keys
        block_keys = {r.block_key for r in results}
        assert block_keys == {"19382", "10001"}

        # Each block should have 2 records
        for r in results:
            block_df = r.df.collect()
            assert len(block_df) == 2

    def test_skips_blocks_with_fewer_than_2_records(self):
        """Blocks with fewer than 2 records are skipped."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=100,
        )
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "first_name": ["John", "Jane", "Bob"],
            "zip": ["19382", "19382", "99999"],  # 99999 has only 1 record
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        # Only the block with 2 records should be returned
        assert len(results) == 1
        assert results[0].block_key == "19382"

    def test_oversized_block_generates_warning(self, caplog):
        """Oversized block generates warning."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=2,
            skip_oversized=False,
        )
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "first_name": ["John", "Jane", "Bob"],
            "zip": ["19382", "19382", "19382"],  # 3 records, exceeds max of 2
        })
        lf = df.lazy()

        with caplog.at_level(logging.WARNING):
            results = build_blocks(lf, config)

        # Block should still be included (skip_oversized=False)
        assert len(results) == 1
        # Warning should be logged
        assert any(
            "exceeds" in record.message.lower() or "oversized" in record.message.lower()
            for record in caplog.records
        )

    def test_skip_oversized_drops_blocks(self):
        """skip_oversized=True drops oversized blocks."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=2,
            skip_oversized=True,
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "first_name": ["John", "Jane", "Bob", "Alice", "Eve"],
            "zip": ["19382", "19382", "19382", "10001", "10001"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        # The 19382 block (3 records) should be skipped; 10001 block (2 records) kept
        assert len(results) == 1
        assert results[0].block_key == "10001"

    def test_multi_field_block_key(self):
        """Multiple fields are concatenated for the block key."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip", "state"], transforms=[])],
            max_block_size=100,
        )
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "zip": ["19382", "19382", "10001"],
            "state": ["PA", "PA", "NY"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        block_keys = {r.block_key for r in results}
        assert "19382||PA" in block_keys
