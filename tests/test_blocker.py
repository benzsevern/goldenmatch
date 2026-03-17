"""Tests for goldenmatch blocker."""

import logging

import polars as pl

from goldenmatch.core.blocker import build_blocks, BlockResult
from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, SortKeyField


class TestBlockResultMetadata:
    def test_default_metadata(self):
        from goldenmatch.core.blocker import BlockResult
        br = BlockResult(block_key="test", df=pl.DataFrame({"a": [1]}).lazy())
        assert br.strategy == "static"
        assert br.depth == 0
        assert br.parent_key is None

    def test_custom_metadata(self):
        from goldenmatch.core.blocker import BlockResult
        br = BlockResult(
            block_key="sub", df=pl.DataFrame({"a": [1]}).lazy(),
            strategy="adaptive", depth=1, parent_key="parent"
        )
        assert br.strategy == "adaptive"
        assert br.depth == 1
        assert br.parent_key == "parent"


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


class TestAdaptiveSubBlocking:
    """Tests for adaptive sub-blocking strategy."""

    def test_oversized_block_is_split_using_sub_block_keys(self):
        """Oversized block is split by sub_block_keys."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=3,
            strategy="adaptive",
            sub_block_keys=[BlockingKeyConfig(fields=["state"], transforms=[])],
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "zip": ["19382"] * 6,
            "state": ["PA", "PA", "PA", "NJ", "NJ", "NJ"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        # The single oversized block (6 records) should be split into 2 blocks of 3
        assert len(results) == 2
        block_keys = {r.block_key for r in results}
        assert "PA" in block_keys
        assert "NJ" in block_keys

    def test_max_depth_3_enforced(self, caplog):
        """Sub-blocking stops at depth 3 even if blocks are still oversized."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=2,
            strategy="adaptive",
            sub_block_keys=[
                BlockingKeyConfig(fields=["state"], transforms=[]),
                BlockingKeyConfig(fields=["city"], transforms=[]),
                BlockingKeyConfig(fields=["street"], transforms=[]),
                BlockingKeyConfig(fields=["apt"], transforms=[]),
            ],
        )
        # All records have same zip, state, city, street - can't split below 4
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "zip": ["19382"] * 4,
            "state": ["PA"] * 4,
            "city": ["WC"] * 4,
            "street": ["Main"] * 4,
            "apt": ["A"] * 4,
        })
        lf = df.lazy()

        with caplog.at_level(logging.WARNING):
            results = build_blocks(lf, config)

        # Should still return the block even though it exceeds max, because depth limit reached
        assert len(results) >= 1
        # Should not crash

    def test_sub_blocked_results_have_correct_metadata(self):
        """Sub-blocked results have strategy=adaptive, correct depth and parent_key."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=3,
            strategy="adaptive",
            sub_block_keys=[BlockingKeyConfig(fields=["state"], transforms=[])],
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "zip": ["19382"] * 6,
            "state": ["PA", "PA", "PA", "NJ", "NJ", "NJ"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        for r in results:
            assert r.strategy == "adaptive"
            assert r.depth == 1
            assert r.parent_key == "19382"

    def test_static_strategy_unchanged(self):
        """Static strategy preserves backwards-compatible behavior."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=["strip"])],
            max_block_size=100,
            strategy="static",
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "first_name": ["John", "Jane", "Bob", "Alice"],
            "zip": ["19382", "10001", "19382", "10001"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        assert len(results) == 2
        for r in results:
            assert r.strategy == "static"
            assert r.depth == 0
            assert r.parent_key is None

    def test_blocks_under_max_not_sub_blocked(self):
        """Blocks under max_block_size are not sub-blocked in adaptive mode."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=10,
            strategy="adaptive",
            sub_block_keys=[BlockingKeyConfig(fields=["state"], transforms=[])],
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "zip": ["19382", "19382", "10001", "10001"],
            "state": ["PA", "PA", "NY", "NY"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        # Blocks are small enough, should remain as primary blocks with static metadata
        assert len(results) == 2
        for r in results:
            assert r.strategy == "static"
            assert r.depth == 0

    def test_recursive_sub_blocking(self):
        """Sub-blocking recurses when first sub_block_key doesn't split enough."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=2,
            strategy="adaptive",
            sub_block_keys=[
                BlockingKeyConfig(fields=["state"], transforms=[]),
                BlockingKeyConfig(fields=["city"], transforms=[]),
            ],
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "zip": ["19382"] * 4,
            "state": ["PA"] * 4,
            "city": ["WC", "WC", "PH", "PH"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        # First sub_block_key (state) produces one block of 4 (all PA), still oversized
        # Second sub_block_key (city) splits into WC(2) and PH(2) at depth 2
        assert len(results) == 2
        for r in results:
            assert r.strategy == "adaptive"
            assert r.depth == 2


class TestSortedNeighborhood:
    """Tests for sorted neighborhood blocking strategy."""

    def test_basic_window_sliding(self):
        """Correct number of windows and sizes with sliding window."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=100,
            strategy="sorted_neighborhood",
            window_size=3,
            sort_key=[SortKeyField(column="last_name", transforms=[])],
        )
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "zip": ["19382"] * 5,
            "last_name": ["Adams", "Brown", "Clark", "Davis", "Evans"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        # 5 records with window_size 3: windows at positions 0-2, 1-3, 2-4 = 3 windows
        assert len(results) == 3
        for r in results:
            assert r.strategy == "sorted_neighborhood"
            collected = r.df.collect()
            assert len(collected) == 3

    def test_small_dataset_single_block(self):
        """Dataset smaller than window_size produces single block."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=100,
            strategy="sorted_neighborhood",
            window_size=10,
            sort_key=[SortKeyField(column="last_name", transforms=[])],
        )
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "zip": ["19382"] * 3,
            "last_name": ["Adams", "Brown", "Clark"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        assert len(results) == 1
        assert results[0].strategy == "sorted_neighborhood"
        assert len(results[0].df.collect()) == 3

    def test_sorted_neighborhood_metadata(self):
        """Sorted neighborhood blocks have correct strategy metadata."""
        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=[])],
            max_block_size=100,
            strategy="sorted_neighborhood",
            window_size=2,
            sort_key=[SortKeyField(column="last_name", transforms=["lowercase"])],
        )
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "zip": ["19382"] * 3,
            "last_name": ["Clark", "Adams", "Brown"],
        })
        lf = df.lazy()

        results = build_blocks(lf, config)

        for r in results:
            assert r.strategy == "sorted_neighborhood"
