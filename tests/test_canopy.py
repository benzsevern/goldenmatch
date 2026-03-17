"""Tests for TF-IDF canopy clustering."""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")

from goldenmatch.core.canopy import build_canopies


class TestBuildCanopies:
    """Unit tests for build_canopies."""

    def test_empty_input(self):
        assert build_canopies([]) == []

    def test_single_record(self):
        canopies = build_canopies(["hello world"])
        # Single record can't form a 2-member canopy in blocker,
        # but canopy builder returns it as a 1-member canopy
        assert len(canopies) >= 1
        assert 0 in canopies[0]

    def test_identical_records_form_single_canopy(self):
        values = ["john smith", "john smith", "john smith"]
        canopies = build_canopies(values, loose_threshold=0.3, tight_threshold=0.7)
        # All identical records should appear in at least one canopy together
        all_members = set()
        for c in canopies:
            if len(c) == 3:
                all_members = set(c)
                break
        assert all_members == {0, 1, 2}

    def test_distinct_records_form_separate_canopies(self):
        values = [
            "john smith new york",
            "john smith new york",
            "xyz abc 12345 qqq",
            "xyz abc 12345 qqq",
        ]
        canopies = build_canopies(values, loose_threshold=0.5, tight_threshold=0.9)
        # Similar records should cluster together
        found_john = False
        found_xyz = False
        for c in canopies:
            if 0 in c and 1 in c:
                found_john = True
            if 2 in c and 3 in c:
                found_xyz = True
        assert found_john, "Similar 'john smith' records should share a canopy"
        assert found_xyz, "Similar 'xyz abc' records should share a canopy"

    def test_overlapping_canopies(self):
        # Middle record should appear in canopies with both ends
        values = [
            "machine learning approach",
            "machine learning methods",  # similar to both
            "deep learning methods",
        ]
        canopies = build_canopies(values, loose_threshold=0.2, tight_threshold=0.9)
        # Record 1 should appear in at least one canopy with record 0
        # and at least one canopy with record 2
        with_0 = any(0 in c and 1 in c for c in canopies)
        with_2 = any(1 in c and 2 in c for c in canopies)
        assert with_0, "Record 1 should share a canopy with record 0"
        assert with_2, "Record 1 should share a canopy with record 2"

    def test_max_canopy_size(self):
        values = [f"similar text variant {i}" for i in range(20)]
        canopies = build_canopies(values, loose_threshold=0.1, max_canopy_size=5)
        for c in canopies:
            assert len(c) <= 5, f"Canopy exceeded max size: {len(c)}"

    def test_none_values_handled(self):
        values = ["hello", None, "hello world", ""]
        canopies = build_canopies(values, loose_threshold=0.1)
        assert len(canopies) >= 1
        # Should not crash

    def test_tight_threshold_removes_centers(self):
        # With tight_threshold=0.0, all records are removed from centers after
        # the first canopy — should produce fewer canopies
        values = ["aaa bbb", "aaa bbb", "ccc ddd", "ccc ddd"]
        canopies_tight = build_canopies(values, loose_threshold=0.3, tight_threshold=0.0)
        canopies_loose = build_canopies(values, loose_threshold=0.3, tight_threshold=1.0)
        # Very tight threshold removes more centers, producing fewer canopies
        assert len(canopies_tight) <= len(canopies_loose)


class TestCanopyBlockerIntegration:
    """Integration tests for canopy strategy through build_blocks."""

    def test_canopy_strategy_through_blocker(self):
        import polars as pl
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, CanopyConfig
        from goldenmatch.core.blocker import build_blocks

        df = pl.DataFrame({
            "__row_id__": [1, 2, 3, 4],
            "name": ["john smith", "john smith", "jane doe", "jane doe"],
        }).lazy()

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["name"])],
            strategy="canopy",
            canopy=CanopyConfig(
                fields=["name"],
                loose_threshold=0.3,
                tight_threshold=0.7,
            ),
        )

        blocks = build_blocks(df, config)
        assert len(blocks) >= 1
        assert all(b.strategy == "canopy" for b in blocks)
        assert all(b.block_key.startswith("canopy_") for b in blocks)

    def test_canopy_strategy_requires_config(self):
        import polars as pl
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        from goldenmatch.core.blocker import build_blocks

        df = pl.DataFrame({
            "__row_id__": [1, 2],
            "name": ["a", "b"],
        }).lazy()

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["name"])],
            strategy="canopy",
        )

        with pytest.raises(ValueError, match="canopy"):
            build_blocks(df, config)

    def test_canopy_blocks_have_at_least_two_members(self):
        import polars as pl
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, CanopyConfig
        from goldenmatch.core.blocker import build_blocks

        df = pl.DataFrame({
            "__row_id__": list(range(6)),
            "title": [
                "machine learning approach",
                "machine learning methods",
                "deep learning for nlp",
                "deep learning for nlp tasks",
                "database optimization",
                "database query optimization",
            ],
        }).lazy()

        config = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["title"])],
            strategy="canopy",
            canopy=CanopyConfig(
                fields=["title"],
                loose_threshold=0.2,
                tight_threshold=0.7,
            ),
        )

        blocks = build_blocks(df, config)
        for block in blocks:
            block_df = block.df.collect()
            assert len(block_df) >= 2, "Canopy blocks should have at least 2 members"
