"""Tests for enhanced auto-configuration -- probabilistic matchkey support."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.core.autoconfig import (
    auto_configure,
    build_probabilistic_matchkeys,
    profile_columns,
)


def _make_df():
    return pl.DataFrame({
        "first_name": ["John", "Jon", "Jane", "Alice"],
        "last_name": ["Smith", "Smith", "Doe", "Brown"],
        "email": ["john@s.com", "jon@s.com", "jane@d.com", "alice@b.com"],
        "zip": ["90210", "90210", "10001", "30301"],
    })


class TestBuildProbabilisticMatchkeys:
    def test_generates_probabilistic_matchkey(self):
        df = _make_df()
        profiles = profile_columns(df)
        mks = build_probabilistic_matchkeys(profiles)
        assert len(mks) == 1
        assert mks[0].type == "probabilistic"
        assert len(mks[0].fields) > 0

    def test_fields_have_levels(self):
        df = _make_df()
        profiles = profile_columns(df)
        mks = build_probabilistic_matchkeys(profiles)
        for f in mks[0].fields:
            assert f.levels in (2, 3)
            assert f.scorer is not None

    def test_exact_fields_get_2_levels(self):
        """Email (exact scorer) should get 2-level comparison."""
        df = _make_df()
        profiles = profile_columns(df)
        mks = build_probabilistic_matchkeys(profiles)
        email_fields = [f for f in mks[0].fields if f.field == "email"]
        if email_fields:
            assert email_fields[0].levels == 2


class TestAutoConfigureZeroConfig:
    def test_auto_configure_produces_valid_config(self, tmp_path):
        import csv
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["first_name", "last_name", "zip"])
            for i in range(20):
                w.writerow([f"Name{i}", f"Last{i}", f"{10000 + i}"])

        config = auto_configure([(str(csv_path), "test")])
        assert config is not None
        mks = config.get_matchkeys()
        assert len(mks) > 0

    def test_auto_configure_handles_single_column(self, tmp_path):
        import csv
        csv_path = tmp_path / "single.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name"])
            for i in range(10):
                w.writerow([f"Person{i}"])

        config = auto_configure([(str(csv_path), "test")])
        assert config is not None
