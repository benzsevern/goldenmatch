"""Tests for PPRL -- privacy-preserving record linkage."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.pprl.protocol import (
    PPRLConfig,
    PartyData,
    compute_bloom_filters,
    link_trusted_third_party,
    link_smc,
    run_pprl,
)
from goldenmatch.utils.transforms import apply_transform


class TestBloomFilterSecurityLevels:
    def test_standard_level(self):
        result = apply_transform("john smith", "bloom_filter:standard")
        # 512 bits = 64 bytes = 128 hex chars
        assert len(result) == 128

    def test_high_level(self):
        result = apply_transform("john smith", "bloom_filter:high")
        # 1024 bits = 128 bytes = 256 hex chars
        assert len(result) == 256

    def test_paranoid_level(self):
        result = apply_transform("john smith", "bloom_filter:paranoid")
        # 2048 bits = 256 bytes = 512 hex chars
        assert len(result) == 512

    def test_default_unchanged(self):
        result = apply_transform("john smith", "bloom_filter")
        # 1024 bits = 128 bytes = 256 hex chars
        assert len(result) == 256

    def test_balanced_padding_short_strings(self):
        """Paranoid mode should pad short strings for consistent filter density."""
        short = apply_transform("Jo", "bloom_filter:paranoid")
        long = apply_transform("Jonathan", "bloom_filter:paranoid")
        # Both should be same length (2048 bits)
        assert len(short) == len(long) == 512

    def test_hmac_produces_different_output(self):
        """High/paranoid mode uses HMAC, should produce different bits than standard."""
        standard = apply_transform("john smith", "bloom_filter:standard")
        high = apply_transform("john smith", "bloom_filter:high")
        # Different hash functions should produce different filters
        assert standard != high

    def test_custom_hmac_key(self):
        """Custom HMAC key via parameter."""
        result_a = apply_transform("john", "bloom_filter:2:20:512:key_a")
        result_b = apply_transform("john", "bloom_filter:2:20:512:key_b")
        # Different keys should produce different filters
        assert result_a != result_b


class TestComputeBloomFilters:
    def test_basic_computation(self):
        df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "name": ["John Smith", "Jane Doe", "Bob Jones"],
        })
        config = PPRLConfig(fields=["name"])
        filters = compute_bloom_filters(df, ["name"], config)
        assert len(filters) == 3
        assert all(isinstance(v, str) for v in filters.values())

    def test_multiple_fields_concatenated(self):
        df = pl.DataFrame({
            "__row_id__": [1],
            "first": ["John"],
            "last": ["Smith"],
        })
        config = PPRLConfig(fields=["first", "last"])
        filters = compute_bloom_filters(df, ["first", "last"], config)
        assert len(filters) == 1

    def test_adds_row_id_if_missing(self):
        df = pl.DataFrame({"name": ["John", "Jane"]})
        config = PPRLConfig(fields=["name"])
        filters = compute_bloom_filters(df, ["name"], config)
        assert len(filters) == 2


class TestTrustedThirdParty:
    @pytest.fixture
    def party_data(self):
        df_a = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["John Smith", "Jane Doe", "Bob Jones"],
        })
        df_b = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["john smith", "Alice Brown", "bob jones"],
        })
        config = PPRLConfig(fields=["name"], threshold=0.7)
        filters_a = compute_bloom_filters(df_a, ["name"], config)
        filters_b = compute_bloom_filters(df_b, ["name"], config)

        party_a = PartyData(party_id="hospital_a", bloom_filters=filters_a, record_count=3)
        party_b = PartyData(party_id="hospital_b", bloom_filters=filters_b, record_count=3)
        return party_a, party_b, config

    def test_finds_matches(self, party_data):
        party_a, party_b, config = party_data
        result = link_trusted_third_party(party_a, party_b, config)
        # John Smith and john smith should match, Bob Jones and bob jones should match
        assert result.match_count >= 2
        assert len(result.clusters) >= 1

    def test_total_comparisons(self, party_data):
        party_a, party_b, config = party_data
        result = link_trusted_third_party(party_a, party_b, config)
        assert result.total_comparisons == 9  # 3x3

    def test_cluster_members_have_party_ids(self, party_data):
        party_a, party_b, config = party_data
        result = link_trusted_third_party(party_a, party_b, config)
        for cid, members in result.clusters.items():
            for party_id, rid in members:
                assert party_id in ("hospital_a", "hospital_b")


class TestSMC:
    def test_smc_matches_ttp(self):
        """SMC should produce the same match decisions as TTP."""
        df_a = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John Smith", "Jane Doe"],
        })
        df_b = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["john smith", "Alice Brown"],
        })
        config = PPRLConfig(fields=["name"], threshold=0.7)
        filters_a = compute_bloom_filters(df_a, ["name"], config)
        filters_b = compute_bloom_filters(df_b, ["name"], config)

        party_a = PartyData(party_id="a", bloom_filters=filters_a, record_count=2)
        party_b = PartyData(party_id="b", bloom_filters=filters_b, record_count=2)

        ttp_result = link_trusted_third_party(party_a, party_b, config)
        smc_result = link_smc(party_a, party_b, config)

        # Same number of clusters (match decisions should agree)
        assert len(smc_result.clusters) == len(ttp_result.clusters)


class TestRunPPRL:
    def test_end_to_end_ttp(self):
        df_a = pl.DataFrame({
            "first_name": ["John", "Jane"],
            "last_name": ["Smith", "Doe"],
        })
        df_b = pl.DataFrame({
            "first_name": ["john", "Bob"],
            "last_name": ["smith", "Jones"],
        })
        config = PPRLConfig(
            fields=["first_name", "last_name"],
            threshold=0.7,
            protocol="trusted_third_party",
        )
        result = run_pprl(df_a, df_b, config)
        # John Smith / john smith should match
        assert result.match_count >= 1

    def test_end_to_end_smc(self):
        df_a = pl.DataFrame({
            "name": ["John Smith", "Jane Doe"],
        })
        df_b = pl.DataFrame({
            "name": ["john smith", "Bob Jones"],
        })
        config = PPRLConfig(
            fields=["name"],
            threshold=0.7,
            protocol="smc",
        )
        result = run_pprl(df_a, df_b, config)
        assert result.match_count >= 1

    def test_no_matches_high_threshold(self):
        df_a = pl.DataFrame({"name": ["John"]})
        df_b = pl.DataFrame({"name": ["completely different"]})
        config = PPRLConfig(fields=["name"], threshold=0.99)
        result = run_pprl(df_a, df_b, config)
        assert result.match_count == 0
        assert len(result.clusters) == 0


class TestPPRLCLI:
    def test_pprl_link_help(self):
        from typer.testing import CliRunner
        from goldenmatch.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["pprl", "link", "--help"])
        assert result.exit_code == 0
        assert "Privacy" in result.stdout or "party" in result.stdout.lower()

    def test_pprl_link_basic(self, tmp_path):
        from typer.testing import CliRunner
        from goldenmatch.cli.main import app

        # Create test files
        a_path = tmp_path / "a.csv"
        b_path = tmp_path / "b.csv"
        out_path = tmp_path / "out.csv"

        pl.DataFrame({"name": ["John Smith", "Jane Doe"]}).write_csv(a_path)
        pl.DataFrame({"name": ["john smith", "Bob Jones"]}).write_csv(b_path)

        runner = CliRunner()
        result = runner.invoke(app, [
            "pprl", "link",
            "--file-a", str(a_path),
            "--file-b", str(b_path),
            "--fields", "name",
            "--threshold", "0.7",
            "--output", str(out_path),
        ])
        assert result.exit_code == 0
        assert "Clusters found" in result.stdout or "clusters" in result.stdout.lower()
