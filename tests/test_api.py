"""Tests for the clean top-level API."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


class TestImports:
    def test_import_goldenmatch(self):
        import goldenmatch as gm
        assert hasattr(gm, "__version__")
        assert hasattr(gm, "dedupe")
        assert hasattr(gm, "match")
        assert hasattr(gm, "pprl_link")
        assert hasattr(gm, "evaluate")
        assert hasattr(gm, "load_config")
        assert hasattr(gm, "DedupeResult")
        assert hasattr(gm, "MatchResult")

    def test_version(self):
        import goldenmatch as gm
        assert gm.__version__

    def test_all_categories_importable(self):
        """Verify all major feature areas are accessible from top-level."""
        import goldenmatch as gm
        # Config
        assert gm.GoldenMatchConfig
        assert gm.MatchkeyConfig
        assert gm.BlockingConfig
        assert gm.LLMScorerConfig
        # Pipeline
        assert gm.run_dedupe
        assert gm.build_clusters
        assert gm.build_blocks
        assert gm.load_file
        # Streaming
        assert gm.match_one
        assert gm.StreamProcessor
        # Evaluation
        assert gm.evaluate_pairs
        assert gm.EvalResult
        # Explain
        assert gm.explain_pair
        # Domain
        assert gm.discover_rulebooks
        assert gm.DomainRulebook
        # Probabilistic
        assert gm.train_em
        # Learned blocking
        assert gm.learn_blocking_rules
        # LLM
        assert gm.llm_score_pairs
        assert gm.llm_cluster_pairs
        assert gm.BudgetTracker
        # PPRL
        assert gm.run_pprl
        assert gm.pprl_auto_config
        assert gm.compute_bloom_filters
        # Other
        assert gm.profile_dataframe
        assert gm.unmerge_record
        assert gm.build_lineage


class TestDedupeResult:
    def test_repr(self):
        from goldenmatch import DedupeResult
        r = DedupeResult(stats={"total_records": 100, "total_clusters": 10, "match_rate": 0.2})
        assert "100" in repr(r)
        assert "10" in repr(r)
        assert "20.0%" in repr(r)

    def test_match_rate(self):
        from goldenmatch import DedupeResult
        r = DedupeResult(stats={"match_rate": 0.15})
        assert r.match_rate == 0.15

    def test_to_csv(self, tmp_path):
        from goldenmatch import DedupeResult
        df = pl.DataFrame({"name": ["John"], "email": ["j@x.com"]})
        r = DedupeResult(golden=df)
        path = r.to_csv(str(tmp_path / "out.csv"))
        assert Path(path).exists()

    def test_to_csv_all(self, tmp_path):
        from goldenmatch import DedupeResult
        df = pl.DataFrame({"x": [1]})
        r = DedupeResult(golden=df, dupes=df, unique=df)
        r.to_csv(str(tmp_path / "results.csv"), which="all")
        assert (tmp_path / "results_golden.csv").exists()
        assert (tmp_path / "results_dupes.csv").exists()
        assert (tmp_path / "results_unique.csv").exists()


class TestMatchResult:
    def test_repr(self):
        from goldenmatch import MatchResult
        df = pl.DataFrame({"x": [1, 2, 3]})
        r = MatchResult(matched=df, unmatched=pl.DataFrame({"x": [4]}))
        assert "matched=3" in repr(r)
        assert "unmatched=1" in repr(r)


class TestDedupeFunction:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        pl.DataFrame({
            "first_name": ["John", "john", "Jane", "JOHN", "Bob"],
            "last_name": ["Smith", "Smith", "Doe", "Smyth", "Jones"],
            "email": ["john@x.com", "john@x.com", "jane@y.com", "john@x.com", "bob@z.com"],
        }).write_csv(path)
        return str(path)

    def test_dedupe_exact(self, sample_csv):
        import goldenmatch as gm
        result = gm.dedupe(sample_csv, exact=["email"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records > 0
        assert result.total_clusters >= 0

    def test_dedupe_with_config_file(self, sample_csv, tmp_path):
        import goldenmatch as gm
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "matchkeys:\n"
            "  - name: exact_email\n"
            "    type: exact\n"
            "    fields:\n"
            "      - field: email\n"
            "        transforms: [lowercase, strip]\n"
        )
        result = gm.dedupe(sample_csv, config=str(config_path))
        assert result.total_records >= 5

    def test_dedupe_returns_clusters(self, sample_csv):
        import goldenmatch as gm
        result = gm.dedupe(sample_csv, exact=["email"])
        # email john@x.com appears 3 times, should form a cluster
        assert len(result.clusters) >= 1


class TestPPRLLink:
    def test_pprl_basic(self, tmp_path):
        import goldenmatch as gm
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        pl.DataFrame({"name": ["John Smith", "Jane Doe"]}).write_csv(a)
        pl.DataFrame({"name": ["john smith", "Bob Jones"]}).write_csv(b)

        result = gm.pprl_link(str(a), str(b), fields=["name"], threshold=0.7)
        assert "clusters" in result
        assert "match_count" in result
        assert result["match_count"] >= 1


class TestLoadConfig:
    def test_load(self, tmp_path):
        import goldenmatch as gm
        p = tmp_path / "config.yaml"
        p.write_text(
            "matchkeys:\n"
            "  - name: test\n"
            "    type: exact\n"
            "    fields:\n"
            "      - field: email\n"
        )
        cfg = gm.load_config(str(p))
        assert cfg is not None
        assert len(cfg.get_matchkeys()) == 1


class TestDedupeDf:
    def test_dedupe_df_exact(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "first_name": ["John", "john", "Jane", "JOHN", "Bob"],
            "email": ["john@x.com", "john@x.com", "jane@y.com", "john@x.com", "bob@z.com"],
        })
        result = gm.dedupe_df(df, exact=["email"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records > 0
        assert result.total_clusters >= 1

    def test_dedupe_df_fuzzy(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "name": ["John Smith", "Jon Smith", "Jane Doe", "Bob Jones"],
            "zip": ["10001", "10001", "20002", "30003"],
        })
        result = gm.dedupe_df(df, fuzzy={"name": 0.80}, blocking=["zip"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records > 0

    def test_dedupe_df_with_config_object(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "email": ["a@x.com", "a@x.com", "b@y.com"],
        })
        cfg = gm.GoldenMatchConfig(
            matchkeys=[gm.MatchkeyConfig(
                name="email",
                type="exact",
                fields=[gm.MatchkeyField(field="email", transforms=["lowercase"])],
            )],
        )
        result = gm.dedupe_df(df, config=cfg)
        assert result.total_clusters >= 1

    def test_dedupe_df_returns_scored_pairs(self):
        import goldenmatch as gm
        df = pl.DataFrame({
            "email": ["a@x.com", "a@x.com", "b@y.com"],
        })
        result = gm.dedupe_df(df, exact=["email"])
        assert isinstance(result.scored_pairs, list)

    def test_dedupe_df_empty(self):
        import goldenmatch as gm
        df = pl.DataFrame({"email": []}).cast({"email": pl.Utf8})
        result = gm.dedupe_df(df, exact=["email"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records == 0

    def test_dedupe_df_missing_column_raises(self):
        import goldenmatch as gm
        df = pl.DataFrame({"name": ["John"]})
        with pytest.raises(Exception):
            gm.dedupe_df(df, exact=["nonexistent_column"])

    def test_dedupe_df_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "dedupe_df")
