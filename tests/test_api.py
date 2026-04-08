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


class TestMatchDf:
    def test_match_df_exact(self):
        import goldenmatch as gm
        target = pl.DataFrame({
            "name": ["John Smith", "Jane Doe"],
            "email": ["john@x.com", "jane@y.com"],
        })
        reference = pl.DataFrame({
            "name": ["JOHN SMITH", "Bob Jones"],
            "email": ["john@x.com", "bob@z.com"],
        })
        result = gm.match_df(target, reference, exact=["email"])
        assert isinstance(result, gm.MatchResult)

    def test_match_df_fuzzy(self):
        import goldenmatch as gm
        target = pl.DataFrame({
            "name": ["John Smith"],
            "zip": ["10001"],
        })
        reference = pl.DataFrame({
            "name": ["Jon Smyth"],
            "zip": ["10001"],
        })
        result = gm.match_df(target, reference, fuzzy={"name": 0.75}, blocking=["zip"])
        assert isinstance(result, gm.MatchResult)

    def test_match_df_no_matches(self):
        import goldenmatch as gm
        target = pl.DataFrame({"email": ["a@x.com"]})
        reference = pl.DataFrame({"email": ["b@y.com"]})
        result = gm.match_df(target, reference, exact=["email"])
        assert isinstance(result, gm.MatchResult)

    def test_match_df_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "match_df")


class TestScoreStrings:
    def test_score_strings_jaro_winkler(self):
        import goldenmatch as gm
        score = gm.score_strings("John Smith", "Jon Smyth", "jaro_winkler")
        assert isinstance(score, float)
        assert 0.7 < score < 1.0

    def test_score_strings_exact(self):
        import goldenmatch as gm
        assert gm.score_strings("hello", "hello", "exact") == 1.0
        assert gm.score_strings("hello", "world", "exact") == 0.0

    def test_score_strings_levenshtein(self):
        import goldenmatch as gm
        score = gm.score_strings("kitten", "sitting", "levenshtein")
        assert isinstance(score, float)
        assert 0.0 < score < 1.0

    def test_score_strings_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "score_strings")


class TestScorePairDf:
    def test_score_pair_basic(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "John Smith", "email": "j@x.com"},
            {"name": "Jon Smyth", "email": "j@x.com"},
            fuzzy={"name": 0.85},
            exact=["email"],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_score_pair_with_scorer(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "John Smith"},
            {"name": "Jon Smyth"},
            fuzzy={"name": 0.85},
        )
        assert isinstance(score, float)
        assert score > 0.7

    def test_score_pair_no_match(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "Alice"},
            {"name": "Zebra"},
            fuzzy={"name": 0.85},
        )
        assert score < 0.5


class TestExplainPairDf:
    def test_explain_basic(self):
        import goldenmatch as gm
        explanation = gm.explain_pair_df(
            {"name": "John Smith", "email": "j@x.com"},
            {"name": "Jon Smyth", "email": "j@x.com"},
            fuzzy={"name": 0.85},
            exact=["email"],
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_importable(self):
        import goldenmatch as gm
        assert hasattr(gm, "explain_pair_df")
        assert hasattr(gm, "score_pair_df")

    def test_explain_no_fields_returns_empty_explanation(self):
        """explain_pair_df with no fuzzy/exact fields returns explanation."""
        import goldenmatch as gm
        explanation = gm.explain_pair_df(
            {"name": "John"},
            {"name": "Jon"},
        )
        # With no fields specified, returns a (possibly empty) explanation
        assert isinstance(explanation, str)

    def test_explain_missing_field_in_record(self):
        """Fields not in record produce None values gracefully."""
        import goldenmatch as gm
        explanation = gm.explain_pair_df(
            {"name": "John"},
            {"name": "Jon"},
            fuzzy={"name": 0.85, "zip": 1.0},
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestScorePairDfEdgeCases:
    def test_auto_detect_common_fields(self):
        """score_pair_df with no fields auto-detects common keys."""
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "John Smith", "zip": "10001"},
            {"name": "Jon Smith", "zip": "10001"},
        )
        assert isinstance(score, float)
        assert score > 0.5

    def test_disjoint_records(self):
        """score_pair_df with no common fields returns 0."""
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"name": "John"},
            {"email": "john@x.com"},
        )
        # No common fields -> no MatchkeyFields -> 0.0
        assert score == 0.0

    def test_exact_only(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"email": "john@x.com"},
            {"email": "john@x.com"},
            exact=["email"],
        )
        assert score == 1.0

    def test_exact_no_match(self):
        import goldenmatch as gm
        score = gm.score_pair_df(
            {"email": "john@x.com"},
            {"email": "jane@y.com"},
            exact=["email"],
        )
        assert score == 0.0


class TestDedupeDfEdgeCases:
    def test_dedupe_df_with_yaml_config_path(self, tmp_path):
        """dedupe_df accepts a YAML config string path."""
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
        df = pl.DataFrame({
            "email": ["john@x.com", "JOHN@x.com", "jane@y.com"],
        })
        result = gm.dedupe_df(df, config=str(config_path))
        assert isinstance(result, gm.DedupeResult)
        assert result.total_records > 0

    def test_dedupe_df_single_record(self):
        """Single record returns no clusters."""
        import goldenmatch as gm
        df = pl.DataFrame({"email": ["a@x.com"]})
        result = gm.dedupe_df(df, exact=["email"])
        assert isinstance(result, gm.DedupeResult)
        assert result.total_clusters == 0


class TestMatchDfEdgeCases:
    def test_match_df_empty_target(self):
        """Empty target DataFrame."""
        import goldenmatch as gm
        target = pl.DataFrame({"email": []}).cast({"email": pl.Utf8})
        reference = pl.DataFrame({"email": ["a@x.com"]})
        result = gm.match_df(target, reference, exact=["email"])
        assert isinstance(result, gm.MatchResult)

    def test_match_df_empty_reference(self):
        """Empty reference DataFrame."""
        import goldenmatch as gm
        target = pl.DataFrame({"email": ["a@x.com"]})
        reference = pl.DataFrame({"email": []}).cast({"email": pl.Utf8})
        result = gm.match_df(target, reference, exact=["email"])
        assert isinstance(result, gm.MatchResult)

    def test_match_df_with_config_string(self, tmp_path):
        """match_df accepts a YAML config string path."""
        import goldenmatch as gm
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "matchkeys:\n"
            "  - name: exact_email\n"
            "    type: exact\n"
            "    fields:\n"
            "      - field: email\n"
            "        transforms: [lowercase]\n"
        )
        target = pl.DataFrame({"email": ["john@x.com"]})
        reference = pl.DataFrame({"email": ["JOHN@x.com"]})
        result = gm.match_df(target, reference, config=str(config_path))
        assert isinstance(result, gm.MatchResult)


class TestDedupeResultHtml:
    def test_repr_html_basic(self):
        from goldenmatch import DedupeResult
        golden = pl.DataFrame({"name": ["John", "Jane"], "email": ["j@x", "d@y"]})
        r = DedupeResult(
            golden=golden,
            stats={"total_records": 5, "total_clusters": 2, "match_rate": 0.4},
        )
        html = r._repr_html_()
        assert "GoldenMatch" in html
        assert "John" in html

    def test_repr_html_empty_golden(self):
        from goldenmatch import DedupeResult
        r = DedupeResult(stats={"total_records": 0, "total_clusters": 0, "match_rate": 0.0})
        html = r._repr_html_()
        assert "GoldenMatch" in html

    def test_repr_html_no_golden(self):
        from goldenmatch import DedupeResult
        r = DedupeResult()
        html = r._repr_html_()
        assert isinstance(html, str)


class TestMatchResultHtml:
    def test_repr_html_with_data(self):
        from goldenmatch import MatchResult
        matched = pl.DataFrame({"name": ["John"], "score": [0.95]})
        r = MatchResult(matched=matched, unmatched=pl.DataFrame({"name": ["Bob"]}))
        html = r._repr_html_()
        assert "Match Result" in html
        assert "John" in html

    def test_repr_html_empty(self):
        from goldenmatch import MatchResult
        r = MatchResult()
        html = r._repr_html_()
        assert isinstance(html, str)


class TestBuildConfig:
    def test_auto_config_no_args(self):
        """_build_config with no args creates auto placeholder."""
        from goldenmatch._api import _build_config
        cfg = _build_config()
        mks = cfg.get_matchkeys()
        assert len(mks) >= 1

    def test_fuzzy_auto_blocking(self):
        """_build_config with fuzzy auto-suggests blocking."""
        from goldenmatch._api import _build_config
        cfg = _build_config(fuzzy={"name": 0.85})
        assert cfg.blocking is not None
        assert cfg.blocking.auto_suggest is True

    def test_explicit_blocking(self):
        """_build_config with explicit blocking."""
        from goldenmatch._api import _build_config
        cfg = _build_config(fuzzy={"name": 0.85}, blocking=["zip"])
        assert cfg.blocking is not None
        assert len(cfg.blocking.keys) == 1

    def test_llm_scorer_flag(self):
        from goldenmatch._api import _build_config
        cfg = _build_config(fuzzy={"name": 0.85}, llm_scorer=True)
        assert cfg.llm_scorer is not None
        assert cfg.llm_scorer.enabled is True

    def test_backend_set(self):
        from goldenmatch._api import _build_config
        cfg = _build_config(fuzzy={"name": 0.85}, backend="ray")
        assert cfg.backend == "ray"


class TestExtractHelpers:
    def test_extract_stats_empty(self):
        from goldenmatch._api import _extract_stats
        stats = _extract_stats({})
        assert stats["total_records"] == 0
        assert stats["total_clusters"] == 0
        assert stats["match_rate"] == 0.0

    def test_extract_stats_with_data(self):
        """total_records counts *input* rows (dupes + unique), not output
        tables. golden is a derived rollup — one canonical per multi-member
        cluster — and must NOT be added on top, or the count exceeds the
        number of rows that were ever in the dataset.
        """
        from goldenmatch._api import _extract_stats
        # Realistic shape: 1 duplicate cluster with 2 source rows (rolled up
        # to 1 golden canonical), plus 2 unique singletons. Input rows = 4.
        golden = pl.DataFrame({"x": [10]})     # 1 canonical
        dupes = pl.DataFrame({"x": [1, 2]})    # 2 source rows in cluster
        unique = pl.DataFrame({"x": [3, 4]})   # 2 singletons
        clusters = {
            0: {"size": 2, "members": [1, 2], "pair_scores": {}},
            1: {"size": 1, "members": [3], "pair_scores": {}},
        }
        stats = _extract_stats({
            "golden": golden, "dupes": dupes, "unique": unique,
            "clusters": clusters,
        })
        assert stats["total_records"] == 4  # dupes(2) + unique(2), NOT + golden
        assert stats["total_clusters"] == 1  # only size > 1
        assert stats["matched_records"] == 2

    def test_extract_pairs(self):
        from goldenmatch._api import _extract_pairs
        clusters = {
            0: {"pair_scores": {(1, 2): 0.95, (1, 3): 0.90}},
        }
        pairs = _extract_pairs({"clusters": clusters})
        assert len(pairs) == 2

    def test_extract_pairs_empty(self):
        from goldenmatch._api import _extract_pairs
        pairs = _extract_pairs({})
        assert pairs == []


class TestScoreStringsEdgeCases:
    def test_token_sort(self):
        import goldenmatch as gm
        score = gm.score_strings("John Smith", "Smith John", "token_sort")
        assert score == 1.0

    def test_soundex_match(self):
        import goldenmatch as gm
        score = gm.score_strings("Robert", "Rupert", "soundex_match")
        assert score == 1.0

    def test_soundex_no_match(self):
        import goldenmatch as gm
        score = gm.score_strings("John", "Mary", "soundex_match")
        assert score == 0.0
