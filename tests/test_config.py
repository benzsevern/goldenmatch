"""Tests for goldenmatch config schemas and loader."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from goldenmatch.config.loader import load_config
from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    FieldTransform,
    GoldenFieldRule,
    GoldenMatchConfig,
    GoldenRulesConfig,
    InputConfig,
    InputFileConfig,
    MatchkeyConfig,
    MatchkeyField,
    MatchSettingsConfig,
    OutputConfig,
)


# ── FieldTransform ──────────────────────────────────────────────────────────


class TestFieldTransform:
    def test_valid_simple_transforms(self):
        for t in [
            "lowercase", "uppercase", "strip", "strip_all", "soundex",
            "metaphone", "digits_only", "alpha_only", "normalize_whitespace",
        ]:
            ft = FieldTransform(transform=t)
            assert ft.transform == t

    def test_valid_substring_transform(self):
        ft = FieldTransform(transform="substring:0:3")
        assert ft.transform == "substring:0:3"

    def test_invalid_transform(self):
        with pytest.raises(ValidationError):
            FieldTransform(transform="invalid_transform")

    def test_invalid_substring_format(self):
        with pytest.raises(ValidationError):
            FieldTransform(transform="substring:abc:def")


# ── MatchkeyField ───────────────────────────────────────────────────────────


class TestMatchkeyField:
    def test_basic_field(self):
        f = MatchkeyField(field="first_name")
        assert f.field == "first_name"
        assert f.transforms == []
        assert f.scorer is None
        assert f.weight is None

    def test_field_with_transforms(self):
        f = MatchkeyField(
            field="first_name",
            transforms=["lowercase", "strip"],
        )
        assert len(f.transforms) == 2

    def test_field_with_scorer(self):
        f = MatchkeyField(field="first_name", scorer="jaro_winkler", weight=0.5)
        assert f.scorer == "jaro_winkler"
        assert f.weight == 0.5

    def test_invalid_scorer(self):
        with pytest.raises(ValidationError):
            MatchkeyField(field="first_name", scorer="bad_scorer")

    def test_valid_scorers(self):
        for s in ["exact", "jaro_winkler", "levenshtein", "token_sort", "soundex_match"]:
            f = MatchkeyField(field="x", scorer=s, weight=1.0)
            assert f.scorer == s


# ── MatchkeyConfig ──────────────────────────────────────────────────────────


class TestMatchkeyConfig:
    def test_exact_matchkey(self):
        mk = MatchkeyConfig(
            name="exact_email",
            type="exact",
            fields=[MatchkeyField(field="email")],
        )
        assert mk.type == "exact"

    def test_weighted_matchkey_requires_threshold(self):
        with pytest.raises(ValidationError):
            MatchkeyConfig(
                name="fuzzy",
                type="weighted",
                fields=[MatchkeyField(field="name", scorer="jaro_winkler", weight=0.5)],
                # missing threshold
            )

    def test_weighted_matchkey_fields_need_scorer_and_weight(self):
        with pytest.raises(ValidationError):
            MatchkeyConfig(
                name="fuzzy",
                type="weighted",
                threshold=0.8,
                fields=[MatchkeyField(field="name")],  # no scorer/weight
            )

    def test_valid_weighted_matchkey(self):
        mk = MatchkeyConfig(
            name="fuzzy",
            type="weighted",
            threshold=0.8,
            fields=[
                MatchkeyField(field="name", scorer="jaro_winkler", weight=0.6),
                MatchkeyField(field="email", scorer="exact", weight=0.4),
            ],
        )
        assert mk.threshold == 0.8


# ── BlockingConfig ──────────────────────────────────────────────────────────


class TestBlockingConfig:
    def test_valid_blocking(self):
        bc = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=["lowercase"])]
        )
        assert len(bc.keys) == 1

    def test_blocking_key_fields_required(self):
        with pytest.raises(ValidationError):
            BlockingKeyConfig(fields=[])


class TestBlockingConfigAdaptive:
    def test_static_strategy_default(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"], transforms=["strip"])])
        assert cfg.strategy == "static"

    def test_adaptive_strategy(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        cfg = BlockingConfig(
            keys=[BlockingKeyConfig(fields=["zip"], transforms=["strip"])],
            strategy="adaptive",
            sub_block_keys=[BlockingKeyConfig(fields=["first_name"], transforms=["lowercase"])],
        )
        assert cfg.strategy == "adaptive"
        assert len(cfg.sub_block_keys) == 1

    def test_sorted_neighborhood_strategy(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, SortKeyField
        cfg = BlockingConfig(
            keys=[],
            strategy="sorted_neighborhood",
            window_size=25,
            sort_key=[
                SortKeyField(column="last_name", transforms=["lowercase", "soundex"]),
                SortKeyField(column="zip", transforms=["substring:0:3"]),
            ],
        )
        assert cfg.window_size == 25
        assert len(cfg.sort_key) == 2

    def test_invalid_strategy_rejected(self):
        from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
        with pytest.raises(ValueError):
            BlockingConfig(keys=[], strategy="invalid")


# ── GoldenFieldRule ─────────────────────────────────────────────────────────


class TestGoldenFieldRule:
    def test_valid_strategies(self):
        for s in ["most_recent", "source_priority", "most_complete", "majority_vote", "first_non_null"]:
            rule = GoldenFieldRule(strategy=s) if s not in ("most_recent", "source_priority") else None
            if s == "most_recent":
                rule = GoldenFieldRule(strategy=s, date_column="updated_at")
            elif s == "source_priority":
                rule = GoldenFieldRule(strategy=s, source_priority=["crm", "web"])
            else:
                rule = GoldenFieldRule(strategy=s)
            assert rule.strategy == s

    def test_most_recent_requires_date_column(self):
        with pytest.raises(ValidationError):
            GoldenFieldRule(strategy="most_recent")

    def test_source_priority_requires_priority_list(self):
        with pytest.raises(ValidationError):
            GoldenFieldRule(strategy="source_priority")

    def test_invalid_strategy(self):
        with pytest.raises(ValidationError):
            GoldenFieldRule(strategy="invalid_strategy")


# ── GoldenRulesConfig ──────────────────────────────────────────────────────


class TestGoldenRulesConfig:
    def test_default_and_field_rules(self):
        gr = GoldenRulesConfig(
            default_strategy="first_non_null",
            field_rules={"email": GoldenFieldRule(strategy="most_complete")},
        )
        assert gr.default_strategy == "first_non_null"

    def test_default_strategy_required(self):
        with pytest.raises(ValidationError):
            GoldenRulesConfig()


# ── InputConfig / OutputConfig ──────────────────────────────────────────────


class TestInputConfig:
    def test_single_file(self):
        ic = InputConfig(file_a=InputFileConfig(path="a.csv"))
        assert ic.file_a.path == "a.csv"
        assert ic.file_b is None

    def test_two_files(self):
        ic = InputConfig(
            file_a=InputFileConfig(path="a.csv"),
            file_b=InputFileConfig(path="b.csv"),
        )
        assert ic.file_b is not None


class TestOutputConfig:
    def test_defaults(self):
        oc = OutputConfig(path="out.csv")
        assert oc.path == "out.csv"


# ── GoldenMatchConfig (top-level) ──────────────────────────────────────────


class TestGoldenMatchConfig:
    def test_minimal_valid(self):
        cfg = GoldenMatchConfig(
            input=InputConfig(file_a=InputFileConfig(path="a.csv")),
            output=OutputConfig(path="out.csv"),
            match_settings=MatchSettingsConfig(
                matchkeys=[
                    MatchkeyConfig(
                        name="exact_email",
                        type="exact",
                        fields=[MatchkeyField(field="email")],
                    )
                ]
            ),
            golden_rules=GoldenRulesConfig(default_strategy="first_non_null"),
        )
        assert cfg is not None

    def test_fuzzy_matchkey_without_blocking_raises(self):
        """Fuzzy/weighted matchkeys require blocking config."""
        with pytest.raises(ValidationError):
            GoldenMatchConfig(
                input=InputConfig(file_a=InputFileConfig(path="a.csv")),
                output=OutputConfig(path="out.csv"),
                match_settings=MatchSettingsConfig(
                    matchkeys=[
                        MatchkeyConfig(
                            name="fuzzy",
                            type="weighted",
                            threshold=0.8,
                            fields=[
                                MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0),
                            ],
                        )
                    ]
                ),
                golden_rules=GoldenRulesConfig(default_strategy="first_non_null"),
                # no blocking config
            )

    def test_fuzzy_matchkey_with_blocking_ok(self):
        cfg = GoldenMatchConfig(
            input=InputConfig(file_a=InputFileConfig(path="a.csv")),
            output=OutputConfig(path="out.csv"),
            match_settings=MatchSettingsConfig(
                matchkeys=[
                    MatchkeyConfig(
                        name="fuzzy",
                        type="weighted",
                        threshold=0.8,
                        fields=[
                            MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0),
                        ],
                    )
                ]
            ),
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["zip"])]
            ),
            golden_rules=GoldenRulesConfig(default_strategy="first_non_null"),
        )
        assert cfg is not None


# ── Config Loader ──────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_load_minimal_yaml(self, tmp_path):
        cfg_data = {
            "input": {"file_a": {"path": "a.csv"}},
            "output": {"path": "out.csv"},
            "match_settings": {
                "matchkeys": [
                    {
                        "name": "exact_email",
                        "type": "exact",
                        "fields": [{"field": "email"}],
                    }
                ]
            },
            "golden_rules": {"default_strategy": "first_non_null"},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_data))
        cfg = load_config(path)
        assert isinstance(cfg, GoldenMatchConfig)
        assert cfg.input.file_a.path == "a.csv"

    def test_load_full_yaml_with_golden_rules_normalization(self, tmp_path):
        """Non-special keys in golden_rules should be normalized into field_rules."""
        cfg_data = {
            "input": {
                "file_a": {"path": "a.csv", "id_column": "id", "source_label": "crm"},
                "file_b": {"path": "b.csv", "id_column": "id", "source_label": "web"},
            },
            "output": {"path": "out.csv"},
            "match_settings": {
                "matchkeys": [
                    {
                        "name": "exact_email",
                        "type": "exact",
                        "fields": [{"field": "email", "transforms": ["lowercase", "strip"]}],
                    },
                    {
                        "name": "fuzzy_name",
                        "type": "weighted",
                        "threshold": 0.85,
                        "fields": [
                            {"field": "first_name", "scorer": "jaro_winkler", "weight": 0.5},
                            {"field": "last_name", "scorer": "jaro_winkler", "weight": 0.5},
                        ],
                    },
                ]
            },
            "blocking": {
                "keys": [{"fields": ["zip"], "transforms": ["lowercase"]}]
            },
            "golden_rules": {
                "default_strategy": "first_non_null",
                "email": {"strategy": "most_complete"},
                "phone": {"strategy": "majority_vote"},
            },
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg_data))
        cfg = load_config(path)
        assert len(cfg.match_settings.matchkeys) == 2
        assert "email" in cfg.golden_rules.field_rules
        assert "phone" in cfg.golden_rules.field_rules
        assert cfg.golden_rules.field_rules["email"].strategy == "most_complete"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))

    def test_load_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(": : : not valid yaml [[[")
        with pytest.raises(ValueError):
            load_config(path)


def test_golden_rules_cluster_quality_defaults():
    """New cluster quality config fields have correct defaults."""
    from goldenmatch.config.schemas import GoldenRulesConfig
    config = GoldenRulesConfig(default_strategy="most_complete")
    assert config.auto_split is True
    assert config.quality_weighting is True
    assert config.weak_cluster_threshold == 0.3
