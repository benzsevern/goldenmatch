"""Tests for the GoldenMatch plugin system."""
from __future__ import annotations

import pytest

from goldenmatch.plugins.registry import PluginRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset plugin registry before each test."""
    PluginRegistry.reset()
    yield
    PluginRegistry.reset()


class TestPluginRegistry:
    def test_singleton(self):
        r1 = PluginRegistry.instance()
        r2 = PluginRegistry.instance()
        assert r1 is r2

    def test_reset(self):
        r1 = PluginRegistry.instance()
        PluginRegistry.reset()
        r2 = PluginRegistry.instance()
        assert r1 is not r2

    def test_list_empty(self):
        r = PluginRegistry.instance()
        plugins = r.list_plugins()
        assert plugins == {
            "scorer": [],
            "transform": [],
            "connector": [],
            "golden_strategy": [],
        }

    def test_register_scorer(self):
        class MyScorer:
            name = "my_scorer"
            def score_pair(self, val_a, val_b):
                return 1.0 if val_a == val_b else 0.0

        r = PluginRegistry.instance()
        r.register_scorer("my_scorer", MyScorer())
        assert r.has_scorer("my_scorer")
        assert r.get_scorer("my_scorer") is not None
        assert "my_scorer" in r.list_plugins()["scorer"]

    def test_register_connector(self):
        class MyConn:
            name = "test_conn"
            def read(self, config):
                pass
            def write(self, df, config):
                pass

        r = PluginRegistry.instance()
        r.register_connector("test_conn", MyConn())
        assert r.has_connector("test_conn")

    def test_unknown_scorer_returns_none(self):
        r = PluginRegistry.instance()
        assert r.get_scorer("nonexistent") is None
        assert not r.has_scorer("nonexistent")

    def test_discover_no_crash(self):
        """discover() should not crash even with no plugins installed."""
        r = PluginRegistry.instance()
        r.discover()
        # Should be empty (no test plugins installed)
        assert isinstance(r.list_plugins(), dict)


class TestPluginScorerIntegration:
    def test_score_field_with_plugin_scorer(self):
        """score_field falls through to plugin scorer."""

        class CustomScorer:
            name = "hamming"
            def score_pair(self, val_a, val_b):
                if val_a is None or val_b is None:
                    return None
                matches = sum(a == b for a, b in zip(val_a, val_b))
                return matches / max(len(val_a), len(val_b))

        r = PluginRegistry.instance()
        r.register_scorer("hamming", CustomScorer())

        from goldenmatch.core.scorer import score_field
        result = score_field("hello", "hallo", "hamming")
        assert result == 4.0 / 5.0

    def test_schema_accepts_plugin_scorer(self):
        """MatchkeyField validator accepts a registered plugin scorer."""

        class FooScorer:
            name = "foo_scorer"
            def score_pair(self, val_a, val_b):
                return 0.5

        r = PluginRegistry.instance()
        r.register_scorer("foo_scorer", FooScorer())

        from goldenmatch.config.schemas import MatchkeyField
        mf = MatchkeyField(field="name", scorer="foo_scorer", weight=1.0)
        assert mf.scorer == "foo_scorer"

    def test_schema_rejects_unknown_scorer(self):
        """MatchkeyField validator still rejects truly unknown scorers."""
        from goldenmatch.config.schemas import MatchkeyField
        with pytest.raises(ValueError, match="Invalid scorer"):
            MatchkeyField(field="name", scorer="totally_nonexistent_scorer", weight=1.0)


class TestPluginDiscovery:
    def test_discover_idempotent(self):
        """Calling discover() multiple times has no side effects."""
        r = PluginRegistry.instance()
        r.discover()
        plugins_1 = r.list_plugins()
        r.discover()
        plugins_2 = r.list_plugins()
        assert plugins_1 == plugins_2
