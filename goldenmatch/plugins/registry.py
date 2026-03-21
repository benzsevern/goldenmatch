"""Plugin registry -- discovers and manages GoldenMatch plugins via entry points."""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

# Entry point group names
_GROUPS = {
    "scorer": "goldenmatch.plugins.scorer",
    "transform": "goldenmatch.plugins.transform",
    "connector": "goldenmatch.plugins.connector",
    "golden_strategy": "goldenmatch.plugins.golden_strategy",
}


class PluginRegistry:
    """Singleton registry for discovered plugins."""

    _instance: PluginRegistry | None = None
    _discovered: bool = False

    def __init__(self) -> None:
        self._scorers: dict[str, object] = {}
        self._transforms: dict[str, object] = {}
        self._connectors: dict[str, object] = {}
        self._golden_strategies: dict[str, object] = {}

    @classmethod
    def instance(cls) -> PluginRegistry:
        """Get or create the singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None
        cls._discovered = False

    def discover(self) -> None:
        """Scan entry points and register all found plugins."""
        if PluginRegistry._discovered:
            return

        for plugin_type, group_name in _GROUPS.items():
            eps = entry_points(group=group_name)
            for ep in eps:
                try:
                    plugin_cls = ep.load()
                    plugin = plugin_cls()
                    name = getattr(plugin, "name", ep.name)
                    self._register(plugin_type, name, plugin)
                    logger.debug("Loaded plugin %s:%s", plugin_type, name)
                except Exception as e:
                    logger.warning("Failed to load plugin %s:%s: %s", plugin_type, ep.name, e)

        PluginRegistry._discovered = True

    def register_scorer(self, name: str, plugin: object) -> None:
        """Manually register a scorer plugin (for testing or built-in extensions)."""
        self._register("scorer", name, plugin)

    def register_transform(self, name: str, plugin: object) -> None:
        """Manually register a transform plugin."""
        self._register("transform", name, plugin)

    def register_connector(self, name: str, plugin: object) -> None:
        """Manually register a connector plugin."""
        self._register("connector", name, plugin)

    def register_golden_strategy(self, name: str, plugin: object) -> None:
        """Manually register a golden strategy plugin."""
        self._register("golden_strategy", name, plugin)

    def _register(self, plugin_type: str, name: str, plugin: object) -> None:
        store = {
            "scorer": self._scorers,
            "transform": self._transforms,
            "connector": self._connectors,
            "golden_strategy": self._golden_strategies,
        }[plugin_type]
        store[name] = plugin

    def get_scorer(self, name: str) -> object | None:
        return self._scorers.get(name)

    def get_transform(self, name: str) -> object | None:
        return self._transforms.get(name)

    def get_connector(self, name: str) -> object | None:
        return self._connectors.get(name)

    def get_golden_strategy(self, name: str) -> object | None:
        return self._golden_strategies.get(name)

    def has_scorer(self, name: str) -> bool:
        return name in self._scorers

    def has_transform(self, name: str) -> bool:
        return name in self._transforms

    def has_connector(self, name: str) -> bool:
        return name in self._connectors

    def list_plugins(self) -> dict[str, list[str]]:
        """Return all registered plugins by type."""
        return {
            "scorer": list(self._scorers.keys()),
            "transform": list(self._transforms.keys()),
            "connector": list(self._connectors.keys()),
            "golden_strategy": list(self._golden_strategies.keys()),
        }
