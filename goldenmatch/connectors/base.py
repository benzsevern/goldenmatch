"""Base connector class for GoldenMatch data sources."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import polars as pl

logger = logging.getLogger(__name__)


class ConnectorError(Exception):
    """Raised when a connector operation fails."""


class BaseConnector(ABC):
    """Base class for all connectors.

    Subclasses implement read() and optionally write().
    Credentials are read from environment variables specified by credentials_env.
    """

    name: str = "base"

    def __init__(self, config: dict) -> None:
        self.config = config
        self._credentials: dict[str, str] = {}
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Load credentials from environment variables."""
        cred_env = self.config.get("credentials_env")
        if not cred_env:
            return

        if isinstance(cred_env, str):
            # Single env var or prefix
            val = os.environ.get(cred_env)
            if val:
                self._credentials["key"] = val
            # Also try common suffixes
            for suffix in ("_USER", "_PASSWORD", "_ACCOUNT", "_DATABASE", "_SCHEMA", "_WAREHOUSE"):
                env_val = os.environ.get(f"{cred_env}{suffix}")
                if env_val:
                    self._credentials[suffix.lstrip("_").lower()] = env_val

    @abstractmethod
    def read(self, config: dict) -> pl.LazyFrame:
        """Read data from the external source.

        Args:
            config: Source configuration dict with query, table, filters, etc.

        Returns:
            Polars LazyFrame with the data.

        Raises:
            ConnectorError: If the read fails.
        """
        ...

    def write(self, df: pl.DataFrame, config: dict) -> None:
        """Write data to the external sink.

        Args:
            df: DataFrame to write.
            config: Write-back configuration dict with table, mode, etc.

        Raises:
            ConnectorError: If the write fails.
        """
        raise ConnectorError(f"Write not implemented for connector '{self.name}'")


def load_connector(connector_name: str, config: dict) -> BaseConnector:
    """Load a connector by name. Checks plugin registry first, then built-in connectors."""
    from goldenmatch.plugins.registry import PluginRegistry
    plugin = PluginRegistry.instance().get_connector(connector_name)
    if plugin is not None:
        return plugin

    # Built-in connectors
    _BUILTIN = {
        "snowflake": "goldenmatch.connectors.snowflake:SnowflakeConnector",
        "databricks": "goldenmatch.connectors.databricks:DatabricksConnector",
        "bigquery": "goldenmatch.connectors.bigquery:BigQueryConnector",
        "hubspot": "goldenmatch.connectors.hubspot:HubSpotConnector",
        "salesforce": "goldenmatch.connectors.salesforce:SalesforceConnector",
    }

    if connector_name not in _BUILTIN:
        raise ConnectorError(
            f"Unknown connector '{connector_name}'. "
            f"Available: {sorted(_BUILTIN.keys())}. "
            f"Or install a plugin that provides it."
        )

    module_path, class_name = _BUILTIN[connector_name].rsplit(":", 1)
    try:
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(config)
    except ImportError as e:
        raise ConnectorError(
            f"Connector '{connector_name}' requires additional dependencies. "
            f"Install with: pip install goldenmatch[{connector_name}]\n"
            f"Error: {e}"
        ) from e
