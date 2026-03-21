"""Tests for the connector framework."""
from __future__ import annotations

import polars as pl
import pytest

from goldenmatch.connectors.base import BaseConnector, ConnectorError, load_connector


class MockConnector(BaseConnector):
    name = "mock"

    def read(self, config: dict) -> pl.LazyFrame:
        return pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).lazy()

    def write(self, df: pl.DataFrame, config: dict) -> None:
        pass


class TestBaseConnector:
    def test_mock_connector_read(self):
        conn = MockConnector(config={})
        lf = conn.read({})
        df = lf.collect()
        assert df.height == 2
        assert "name" in df.columns

    def test_base_write_raises(self):
        class ReadOnlyConn(BaseConnector):
            name = "readonly"
            def read(self, config):
                return pl.DataFrame().lazy()

        conn = ReadOnlyConn(config={})
        with pytest.raises(ConnectorError, match="Write not implemented"):
            conn.write(pl.DataFrame(), {})


class TestLoadConnector:
    def test_unknown_connector(self):
        with pytest.raises(ConnectorError, match="Unknown connector"):
            load_connector("nonexistent_db", {})

    def test_connector_error_without_creds(self):
        """Built-in connectors raise errors when credentials are missing."""
        # Connector loads but fails on read without credentials
        with pytest.raises(Exception):
            conn = load_connector("snowflake", {})
            conn.read({"query": "SELECT 1"})

    def test_plugin_connector(self):
        """Plugin connectors are found via registry."""
        from goldenmatch.plugins.registry import PluginRegistry
        PluginRegistry.reset()
        r = PluginRegistry.instance()

        mock = MockConnector(config={})
        r.register_connector("mock", mock)

        result = load_connector("mock", {})
        assert result is mock

        PluginRegistry.reset()


class TestConnectorCredentials:
    def test_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_CRED", "my-secret-key")
        conn = MockConnector(config={"credentials_env": "TEST_CRED"})
        assert conn._credentials.get("key") == "my-secret-key"

    def test_loads_suffixed_vars(self, monkeypatch):
        monkeypatch.setenv("SF_USER", "admin")
        monkeypatch.setenv("SF_PASSWORD", "pass123")
        conn = MockConnector(config={"credentials_env": "SF"})
        assert conn._credentials.get("user") == "admin"
        assert conn._credentials.get("password") == "pass123"
