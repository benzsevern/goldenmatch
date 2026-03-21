"""GoldenMatch connectors -- read/write data from external sources.

Each connector implements the ConnectorPlugin protocol from goldenmatch.plugins.base.
Connectors are optional dependencies: pip install goldenmatch[snowflake], etc.

Built-in connectors auto-register with the plugin registry on import.
"""
from __future__ import annotations

from goldenmatch.connectors.base import BaseConnector, ConnectorError

__all__ = ["BaseConnector", "ConnectorError"]
