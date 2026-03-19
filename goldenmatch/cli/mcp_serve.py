"""CLI command for starting the GoldenMatch MCP server."""

from __future__ import annotations

from typing import Optional

import typer


def mcp_serve_cmd(
    files: list[str] = typer.Argument(..., help="Data files to load"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML file"),
) -> None:
    """Start the GoldenMatch MCP server (for Claude Desktop integration)."""
    import asyncio
    from goldenmatch.mcp.server import run_server

    asyncio.run(run_server(files, config))
