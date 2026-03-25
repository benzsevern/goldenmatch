"""CLI command for the A2A agent server."""
from __future__ import annotations

import typer


def agent_serve_cmd(
    port: int = typer.Option(8200, help="Port for the A2A server"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
):
    """Start the GoldenMatch A2A agent server.

    Exposes GoldenMatch as a discoverable agent via the A2A protocol.
    Other AI agents can find and invoke entity resolution skills.

    Agent card: http://{host}:{port}/.well-known/agent.json
    """
    try:
        from goldenmatch.a2a.server import create_app
        import aiohttp.web
    except ImportError:
        typer.echo("A2A server requires aiohttp. Install with: pip install goldenmatch[agent]")
        raise typer.Exit(1)

    typer.echo(f"GoldenMatch A2A agent server starting on {host}:{port}")
    typer.echo(f"Agent card: http://{host}:{port}/.well-known/agent.json")
    typer.echo("Storage: memory (set DATABASE_URL for Postgres, create .goldenmatch/ for SQLite)")
    app = create_app(host, port)
    aiohttp.web.run_app(app, host=host, port=port, print=None)
