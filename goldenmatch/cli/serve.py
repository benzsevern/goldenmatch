"""CLI command for starting the GoldenMatch REST API server."""

from __future__ import annotations

from typing import Optional

import typer


def serve_cmd(
    files: list[str] = typer.Argument(..., help="Data files to load"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML file"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", "-p", help="Server port"),
) -> None:
    """Start the GoldenMatch REST API server."""
    from goldenmatch.tui.engine import MatchEngine

    # Load engine
    engine = MatchEngine(files)
    print(f"Loaded {engine.row_count:,} records from {len(files)} file(s)")

    # Load config
    if config:
        from goldenmatch.config.loader import load_config
        cfg = load_config(config)
    else:
        from goldenmatch.core.autoconfig import auto_configure
        parsed = [(f, f.split("/")[-1].split("\\")[-1].split(".")[0]) for f in files]
        cfg = auto_configure(parsed)
        print("Auto-configured matching rules")

    # Start server
    from goldenmatch.api.server import start_server
    start_server(engine, cfg, host=host, port=port)
