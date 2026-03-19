"""CLI command for live stream mode — poll and match continuously."""

from __future__ import annotations

from typing import Optional

import typer


def watch_cmd(
    source_type: str = typer.Option("postgres", "--source-type", help="Database type"),
    connection_string: Optional[str] = typer.Option(None, "--connection-string", help="Database URL"),
    table: str = typer.Option(..., "--table", help="Table to watch"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML file"),
    poll_interval: int = typer.Option(30, "--interval", help="Seconds between polls"),
    output_mode: str = typer.Option("separate", "--output-mode", help="separate or in_place"),
    incremental_column: Optional[str] = typer.Option(None, "--incremental-column", help="Column for change detection"),
) -> None:
    """Watch a database table and match new records continuously."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Load config
    if config:
        from goldenmatch.config.loader import load_config
        cfg = load_config(config)
    else:
        typer.echo("Error: --config is required for watch mode.", err=True)
        raise typer.Exit(code=1)

    # Create connector
    source_config = {"type": source_type, "connection": connection_string}
    from goldenmatch.db.connector import create_connector
    connector = create_connector(source_config)

    try:
        connector.connect()

        from goldenmatch.db.watch import watch
        watch(
            connector=connector,
            source_table=table,
            config=cfg,
            poll_interval=poll_interval,
            output_mode=output_mode,
            incremental_column=incremental_column,
        )
    finally:
        connector.close()
