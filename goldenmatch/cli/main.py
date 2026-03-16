"""GoldenMatch CLI application."""

from __future__ import annotations

import typer

from goldenmatch.cli.dedupe import dedupe_cmd
from goldenmatch.cli.match import match_cmd

app = typer.Typer(
    name="goldenmatch",
    help="GoldenMatch: deduplication and list-matching toolkit.",
    no_args_is_help=True,
)

app.command("dedupe", help="Run deduplication on one or more files.")(dedupe_cmd)
app.command("match", help="Match a target file against reference files.")(match_cmd)
