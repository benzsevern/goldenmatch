"""Tests for CLI dedupe command."""

from __future__ import annotations

import textwrap
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from goldenmatch.cli.main import app
from goldenmatch.cli.dedupe import _parse_file_source

runner = CliRunner()


@pytest.fixture
def cli_csv(tmp_path) -> Path:
    path = tmp_path / "data.csv"
    df = pl.DataFrame({
        "first_name": ["John", "john", "Jane", "JOHN", "Bob"],
        "last_name": ["Smith", "Smith", "Doe", "Smyth", "Jones"],
        "email": ["john@ex.com", "john@ex.com", "jane@t.com", "john@ex.com", "bob@t.com"],
    })
    df.write_csv(path)
    return path


@pytest.fixture
def simple_config(tmp_path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent("""\
        matchkeys:
          - name: exact_email
            type: exact
            fields:
              - field: email
                transforms: [lowercase, strip]
    """))
    return path


class TestParseFileSource:
    def test_plain_path(self):
        path, source = _parse_file_source("/data/file.csv")
        assert path == "/data/file.csv"
        assert source == "file"

    def test_path_with_source(self):
        path, source = _parse_file_source("/data/file.csv:my_source")
        assert path == "/data/file.csv"
        assert source == "my_source"

    def test_windows_drive_letter(self):
        path, source = _parse_file_source("C:\\data\\file.csv")
        assert path == "C:\\data\\file.csv"
        assert source == "file"

    def test_windows_with_source(self):
        path, source = _parse_file_source("C:\\data\\file.csv:my_source")
        # The last colon separates source, but C: at start is a drive
        # rfind finds last colon -> "C:\\data\\file.csv" : "my_source"
        assert path == "C:\\data\\file.csv"
        assert source == "my_source"


class TestDedupeCmd:
    def test_basic_dedupe(self, cli_csv, simple_config):
        result = runner.invoke(app, [
            "dedupe",
            str(cli_csv),
            "--config", str(simple_config),
            "--quiet",
        ])
        assert result.exit_code == 0

    def test_missing_config(self, cli_csv):
        result = runner.invoke(app, [
            "dedupe",
            str(cli_csv),
            "--config", "/nonexistent/config.yaml",
        ])
        assert result.exit_code == 1
        assert "Config error" in result.output

    def test_help(self):
        result = runner.invoke(app, ["dedupe", "--help"])
        assert result.exit_code == 0
        assert "dedupe" in result.output.lower()

    def test_output_report(self, cli_csv, simple_config):
        result = runner.invoke(app, [
            "dedupe",
            str(cli_csv),
            "--config", str(simple_config),
            "--output-report",
        ])
        assert result.exit_code == 0
        assert "Dedupe Report" in result.output

    def test_output_all(self, cli_csv, simple_config, tmp_path):
        result = runner.invoke(app, [
            "dedupe",
            str(cli_csv),
            "--config", str(simple_config),
            "--output-all",
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0


class TestMatchCmd:
    def test_help(self):
        result = runner.invoke(app, ["match", "--help"])
        assert result.exit_code == 0
        assert "match" in result.output.lower()
