"""Tests for CLI match command."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from goldenmatch.cli.main import app

runner = CliRunner()


@pytest.fixture
def match_config(tmp_path) -> Path:
    path = tmp_path / "match_config.yaml"
    path.write_text(textwrap.dedent("""\
        matchkeys:
          - name: exact_email
            type: exact
            fields:
              - field: email
                transforms: [lowercase, strip]
    """))
    return path


class TestMatchCmd:
    def test_basic_match(self, sample_csv, sample_csv_b, match_config, tmp_path):
        result = runner.invoke(app, [
            "match",
            str(sample_csv),
            "--against", str(sample_csv_b),
            "--config", str(match_config),
            "--output-all",
            "--output-dir", str(tmp_path / "out"),
            "--quiet",
        ])
        assert result.exit_code == 0

    def test_help(self):
        result = runner.invoke(app, ["match", "--help"])
        assert result.exit_code == 0
        assert "match" in result.output.lower()

    def test_missing_config(self, sample_csv, sample_csv_b):
        result = runner.invoke(app, [
            "match",
            str(sample_csv),
            "--against", str(sample_csv_b),
            "--config", "/nonexistent/config.yaml",
        ])
        assert result.exit_code == 1
        assert "Config error" in result.output

    def test_output_report(self, sample_csv, sample_csv_b, match_config):
        result = runner.invoke(app, [
            "match",
            str(sample_csv),
            "--against", str(sample_csv_b),
            "--config", str(match_config),
            "--output-report",
        ])
        assert result.exit_code == 0
        assert "Match Report" in result.output

    def test_match_mode_all(self, sample_csv, sample_csv_b, match_config):
        result = runner.invoke(app, [
            "match",
            str(sample_csv),
            "--against", str(sample_csv_b),
            "--config", str(match_config),
            "--match-mode", "all",
            "--quiet",
        ])
        assert result.exit_code == 0
