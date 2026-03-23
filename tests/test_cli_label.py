"""Tests for label CLI command."""
from __future__ import annotations

from typer.testing import CliRunner
from goldenmatch.cli.main import app

runner = CliRunner()


class TestLabelCLI:
    def test_label_help(self):
        result = runner.invoke(app, ["label", "--help"])
        assert result.exit_code == 0
        assert "ground truth" in result.stdout.lower() or "label" in result.stdout.lower()

    def test_label_missing_config(self):
        result = runner.invoke(app, ["label", "nonexistent.csv", "--config", "nonexistent.yaml"])
        assert result.exit_code != 0
