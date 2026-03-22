"""Tests for incremental CLI command."""
from __future__ import annotations

import textwrap
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from goldenmatch.cli.main import app

runner = CliRunner()


@pytest.fixture
def base_csv(tmp_path) -> Path:
    path = tmp_path / "base.csv"
    pl.DataFrame({
        "first_name": ["John", "Jane", "Bob"],
        "last_name": ["Smith", "Doe", "Jones"],
        "email": ["john@ex.com", "jane@t.com", "bob@t.com"],
    }).write_csv(path)
    return path


@pytest.fixture
def new_csv(tmp_path) -> Path:
    path = tmp_path / "new.csv"
    pl.DataFrame({
        "first_name": ["john", "Alice"],
        "last_name": ["Smith", "Brown"],
        "email": ["john@ex.com", "alice@t.com"],
    }).write_csv(path)
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


class TestIncrementalCLI:
    def test_basic_incremental(self, base_csv, new_csv, simple_config, tmp_path):
        output = tmp_path / "output.csv"
        result = runner.invoke(app, [
            "incremental",
            str(base_csv),
            "--new-records", str(new_csv),
            "--config", str(simple_config),
            "--output", str(output),
        ])
        assert result.exit_code == 0
        assert output.exists()
        df = pl.read_csv(output)
        assert df.height > 0

    def test_incremental_shows_stats(self, base_csv, new_csv, simple_config):
        result = runner.invoke(app, [
            "incremental",
            str(base_csv),
            "--new-records", str(new_csv),
            "--config", str(simple_config),
        ])
        assert result.exit_code == 0
        out = result.stdout.lower()
        assert "matched" in out or "new" in out or "processed" in out

    def test_incremental_missing_new(self, base_csv, simple_config, tmp_path):
        result = runner.invoke(app, [
            "incremental",
            str(base_csv),
            "--new-records", str(tmp_path / "does_not_exist.csv"),
            "--config", str(simple_config),
        ])
        assert result.exit_code != 0
