"""Tests for CLI --preview flag on dedupe and match commands."""

import pytest
import yaml
from typer.testing import CliRunner
from goldenmatch.cli.main import app

runner = CliRunner()


@pytest.fixture
def preview_config(tmp_path, sample_csv):
    cfg_path = tmp_path / "goldenmatch.yaml"
    cfg_path.write_text(yaml.dump({
        "matchkeys": [
            {
                "name": "email_key",
                "fields": [{"column": "email", "transforms": ["lowercase"]}],
                "comparison": "exact",
            }
        ],
        "output": {
            "format": "csv",
            "directory": str(tmp_path),
            "run_name": "preview_test",
        },
    }))
    return cfg_path


class TestDedupePreview:
    def test_preview_flag(self, sample_csv, preview_config):
        result = runner.invoke(
            app,
            ["dedupe", str(sample_csv), "--config", str(preview_config), "--preview"],
            input="N\n",
        )
        assert result.exit_code == 0

    def test_preview_size(self, sample_csv, preview_config):
        result = runner.invoke(
            app,
            [
                "dedupe", str(sample_csv),
                "--config", str(preview_config),
                "--preview", "--preview-size", "3",
            ],
            input="N\n",
        )
        assert result.exit_code == 0

    def test_no_preview_still_works(self, sample_csv, preview_config):
        result = runner.invoke(
            app,
            ["dedupe", str(sample_csv), "--config", str(preview_config), "--output-report"],
        )
        assert result.exit_code == 0


class TestMatchPreview:
    def test_preview_flag(self, sample_csv, sample_csv_b, preview_config):
        result = runner.invoke(
            app,
            [
                "match", str(sample_csv),
                "--against", str(sample_csv_b),
                "--config", str(preview_config),
                "--preview",
            ],
            input="N\n",
        )
        assert result.exit_code == 0
