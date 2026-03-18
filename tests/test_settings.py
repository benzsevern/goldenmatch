"""Tests for settings persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from goldenmatch.config.settings import (
    UserSettings,
    load_global_settings,
    load_project_settings,
    save_global_settings,
    save_project_settings,
)


class TestUserSettings:
    def test_defaults(self):
        s = UserSettings()
        assert s.output_mode == "tui"
        assert s.output_format == "csv"
        assert s.embedding_model == "auto"

    def test_round_trip(self):
        s = UserSettings(output_mode="files", embedding_model="all-mpnet-base-v2")
        d = s.to_dict()
        s2 = UserSettings.from_dict(d)
        assert s2.output_mode == "files"
        assert s2.embedding_model == "all-mpnet-base-v2"

    def test_from_empty_dict(self):
        s = UserSettings.from_dict({})
        assert s.output_mode == "tui"


class TestGlobalSettings:
    def test_save_and_load(self, tmp_path, monkeypatch):
        settings_file = tmp_path / "settings.yaml"
        monkeypatch.setattr(
            "goldenmatch.config.settings.GLOBAL_SETTINGS_FILE", settings_file
        )
        monkeypatch.setattr(
            "goldenmatch.config.settings.GLOBAL_SETTINGS_DIR", tmp_path
        )

        s = UserSettings(output_mode="console", output_format="parquet")
        save_global_settings(s)
        assert settings_file.exists()

        loaded = load_global_settings()
        assert loaded.output_mode == "console"
        assert loaded.output_format == "parquet"

    def test_missing_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "goldenmatch.config.settings.GLOBAL_SETTINGS_FILE",
            tmp_path / "nonexistent.yaml",
        )
        s = load_global_settings()
        assert s.output_mode == "tui"


class TestProjectSettings:
    def test_save_and_load(self, tmp_path):
        config_dict = {
            "matchkeys": [{"name": "test", "type": "exact", "fields": [{"field": "name"}]}],
        }
        save_project_settings(config_dict, tmp_path)
        assert (tmp_path / ".goldenmatch.yaml").exists()

        loaded = load_project_settings(tmp_path)
        assert loaded is not None
        assert "matchkeys" in loaded

    def test_missing_returns_none(self, tmp_path):
        result = load_project_settings(tmp_path)
        assert result is None
