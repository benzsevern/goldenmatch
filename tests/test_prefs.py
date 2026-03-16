"""Tests for the preset store."""

from __future__ import annotations

from pathlib import Path

import pytest

from goldenmatch.prefs.store import PresetStore


@pytest.fixture
def store(tmp_path) -> PresetStore:
    return PresetStore(base_dir=tmp_path / "presets")


@pytest.fixture
def sample_yaml(tmp_path) -> Path:
    p = tmp_path / "my_config.yaml"
    p.write_text("matchkeys:\n  - name: test\n    type: exact\n    fields:\n      - field: email\n")
    return p


class TestPresetStore:
    def test_save_and_list(self, store, sample_yaml):
        store.save("my_preset", sample_yaml)
        assert "my_preset" in store.list_presets()

    def test_save_creates_directory(self, store, sample_yaml):
        assert not store.base_dir.exists()
        store.save("first", sample_yaml)
        assert store.base_dir.exists()

    def test_load(self, store, sample_yaml, tmp_path):
        store.save("loadme", sample_yaml)
        dest = tmp_path / "loaded.yaml"
        store.load("loadme", dest)
        assert dest.exists()
        assert dest.read_text() == sample_yaml.read_text()

    def test_delete(self, store, sample_yaml):
        store.save("deleteme", sample_yaml)
        assert "deleteme" in store.list_presets()
        store.delete("deleteme")
        assert "deleteme" not in store.list_presets()

    def test_show(self, store, sample_yaml):
        store.save("showme", sample_yaml)
        content = store.show("showme")
        assert "matchkeys" in content

    def test_list_empty(self, store):
        assert store.list_presets() == []

    def test_list_sorted(self, store, sample_yaml):
        store.save("zebra", sample_yaml)
        store.save("alpha", sample_yaml)
        store.save("middle", sample_yaml)
        assert store.list_presets() == ["alpha", "middle", "zebra"]

    def test_load_nonexistent(self, store, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            store.load("nope", tmp_path / "out.yaml")

    def test_delete_nonexistent(self, store):
        with pytest.raises(FileNotFoundError, match="not found"):
            store.delete("nope")

    def test_show_nonexistent(self, store):
        with pytest.raises(FileNotFoundError, match="not found"):
            store.show("nope")

    def test_save_nonexistent_source(self, store):
        with pytest.raises(FileNotFoundError):
            store.save("bad", "/nonexistent/file.yaml")
