"""Persistent preset store for saving/loading GoldenMatch configs."""

from __future__ import annotations

import shutil
from pathlib import Path


class PresetStore:
    """Manages named config presets stored as YAML files."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        if base_dir is None:
            base_dir = Path.home() / ".goldenmatch" / "presets"
        self._base_dir = Path(base_dir)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def save(self, name: str, config_path: str | Path) -> Path:
        """Copy a config file into the presets directory under the given name.

        Returns the destination path.
        """
        self._base_dir.mkdir(parents=True, exist_ok=True)
        src = Path(config_path)
        if not src.exists():
            raise FileNotFoundError(f"Config file not found: {src}")
        dest = self._base_dir / f"{name}.yaml"
        shutil.copy2(src, dest)
        return dest

    def load(self, name: str, dest: str | Path) -> Path:
        """Copy a named preset to the destination path.

        Returns the destination path.
        """
        preset = self._base_dir / f"{name}.yaml"
        if not preset.exists():
            raise FileNotFoundError(f"Preset '{name}' not found.")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preset, dest)
        return dest

    def list_presets(self) -> list[str]:
        """Return sorted list of available preset names."""
        if not self._base_dir.exists():
            return []
        return sorted(p.stem for p in self._base_dir.glob("*.yaml"))

    def delete(self, name: str) -> None:
        """Remove a preset by name."""
        preset = self._base_dir / f"{name}.yaml"
        if not preset.exists():
            raise FileNotFoundError(f"Preset '{name}' not found.")
        preset.unlink()

    def show(self, name: str) -> str:
        """Return the contents of a preset as a string."""
        preset = self._base_dir / f"{name}.yaml"
        if not preset.exists():
            raise FileNotFoundError(f"Preset '{name}' not found.")
        return preset.read_text(encoding="utf-8")
