"""User settings persistence for GoldenMatch (global + project level)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

GLOBAL_SETTINGS_DIR = Path.home() / ".goldenmatch"
GLOBAL_SETTINGS_FILE = GLOBAL_SETTINGS_DIR / "settings.yaml"
PROJECT_SETTINGS_FILE = ".goldenmatch.yaml"


@dataclass
class UserSettings:
    """User preferences that persist across sessions."""

    output_mode: str = "tui"  # tui | files | console
    output_dir: str = "./goldenmatch_output"
    output_format: str = "csv"  # csv | parquet
    embedding_model: str = "auto"  # auto | model name
    auto_model_threshold: int = 50000

    def to_dict(self) -> dict:
        return {
            "defaults": {
                "output_mode": self.output_mode,
                "output_dir": self.output_dir,
                "output_format": self.output_format,
                "embedding_model": self.embedding_model,
                "auto_model_threshold": self.auto_model_threshold,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserSettings":
        defaults = data.get("defaults", {})
        return cls(
            output_mode=defaults.get("output_mode", "tui"),
            output_dir=defaults.get("output_dir", "./goldenmatch_output"),
            output_format=defaults.get("output_format", "csv"),
            embedding_model=defaults.get("embedding_model", "auto"),
            auto_model_threshold=defaults.get("auto_model_threshold", 50000),
        )


def load_global_settings() -> UserSettings:
    """Load settings from ~/.goldenmatch/settings.yaml."""
    if not GLOBAL_SETTINGS_FILE.exists():
        return UserSettings()
    try:
        with open(GLOBAL_SETTINGS_FILE) as f:
            data = yaml.safe_load(f) or {}
        return UserSettings.from_dict(data)
    except Exception as e:
        logger.warning("Failed to load global settings: %s", e)
        return UserSettings()


def save_global_settings(settings: UserSettings) -> None:
    """Save settings to ~/.goldenmatch/settings.yaml."""
    GLOBAL_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_SETTINGS_FILE, "w") as f:
        yaml.dump(settings.to_dict(), f, default_flow_style=False, sort_keys=False)
    logger.info("Saved global settings to %s", GLOBAL_SETTINGS_FILE)


def load_project_settings(directory: Path | None = None) -> dict | None:
    """Load project settings from .goldenmatch.yaml in the given directory.

    Returns the raw dict (to be passed to load_config), or None if not found.
    """
    d = directory or Path.cwd()
    path = d / PROJECT_SETTINGS_FILE
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("Loaded project settings from %s", path)
        return data
    except Exception as e:
        logger.warning("Failed to load project settings: %s", e)
        return None


def save_project_settings(config_dict: dict, directory: Path | None = None) -> None:
    """Save config dict to .goldenmatch.yaml in the given directory."""
    d = directory or Path.cwd()
    path = d / PROJECT_SETTINGS_FILE
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved project settings to %s", path)


def load_settings() -> UserSettings:
    """Load merged settings: global, overlaid with project-level."""
    settings = load_global_settings()

    project = load_project_settings()
    if project and "_autoconfig" in project:
        meta = project["_autoconfig"]
        if "embedding_model" in meta:
            settings.embedding_model = meta["embedding_model"]

    return settings
