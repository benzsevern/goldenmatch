"""YAML config loader for GoldenMatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from goldenmatch.config.schemas import GoldenMatchConfig

# Keys in golden_rules that are part of the schema (not field names)
_GOLDEN_RULES_SPECIAL_KEYS = frozenset({"default_strategy", "field_rules"})


def _normalize_golden_rules(raw: dict[str, Any]) -> dict[str, Any]:
    """Move non-special keys in golden_rules into the field_rules dict."""
    golden = raw.get("golden_rules")
    if golden is None or not isinstance(golden, dict):
        return raw

    field_rules: dict[str, Any] = golden.pop("field_rules", {})
    extra_keys = [k for k in golden if k not in _GOLDEN_RULES_SPECIAL_KEYS]
    for key in extra_keys:
        field_rules[key] = golden.pop(key)

    if field_rules:
        golden["field_rules"] = field_rules

    return raw


def _normalize_standardization(raw: dict[str, Any]) -> dict[str, Any]:
    """Allow flat standardization format without explicit 'rules' key.

    Users can write either:
        standardization:
          rules:
            email: [email]
    Or the shorthand:
        standardization:
          email: [email]
    """
    std = raw.get("standardization")
    if std is None or not isinstance(std, dict):
        return raw
    if "rules" not in std:
        # Everything is a column->standardizers mapping
        raw["standardization"] = {"rules": std}
    return raw


def load_config(path: str | Path) -> GoldenMatchConfig:
    """Load and validate a GoldenMatch YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated GoldenMatchConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    raw = _normalize_golden_rules(raw)
    raw = _normalize_standardization(raw)

    try:
        return GoldenMatchConfig(**raw)
    except Exception as exc:
        raise ValueError(f"Config validation failed: {exc}") from exc
