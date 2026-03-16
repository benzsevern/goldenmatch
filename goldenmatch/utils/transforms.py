"""Field transform utilities for GoldenMatch."""

from __future__ import annotations

import re

import jellyfish


def apply_transform(value: str | None, transform: str) -> str | None:
    """Apply a single named transform to a string value.

    Args:
        value: The input string, or None.
        transform: The transform name (e.g. "lowercase", "soundex", "substring:0:3").

    Returns:
        The transformed string, or None if value is None.

    Raises:
        ValueError: If the transform name is not recognised.
    """
    if value is None:
        return None

    if transform == "lowercase":
        return value.lower()
    elif transform == "uppercase":
        return value.upper()
    elif transform == "strip":
        return value.strip()
    elif transform == "strip_all":
        return re.sub(r"\s+", "", value)
    elif transform.startswith("substring:"):
        parts = transform.split(":")
        start = int(parts[1])
        end = int(parts[2])
        return value[start:end]
    elif transform == "soundex":
        return jellyfish.soundex(value)
    elif transform == "metaphone":
        return jellyfish.metaphone(value)
    elif transform == "digits_only":
        return re.sub(r"[^0-9]", "", value)
    elif transform == "alpha_only":
        return re.sub(r"[^a-zA-Z]", "", value)
    elif transform == "normalize_whitespace":
        return re.sub(r"\s+", " ", value).strip()
    else:
        raise ValueError(f"Unknown transform: {transform!r}")


def apply_transforms(value: str | None, transforms: list[str]) -> str | None:
    """Apply a chain of transforms to a string value.

    Args:
        value: The input string, or None.
        transforms: List of transform names to apply in order.

    Returns:
        The transformed string, or None if value is None.
    """
    if value is None:
        return None

    for t in transforms:
        value = apply_transform(value, t)
    return value
