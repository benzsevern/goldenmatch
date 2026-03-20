"""Field transform utilities for GoldenMatch."""

from __future__ import annotations

import hashlib
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
    elif transform == "token_sort":
        tokens = value.strip().split()
        return " ".join(sorted(tokens))
    elif transform.startswith("qgram:"):
        q = int(transform.split(":")[1])
        padded = f"##{value}##"
        grams = sorted(set(padded[i:i + q] for i in range(len(padded) - q + 1)))
        return " ".join(grams[:5])
    elif transform == "first_token":
        tokens = value.strip().split()
        return tokens[0] if tokens else value
    elif transform == "last_token":
        tokens = value.strip().split()
        return tokens[-1] if tokens else value
    elif transform == "bloom_filter" or transform.startswith("bloom_filter:"):
        return _bloom_filter_transform(value, transform)
    else:
        raise ValueError(f"Unknown transform: {transform!r}")


def _bloom_filter_transform(value: str, transform: str) -> str:
    """Convert a string to a CLK (Cryptographic Longterm Key) as hex string.

    Generates character-level n-grams, hashes each with k hash functions
    into a fixed-size bit array, returns the bit array as a hex string.

    Parameterized: bloom_filter or bloom_filter:ngram:k:size
    """
    # Parse parameters
    if transform == "bloom_filter":
        ngram_size, num_hashes, filter_size = 2, 20, 1024
    else:
        parts = transform.split(":")
        ngram_size = int(parts[1])
        num_hashes = int(parts[2])
        filter_size = int(parts[3])

    filter_bytes = filter_size // 8
    bits = bytearray(filter_bytes)

    # Generate character n-grams
    padded = value.lower().strip()
    if len(padded) < ngram_size:
        padded = padded.ljust(ngram_size, "_")
    ngrams = [padded[i:i + ngram_size] for i in range(len(padded) - ngram_size + 1)]

    # Hash each n-gram with k different hash functions
    for ngram in ngrams:
        for k in range(num_hashes):
            h = hashlib.sha256(f"{k}:{ngram}".encode()).hexdigest()
            bit_pos = int(h, 16) % filter_size
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            bits[byte_idx] |= (1 << bit_idx)

    return bits.hex()


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
