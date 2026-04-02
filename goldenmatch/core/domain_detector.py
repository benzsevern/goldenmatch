"""Domain detection for auto-config — scores column profiles against registered domain packs."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from goldenmatch.core.autoconfig import ColumnProfile
from goldenmatch.core.domain_registry import discover_rulebooks, DomainRulebook

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.2


@dataclass
class DomainDetectionResult:
    """Result of domain detection."""
    domain: str
    confidence: float
    rulebook: DomainRulebook | None = None
    preset: dict | None = None


def detect_domain(
    profiles: list[ColumnProfile],
    rulebooks: dict[str, DomainRulebook] | None = None,
) -> DomainDetectionResult:
    """Detect the data domain from column profiles.

    Scores each registered domain by signal overlap with column names.
    Uses word-boundary matching to avoid false positives (e.g., "mp" in "company").
    Returns the best match or "generic" fallback.
    """
    if rulebooks is None:
        try:
            rulebooks = discover_rulebooks()
        except Exception as exc:
            logger.warning("Domain detection: failed to discover rulebooks: %s", exc)
            return DomainDetectionResult(domain="generic", confidence=0.0)

    if not rulebooks:
        return DomainDetectionResult(domain="generic", confidence=0.0)

    # Build a set of individual column name tokens for word-boundary matching.
    # Split on underscores and non-alphanumeric chars so "first_name" yields {"first", "name"}.
    col_tokens: set[str] = set()
    for p in profiles:
        col_tokens.update(re.split(r"[^a-zA-Z0-9]+", p.name.lower()))
    col_tokens.discard("")

    best_rb: DomainRulebook | None = None
    best_score = 0.0

    for rb in rulebooks.values():
        if not rb.signals:
            continue
        hits = sum(1 for s in rb.signals if s.lower() in col_tokens)
        score = hits / len(rb.signals)
        if score > best_score:
            best_score = score
            best_rb = rb

    if best_score < _CONFIDENCE_THRESHOLD or best_rb is None:
        logger.info("No domain detected (best score %.2f < %.2f)", best_score, _CONFIDENCE_THRESHOLD)
        return DomainDetectionResult(domain="generic", confidence=best_score)

    preset = getattr(best_rb, "autoconfig_preset", None)
    logger.info("Detected domain '%s' (confidence %.2f)", best_rb.name, best_score)
    return DomainDetectionResult(
        domain=best_rb.name,
        confidence=best_score,
        rulebook=best_rb,
        preset=preset,
    )
