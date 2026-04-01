"""Domain detection for auto-config — scores column profiles against registered domain packs."""
from __future__ import annotations

import logging
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
    Returns the best match or "generic" fallback.
    """
    if rulebooks is None:
        rulebooks = discover_rulebooks()

    if not rulebooks:
        return DomainDetectionResult(domain="generic", confidence=0.0)

    col_str = " ".join(p.name.lower() for p in profiles)

    best_rb: DomainRulebook | None = None
    best_score = 0.0

    for rb in rulebooks.values():
        if not rb.signals:
            continue
        hits = sum(1 for s in rb.signals if s.lower() in col_str)
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
