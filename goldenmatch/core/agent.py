"""Intelligence layer for autonomous entity resolution.

Profiles data, selects matching strategy, orchestrates pipelines,
and gates borderline pairs through the review queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl

from goldenmatch.core.review_queue import ReviewQueue, gate_pairs

logger = logging.getLogger(__name__)

# Column-name patterns that indicate sensitive PII
_SENSITIVE_PATTERNS = frozenset({
    "ssn", "social_security", "dob", "date_of_birth",
    "birth_date", "drivers_license", "dl_number",
})


# ── Data profiles ────────────────────────────────────────────────────────────


@dataclass
class FieldProfile:
    """Statistical profile of a single column."""

    name: str
    type: str  # "string", "numeric", "other"
    uniqueness: float  # n_unique / row_count, 0-1
    null_rate: float  # fraction null, 0-1
    avg_length: float  # mean string length (0 for non-string)


@dataclass
class DataProfile:
    """Aggregate profile of a DataFrame."""

    row_count: int
    fields: list[FieldProfile]
    has_sensitive: bool


@dataclass
class StrategyDecision:
    """Result of automatic strategy selection."""

    strategy: str
    why: str
    domain: Optional[str] = None
    strong_ids: list[str] = field(default_factory=list)
    fuzzy_fields: list[str] = field(default_factory=list)
    backend: Optional[str] = None
    auto_execute: bool = True


# ── Profiling ────────────────────────────────────────────────────────────────


def profile_for_agent(df: pl.DataFrame) -> DataProfile:
    """Profile a DataFrame for strategy selection.

    For each column computes uniqueness, null rate, and average string length.
    Detects sensitive fields by column name pattern matching.
    """
    height = df.height
    fields: list[FieldProfile] = []
    has_sensitive = False

    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        if col_lower in _SENSITIVE_PATTERNS:
            has_sensitive = True

        series = df[col]
        dtype = series.dtype

        # Determine type category
        if dtype in (pl.Utf8, pl.String, pl.Categorical):
            ftype = "string"
        elif dtype.is_numeric():
            ftype = "numeric"
        else:
            ftype = "other"

        # Uniqueness
        n_unique = series.n_unique()
        uniqueness = n_unique / height if height > 0 else 0.0

        # Null rate
        null_count = series.null_count()
        null_rate = null_count / height if height > 0 else 0.0

        # Average length (string columns only)
        if ftype == "string":
            lengths = series.cast(pl.Utf8).str.len_bytes()
            avg_length = lengths.drop_nulls().mean() or 0.0
        else:
            avg_length = 0.0

        fields.append(FieldProfile(
            name=col,
            type=ftype,
            uniqueness=uniqueness,
            null_rate=null_rate,
            avg_length=float(avg_length),
        ))

    return DataProfile(row_count=height, fields=fields, has_sensitive=has_sensitive)


# ── Strategy selection ───────────────────────────────────────────────────────


def select_strategy(profile: DataProfile) -> StrategyDecision:
    """Choose a matching strategy based on data profile.

    Decision tree:
    1. Sensitive fields detected -> PPRL (manual review required).
    2. Strong IDs only (high uniqueness, low nulls) -> exact_only.
    3. Strong IDs + fuzzy candidates -> exact_then_fuzzy.
    4. Fuzzy candidates available -> fuzzy.
    5. Domain detected with confidence -> domain_extraction.
    6. Fallback -> fuzzy.
    """
    # Sensitive data -> PPRL
    if profile.has_sensitive:
        return StrategyDecision(
            strategy="pprl",
            why="Sensitive fields detected; using privacy-preserving record linkage.",
            auto_execute=False,
        )

    # Detect domain
    domain_name: str | None = None
    domain_confidence: float = 0.0
    try:
        from goldenmatch.core.domain_registry import match_domain
        col_names = [f.name for f in profile.fields]
        rb = match_domain(col_names)
        if rb is not None:
            domain_name = rb.name
            # Confidence = fraction of signals that matched
            col_str = " ".join(c.lower() for c in col_names)
            hits = sum(1 for s in rb.signals if s.lower() in col_str)
            domain_confidence = hits / len(rb.signals) if rb.signals else 0.0
    except Exception:
        pass

    # Identify strong IDs and fuzzy candidates
    strong_ids: list[str] = []
    fuzzy_candidates: list[str] = []

    for f in profile.fields:
        if f.type == "string":
            if f.uniqueness > 0.90 and f.null_rate < 0.05:
                strong_ids.append(f.name)
            elif f.uniqueness < 0.90 and f.avg_length > 3 and f.null_rate < 0.50:
                fuzzy_candidates.append(f.name)

    # Backend recommendation
    backend = "ray" if profile.row_count > 500_000 else None

    # Decision tree
    if strong_ids and not fuzzy_candidates:
        return StrategyDecision(
            strategy="exact_only",
            why=f"High-uniqueness fields ({', '.join(strong_ids)}) with no fuzzy candidates.",
            domain=domain_name,
            strong_ids=strong_ids,
            fuzzy_fields=[],
            backend=backend,
        )

    if strong_ids and fuzzy_candidates:
        return StrategyDecision(
            strategy="exact_then_fuzzy",
            why=(
                f"Exact on {', '.join(strong_ids)}; "
                f"fuzzy on {', '.join(fuzzy_candidates)}."
            ),
            domain=domain_name,
            strong_ids=strong_ids,
            fuzzy_fields=fuzzy_candidates,
            backend=backend,
        )

    if fuzzy_candidates:
        return StrategyDecision(
            strategy="fuzzy",
            why=f"Fuzzy matching on {', '.join(fuzzy_candidates)}.",
            domain=domain_name,
            strong_ids=[],
            fuzzy_fields=fuzzy_candidates,
            backend=backend,
        )

    if domain_name and domain_confidence > 0.5:
        return StrategyDecision(
            strategy="domain_extraction",
            why=f"Domain '{domain_name}' detected (confidence {domain_confidence:.0%}).",
            domain=domain_name,
            strong_ids=[],
            fuzzy_fields=[],
            backend=backend,
        )

    # Fallback
    return StrategyDecision(
        strategy="fuzzy",
        why="No strong identifiers found; defaulting to fuzzy matching.",
        domain=domain_name,
        strong_ids=[],
        fuzzy_fields=[f.name for f in profile.fields if f.type == "string"],
        backend=backend,
    )


# ── Alternatives ─────────────────────────────────────────────────────────────


def build_alternatives(
    decision: StrategyDecision,
    profile: DataProfile,
) -> list[dict[str, str]]:
    """Generate alternative strategies the user might consider."""
    alts: list[dict[str, str]] = []

    if decision.strategy != "pprl":
        alts.append({
            "strategy": "pprl",
            "why_not": "No sensitive fields detected, but PPRL is available if data leaves your network.",
        })

    if decision.strategy != "fellegi_sunter":
        alts.append({
            "strategy": "fellegi_sunter",
            "why_not": "Probabilistic model available for automatic parameter estimation.",
        })

    return alts


# ── Config builder ───────────────────────────────────────────────────────────


def _decision_to_config(decision: StrategyDecision) -> Any:
    """Translate a StrategyDecision into a GoldenMatchConfig."""
    from goldenmatch.config.schemas import (
        GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
        BlockingConfig, BlockingKeyConfig,
    )

    matchkeys: list[MatchkeyConfig] = []

    # Exact matchkeys from strong IDs
    for col in decision.strong_ids:
        matchkeys.append(MatchkeyConfig(
            name=f"exact_{col}",
            type="exact",
            fields=[MatchkeyField(field=col, transforms=["lowercase", "strip"])],
        ))

    # Fuzzy matchkey from fuzzy fields
    if decision.fuzzy_fields:
        fields = [
            MatchkeyField(
                field=col,
                scorer="jaro_winkler",
                weight=1.0,
                transforms=["lowercase", "strip"],
            )
            for col in decision.fuzzy_fields
        ]
        matchkeys.append(MatchkeyConfig(
            name="fuzzy",
            type="weighted",
            threshold=0.85,
            fields=fields,
        ))

    # Fallback placeholder
    if not matchkeys:
        matchkeys.append(MatchkeyConfig(
            name="auto",
            type="exact",
            fields=[MatchkeyField(field="__placeholder__")],
        ))

    # Blocking from first fuzzy field
    blocking_config = None
    if decision.fuzzy_fields:
        blocking_config = BlockingConfig(
            keys=[BlockingKeyConfig(
                fields=[decision.fuzzy_fields[0]],
                transforms=["lowercase", "first_token"],
            )],
        )

    return GoldenMatchConfig(
        matchkeys=matchkeys,
        blocking=blocking_config,
        backend=decision.backend,
    )


# ── Agent session ────────────────────────────────────────────────────────────


class AgentSession:
    """Stateful session for autonomous entity resolution.

    Holds data, config, results, and a review queue.
    """

    def __init__(self) -> None:
        self.data: pl.DataFrame | None = None
        self.config: Any = None
        self.result: Any = None
        self.review_queue = ReviewQueue(backend="memory")
        self.reasoning: dict[str, Any] = {}

    # ── analyze ──────────────────────────────────────────────────────────

    def analyze(self, file_path: str) -> dict[str, Any]:
        """Load a CSV and return profiling + strategy analysis."""
        df = pl.read_csv(file_path, encoding="utf8", ignore_errors=True)
        self.data = df

        profile = profile_for_agent(df)
        decision = select_strategy(profile)
        alternatives = build_alternatives(decision, profile)

        self.reasoning = {
            "profile": {
                "row_count": profile.row_count,
                "fields": [
                    {
                        "name": f.name,
                        "type": f.type,
                        "uniqueness": round(f.uniqueness, 4),
                        "null_rate": round(f.null_rate, 4),
                        "avg_length": round(f.avg_length, 1),
                    }
                    for f in profile.fields
                ],
                "has_sensitive": profile.has_sensitive,
            },
            "strategy": decision.strategy,
            "why": decision.why,
            "domain": decision.domain,
            "strong_ids": decision.strong_ids,
            "fuzzy_fields": decision.fuzzy_fields,
            "backend": decision.backend,
            "auto_execute": decision.auto_execute,
            "alternatives": alternatives,
        }

        return self.reasoning

    # ── deduplicate ──────────────────────────────────────────────────────

    def deduplicate(
        self,
        file_path: str,
        config: Any = None,
    ) -> dict[str, Any]:
        """Full deduplication with profiling, strategy, gating, and review queue."""
        from goldenmatch._api import dedupe_df

        df = pl.read_csv(file_path, encoding="utf8", ignore_errors=True)
        self.data = df

        profile = profile_for_agent(df)
        decision = select_strategy(profile)

        if config is not None:
            cfg = config
        else:
            cfg = _decision_to_config(decision)
        self.config = cfg

        result = dedupe_df(df, config=cfg)
        self.result = result

        # Gate scored pairs through review queue
        scored_pairs = result.scored_pairs or []
        auto_merged, review, auto_rejected = gate_pairs(scored_pairs)

        # Populate review queue
        job_name = f"dedupe_{file_path}"
        for id_a, id_b, score in review:
            self.review_queue.add(
                job_name=job_name,
                id_a=id_a,
                id_b=id_b,
                score=score,
                explanation=f"Score {score:.3f} needs human review.",
            )

        # Confidence distribution
        scores = [s for _, _, s in scored_pairs]
        confidence_distribution = {
            "auto_merged": len(auto_merged),
            "review": len(review),
            "auto_rejected": len(auto_rejected),
            "total_pairs": len(scored_pairs),
        }

        self.reasoning = {
            "strategy": decision.strategy,
            "why": decision.why,
            "domain": decision.domain,
            "strong_ids": decision.strong_ids,
            "fuzzy_fields": decision.fuzzy_fields,
        }

        return {
            "results": result,
            "reasoning": self.reasoning,
            "confidence_distribution": confidence_distribution,
            "storage": self.review_queue.storage_tier,
        }

    # ── match_sources ────────────────────────────────────────────────────

    def match_sources(
        self,
        file_a: str,
        file_b: str,
        config: Any = None,
    ) -> dict[str, Any]:
        """Match two CSV sources with profiling, strategy, and gating."""
        from goldenmatch._api import match_df

        df_a = pl.read_csv(file_a, encoding="utf8", ignore_errors=True)
        df_b = pl.read_csv(file_b, encoding="utf8", ignore_errors=True)

        # Profile the target (first file)
        profile = profile_for_agent(df_a)
        decision = select_strategy(profile)

        if config is not None:
            cfg = config
        else:
            cfg = _decision_to_config(decision)
        self.config = cfg

        result = match_df(df_a, df_b, config=cfg)
        self.result = result

        self.reasoning = {
            "strategy": decision.strategy,
            "why": decision.why,
            "domain": decision.domain,
            "strong_ids": decision.strong_ids,
            "fuzzy_fields": decision.fuzzy_fields,
        }

        return {
            "results": result,
            "reasoning": self.reasoning,
        }

    # ── compare_strategies ───────────────────────────────────────────────

    def compare_strategies(
        self,
        file_path: str,
        ground_truth: str | None = None,
    ) -> dict[str, Any]:
        """Run multiple strategies on the same data and compare proxy metrics.

        If ground_truth is provided, evaluates against it.
        Otherwise, returns cluster-count / match-rate proxies.
        """
        from goldenmatch._api import dedupe_df

        df = pl.read_csv(file_path, encoding="utf8", ignore_errors=True)
        self.data = df

        profile = profile_for_agent(df)
        decision = select_strategy(profile)

        # Strategies to try
        strategies_to_run: list[StrategyDecision] = [decision]

        # Add an exact-only variant if there are strong IDs
        if decision.strong_ids and decision.strategy != "exact_only":
            strategies_to_run.append(StrategyDecision(
                strategy="exact_only",
                why="Comparison: exact matching only.",
                strong_ids=decision.strong_ids,
                fuzzy_fields=[],
            ))

        # Add a fuzzy-only variant if there are fuzzy fields
        if decision.fuzzy_fields and decision.strategy != "fuzzy":
            strategies_to_run.append(StrategyDecision(
                strategy="fuzzy",
                why="Comparison: fuzzy matching only.",
                strong_ids=[],
                fuzzy_fields=decision.fuzzy_fields,
            ))

        results_per_strategy: dict[str, dict] = {}

        for strat in strategies_to_run:
            cfg = _decision_to_config(strat)
            try:
                res = dedupe_df(df, config=cfg)
                clusters = res.clusters or {}
                multi_clusters = sum(1 for c in clusters.values() if c.get("size", 0) > 1)
                total_matched = sum(c.get("size", 0) for c in clusters.values() if c.get("size", 0) > 1)
                match_rate = total_matched / df.height if df.height > 0 else 0.0

                metrics: dict[str, Any] = {
                    "clusters": multi_clusters,
                    "match_rate": round(match_rate, 4),
                    "total_pairs": len(res.scored_pairs),
                }

                # Evaluate against ground truth if provided
                if ground_truth is not None:
                    try:
                        from goldenmatch.core.evaluate import evaluate_clusters, load_ground_truth_csv
                        gt = load_ground_truth_csv(ground_truth)
                        eval_result = evaluate_clusters(clusters, gt)
                        metrics["precision"] = round(eval_result.precision, 4)
                        metrics["recall"] = round(eval_result.recall, 4)
                        metrics["f1"] = round(eval_result.f1, 4)
                    except Exception as exc:
                        metrics["eval_error"] = str(exc)

                results_per_strategy[strat.strategy] = metrics
            except Exception as exc:
                results_per_strategy[strat.strategy] = {"error": str(exc)}

        return {
            "recommended": decision.strategy,
            "strategies": results_per_strategy,
        }
