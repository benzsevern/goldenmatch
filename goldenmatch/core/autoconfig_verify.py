"""Auto-configuration verification: preflight + postflight checks.

See PR #44 design notes for the broader context.

Preflight runs right before `auto_configure_df` returns. It validates that the
generated `GoldenMatchConfig` is internally consistent with the DataFrame it
was derived from, auto-repairs issues where it can, and raises
`ConfigValidationError` on unrepairable errors.

Every check produces a `PreflightFinding`. A `PreflightReport` aggregates them
and is attached to the returned config as ``config._preflight_report`` for
downstream introspection (Postflight, diagnostics, tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    import polars as pl

    from goldenmatch.config.schemas import GoldenMatchConfig
    from goldenmatch.core.autoconfig import ColumnProfile


# Closed enumeration of preflight check names. A Literal (rather than free-form
# str) gives static protection against typos in producer/consumer code and
# makes the set of checks grep-friendly.
PreflightCheckName = Literal[
    "missing_column",
    "cardinality_high",
    "cardinality_low",
    "block_size",
    "remote_asset",
    "weight_confidence",
    "no_matchkeys_remain",
    "remote_asset_matchkey_empty",
]


@dataclass
class PreflightFinding:
    """One check result — informational, warning, or hard error."""

    check: PreflightCheckName
    severity: Literal["error", "warning", "info"]
    subject: str
    message: str
    repaired: bool
    repair_note: str | None


@dataclass
class PreflightReport:
    """Aggregated result of running all preflight checks."""

    findings: list[PreflightFinding] = field(default_factory=list)
    config_was_modified: bool = False

    @property
    def has_errors(self) -> bool:
        """True if any unrepaired error-severity finding exists."""
        return any(
            f.severity == "error" and not f.repaired for f in self.findings
        )


# ── Postflight signals schema (stable public contract) ──
#
# The shape below is the stable schema of ``PostflightReport.signals``. Using
# TypedDict rather than ``dict[str, Any]`` makes the contract legible and
# mechanically enforceable, without changing runtime behavior.


class ScoreHistogram(TypedDict):
    bins: list[float]
    counts: list[int]


class BlockSizePercentiles(TypedDict):
    p50: int
    p95: int
    p99: int
    max: int


class ClusterSizePercentiles(TypedDict):
    p50: int
    p95: int
    p99: int
    max: int
    count: int


class OversizedCluster(TypedDict, total=False):
    cluster_id: int
    size: int
    bottleneck_pair: list[int]  # [int, int]


class PostflightSignals(TypedDict):
    score_histogram: ScoreHistogram
    blocking_recall: float | Literal["deferred"]
    block_size_percentiles: BlockSizePercentiles
    threshold_overlap_pct: float
    total_pairs_scored: int
    current_threshold: float
    preliminary_cluster_sizes: ClusterSizePercentiles
    oversized_clusters: list[OversizedCluster]


@dataclass
class PostflightAdjustment:
    """A single auto-applied adjustment produced by postflight.

    Each adjustment is keyed by the ``signal`` that motivated it so callers
    can trace which signal drove a change.
    """

    field: str
    from_value: Any
    to_value: Any
    reason: str
    signal: str


def _empty_signals() -> "PostflightSignals":
    """Factory for PostflightReport.signals.

    Returns an empty dict typed as PostflightSignals; ``postflight()``
    populates every required key before the report is returned. Tests may
    also build a ``PostflightReport`` with partial signals.
    """
    return {}  # type: ignore[typeddict-item]


@dataclass
class PostflightReport:
    """Aggregated result of running all postflight signals.

    ``signals`` is the stable schema documented in ``PostflightSignals``.
    ``adjustments`` is the list of auto-applied config tweaks (suppressed in
    strict mode). ``advisories`` are human-readable hints (e.g. "consider
    --llm-auto").
    """

    signals: "PostflightSignals" = field(default_factory=_empty_signals)
    adjustments: list[PostflightAdjustment] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)


class ConfigValidationError(Exception):
    """Raised when preflight finds unrepairable configuration errors.

    The full `PreflightReport` is attached as ``self.report`` for callers that
    want to inspect findings programmatically.
    """

    def __init__(self, *, report: PreflightReport) -> None:
        self.report = report
        unrepaired = [
            f for f in report.findings
            if f.severity == "error" and not f.repaired
        ]
        msg_parts = [f"{f.check}: {f.message}" for f in unrepaired]
        super().__init__(
            f"auto-config produced {len(unrepaired)} unrepairable error(s): "
            + "; ".join(msg_parts)
        )


# ── Column collection ────────────────────────────────────────────────────


def _collect_referenced_columns(config: "GoldenMatchConfig") -> set[str]:
    """Walk the config and return every raw column name it references.

    Collects from blocking (keys + passes) and all matchkey fields.
    """
    cols: set[str] = set()
    if config.blocking is not None:
        for key in config.blocking.keys or []:
            cols.update(key.fields)
        for key in (config.blocking.passes or []):
            cols.update(key.fields)
    for mk in config.get_matchkeys():
        for f in mk.fields:
            if f.field is not None and f.field != "__record__":
                cols.add(f.field)
            if f.column is not None:
                cols.add(f.column)
            if f.columns:
                cols.update(f.columns)
    return cols


def _check_columns(
    df: "pl.DataFrame", config: "GoldenMatchConfig", report: PreflightReport
) -> None:
    """Check 1: every referenced column exists, or is pipeline-synthesized,
    or is a domain-extracted column recoverable via domain repair.
    """
    from goldenmatch.core.domain import _DOMAIN_EXTRACTED_COLS

    df_cols = set(df.columns)
    referenced = _collect_referenced_columns(config)
    domain_profile = getattr(config, "_domain_profile", None)

    for col in sorted(referenced):
        if col in df_cols:
            continue
        # Pipeline-synthesized matchkey columns — safe, created at runtime.
        if col.startswith("__mk_"):
            continue
        # Domain-extracted column: auto-repair by enabling DomainConfig if
        # a domain profile was stashed by auto_configure_df.
        if col in _DOMAIN_EXTRACTED_COLS and domain_profile is not None:
            _repair_domain(config, domain_profile, report, subject=col)
            continue
        # Unrepairable: raw column does not exist in the DataFrame.
        report.findings.append(
            PreflightFinding(
                check="missing_column",
                severity="error",
                subject=col,
                message=(
                    f"column '{col}' referenced by config but not present in "
                    f"DataFrame (columns: {sorted(df_cols)[:10]}...)"
                ),
                repaired=False,
                repair_note=None,
            )
        )


def _repair_domain(
    config: "GoldenMatchConfig",
    domain_profile: object,
    report: PreflightReport,
    *,
    subject: str,
) -> None:
    """Enable DomainConfig so the pipeline produces the extracted columns."""
    from goldenmatch.config.schemas import DomainConfig

    already_enabled = (
        config.domain is not None and config.domain.enabled is True
    )
    if not already_enabled:
        config.domain = DomainConfig(
            enabled=True, mode=getattr(domain_profile, "name", None)
        )
        report.config_was_modified = True

    report.findings.append(
        PreflightFinding(
            check="missing_column",
            severity="error",
            subject=subject,
            message=(
                f"column '{subject}' is produced by domain extraction; "
                f"enabled config.domain (mode={getattr(domain_profile, 'name', None)})"
            ),
            repaired=True,
            repair_note=(
                "config.domain enabled so the pipeline's domain-extraction "
                "step produces this column at runtime"
            ),
        )
    )


# ── Entry point ──────────────────────────────────────────────────────────


def _check_cardinality(
    df: "pl.DataFrame", config: "GoldenMatchConfig", report: PreflightReport
) -> None:
    """Checks 2 & 3: drop exact matchkeys whose column cardinality is useless.

    - ratio >= 0.99 → near-unique, no pair ever agrees (warning + drop).
    - ratio <  0.01 → always-same, every pair agrees trivially (warning + drop).
    """
    if df.height == 0:
        return

    mks = config.get_matchkeys()
    kept: list = []
    df_cols = set(df.columns)

    for mk in mks:
        if mk.type != "exact":
            kept.append(mk)
            continue

        # Exact matchkey: inspect the fields' cardinality. A multi-field exact
        # matchkey is dropped only if *every* field is out of bounds; otherwise
        # it probably still discriminates.
        high_hits: list[str] = []
        low_hits: list[str] = []
        checked = 0
        for f in mk.fields:
            col = f.field
            if not col or col not in df_cols:
                continue
            checked += 1
            n_unique = df[col].n_unique()
            ratio = n_unique / df.height
            if ratio >= 0.99:
                high_hits.append(col)
            elif ratio <= 0.01:
                low_hits.append(col)

        drop = False
        reason_check: str | None = None
        reason_msg: str | None = None
        if checked > 0 and len(high_hits) == checked:
            drop = True
            reason_check = "cardinality_high"
            reason_msg = (
                f"exact matchkey '{mk.name}' dropped: column(s) {high_hits} "
                f"have cardinality_ratio >= 0.99 (near-unique, never agree)"
            )
        elif checked > 0 and len(low_hits) == checked:
            drop = True
            reason_check = "cardinality_low"
            reason_msg = (
                f"exact matchkey '{mk.name}' dropped: column(s) {low_hits} "
                f"have cardinality_ratio < 0.01 (always-same, trivially agree)"
            )

        if drop:
            report.findings.append(
                PreflightFinding(
                    check=reason_check or "cardinality",
                    severity="warning",
                    subject=mk.name,
                    message=reason_msg or "",
                    repaired=True,
                    repair_note="matchkey dropped from config",
                )
            )
            report.config_was_modified = True
        else:
            kept.append(mk)

    # Write back if we actually dropped anything.
    if len(kept) != len(mks):
        if config.matchkeys is not None:
            config.matchkeys = kept
        elif config.match_settings is not None:
            config.match_settings.matchkeys = kept

    # Hard-error if no matchkeys remain.
    if not config.get_matchkeys():
        report.findings.append(
            PreflightFinding(
                check="no_matchkeys_remain",
                severity="error",
                subject="<config>",
                message=(
                    "no matchkeys remain after preflight cardinality repair; "
                    "auto-config cannot produce a usable config for this data"
                ),
                repaired=False,
                repair_note=None,
            )
        )


def _check_block_sizes(
    df: "pl.DataFrame", config: "GoldenMatchConfig", report: PreflightReport
) -> None:
    """Check 4: per-key block-size sanity (P50/P99 distribution).

    Warns — does not auto-repair. Blocking strategy choice is usually
    intentional; preflight's job is to surface the trade-off, not override it.
    """
    if config.blocking is None or df.height == 0:
        return

    import polars as pl
    from goldenmatch.core.blocker import _build_block_key_expr

    # Cap sample at 10K rows for speed. Sample via ``df.head(n)`` (not random)
    # for determinism — for pre-sorted inputs the distribution may over-
    # represent the head's block skew; acceptable as a first-line indicator.
    n = min(df.height, 10_000)
    sample = df.head(n) if df.height > n else df

    keys = list(config.blocking.keys or [])
    keys.extend(config.blocking.passes or [])

    for key in keys:
        # Skip if the blocking key references a column the df doesn't have;
        # that's Check 1's problem.
        if not all(f in df.columns for f in key.fields):
            continue
        try:
            expr = _build_block_key_expr(key)
            sizes = (
                sample.with_columns(expr)
                .group_by("__block_key__")
                .len()
                .get_column("len")
            )
        except Exception as exc:
            # Transform failure — don't crash preflight; Check 1 or runtime
            # will surface it.
            report.findings.append(
                PreflightFinding(
                    check="block_size",
                    severity="info",
                    subject=",".join(key.fields),
                    message=f"could not sample block sizes: {exc!r}",
                    repaired=False,
                    repair_note=None,
                )
            )
            continue

        if sizes.len() == 0:
            continue

        p50 = float(sizes.quantile(0.5) or 0)
        p99 = float(sizes.quantile(0.99) or 0)

        if p99 > 5000 or p50 < 2:
            verdict = []
            if p99 > 5000:
                verdict.append(f"P99={p99:.0f} > 5000 (mega-blocks will dominate runtime)")
            if p50 < 2:
                verdict.append(f"P50={p50:.0f} < 2 (most blocks too small to produce pairs)")
            report.findings.append(
                PreflightFinding(
                    check="block_size",
                    severity="warning",
                    subject=",".join(key.fields),
                    message=(
                        f"blocking key {key.fields}: "
                        f"P50={p50:.0f}, P99={p99:.0f}, "
                        f"n_blocks={sizes.len()} (sampled {n} rows). "
                        + "; ".join(verdict)
                    ),
                    repaired=False,
                    repair_note=None,
                )
            )


_REMOTE_SCORERS = frozenset({"embedding", "record_embedding"})


def _check_remote_assets(
    config: "GoldenMatchConfig",
    report: PreflightReport,
    *,
    allow_remote_assets: bool,
) -> None:
    """Check 5: demote or drop matchkey fields that would load remote assets
    (embedding models, cross-encoders) unless explicitly opted in.

    Behaviors:
      - ``scorer='embedding'`` → demoted to ``'ensemble'`` (offline-safe).
      - ``scorer='record_embedding'`` → the field is REMOVED (not demoted —
        it relies on the synthetic ``__record__`` placeholder column with
        ``columns=[...]`` and has no ensemble-compatible fallback).
      - ``rerank=True`` → disabled (cross-encoder download).
      - If a matchkey's only fields were ``record_embedding`` and they all
        get removed, the matchkey itself is dropped (separate
        ``remote_asset_matchkey_empty`` finding).

    Skipped entirely when ``allow_remote_assets`` is True, or when an LLM
    scorer is already enabled (implies online operation is fine).
    """
    if allow_remote_assets:
        return

    llm_enabled = (
        config.llm_scorer is not None and config.llm_scorer.enabled is True
    )
    if llm_enabled:
        return

    mks = config.get_matchkeys()
    to_drop: list = []
    for mk in mks:
        uses_remote = any(f.scorer in _REMOTE_SCORERS for f in mk.fields)
        uses_rerank = bool(mk.rerank)
        if not (uses_remote or uses_rerank):
            continue

        # record_embedding uses the synthetic '__record__' placeholder column —
        # no real data behind it. Demoting to a per-field scorer like 'ensemble'
        # would leave the pipeline looking for a column that doesn't exist. Drop
        # those fields entirely; the remaining per-field fuzzy scorers in the
        # matchkey already cover offline scoring.
        kept_fields = []
        for f in mk.fields:
            if f.scorer == "record_embedding":
                report.findings.append(
                    PreflightFinding(
                        check="remote_asset",
                        severity="warning",
                        subject=f"{mk.name}.{f.field}",
                        message=(
                            "scorer 'record_embedding' dropped (requires model "
                            "download and the synthetic '__record__' column). "
                            "Pass allow_remote_assets=True to opt in."
                        ),
                        repaired=True,
                        repair_note="record_embedding field dropped",
                    )
                )
                report.config_was_modified = True
                continue
            if f.scorer == "embedding":
                original = f.scorer
                f.scorer = "ensemble"
                report.findings.append(
                    PreflightFinding(
                        check="remote_asset",
                        severity="warning",
                        subject=f"{mk.name}.{f.field}",
                        message=(
                            f"scorer '{original}' requires downloading a model; "
                            f"demoted to 'ensemble' (offline-safe). "
                            f"Pass allow_remote_assets=True to opt in."
                        ),
                        repaired=True,
                        repair_note=f"scorer: {original} → ensemble",
                    )
                )
                report.config_was_modified = True
            kept_fields.append(f)
        mk.fields = kept_fields

        if uses_rerank:
            mk.rerank = False
            report.findings.append(
                PreflightFinding(
                    check="remote_asset",
                    severity="warning",
                    subject=mk.name,
                    message=(
                        f"matchkey '{mk.name}' had rerank=True (cross-encoder "
                        f"model download); disabled. "
                        f"Pass allow_remote_assets=True to opt in."
                    ),
                    repaired=True,
                    repair_note="rerank: True → False",
                )
            )
            report.config_was_modified = True

        # If all fields got dropped (e.g. a weighted matchkey whose only field
        # was record_embedding), remove the zombie matchkey entirely.
        if not mk.fields:
            to_drop.append(mk)
            report.findings.append(
                PreflightFinding(
                    check="remote_asset_matchkey_empty",
                    severity="info",
                    subject=mk.name,
                    message=(
                        f"matchkey '{mk.name}' has no fields left after "
                        f"record_embedding removal; dropped from config"
                    ),
                    repaired=True,
                    repair_note=(
                        "matchkey left with zero fields after "
                        "record_embedding removal — dropped"
                    ),
                )
            )
            report.config_was_modified = True

    if to_drop:
        kept_mks = [mk for mk in mks if mk not in to_drop]
        if config.matchkeys is not None:
            config.matchkeys = kept_mks
        elif config.match_settings is not None:
            config.match_settings.matchkeys = kept_mks


def _check_weight_confidence(
    config: "GoldenMatchConfig",
    profiles: "list[ColumnProfile]",
    report: PreflightReport,
) -> None:
    """Check 6: cap weight at 0.5 for fields whose column profile has
    confidence < 0.5. Prevents a column that auto-config is unsure about
    from dominating the weighted score.
    """
    conf_by_col = {p.name: p.confidence for p in profiles}

    for mk in config.get_matchkeys():
        if mk.type != "weighted":
            continue
        for f in mk.fields:
            if f.field is None or f.weight is None:
                continue
            conf = conf_by_col.get(f.field)
            if conf is None:
                continue
            if conf < 0.5 and f.weight > 0.5:
                original = f.weight
                f.weight = 0.5
                report.findings.append(
                    PreflightFinding(
                        check="weight_confidence",
                        severity="warning",
                        subject=f"{mk.name}.{f.field}",
                        message=(
                            f"field '{f.field}' has column profile confidence "
                            f"{conf:.2f} < 0.5; capped weight {original:.2f} → 0.50"
                        ),
                        repaired=True,
                        repair_note=f"weight: {original:.2f} → 0.50",
                    )
                )
                report.config_was_modified = True


# ── Postflight helpers ──────────────────────────────────────────────────


def _signal_score_histogram(
    pair_scores: "list[tuple[int, int, float]]",
    current_threshold: float,
) -> dict[str, Any]:
    """Build a 100-bin histogram of pair scores and detect bimodality.

    Returns a dict with:
    - ``histogram``: {"bins": list[float], "counts": list[int]}
    - ``valley_location``: midpoint of the deepest valley between two local
      maxima that are >10 bins apart, or None if unimodal.
    - ``valley_depth_ratio``: valley_count / max(peak_left, peak_right).
      Lower = deeper valley. Bimodality threshold is ratio < 0.5.

    A distribution is considered bimodal when both a valley and two peaks
    (separated by >10 bins) exist AND ``valley_depth_ratio < 0.5``.
    """
    n_bins = 100
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for _a, _b, s in pair_scores:
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        idx = int(s * n_bins)
        if idx == n_bins:
            idx = n_bins - 1
        counts[idx] += 1

    # Smooth with a 5-bin window to reduce sampling noise — bimodality is a
    # shape property, not a per-bin accident. Raw counts stay in the reported
    # histogram; smoothing is only used internally for peak/valley detection.
    def _smooth(vals: list[int]) -> list[float]:
        out: list[float] = []
        for i in range(len(vals)):
            lo = max(0, i - 2)
            hi = min(len(vals), i + 3)
            window = vals[lo:hi]
            out.append(sum(window) / len(window))
        return out

    smoothed = _smooth(counts)
    total = sum(counts)
    mean_bin = total / n_bins if n_bins > 0 else 0.0
    max_count = max(smoothed) if smoothed else 0.0
    # A real mode is well above the average bin-count; this filters out the
    # ~uniform-noise peaks that appear in 1000-sample random data.
    min_peak_height = max(max_count * 0.3, mean_bin * 2.0)

    peaks: list[int] = []
    for i in range(n_bins):
        left = smoothed[i - 1] if i > 0 else -1.0
        right = smoothed[i + 1] if i < n_bins - 1 else -1.0
        if smoothed[i] >= left and smoothed[i] >= right and smoothed[i] >= min_peak_height:
            if smoothed[i] > left or smoothed[i] > right:
                peaks.append(i)

    # Pick the two tallest peaks that are separated by >10 bins.
    best_pair: tuple[int, int] | None = None
    if len(peaks) >= 2:
        peaks_sorted = sorted(peaks, key=lambda i: smoothed[i], reverse=True)
        for i_idx, i in enumerate(peaks_sorted):
            for j in peaks_sorted[i_idx + 1:]:
                if abs(i - j) > 10:
                    best_pair = (min(i, j), max(i, j))
                    break
            if best_pair is not None:
                break

    valley_location: float | None = None
    valley_depth_ratio: float | None = None
    if best_pair is not None:
        left_peak, right_peak = best_pair
        valley_idx = left_peak + 1
        valley_count = smoothed[valley_idx]
        for k in range(left_peak + 1, right_peak):
            if smoothed[k] < valley_count:
                valley_count = smoothed[k]
                valley_idx = k
        peak_min = min(smoothed[left_peak], smoothed[right_peak])
        if peak_min > 0:
            # Ratio against the SHALLOWER peak — a true valley has to sag
            # well below both sides, not just below the taller one. This also
            # filters uniform noise where one "peak" is a chance spike.
            valley_depth_ratio = valley_count / peak_min
            valley_location = (valley_idx + 0.5) / n_bins

    return {
        "histogram": {"bins": bin_edges, "counts": counts},
        "valley_location": valley_location,
        "valley_depth_ratio": valley_depth_ratio,
    }


def _resolve_current_threshold(
    config: "GoldenMatchConfig", override: float | None
) -> float:
    """Return the threshold to evaluate postflight against.

    Order: explicit override > first weighted matchkey's threshold > 0.7.
    """
    if override is not None:
        return float(override)
    for mk in config.get_matchkeys():
        if mk.type == "weighted" and mk.threshold is not None:
            return float(mk.threshold)
    return 0.7


def _signal_blocking_recall(
    df: "pl.DataFrame",
    config: "GoldenMatchConfig",
    pair_scores: "list[tuple[int, int, float]]",
    current_threshold: float,
) -> float | Literal["deferred"]:
    """Estimate blocking recall by brute-forcing a sample.

    Returns the string sentinel ``"deferred"`` when recall cannot be measured
    (gated at df.height < 10_000, and deferred-always today until the
    iterative autoconfig loop lands). Consumers of
    ``PostflightReport.signals["blocking_recall"]`` should handle the value
    as either a float in [0, 1] OR the literal string ``"deferred"``.

    TODO(autoconfig-iterative): implement brute-force recall estimation over
    a uniform 1000-row sample when df.height >= 10_000. Reserved for the
    iterative autoconfig loop so postflight has a meaningful recall estimate.
    """
    if df.height < 10_000:
        return "deferred"
    _ = (config, pair_scores, current_threshold)  # silence unused
    return "deferred"


def _signal_cluster_sizes(
    pair_scores: "list[tuple[int, int, float]]",
    current_threshold: float,
) -> dict[str, Any]:
    """Union-find over above-threshold pairs; report size percentiles and
    identify oversized clusters (size > 100) with their bottleneck pair.
    """
    filtered = [(a, b, s) for a, b, s in pair_scores if s >= current_threshold]
    if not filtered:
        return {
            "preliminary_cluster_sizes": {
                "p50": 0, "p95": 0, "p99": 0, "max": 0, "count": 0,
            },
            "oversized_clusters": [],
        }

    # Union-find
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b, _ in filtered:
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    # Components: root -> members
    components: dict[int, list[int]] = {}
    for node in parent:
        r = find(node)
        components.setdefault(r, []).append(node)

    sizes = sorted(len(m) for m in components.values())

    def _percentile(sorted_vals: list[int], q: float) -> int:
        if not sorted_vals:
            return 0
        idx = int(round(q * (len(sorted_vals) - 1)))
        return sorted_vals[idx]

    percentiles = {
        "p50": _percentile(sizes, 0.50),
        "p95": _percentile(sizes, 0.95),
        "p99": _percentile(sizes, 0.99),
        "max": sizes[-1] if sizes else 0,
        "count": len(sizes),
    }

    oversized: list[dict[str, Any]] = []
    for cluster_id, members in components.items():
        if len(members) <= 100:
            continue
        member_set = set(members)
        weakest_pair: tuple[int, int] | None = None
        weakest_score: float | None = None
        for a, b, s in filtered:
            if a in member_set and b in member_set:
                if weakest_score is None or s < weakest_score:
                    weakest_score = s
                    weakest_pair = (a, b)
        entry: dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "size": len(members),
        }
        if weakest_pair is not None:
            entry["bottleneck_pair"] = [int(weakest_pair[0]), int(weakest_pair[1])]
        oversized.append(entry)

    return {
        "preliminary_cluster_sizes": percentiles,
        "oversized_clusters": oversized,
    }


def _signal_threshold_overlap(
    pair_scores: "list[tuple[int, int, float]]",
    current_threshold: float,
) -> float:
    """Fraction of pairs whose score lies in [threshold - 0.02, threshold + 0.02]."""
    if not pair_scores:
        return 0.0
    lo = current_threshold - 0.02
    hi = current_threshold + 0.02
    in_band = sum(1 for _a, _b, s in pair_scores if lo <= s <= hi)
    return in_band / len(pair_scores)


def _signal_block_size_percentiles(
    df: "pl.DataFrame", config: "GoldenMatchConfig"
) -> dict[str, Any]:
    """Compute P50/P95/P99/max block sizes across all blocking keys.

    Mirrors the sampling approach in _check_block_sizes (Preflight Check 4):
    sample ``min(df.height, 10_000)`` via ``df.head(n)`` (deterministic, not
    random — for pre-sorted inputs the distribution may over-represent the
    head's block skew; acceptable as a first-line indicator), build the
    blocking key expr, group, count. Returns zeros on failure or when
    blocking is absent.
    """
    zero = {"p50": 0, "p95": 0, "p99": 0, "max": 0}
    if config.blocking is None or df.height == 0:
        return zero

    from goldenmatch.core.blocker import _build_block_key_expr

    n = min(df.height, 10_000)
    sample = df.head(n) if df.height > n else df

    keys = list(config.blocking.keys or [])
    keys.extend(config.blocking.passes or [])

    all_sizes: list[int] = []
    for key in keys:
        if not all(f in df.columns for f in key.fields):
            continue
        try:
            expr = _build_block_key_expr(key)
            sizes_series = (
                sample.with_columns(expr)
                .group_by("__block_key__")
                .len()
                .get_column("len")
            )
        except Exception:
            continue
        all_sizes.extend(int(s) for s in sizes_series.to_list())

    if not all_sizes:
        return zero
    all_sizes.sort()

    def _percentile(vals: list[int], q: float) -> int:
        idx = int(round(q * (len(vals) - 1)))
        return vals[idx]

    return {
        "p50": _percentile(all_sizes, 0.50),
        "p95": _percentile(all_sizes, 0.95),
        "p99": _percentile(all_sizes, 0.99),
        "max": all_sizes[-1],
    }


def postflight(
    df: "pl.DataFrame",
    config: "GoldenMatchConfig",
    *,
    pair_scores: "list[tuple[int, int, float]]",
    current_threshold: float | None = None,
) -> PostflightReport:
    """Run all postflight signals on (df, config, pair_scores).

    Populates ``report.signals`` with the stable schema documented in
    ``PostflightSignals``.
    When ``config._strict_autoconfig`` is True, signals are still computed
    but no adjustments are emitted (advisories may still accrue).
    """
    report = PostflightReport()
    threshold = _resolve_current_threshold(config, current_threshold)
    strict = bool(getattr(config, "_strict_autoconfig", False))

    # Score histogram + bimodality
    hist = _signal_score_histogram(pair_scores, threshold)
    report.signals["score_histogram"] = hist["histogram"]
    valley = hist["valley_location"]
    depth_ratio = hist["valley_depth_ratio"]

    is_bimodal = (
        valley is not None
        and depth_ratio is not None
        and depth_ratio < 0.5
    )
    if is_bimodal and valley is not None:
        if abs(valley - threshold) > 0.05 and not strict:
            report.adjustments.append(
                PostflightAdjustment(
                    field="threshold",
                    from_value=threshold,
                    to_value=round(float(valley), 3),
                    reason=(
                        f"score distribution is bimodal; valley at "
                        f"{valley:.3f} (depth ratio {depth_ratio:.2f}) is far "
                        f"from current threshold {threshold:.3f}"
                    ),
                    signal="score_histogram",
                )
            )
    elif not is_bimodal:
        report.advisories.append(
            "score distribution is unimodal; threshold cannot be auto-set."
        )

    # Blocking recall (gated >=10K rows; otherwise None)
    report.signals["blocking_recall"] = _signal_blocking_recall(
        df, config, pair_scores, threshold
    )

    # Block size percentiles
    report.signals["block_size_percentiles"] = _signal_block_size_percentiles(
        df, config
    )

    # Threshold-band overlap
    overlap = _signal_threshold_overlap(pair_scores, threshold)
    report.signals["threshold_overlap_pct"] = overlap
    llm_enabled = (
        config.llm_scorer is not None
        and getattr(config.llm_scorer, "enabled", False) is True
    )
    if overlap > 0.20 and not llm_enabled:
        report.advisories.append(
            f"{overlap:.1%} of pairs lie within 0.02 of the threshold; "
            f"consider --llm-auto for threshold-band calibration."
        )

    # Total pairs scored + current threshold (signal fingerprint)
    report.signals["total_pairs_scored"] = len(pair_scores)
    report.signals["current_threshold"] = threshold

    # Preliminary cluster sizes + oversized clusters
    cluster_info = _signal_cluster_sizes(pair_scores, threshold)
    report.signals["preliminary_cluster_sizes"] = cluster_info[
        "preliminary_cluster_sizes"
    ]
    report.signals["oversized_clusters"] = cluster_info["oversized_clusters"]

    return report


def preflight(
    df: "pl.DataFrame",
    config: "GoldenMatchConfig",
    *,
    profiles: "list[ColumnProfile] | None" = None,
    allow_remote_assets: bool = False,
) -> PreflightReport:
    """Run all preflight checks on (df, config).

    Auto-repairs what it can (annotating each finding with ``repaired=True``)
    and records unrepairable issues as error findings. Callers should inspect
    ``report.has_errors`` and raise `ConfigValidationError` if strict behavior
    is desired — `auto_configure_df` does this.
    """
    report = PreflightReport()
    _check_columns(df, config, report)
    _check_cardinality(df, config, report)
    _check_block_sizes(df, config, report)
    _check_remote_assets(config, report, allow_remote_assets=allow_remote_assets)
    if profiles is not None:
        _check_weight_confidence(config, profiles, report)
    return report
