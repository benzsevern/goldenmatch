"""Auto-configuration verification: preflight (and later, postflight) checks.

See spec: docs/superpowers/specs/2026-04-14-autoconfig-verification-design.md §4.

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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl

    from goldenmatch.config.schemas import GoldenMatchConfig
    from goldenmatch.core.autoconfig import ColumnProfile


@dataclass
class PreflightFinding:
    """One check result — informational, warning, or hard error."""

    check: str
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

    # Cap sample at 10K rows — block-size distribution converges well below
    # full-dataset sizes and the transforms are O(n).
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
    """Check 5: demote scorers that require downloading models or hitting the
    network, unless the caller explicitly allowed them or an LLM scorer is
    already enabled (which implies online operation is fine).
    """
    if allow_remote_assets:
        return

    llm_enabled = (
        config.llm_scorer is not None and config.llm_scorer.enabled is True
    )
    if llm_enabled:
        return

    for mk in config.get_matchkeys():
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
