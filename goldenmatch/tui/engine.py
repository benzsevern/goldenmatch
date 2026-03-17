"""MatchEngine — shared foundation for TUI and preview mode.

Wraps the existing pipeline modules into a clean API with sample
extraction, scored-pairs caching, and threshold re-clustering.
No Textual dependency — pure Python + Polars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass
class EngineStats:
    total_records: int
    total_clusters: int
    singleton_count: int
    match_rate: float
    cluster_sizes: list[int]
    avg_cluster_size: float
    max_cluster_size: int
    oversized_count: int
    hit_rate: float | None = None
    avg_score: float | None = None


@dataclass
class EngineResult:
    clusters: dict[int, dict]
    golden: pl.DataFrame | None
    unique: pl.DataFrame | None
    dupes: pl.DataFrame | None
    quarantine: pl.DataFrame | None
    matched: pl.DataFrame | None
    unmatched: pl.DataFrame | None
    scored_pairs: list[tuple[int, int, float]]
    stats: EngineStats


class MatchEngine:
    """Wraps the pipeline into a clean API for the TUI and preview mode."""

    def __init__(self, files: list[Path | str]):
        self._files = [Path(f) for f in files]
        self._data: pl.DataFrame | None = None
        self._profile: dict | None = None
        self._last_result: EngineResult | None = None
        self._load()

    def _load(self) -> None:
        from goldenmatch.core.ingest import load_file
        from goldenmatch.core.profiler import profile_dataframe

        frames = []
        for f in self._files:
            lf = load_file(f)
            lf = lf.with_columns(pl.lit(f.stem).alias("__source__"))
            frames.append(lf.collect())
        combined = pl.concat(frames)
        # Add row IDs
        combined = combined.with_row_index("__row_id__").with_columns(
            pl.col("__row_id__").cast(pl.Int64)
        )
        self._data = combined
        # Profile without internal columns
        profile_cols = [c for c in combined.columns if not c.startswith("__")]
        self._profile = profile_dataframe(combined.select(profile_cols))

    @property
    def data(self) -> pl.DataFrame:
        return self._data

    @property
    def profile(self) -> dict:
        return self._profile

    @property
    def columns(self) -> list[str]:
        return [c for c in self._data.columns if not c.startswith("__")]

    @property
    def row_count(self) -> int:
        return self._data.height

    def get_sample(self, n: int) -> pl.DataFrame:
        if n >= self._data.height:
            return self._data
        return self._data.head(n)

    def _compute_stats(self, clusters: dict[int, dict], total_records: int) -> EngineStats:
        """Compute statistics from cluster results."""
        multi = [cid for cid, c in clusters.items() if c["size"] > 1]
        singletons = len(clusters) - len(multi)
        cluster_sizes = [clusters[cid]["size"] for cid in multi]
        oversized_count = sum(1 for cid in multi if clusters[cid]["oversized"])
        avg_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0
        max_size = max(cluster_sizes) if cluster_sizes else 0
        matched_records = sum(cluster_sizes)
        match_rate = matched_records / total_records if total_records > 0 else 0.0

        return EngineStats(
            total_records=total_records,
            total_clusters=len(multi),
            singleton_count=singletons,
            match_rate=match_rate,
            cluster_sizes=cluster_sizes,
            avg_cluster_size=avg_size,
            max_cluster_size=max_size,
            oversized_count=oversized_count,
        )

    def _run_pipeline(self, df: pl.DataFrame, config) -> EngineResult:
        """Core pipeline logic — mirrors run_dedupe but returns EngineResult."""
        from goldenmatch.core.autofix import auto_fix_dataframe
        from goldenmatch.core.blocker import build_blocks
        from goldenmatch.core.cluster import build_clusters
        from goldenmatch.core.golden import build_golden_record
        from goldenmatch.core.matchkey import compute_matchkeys
        from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
        from goldenmatch.core.standardize import apply_standardization
        from goldenmatch.core.validate import ValidationRule, validate_dataframe
        from goldenmatch.config.schemas import GoldenRulesConfig

        combined_lf = df.lazy()
        quarantine_df = None

        # ── Auto-fix ──
        if config.validation and config.validation.auto_fix:
            combined_df_tmp = combined_lf.collect()
            combined_df_tmp, _fix_log = auto_fix_dataframe(combined_df_tmp)
            combined_lf = combined_df_tmp.lazy()

        # ── Validation ──
        if config.validation and config.validation.rules:
            rules = [
                ValidationRule(
                    column=rc.column,
                    rule_type=rc.rule_type,
                    params=rc.params,
                    action=rc.action,
                )
                for rc in config.validation.rules
            ]
            combined_df_tmp = combined_lf.collect()
            valid_df, quarantine_df, _val_report = validate_dataframe(combined_df_tmp, rules)
            combined_lf = valid_df.lazy()

        # ── Standardization ──
        if config.standardization and config.standardization.rules:
            combined_lf = apply_standardization(combined_lf, config.standardization.rules)

        # ── Compute matchkeys ──
        matchkeys = config.get_matchkeys()
        combined_lf = compute_matchkeys(combined_lf, matchkeys)

        # ── Score all pairs (cascading: exact first, then fuzzy) ──
        all_pairs: list[tuple[int, int, float]] = []
        matched_pairs: set[tuple[int, int]] = set()
        collected_df = combined_lf.collect()

        # Phase 1: Exact matchkeys (fast)
        for mk in matchkeys:
            if mk.type == "exact":
                pairs = find_exact_matches(combined_lf, mk)
                all_pairs.extend(pairs)
                for a, b, s in pairs:
                    matched_pairs.add((min(a, b), max(a, b)))

        # Phase 2: Fuzzy matchkeys (slow — skip already-matched pairs)
        for mk in matchkeys:
            if mk.type == "weighted":
                if config.blocking is None:
                    continue
                blocks = build_blocks(combined_lf, config.blocking)
                for block in blocks:
                    block_df = block.df.collect()
                    pairs = find_fuzzy_matches(block_df, mk, exclude_pairs=matched_pairs)
                    all_pairs.extend(pairs)
                    for a, b, s in pairs:
                        matched_pairs.add((min(a, b), max(a, b)))

        # ── Cluster ──
        all_ids = collected_df["__row_id__"].to_list()
        max_cluster_size = 100
        if config.golden_rules and hasattr(config.golden_rules, "max_cluster_size"):
            max_cluster_size = config.golden_rules.max_cluster_size

        clusters = build_clusters(all_pairs, all_ids, max_cluster_size=max_cluster_size)

        # ── Golden records ──
        golden_rules = config.golden_rules or GoldenRulesConfig(default_strategy="most_complete")
        golden_records = []
        for cluster_id, cluster_info in clusters.items():
            if cluster_info["size"] > 1 and not cluster_info["oversized"]:
                member_ids = cluster_info["members"]
                cluster_df = collected_df.filter(pl.col("__row_id__").is_in(member_ids))
                golden = build_golden_record(cluster_df, golden_rules)
                golden["__cluster_id__"] = cluster_id
                golden_records.append(golden)

        golden_df = None
        if golden_records:
            golden_rows = []
            for rec in golden_records:
                row = {"__cluster_id__": rec["__cluster_id__"]}
                row["__golden_confidence__"] = rec.get("__golden_confidence__", 0.0)
                for col, val_info in rec.items():
                    if col in ("__cluster_id__", "__golden_confidence__"):
                        continue
                    if isinstance(val_info, dict) and "value" in val_info:
                        row[col] = val_info["value"]
                golden_rows.append(row)
            golden_df = pl.DataFrame(golden_rows)

        # ── Classify dupes / unique ──
        multi_cluster_ids = [cid for cid, c in clusters.items() if c["size"] > 1]
        dupe_row_ids = set()
        for cid in multi_cluster_ids:
            dupe_row_ids.update(clusters[cid]["members"])
        unique_row_ids = set(all_ids) - dupe_row_ids

        dupes_df = collected_df.filter(pl.col("__row_id__").is_in(list(dupe_row_ids)))
        unique_df = collected_df.filter(pl.col("__row_id__").is_in(list(unique_row_ids)))

        # ── Stats ──
        stats = self._compute_stats(clusters, collected_df.height)

        return EngineResult(
            clusters=clusters,
            golden=golden_df,
            unique=unique_df,
            dupes=dupes_df,
            quarantine=quarantine_df,
            matched=None,
            unmatched=None,
            scored_pairs=all_pairs,
            stats=stats,
        )

    def run_sample(self, config, sample_size: int = 1000) -> EngineResult:
        """Run the pipeline on a sample of the data."""
        sample_df = self.get_sample(sample_size)
        result = self._run_pipeline(sample_df, config)
        self._last_result = result
        return result

    def run_full(self, config) -> EngineResult:
        """Run the pipeline on the full dataset."""
        result = self._run_pipeline(self._data, config)
        self._last_result = result
        return result

    def recluster_at_threshold(self, threshold: float) -> EngineStats:
        """Re-cluster cached scored pairs at a new threshold. No re-scoring."""
        if self._last_result is None:
            raise RuntimeError("No previous run exists. Call run_sample or run_full first.")

        from goldenmatch.core.cluster import build_clusters

        filtered_pairs = [
            (a, b, s) for a, b, s in self._last_result.scored_pairs if s >= threshold
        ]

        # Gather all IDs from the last result's clusters
        all_ids = []
        for cluster_info in self._last_result.clusters.values():
            all_ids.extend(cluster_info["members"])
        all_ids = sorted(set(all_ids))

        clusters = build_clusters(filtered_pairs, all_ids)
        return self._compute_stats(clusters, len(all_ids))
