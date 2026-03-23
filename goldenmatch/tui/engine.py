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
    llm_cost: dict | None = None


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
        from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches, score_blocks_parallel, rerank_top_pairs
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

        # ── Auto-suggest blocking keys ──
        collected_df = combined_lf.collect()
        if config.blocking and config.blocking.auto_suggest:
            from goldenmatch.core.pipeline import _run_auto_suggest
            _run_auto_suggest(collected_df, config)

        # ── Score all pairs (cascading: exact first, then fuzzy) ──
        all_pairs: list[tuple[int, int, float]] = []
        matched_pairs: set[tuple[int, int]] = set()

        # Phase 1: Exact matchkeys (fast)
        for mk in matchkeys:
            if mk.type == "exact":
                pairs = find_exact_matches(combined_lf, mk)
                all_pairs.extend(pairs)
                for a, b, s in pairs:
                    matched_pairs.add((min(a, b), max(a, b)))

        # Phase 2: Fuzzy matchkeys (parallel block scoring)
        for mk in matchkeys:
            if mk.type == "weighted":
                if config.blocking is None:
                    continue
                blocks = build_blocks(combined_lf, config.blocking)
                pairs = score_blocks_parallel(blocks, mk, matched_pairs)
                all_pairs.extend(pairs)

        # Phase 2b: Probabilistic matchkeys (Fellegi-Sunter with EM)
        for mk in matchkeys:
            if mk.type == "probabilistic":
                if config.blocking is None:
                    continue
                from goldenmatch.core.probabilistic import train_em, score_probabilistic
                blocks = build_blocks(combined_lf, config.blocking)
                blocking_fields = []
                if config.blocking and config.blocking.keys:
                    for bk in config.blocking.keys:
                        blocking_fields.extend(bk.fields)
                em_result = train_em(
                    collected_df, mk,
                    max_iterations=mk.em_iterations,
                    convergence=mk.convergence_threshold,
                    blocks=blocks,
                    blocking_fields=blocking_fields,
                )
                for block in blocks:
                    block_df = block.df.collect() if hasattr(block.df, 'collect') else block.df
                    pairs = score_probabilistic(block_df, mk, em_result, exclude_pairs=matched_pairs)
                    all_pairs.extend(pairs)
                    for a, b, s in pairs:
                        matched_pairs.add((min(a, b), max(a, b)))

        # ── Rerank (optional) ──
        for mk in matchkeys:
            if mk.type == "weighted" and mk.rerank:
                all_pairs = rerank_top_pairs(all_pairs, collected_df, mk)
                break

        # ── LLM scorer (optional) ──
        llm_budget_summary = None
        if config.llm_scorer and config.llm_scorer.enabled and all_pairs:
            has_budget = config.llm_scorer.budget is not None
            if config.llm_scorer.mode == "cluster":
                from goldenmatch.core.llm_cluster import llm_cluster_pairs
                result = llm_cluster_pairs(
                    all_pairs, collected_df,
                    config=config.llm_scorer,
                    return_budget=has_budget,
                )
            else:
                from goldenmatch.core.llm_scorer import llm_score_pairs
                result = llm_score_pairs(
                    all_pairs, collected_df,
                    config=config.llm_scorer,
                    return_budget=has_budget,
                )
            if has_budget:
                all_pairs, llm_budget_summary = result
            else:
                all_pairs = result
            all_pairs = [(a, b, s) for a, b, s in all_pairs if s > 0.5]

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
        if llm_budget_summary:
            stats.llm_cost = llm_budget_summary

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

    def unmerge_record(self, record_id: int, threshold: float = 0.0) -> EngineResult | None:
        """Remove a record from its cluster and return updated results."""
        if self._last_result is None:
            return None

        from goldenmatch.core.cluster import unmerge_record

        clusters = unmerge_record(record_id, self._last_result.clusters, threshold)
        stats = self._compute_stats(clusters, self._data.height)

        self._last_result = EngineResult(
            clusters=clusters,
            golden=self._last_result.golden,
            unique=self._last_result.unique,
            dupes=self._last_result.dupes,
            quarantine=self._last_result.quarantine,
            matched=self._last_result.matched,
            unmatched=self._last_result.unmatched,
            scored_pairs=self._last_result.scored_pairs,
            stats=stats,
        )
        return self._last_result

    def unmerge_cluster(self, cluster_id: int) -> EngineResult | None:
        """Shatter a cluster into singletons and return updated results."""
        if self._last_result is None:
            return None

        from goldenmatch.core.cluster import unmerge_cluster

        clusters = unmerge_cluster(cluster_id, self._last_result.clusters)
        stats = self._compute_stats(clusters, self._data.height)

        self._last_result = EngineResult(
            clusters=clusters,
            golden=self._last_result.golden,
            unique=self._last_result.unique,
            dupes=self._last_result.dupes,
            quarantine=self._last_result.quarantine,
            matched=self._last_result.matched,
            unmatched=self._last_result.unmatched,
            scored_pairs=self._last_result.scored_pairs,
            stats=stats,
        )
        return self._last_result

    def match_one(self, record: dict, config) -> list[tuple[int, float]]:
        """Match a single record against the loaded dataset.

        Uses brute-force scoring against the full dataset. For ANN-accelerated
        matching, use core.match_one directly with a pre-built ANNBlocker.
        """
        from goldenmatch.core.match_one import match_one

        matchkeys = config.get_matchkeys()
        results = []
        for mk in matchkeys:
            if mk.type == "weighted":
                matches = match_one(record, self._data, mk)
                results.extend(matches)
        # Deduplicate by row_id, keep highest score
        best: dict[int, float] = {}
        for row_id, score in results:
            if row_id not in best or score > best[row_id]:
                best[row_id] = score
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

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
