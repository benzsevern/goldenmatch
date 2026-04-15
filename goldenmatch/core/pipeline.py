"""Pipeline orchestrator for GoldenMatch dedupe and list-match workflows."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig, GoldenRulesConfig
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.ingest import load_file, validate_columns, apply_column_map
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.validate import ValidationRule, validate_dataframe
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.block_analyzer import analyze_blocking
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches, score_blocks_parallel, rerank_top_pairs


def _get_block_scorer(config):
    """Return the block scoring function based on configured backend."""
    backend = getattr(config, "backend", None)
    if backend == "ray":
        from goldenmatch.backends.ray_backend import score_blocks_ray
        return score_blocks_ray
    return score_blocks_parallel
from goldenmatch.core.cluster import build_clusters
from goldenmatch.core.golden import build_golden_record
from goldenmatch.output.writer import write_output
from goldenmatch.output.report import generate_dedupe_report, generate_match_report

logger = logging.getLogger(__name__)


def _extract_matchkey_columns(config: GoldenMatchConfig) -> list[str]:
    """Extract unique field names from all matchkeys in config."""
    cols = set()
    for mk in config.get_matchkeys():
        for f in mk.fields:
            cols.add(f.field)
    return sorted(cols)


def _run_auto_suggest(df: pl.DataFrame, config: GoldenMatchConfig) -> None:
    """Run block analyzer auto-suggest if enabled in config.

    Logs the top 3 suggestions. If config.blocking.keys is empty,
    populates it from the top suggestion.
    """
    if not config.blocking or not config.blocking.auto_suggest:
        return

    matchkey_columns = _extract_matchkey_columns(config)
    if not matchkey_columns:
        return

    suggestions = analyze_blocking(df, matchkey_columns)
    if not suggestions:
        logger.info("Auto-suggest: no blocking suggestions found")
        return

    # Log top 3 suggestions
    for i, s in enumerate(suggestions[:3]):
        logger.info(
            "Auto-suggest #%d: %s (blocks=%d, max_size=%d, comparisons=%d, recall=%.2f, score=%.4f)",
            i + 1,
            s.description,
            s.group_count,
            s.max_group_size,
            s.total_comparisons,
            s.estimated_recall,
            s.score,
        )

    # If no user-configured keys, use the top suggestion
    if not config.blocking.keys:
        top = suggestions[0]
        from goldenmatch.config.schemas import BlockingKeyConfig

        new_keys = []
        for cand in top.keys:
            new_keys.append(BlockingKeyConfig(
                fields=cand["key_fields"],
                transforms=cand.get("transforms", []) if isinstance(cand.get("transforms", []), list) and all(isinstance(t, str) for t in cand.get("transforms", [])) else [],
            ))
        config.blocking.keys = new_keys
        logger.info("Auto-suggest: using top suggestion '%s' as blocking keys", top.description)


def _add_row_ids(lf: pl.LazyFrame, offset: int = 0) -> pl.LazyFrame:
    """Add __row_id__ column using with_row_index + offset."""
    lf = lf.with_row_index("__row_id__")
    if offset > 0:
        lf = lf.with_columns((pl.col("__row_id__") + offset).alias("__row_id__"))
    # Cast to Int64 for consistency
    lf = lf.with_columns(pl.col("__row_id__").cast(pl.Int64))
    return lf


def _get_required_columns(config: GoldenMatchConfig) -> list[str]:
    """Extract all column names referenced in matchkeys and blocking config."""
    cols = set()
    for mk in config.get_matchkeys():
        for f in mk.fields:
            if f.columns:
                cols.update(f.columns)
            elif f.field and f.field != "__record__":
                cols.add(f.field)
    if config.blocking:
        for key_config in config.blocking.keys:
            for field_name in key_config.fields:
                cols.add(field_name)
    return sorted(cols)


def run_dedupe(
    files: list[tuple],
    config: GoldenMatchConfig,
    output_golden: bool = False,
    output_clusters: bool = False,
    output_dupes: bool = False,
    output_unique: bool = False,
    output_report: bool = False,
    across_files_only: bool = False,
    llm_retrain: bool = False,
    llm_provider: str | None = None,
    llm_max_labels: int = 500,
) -> dict:
    """Run the dedupe pipeline.

    Args:
        files: List of (file_path, source_name) tuples.
        config: GoldenMatch configuration.
        output_golden: Whether to output golden records.
        output_clusters: Whether to output cluster info.
        output_dupes: Whether to output duplicate records.
        output_unique: Whether to output unique records.
        output_report: Whether to generate a report.
        across_files_only: If True, only match across different sources.

    Returns:
        Dict with keys: clusters, golden, unique, dupes, report.
    """
    matchkeys = config.get_matchkeys()

    # ── Step 1: INGEST ──
    frames = []
    offset = 0
    for file_spec in files:
        if len(file_spec) == 3:
            file_path, source_name, column_map = file_spec
        else:
            file_path, source_name = file_spec[0], file_spec[1]
            column_map = None
        lf = load_file(file_path)
        if column_map:
            lf = apply_column_map(lf, column_map)
        required = _get_required_columns(config)
        validate_columns(lf, required)
        lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
        lf = _add_row_ids(lf, offset=offset)
        collected = lf.collect()
        offset += len(collected)
        frames.append(collected.lazy())

    combined_lf = pl.concat([f.collect() for f in frames]).lazy()

    return _run_dedupe_pipeline(
        combined_lf, config, matchkeys,
        output_golden, output_clusters,
        output_dupes, output_unique, output_report,
        across_files_only, llm_retrain, llm_provider, llm_max_labels,
    )


def _run_dedupe_pipeline(
    combined_lf: pl.LazyFrame,
    config: GoldenMatchConfig,
    matchkeys: list,
    output_golden: bool = False,
    output_clusters: bool = False,
    output_dupes: bool = False,
    output_unique: bool = False,
    output_report: bool = False,
    across_files_only: bool = False,
    llm_retrain: bool = False,
    llm_provider: str | None = None,
    llm_max_labels: int = 500,
    auto_config: bool = False,
    auto_config_llm_provider: str | None = None,
) -> dict:
    """Shared dedupe pipeline logic (post-ingest).

    This function contains all pipeline steps from auto-fix/validation through
    output. Both run_dedupe() and run_dedupe_df() delegate to this function.
    """
    # ── Step 1.4: GOLDENCHECK QUALITY SCAN (if available) ──
    if config.quality is None or config.quality.mode != "disabled":
        from goldenmatch.core.quality import run_quality_check
        combined_df_tmp = combined_lf.collect()
        combined_df_tmp, gc_fixes = run_quality_check(combined_df_tmp, config.quality)
        if gc_fixes:
            logger.info("GoldenCheck: %d fixes applied", len(gc_fixes))
        combined_lf = combined_df_tmp.lazy()

    # ── Step 1.4b: GOLDENFLOW TRANSFORM (if available) ──
    # Runs after GoldenCheck (validates) and before autofix (remaining cleanup).
    # Not in _run_match_pipeline -- add there if match pipeline gains a quality step.
    if config.transform is None or config.transform.mode != "disabled":
        from goldenmatch.core.transform import run_transform
        combined_df_tmp = combined_lf.collect()
        combined_df_tmp, gf_fixes = run_transform(combined_df_tmp, config.transform)
        if gf_fixes:
            logger.info("GoldenFlow: %d transforms applied", len(gf_fixes))
        combined_lf = combined_df_tmp.lazy()

    # ── Step 1.5a: AUTO-FIX + VALIDATION ──
    if config.validation and config.validation.auto_fix:
        combined_df_tmp = combined_lf.collect()
        combined_df_tmp, fix_log = auto_fix_dataframe(combined_df_tmp)
        logger.info("Auto-fix applied: %d fix type(s)", len(fix_log))
        combined_lf = combined_df_tmp.lazy()

    # ── Step 1.5b: AUTO-CONFIG ON CLEANED DATA (if zero-config) ──
    if auto_config:
        from goldenmatch.core.autoconfig import auto_configure_df
        combined_df_tmp = combined_lf.collect()
        auto_cfg = auto_configure_df(
            combined_df_tmp,
            llm_provider=auto_config_llm_provider,
            llm_auto=config.llm_auto,
        )
        config.matchkeys = auto_cfg.matchkeys
        config.match_settings = auto_cfg.match_settings
        config.blocking = auto_cfg.blocking
        config.golden_rules = auto_cfg.golden_rules
        config.llm_scorer = auto_cfg.llm_scorer
        config.memory = auto_cfg.memory
        # Propagate domain config so pipeline's domain-extraction step runs
        # when auto_configure_df (via preflight Check 1) decided it should.
        if auto_cfg.domain is not None:
            config.domain = auto_cfg.domain
        # Propagate preflight-verification markers so postflight (later in
        # this function) knows auto-config was used and whether strict mode
        # is on. These are underscore-private attrs, not Pydantic fields.
        if getattr(auto_cfg, "_preflight_report", None) is not None:
            config._preflight_report = auto_cfg._preflight_report
        if getattr(auto_cfg, "_strict_autoconfig", False):
            config._strict_autoconfig = True
        matchkeys = config.get_matchkeys()
        logger.info("Auto-configured from cleaned data: %d matchkeys", len(matchkeys))
        combined_lf = combined_df_tmp.lazy()

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
        valid_df, quarantine_df, val_report = validate_dataframe(combined_df_tmp, rules)
        logger.info("Validation: %d quarantined rows", quarantine_df.height)
        combined_lf = valid_df.lazy()
    else:
        quarantine_df = None

    # ── Step 1.5b: STANDARDIZE ──
    if config.standardization and config.standardization.rules:
        combined_lf = apply_standardization(combined_lf, config.standardization.rules)

    # ── Step 1.5c: DOMAIN FEATURE EXTRACTION ──
    domain_cfg = config.domain
    if domain_cfg and domain_cfg.enabled:
        from goldenmatch.core.domain import detect_domain, extract_features
        combined_df_tmp = combined_lf.collect()
        user_cols = [c for c in combined_df_tmp.columns if not c.startswith("__")]

        if domain_cfg.mode == "auto" or domain_cfg.mode is None:
            domain_profile = detect_domain(user_cols)
        else:
            from goldenmatch.core.domain import DomainProfile
            domain_profile = DomainProfile(
                name=domain_cfg.mode, confidence=1.0,
                text_columns=[c for c in user_cols if any(p in c.lower() for p in ("name", "title", "description", "product"))],
            )

        if domain_profile.confidence > 0.3:
            logger.info("Domain detected: %s (confidence %.2f)", domain_profile.name, domain_profile.confidence)
            combined_df_tmp, low_conf_ids = extract_features(
                combined_df_tmp, domain_profile, domain_cfg.confidence_threshold,
            )

            # LLM validation for low-confidence extractions
            if domain_cfg.llm_validation and low_conf_ids:
                from goldenmatch.core.llm_extract import llm_extract_features, apply_llm_extractions
                budget = None
                if domain_cfg.budget:
                    from goldenmatch.core.llm_budget import BudgetTracker
                    budget = BudgetTracker(domain_cfg.budget)
                text_col = domain_profile.text_columns[0] if domain_profile.text_columns else user_cols[0]
                extractions = llm_extract_features(
                    combined_df_tmp, low_conf_ids, text_col,
                    domain=domain_profile.name, budget_tracker=budget,
                )
                combined_df_tmp = apply_llm_extractions(combined_df_tmp, extractions, domain_profile.name)
                logger.info("LLM validated %d/%d low-confidence extractions", len(extractions), len(low_conf_ids))

            combined_lf = combined_df_tmp.lazy()

    # ── Step 2: TRANSFORM ──
    combined_lf = compute_matchkeys(combined_lf, matchkeys)

    # ── Step 2.5: AUTO-SUGGEST blocking keys ──
    collected_df = combined_lf.collect()
    _run_auto_suggest(collected_df, config)

    # ── Step 3: BLOCK + COMPARE (cascading: exact first, then fuzzy) ──
    all_pairs: list[tuple[int, int, float]] = []
    matched_pairs: set[tuple[int, int]] = set()

    # Build source lookup for across_files_only filtering
    source_lookup = {}
    if across_files_only:
        for row in collected_df.select("__row_id__", "__source__").to_dicts():
            source_lookup[row["__row_id__"]] = row["__source__"]

    # Phase 1: Exact matchkeys (fast)
    for mk in matchkeys:
        if mk.type == "exact":
            pairs = find_exact_matches(combined_lf, mk)
            if across_files_only:
                pairs = [
                    (a, b, s) for a, b, s in pairs
                    if source_lookup.get(a) != source_lookup.get(b)
                ]
            all_pairs.extend(pairs)
            for a, b, s in pairs:
                matched_pairs.add((min(a, b), max(a, b)))

    logger.info("Exact matching found %d pairs", len(all_pairs))

    # Phase 2: Fuzzy matchkeys (parallel block scoring)
    for mk in matchkeys:
        if mk.type == "weighted":
            if config.blocking is None:
                continue
            blocks = build_blocks(combined_lf, config.blocking)
            block_scorer = _get_block_scorer(config)
            pairs = block_scorer(
                blocks, mk, matched_pairs,
                across_files_only=across_files_only,
                source_lookup=source_lookup if across_files_only else None,
            )
            all_pairs.extend(pairs)

    # Phase 2b: Probabilistic matchkeys (Fellegi-Sunter with EM)
    for mk in matchkeys:
        if mk.type == "probabilistic":
            if config.blocking is None:
                continue
            from goldenmatch.core.probabilistic import train_em, score_probabilistic
            # Build blocks first, then train EM on within-block pairs
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
            logger.info(
                "F-S EM: converged=%s, iterations=%d, match_rate=%.4f",
                em_result.converged, em_result.iterations, em_result.proportion_matched,
            )
            for block in blocks:
                block_df = block.df.collect() if hasattr(block.df, 'collect') else block.df
                pairs = score_probabilistic(block_df, mk, em_result, exclude_pairs=matched_pairs)
                if across_files_only:
                    pairs = [
                        (a, b, s) for a, b, s in pairs
                        if source_lookup.get(a) != source_lookup.get(b)
                    ]
                all_pairs.extend(pairs)
                for a, b, s in pairs:
                    matched_pairs.add((min(a, b), max(a, b)))

    # ── Step 3.3: CROSS-ENCODER RERANKING (optional) ──
    for mk in matchkeys:
        if mk.type == "weighted" and mk.rerank:
            all_pairs = rerank_top_pairs(all_pairs, collected_df, mk)
            break  # rerank once with the first rerank-enabled matchkey

    # ── Step 3.4: LLM SCORER (optional) ──
    if config.llm_scorer and config.llm_scorer.enabled and all_pairs:
        if config.llm_scorer.mode == "cluster":
            from goldenmatch.core.llm_cluster import llm_cluster_pairs
            all_pairs = llm_cluster_pairs(all_pairs, collected_df, config=config.llm_scorer)
        else:
            from goldenmatch.core.llm_scorer import llm_score_pairs
            all_pairs = llm_score_pairs(all_pairs, collected_df, config=config.llm_scorer)
        # Filter to scored matches only
        all_pairs = [(a, b, s) for a, b, s in all_pairs if s > 0.5]

    # ── Step 3.5: LLM BOOST (optional) ──
    if config.llm_boost and all_pairs:
        try:
            from goldenmatch.core.boost import boost_accuracy
            matchable_cols = [
                c for c in collected_df.columns if not c.startswith("__")
            ]
            all_pairs = boost_accuracy(
                all_pairs, collected_df, matchable_cols,
                provider=llm_provider,
                api_key=None,  # auto-detect from env/settings
                model_name=None,  # auto-detect
                max_labels=llm_max_labels,
                retrain=llm_retrain,
            )
        except ImportError as e:
            logger.warning("LLM boost unavailable: %s", e)

    # ── Step 3.6: POSTFLIGHT (auto-config only) ──
    # Postflight verification (spec §6). Signals are computed from the
    # unadjusted pair list; threshold adjustments (if any, non-strict only)
    # are then applied to all_pairs before clustering. This is intentional:
    # signals reflect what postflight actually observed; downstream clustering
    # reflects the adjusted threshold. Documented in spec §6.1.
    postflight_report = None
    if getattr(config, "_preflight_report", None) is not None:
        from goldenmatch.core.autoconfig_verify import postflight as _postflight
        postflight_report = _postflight(collected_df, config, pair_scores=all_pairs)
        if not getattr(config, "_strict_autoconfig", False):
            for adj in postflight_report.adjustments:
                if adj.field == "threshold":
                    all_pairs = [p for p in all_pairs if p[2] >= adj.to_value]

    # ── Step 4: CLUSTER ──
    all_ids = collected_df["__row_id__"].to_list()
    max_cluster_size = 100
    weak_threshold = 0.3
    auto_split = True
    if config.golden_rules:
        if hasattr(config.golden_rules, "max_cluster_size"):
            max_cluster_size = config.golden_rules.max_cluster_size
        if hasattr(config.golden_rules, "weak_cluster_threshold"):
            weak_threshold = config.golden_rules.weak_cluster_threshold
        if hasattr(config.golden_rules, "auto_split"):
            auto_split = config.golden_rules.auto_split

    clusters = build_clusters(
        all_pairs, all_ids,
        max_cluster_size=max_cluster_size,
        weak_cluster_threshold=weak_threshold,
        auto_split=auto_split,
    )

    # ── Step 5: GOLDEN ──
    golden_records = []
    golden_rules = config.golden_rules or GoldenRulesConfig(default_strategy="most_complete")

    for cluster_id, cluster_info in clusters.items():
        if cluster_info["size"] > 1 and not cluster_info["oversized"]:
            member_ids = cluster_info["members"]
            cluster_df = collected_df.filter(pl.col("__row_id__").is_in(member_ids))
            golden = build_golden_record(cluster_df, golden_rules)
            golden["__cluster_id__"] = cluster_id
            golden_records.append(golden)

    # Build golden DataFrame
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
        # Build explicit schema to prevent mixed-type inference errors.
        # Golden records from different clusters may have different value types
        # for the same column (e.g. "0" str vs 0 int).
        all_keys: set[str] = set()
        for row in golden_rows:
            all_keys.update(row.keys())
        schema_overrides = {
            k: pl.Utf8 for k in all_keys
            if k not in ("__cluster_id__", "__golden_confidence__")
        }
        golden_df = pl.DataFrame(golden_rows, schema_overrides=schema_overrides)

    # Classify records
    multi_cluster_ids = [
        cid for cid, cinfo in clusters.items() if cinfo["size"] > 1
    ]
    dupe_row_ids = set()
    for cid in multi_cluster_ids:
        dupe_row_ids.update(clusters[cid]["members"])
    unique_row_ids = set(all_ids) - dupe_row_ids

    dupes_df = collected_df.filter(pl.col("__row_id__").is_in(list(dupe_row_ids)))
    unique_df = collected_df.filter(pl.col("__row_id__").is_in(list(unique_row_ids)))

    # ── Step 6: REPORT ──
    report = None
    if output_report:
        cluster_sizes = [c["size"] for c in clusters.values()]
        oversized_count = sum(1 for c in clusters.values() if c["oversized"])
        report = generate_dedupe_report(
            total_records=len(collected_df),
            total_clusters=len(clusters),
            cluster_sizes=cluster_sizes,
            oversized_clusters=oversized_count,
            matchkeys_used=[mk.name for mk in matchkeys],
        )

    # ── Step 7: OUTPUT ──
    run_name = config.output.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = config.output.format or "csv"
    directory = config.output.directory or config.output.path or "."

    if output_golden and golden_df is not None:
        write_output(golden_df, directory, run_name, "golden", fmt)

    if output_clusters:
        # Build clusters DataFrame
        cluster_rows = []
        for cid, cinfo in clusters.items():
            for member_id in cinfo["members"]:
                cluster_rows.append({
                    "__cluster_id__": cid,
                    "__row_id__": member_id,
                    "__cluster_size__": cinfo["size"],
                    "__oversized__": cinfo["oversized"],
                })
        if cluster_rows:
            clusters_df = pl.DataFrame(cluster_rows)
            write_output(clusters_df, directory, run_name, "clusters", fmt)

    if output_dupes and len(dupes_df) > 0:
        write_output(dupes_df, directory, run_name, "dupes", fmt)

    if output_unique and len(unique_df) > 0:
        write_output(unique_df, directory, run_name, "unique", fmt)

    # ── Step 7.5: LINEAGE (always save when outputting) ──
    if output_golden or output_clusters or output_dupes:
        try:
            from goldenmatch.core.lineage import build_lineage, save_lineage
            lineage = build_lineage(all_pairs, collected_df, matchkeys, clusters)
            save_lineage(lineage, directory, run_name)
        except Exception as e:
            logger.warning("Lineage generation failed: %s", e)

    results = {
        "clusters": clusters,
        "golden": golden_df,
        "unique": unique_df,
        "dupes": dupes_df,
        "report": report,
        "quarantine": quarantine_df,
        "postflight_report": postflight_report,
    }

    return results


def run_dedupe_df(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    source_name: str = "dataframe",
    output_golden: bool = False,
    output_clusters: bool = False,
    output_dupes: bool = False,
    output_unique: bool = False,
    output_report: bool = False,
    auto_config: bool = False,
    auto_config_llm_provider: str | None = None,
) -> dict:
    """Run dedupe pipeline on a DataFrame directly (no file I/O)."""
    # Cast all columns to string to prevent schema mismatch errors when
    # mixed-type columns (e.g. birth_year inferred as i64 in some rows,
    # str in others) reach blocking/scoring operations.
    df = df.cast({col: pl.Utf8 for col in df.columns if not col.startswith("__")})
    matchkeys = [] if auto_config else config.get_matchkeys()
    lf = df.lazy()
    if not auto_config:
        required = _get_required_columns(config)
        validate_columns(lf, required)
    lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
    lf = _add_row_ids(lf, offset=0)
    combined_lf = lf.collect().lazy()
    return _run_dedupe_pipeline(combined_lf, config, matchkeys,
                                output_golden, output_clusters,
                                output_dupes, output_unique, output_report,
                                auto_config=auto_config,
                                auto_config_llm_provider=auto_config_llm_provider)


def run_match(
    target_file: tuple,
    reference_files: list[tuple],
    config: GoldenMatchConfig,
    output_matched: bool = False,
    output_unmatched: bool = False,
    output_scores: bool = False,
    output_report: bool = False,
    match_mode: str = "best",
) -> dict:
    """Run the list-match pipeline.

    Args:
        target_file: (file_path, source_name) for the target file.
        reference_files: List of (file_path, source_name) for reference files.
        config: GoldenMatch configuration.
        output_matched: Whether to output matched records.
        output_unmatched: Whether to output unmatched records.
        output_scores: Whether to output score details.
        output_report: Whether to generate a report.
        match_mode: "best" (top score per target) or "all" (all matches).

    Returns:
        Dict with keys: matched, unmatched, report.
    """
    matchkeys = config.get_matchkeys()

    # ── Step 1: Load target ──
    if len(target_file) == 3:
        target_path, target_source, target_col_map = target_file
    else:
        target_path, target_source = target_file[0], target_file[1]
        target_col_map = None
    target_lf = load_file(target_path)
    if target_col_map:
        target_lf = apply_column_map(target_lf, target_col_map)
    target_lf = target_lf.with_columns(pl.lit(target_source).alias("__source__"))
    target_lf = _add_row_ids(target_lf, offset=0)
    target_df = target_lf.collect()
    target_ids = set(target_df["__row_id__"].to_list())
    offset = len(target_df)

    # ── Step 2: Load references ──
    ref_frames = []
    ref_sources = set()
    for ref_spec in reference_files:
        if len(ref_spec) == 3:
            ref_path, ref_source, ref_col_map = ref_spec
        else:
            ref_path, ref_source = ref_spec[0], ref_spec[1]
            ref_col_map = None
        ref_lf = load_file(ref_path)
        if ref_col_map:
            ref_lf = apply_column_map(ref_lf, ref_col_map)
        ref_lf = ref_lf.with_columns(pl.lit(ref_source).alias("__source__"))
        ref_lf = _add_row_ids(ref_lf, offset=offset)
        ref_df = ref_lf.collect()
        offset += len(ref_df)
        ref_frames.append(ref_df)
        ref_sources.add(ref_source)

    # Concat all
    all_frames = [target_df] + ref_frames
    combined_df = pl.concat(all_frames)
    combined_lf = combined_df.lazy()

    return _run_match_pipeline(
        combined_lf, config, matchkeys, target_ids,
        output_matched, output_unmatched, output_scores,
        output_report, match_mode,
    )


def _run_match_pipeline(
    combined_lf: pl.LazyFrame,
    config: GoldenMatchConfig,
    matchkeys: list,
    target_ids: set,
    output_matched: bool = False,
    output_unmatched: bool = False,
    output_scores: bool = False,
    output_report: bool = False,
    match_mode: str = "best",
) -> dict:
    """Shared match pipeline logic (post-ingest).

    This function contains all pipeline steps from auto-fix/validation through
    output. Both run_match() and run_match_df() delegate to this function.
    """
    # ── Step 2.5a: AUTO-FIX + VALIDATION ──
    if config.validation and config.validation.auto_fix:
        combined_df_tmp = combined_lf.collect()
        combined_df_tmp, fix_log = auto_fix_dataframe(combined_df_tmp)
        logger.info("Auto-fix applied: %d fix type(s)", len(fix_log))
        combined_lf = combined_df_tmp.lazy()

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
        valid_df, quarantine_df_match, val_report = validate_dataframe(combined_df_tmp, rules)
        logger.info("Validation: %d quarantined rows", quarantine_df_match.height)
        combined_lf = valid_df.lazy()
    else:
        quarantine_df_match = None

    # ── Step 2.5b: STANDARDIZE ──
    if config.standardization and config.standardization.rules:
        combined_lf = apply_standardization(combined_lf, config.standardization.rules)

    # ── Step 3: Compute matchkeys ──
    combined_lf = compute_matchkeys(combined_lf, matchkeys)
    combined_df = combined_lf.collect()

    # ── Step 3.5: AUTO-SUGGEST blocking keys ──
    _run_auto_suggest(combined_df, config)

    # Build source lookup
    source_lookup = {}
    for row in combined_df.select("__row_id__", "__source__").to_dicts():
        source_lookup[row["__row_id__"]] = row["__source__"]

    # ── Step 4: Find matches (cascading: exact first, then fuzzy) ──
    all_pairs: list[tuple[int, int, float]] = []
    matched_pairs: set[tuple[int, int]] = set()

    # Phase 1: Exact matchkeys (fast)
    for mk in matchkeys:
        if mk.type == "exact":
            pairs = find_exact_matches(combined_lf, mk)
            # Filter to cross target/ref pairs only
            pairs = [
                (a, b, s) for a, b, s in pairs
                if (a in target_ids) != (b in target_ids)
            ]
            all_pairs.extend(pairs)
            for a, b, s in pairs:
                matched_pairs.add((min(a, b), max(a, b)))

    # Phase 2: Fuzzy matchkeys (parallel block scoring)
    for mk in matchkeys:
        if mk.type == "weighted":
            if config.blocking is None:
                continue
            blocks = build_blocks(combined_lf, config.blocking)
            block_scorer = _get_block_scorer(config)
            pairs = block_scorer(
                blocks, mk, matched_pairs,
                target_ids=target_ids,
            )
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
            from goldenmatch.core.probabilistic import train_em, score_probabilistic
            em_result = train_em(
                combined_df, mk,
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

    # ── Step 4.5: CROSS-ENCODER RERANKING (optional) ──
    for mk in matchkeys:
        if mk.type == "weighted" and mk.rerank:
            all_pairs = rerank_top_pairs(all_pairs, combined_df, mk)
            break

    # ── Step 4.6: LLM SCORER (optional) ──
    if config.llm_scorer and config.llm_scorer.enabled and all_pairs:
        if config.llm_scorer.mode == "cluster":
            from goldenmatch.core.llm_cluster import llm_cluster_pairs
            all_pairs = llm_cluster_pairs(all_pairs, combined_df, config=config.llm_scorer)
        else:
            from goldenmatch.core.llm_scorer import llm_score_pairs
            all_pairs = llm_score_pairs(all_pairs, combined_df, config=config.llm_scorer)
        all_pairs = [(a, b, s) for a, b, s in all_pairs if s > 0.5]

    # ── Step 5: Normalize pairs so target ID is always first ──
    normalized: list[tuple[int, int, float]] = []
    for a, b, score in all_pairs:
        if a in target_ids:
            normalized.append((a, b, score))
        else:
            normalized.append((b, a, score))

    # ── Step 6: Group by target, apply match_mode ──
    target_matches: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for target_id, ref_id, score in normalized:
        target_matches[target_id].append((ref_id, score))

    if match_mode == "best":
        # Keep only highest score per target
        for tid in target_matches:
            matches = target_matches[tid]
            best = max(matches, key=lambda x: x[1])
            target_matches[tid] = [best]

    # ── Step 7: Build output ──
    matched_rows = []
    all_scores = []
    for target_id, matches in target_matches.items():
        target_row = combined_df.filter(pl.col("__row_id__") == target_id).to_dicts()[0]
        for ref_id, score in matches:
            ref_row = combined_df.filter(pl.col("__row_id__") == ref_id).to_dicts()[0]
            row = {"__target_row_id__": target_id, "__ref_row_id__": ref_id, "__match_score__": score}
            # Add target fields with target_ prefix
            for col, val in target_row.items():
                if not col.startswith("__"):
                    row[f"target_{col}"] = val
            # Add ref fields with ref_ prefix
            for col, val in ref_row.items():
                if not col.startswith("__"):
                    row[f"ref_{col}"] = val
            matched_rows.append(row)
            all_scores.append(score)

    matched_df = pl.DataFrame(matched_rows) if matched_rows else None

    # Unmatched targets
    matched_target_ids = set(target_matches.keys())
    unmatched_ids = target_ids - matched_target_ids
    unmatched_df = combined_df.filter(pl.col("__row_id__").is_in(list(unmatched_ids)))

    # Report
    report = None
    if output_report:
        report = generate_match_report(
            total_targets=len(target_ids),
            matched=len(matched_target_ids),
            unmatched=len(unmatched_ids),
            scores=all_scores,
        )

    # Write outputs
    run_name = config.output.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = config.output.format or "csv"
    directory = config.output.directory or config.output.path or "."

    if output_matched and matched_df is not None:
        write_output(matched_df, directory, run_name, "matched", fmt)

    if output_unmatched and len(unmatched_df) > 0:
        write_output(unmatched_df, directory, run_name, "unmatched", fmt)

    if output_scores and matched_df is not None:
        write_output(matched_df, directory, run_name, "scores", fmt)

    return {
        "matched": matched_df,
        "unmatched": unmatched_df,
        "report": report,
        "quarantine": quarantine_df_match,
    }


def run_match_df(
    target_df: pl.DataFrame,
    reference_df: pl.DataFrame,
    config: GoldenMatchConfig,
    target_name: str = "target",
    reference_name: str = "reference",
) -> dict:
    """Run match pipeline on DataFrames directly (no file I/O)."""
    matchkeys = config.get_matchkeys()
    required = _get_required_columns(config)
    validate_columns(target_df.lazy(), required)
    validate_columns(reference_df.lazy(), required)

    target_lf = target_df.lazy()
    target_lf = target_lf.with_columns(pl.lit(target_name).alias("__source__"))
    target_lf = _add_row_ids(target_lf, offset=0)
    target_collected = target_lf.collect()
    target_ids = set(target_collected["__row_id__"].to_list())

    ref_lf = reference_df.lazy()
    ref_lf = ref_lf.with_columns(pl.lit(reference_name).alias("__source__"))
    ref_lf = _add_row_ids(ref_lf, offset=len(target_collected))
    ref_collected = ref_lf.collect()

    combined_lf = pl.concat([target_collected, ref_collected]).lazy()

    return _run_match_pipeline(combined_lf, config, matchkeys, target_ids)
