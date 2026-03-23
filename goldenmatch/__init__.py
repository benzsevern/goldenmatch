"""GoldenMatch -- entity resolution toolkit.

Quick start:
    import goldenmatch as gm

    # Deduplicate a CSV
    result = gm.dedupe("customers.csv", exact=["email"], fuzzy={"name": 0.85})
    result.golden.write_csv("deduped.csv")

    # Match across files
    result = gm.match("targets.csv", "reference.csv", fuzzy={"name": 0.85})

    # Privacy-preserving linkage
    result = gm.pprl_link("hospital_a.csv", "hospital_b.csv", fields=["name", "dob", "zip"])

    # Evaluate accuracy
    metrics = gm.evaluate("data.csv", config="config.yaml", ground_truth="gt.csv")

    # Streaming single-record matching
    matches = gm.match_one(record, df, matchkey)

    # Domain extraction
    rulebooks = gm.discover_rulebooks()

    # Explain a match
    explanation = gm.explain_pair(record_a, record_b, matchkey)

All features are accessible via `import goldenmatch as gm`.
"""
__version__ = "1.0.0"

# ── High-level API (convenience functions) ────────────────────────────────
from goldenmatch._api import (
    dedupe,
    match,
    pprl_link,
    evaluate,
    load_config,
    DedupeResult,
    MatchResult,
)

# ── Config schemas (for building configs programmatically) ────────────────
from goldenmatch.config.schemas import (
    GoldenMatchConfig,
    MatchkeyConfig,
    MatchkeyField,
    BlockingConfig,
    BlockingKeyConfig,
    GoldenRulesConfig,
    GoldenFieldRule,
    LLMScorerConfig,
    BudgetConfig,
    DomainConfig,
    StandardizationConfig,
    ValidationConfig,
    OutputConfig,
)

# ── Core pipeline functions ───────────────────────────────────────────────
from goldenmatch.core.pipeline import run_dedupe, run_match
from goldenmatch.core.scorer import (
    find_exact_matches,
    find_fuzzy_matches,
    score_pair,
    score_blocks_parallel,
)
from goldenmatch.core.cluster import (
    build_clusters,
    add_to_cluster,
    unmerge_record,
    unmerge_cluster,
    compute_cluster_confidence,
)
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.golden import build_golden_record
from goldenmatch.core.ingest import load_file, load_files
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys

# ── Streaming / incremental ──────────────────────────────────────────────
from goldenmatch.core.match_one import match_one
from goldenmatch.core.streaming import StreamProcessor, run_stream

# ── Evaluation ───────────────────────────────────────────────────────────
from goldenmatch.core.evaluate import (
    evaluate_pairs,
    evaluate_clusters,
    load_ground_truth_csv,
    EvalResult,
)

# ── Explainability ───────────────────────────────────────────────────────
from goldenmatch.core.explain import explain_pair_nl, explain_cluster_nl

# ── Domain extraction ────────────────────────────────────────────────────
from goldenmatch.core.domain_registry import (
    discover_rulebooks,
    load_rulebook,
    save_rulebook,
    match_domain,
    extract_with_rulebook,
    DomainRulebook,
)

# ── Probabilistic (Fellegi-Sunter) ───────────────────────────────────────
from goldenmatch.core.probabilistic import train_em, score_probabilistic

# ── Learned blocking ─────────────────────────────────────────────────────
from goldenmatch.core.learned_blocking import learn_blocking_rules, apply_learned_blocks

# ── LLM scoring ──────────────────────────────────────────────────────────
from goldenmatch.core.llm_scorer import llm_score_pairs
from goldenmatch.core.llm_cluster import llm_cluster_pairs
from goldenmatch.core.llm_budget import BudgetTracker
from goldenmatch.core.llm_labeler import label_pairs as llm_label_pairs
from goldenmatch.core.llm_extract import llm_extract_features

# ── PPRL ─────────────────────────────────────────────────────────────────
from goldenmatch.pprl.protocol import (
    PPRLConfig,
    run_pprl,
    compute_bloom_filters,
    link_trusted_third_party,
    link_smc,
    PartyData,
    LinkageResult,
)
from goldenmatch.pprl.autoconfig import (
    auto_configure_pprl,
    auto_configure_pprl_llm,
    profile_for_pprl,
)

# ── Profiling ────────────────────────────────────────────────────────────
from goldenmatch.core.profiler import profile_dataframe

# ── Lineage ──────────────────────────────────────────────────────────────
from goldenmatch.core.lineage import build_lineage, save_lineage

# ── Active learning / boost ──────────────────────────────────────────────
from goldenmatch.core.boost import boost_accuracy

# ── Auto-configuration ──────────────────────────────────────────────────
from goldenmatch.core.autoconfig import auto_configure
from goldenmatch.core.threshold import suggest_threshold

# ── Data quality ─────────────────────────────────────────────────────────
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.validate import validate_dataframe
from goldenmatch.core.anomaly import detect_anomalies

# ── Schema matching ──────────────────────────────────────────────────────
from goldenmatch.core.schema_match import auto_map_columns

# ── Graph ER ─────────────────────────────────────────────────────────────
from goldenmatch.core.graph_er import run_graph_er

# ── Reranking ────────────────────────────────────────────────────────────
from goldenmatch.core.scorer import rerank_top_pairs

# ── Diff / Rollback ─────────────────────────────────────────────────────
from goldenmatch.core.diff import generate_diff
from goldenmatch.core.rollback import rollback_run

# ── Output ───────────────────────────────────────────────────────────────
from goldenmatch.output.writer import write_output
from goldenmatch.output.report import generate_dedupe_report

# ── REST API Client ──────────────────────────────────────────────────────
from goldenmatch.client import Client

# ── Shortcuts ────────────────────────────────────────────────────────────
explain_pair = explain_pair_nl
explain_cluster = explain_cluster_nl
pprl_auto_config = auto_configure_pprl

__all__ = [
    # Version
    "__version__",
    # High-level API
    "dedupe", "match", "pprl_link", "evaluate", "load_config",
    "DedupeResult", "MatchResult",
    # Config
    "GoldenMatchConfig", "MatchkeyConfig", "MatchkeyField",
    "BlockingConfig", "BlockingKeyConfig",
    "GoldenRulesConfig", "GoldenFieldRule",
    "LLMScorerConfig", "BudgetConfig",
    "DomainConfig", "StandardizationConfig", "ValidationConfig", "OutputConfig",
    # Pipeline
    "run_dedupe", "run_match",
    "find_exact_matches", "find_fuzzy_matches", "score_pair", "score_blocks_parallel",
    "build_clusters", "add_to_cluster", "unmerge_record", "unmerge_cluster",
    "compute_cluster_confidence",
    "build_blocks", "build_golden_record",
    "load_file", "load_files",
    "apply_standardization", "compute_matchkeys",
    # Streaming
    "match_one", "StreamProcessor", "run_stream",
    # Evaluation
    "evaluate_pairs", "evaluate_clusters", "load_ground_truth_csv", "EvalResult",
    # Explain
    "explain_pair", "explain_pair_nl", "explain_cluster", "explain_cluster_nl",
    # Domain
    "discover_rulebooks", "load_rulebook", "save_rulebook",
    "match_domain", "extract_with_rulebook", "DomainRulebook",
    # Probabilistic
    "train_em", "score_probabilistic",
    # Learned blocking
    "learn_blocking_rules", "apply_learned_blocks",
    # LLM
    "llm_score_pairs", "llm_cluster_pairs", "BudgetTracker",
    "llm_label_pairs", "llm_extract_features",
    # PPRL
    "PPRLConfig", "run_pprl", "compute_bloom_filters",
    "link_trusted_third_party", "link_smc",
    "PartyData", "LinkageResult",
    "auto_configure_pprl", "auto_configure_pprl_llm", "profile_for_pprl",
    "pprl_auto_config",
    # Profiling
    "profile_dataframe",
    # Lineage
    "build_lineage", "save_lineage",
    # Active learning / boost
    "boost_accuracy",
    # Auto-configuration
    "auto_configure", "suggest_threshold",
    # Data quality
    "auto_fix_dataframe", "validate_dataframe", "detect_anomalies",
    # Schema matching
    "auto_map_columns",
    # Graph ER
    "run_graph_er",
    # Reranking
    "rerank_top_pairs",
    # Diff / Rollback
    "generate_diff", "rollback_run",
    # Output
    "write_output", "generate_dedupe_report",
    # REST API Client
    "Client",
]
