"""Agent-level MCP tools for autonomous entity resolution.

Each tool creates its own AgentSession (no shared global state),
delegates to the appropriate AgentSession method, and returns
results as JSON in TextContent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

AGENT_TOOLS = [
    Tool(
        name="analyze_data",
        description="Profile data, detect domain, recommend ER strategy",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="auto_configure",
        description="Generate optimal matching config from data analysis",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "constraints": {"type": "object"},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="agent_deduplicate",
        description="Run full ER pipeline with confidence gating and reasoning",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="agent_match_sources",
        description="Match two files with intelligent strategy selection",
        inputSchema={
            "type": "object",
            "properties": {
                "file_a": {"type": "string"},
                "file_b": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["file_a", "file_b"],
        },
    ),
    Tool(
        name="agent_explain_pair",
        description="Natural language explanation for a record pair",
        inputSchema={
            "type": "object",
            "properties": {
                "record_a": {"type": "object"},
                "record_b": {"type": "object"},
                "fuzzy": {"type": "object"},
                "exact": {"type": "array"},
            },
            "required": ["record_a", "record_b"],
        },
    ),
    Tool(
        name="agent_explain_cluster",
        description="Explain why records are in the same cluster",
        inputSchema={
            "type": "object",
            "properties": {
                "cluster_id": {"type": "integer"},
            },
            "required": ["cluster_id"],
        },
    ),
    Tool(
        name="agent_review_queue",
        description="Get borderline pairs awaiting approval",
        inputSchema={
            "type": "object",
            "properties": {
                "job_name": {"type": "string"},
            },
            "required": ["job_name"],
        },
    ),
    Tool(
        name="agent_approve_reject",
        description="Approve or reject a review queue pair",
        inputSchema={
            "type": "object",
            "properties": {
                "job_name": {"type": "string"},
                "id_a": {"type": "integer"},
                "id_b": {"type": "integer"},
                "decision": {"type": "string"},
                "decided_by": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["job_name", "id_a", "id_b", "decision", "decided_by"],
        },
    ),
    Tool(
        name="agent_compare_strategies",
        description="Compare ER strategies on your data",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "ground_truth": {"type": "string"},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="suggest_pprl",
        description="Check if data needs privacy-preserving matching",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="scan_quality",
        description=(
            "Run GoldenCheck data quality scan on a CSV file. "
            "Returns issues found (encoding errors, Unicode problems, format violations) "
            "without applying fixes. Requires goldencheck: pip install goldenmatch[quality]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file to scan",
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain hint (healthcare, finance, ecommerce)",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="fix_quality",
        description=(
            "Run GoldenCheck scan and apply fixes to a CSV file. "
            "Returns the fixed data summary and a manifest of all fixes applied. "
            "Requires goldencheck: pip install goldenmatch[quality]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file to fix",
                },
                "fix_mode": {
                    "type": "string",
                    "enum": ["safe", "moderate"],
                    "description": "Fix aggressiveness: safe (conservative) or moderate (balanced). Default: safe",
                    "default": "safe",
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain hint (healthcare, finance, ecommerce)",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save the fixed CSV. If omitted, returns summary only.",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="run_transforms",
        description=(
            "Run GoldenFlow data transforms on a CSV file. "
            "Normalizes phone numbers (E.164), dates (ISO), categorical spelling, "
            "and Unicode issues. Returns a manifest of transforms applied. "
            "Requires goldenflow: pip install goldenmatch[transform]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file to transform",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional path to save the transformed CSV. If omitted, returns summary only.",
                },
            },
            "required": ["file_path"],
        },
    ),
]

_AGENT_TOOL_NAMES = frozenset(t.name for t in AGENT_TOOLS)


def _serialize_result(result: Any) -> dict:
    """Convert pipeline result objects to JSON-safe dicts."""
    if hasattr(result, "clusters") and hasattr(result, "stats"):
        # DedupeResult / MatchResult
        clusters = result.clusters or {}
        multi = sum(1 for c in clusters.values() if c.get("size", 0) > 1)
        total_matched = sum(
            c.get("size", 0) for c in clusters.values() if c.get("size", 0) > 1
        )
        stats = result.stats if isinstance(result.stats, dict) else {}
        return {
            "total_records": stats.get("total_records", 0),
            "total_clusters": multi,
            "total_matched_records": total_matched,
            "match_rate": stats.get("match_rate", 0.0),
            "scored_pairs": len(result.scored_pairs) if result.scored_pairs else 0,
        }
    if isinstance(result, dict):
        return result
    return {"value": str(result)}


def handle_agent_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route an agent-level MCP tool call to the appropriate handler.

    Creates a fresh AgentSession per call (stateless).
    Returns results as JSON in TextContent.
    """
    from goldenmatch.core.agent import AgentSession

    try:
        result = _dispatch(name, arguments, AgentSession)
        return [TextContent(
            type="text",
            text=json.dumps(result, default=str, indent=2),
        )]
    except Exception as exc:
        logger.exception("Agent tool %s failed", name)
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(exc)}),
        )]


def _dispatch(name: str, args: dict, session_cls: type) -> dict:
    """Dispatch to the appropriate handler by tool name."""

    if name == "analyze_data":
        session = session_cls()
        return session.analyze(args["file_path"])

    if name == "auto_configure":
        session = session_cls()
        analysis = session.analyze(args["file_path"])
        # Build config from the analysis
        from goldenmatch.core.agent import _decision_to_config, select_strategy, profile_for_agent
        profile = profile_for_agent(session.data)
        decision = select_strategy(profile)
        config = _decision_to_config(decision)
        # Serialize config to dict
        config_dict = config.model_dump() if hasattr(config, "model_dump") else {}
        return {
            "analysis": analysis,
            "config": config_dict,
        }

    if name == "agent_deduplicate":
        session = session_cls()
        config_arg = args.get("config")
        raw = session.deduplicate(args["file_path"], config=config_arg)
        return {
            "reasoning": raw.get("reasoning", {}),
            "confidence_distribution": raw.get("confidence_distribution", {}),
            "storage": raw.get("storage", "memory"),
            "results": _serialize_result(raw.get("results")),
        }

    if name == "agent_match_sources":
        session = session_cls()
        config_arg = args.get("config")
        raw = session.match_sources(args["file_a"], args["file_b"], config=config_arg)
        return {
            "reasoning": raw.get("reasoning", {}),
            "results": _serialize_result(raw.get("results")),
        }

    if name == "agent_explain_pair":
        from goldenmatch._api import explain_pair_df
        fuzzy = args.get("fuzzy")
        exact = args.get("exact")
        explanation = explain_pair_df(
            args["record_a"],
            args["record_b"],
            fuzzy=fuzzy,
            exact=exact,
        )
        return {"explanation": explanation}

    if name == "agent_explain_cluster":
        from goldenmatch.core.explain import explain_cluster_nl
        cluster_id = args["cluster_id"]
        # With no global state, return a descriptive message
        return {
            "cluster_id": cluster_id,
            "note": (
                "agent_explain_cluster requires a prior agent_deduplicate call. "
                "Each MCP tool call is stateless; run agent_deduplicate first, "
                "then inspect the clusters dict directly."
            ),
        }

    if name == "agent_review_queue":
        session = session_cls()
        job_name = args["job_name"]
        pending = session.review_queue.list_pending(job_name)
        return {
            "job_name": job_name,
            "pending": [
                {
                    "id_a": item.id_a,
                    "id_b": item.id_b,
                    "score": item.score,
                    "explanation": item.explanation,
                }
                for item in pending
            ],
            "count": len(pending),
        }

    if name == "agent_approve_reject":
        session = session_cls()
        job_name = args["job_name"]
        decision = args["decision"]
        decided_by = args["decided_by"]
        reason = args.get("reason", "")

        if decision == "approve":
            session.review_queue.approve(
                job_name, args["id_a"], args["id_b"], decided_by,
            )
        elif decision == "reject":
            session.review_queue.reject(
                job_name, args["id_a"], args["id_b"], decided_by, reason,
            )
        else:
            return {"error": f"Invalid decision: {decision!r}. Use 'approve' or 'reject'."}

        return {
            "status": "ok",
            "decision": decision,
            "job_name": job_name,
            "id_a": args["id_a"],
            "id_b": args["id_b"],
            "decided_by": decided_by,
        }

    if name == "agent_compare_strategies":
        session = session_cls()
        ground_truth = args.get("ground_truth")
        return session.compare_strategies(args["file_path"], ground_truth)

    if name == "suggest_pprl":
        session = session_cls()
        analysis = session.analyze(args["file_path"])
        needs_pprl = analysis.get("strategy") == "pprl"
        return {
            "needs_pprl": needs_pprl,
            "strategy": analysis.get("strategy"),
            "why": analysis.get("why"),
            "has_sensitive": analysis.get("profile", {}).get("has_sensitive", False),
            "recommendation": (
                "Use PPRL (privacy-preserving record linkage) for this data."
                if needs_pprl
                else "Standard matching is safe for this data. PPRL is optional."
            ),
        }

    if name == "scan_quality":
        import polars as pl
        from goldenmatch.core.quality import _goldencheck_available, run_quality_check
        from goldenmatch.config.schemas import QualityConfig

        if not _goldencheck_available():
            return {
                "error": "goldencheck is not installed. Install with: pip install goldenmatch[quality]",
            }

        df = pl.read_csv(args["file_path"], encoding="utf8-lossy", ignore_errors=True)
        qc = QualityConfig(mode="silent", fix_mode="none", domain=args.get("domain"))
        _, _ = run_quality_check(df, qc)

        # Re-run scan to capture findings for the response
        from goldencheck.engine.scanner import scan_file
        from goldencheck.engine.confidence import apply_confidence_downgrade
        import tempfile
        from pathlib import Path as _Path

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            df.write_csv(tmp.name)
            tmp_path = _Path(tmp.name)
        try:
            findings, _ = scan_file(tmp_path, domain=args.get("domain"))
            findings = apply_confidence_downgrade(findings, llm_boost=False)
        finally:
            tmp_path.unlink(missing_ok=True)

        issues = []
        for f in findings:
            issues.append({
                "rule": f.rule_id,
                "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                "column": f.column,
                "message": f.message,
                "rows_affected": f.rows_affected,
                "confidence": round(f.confidence, 2) if hasattr(f, "confidence") else None,
            })

        return {
            "file": args["file_path"],
            "total_records": df.height,
            "issues_found": len(issues),
            "issues": issues,
        }

    if name == "fix_quality":
        import polars as pl
        from goldenmatch.core.quality import _goldencheck_available, run_quality_check
        from goldenmatch.config.schemas import QualityConfig

        if not _goldencheck_available():
            return {
                "error": "goldencheck is not installed. Install with: pip install goldenmatch[quality]",
            }

        df = pl.read_csv(args["file_path"], encoding="utf8-lossy", ignore_errors=True)
        fix_mode = args.get("fix_mode", "safe")
        domain = args.get("domain")
        qc = QualityConfig(mode="silent", fix_mode=fix_mode, domain=domain)
        fixed_df, fixes = run_quality_check(df, qc)

        output_path = args.get("output_path")
        if output_path:
            fixed_df.write_csv(output_path)

        return {
            "file": args["file_path"],
            "fix_mode": fix_mode,
            "total_records": fixed_df.height,
            "fixes_applied": len(fixes),
            "fixes": fixes,
            "output_path": output_path,
        }

    if name == "run_transforms":
        import polars as pl
        from goldenmatch.core.transform import _goldenflow_available, run_transform
        from goldenmatch.config.schemas import TransformConfig

        if not _goldenflow_available():
            return {
                "error": "goldenflow is not installed. Install with: pip install goldenmatch[transform]",
            }

        df = pl.read_csv(args["file_path"], encoding="utf8-lossy", ignore_errors=True)
        tc = TransformConfig(mode="silent")
        transformed_df, fixes = run_transform(df, tc)

        output_path = args.get("output_path")
        if output_path:
            transformed_df.write_csv(output_path)

        return {
            "file": args["file_path"],
            "total_records": transformed_df.height,
            "transforms_applied": len(fixes),
            "transforms": fixes,
            "output_path": output_path,
        }

    return {"error": f"Unknown agent tool: {name}"}
