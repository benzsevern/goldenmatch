"""GoldenMatch MCP Server — tools for entity resolution via Claude Desktop.

Usage:
    goldenmatch mcp-serve --file customers.csv --config config.yaml

Or add to Claude Desktop config (claude_desktop_config.json):
    {
        "mcpServers": {
            "goldenmatch": {
                "command": "goldenmatch",
                "args": ["mcp-serve", "--file", "customers.csv"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from goldenmatch.mcp.agent_tools import AGENT_TOOLS, handle_agent_tool

logger = logging.getLogger(__name__)

_AGENT_TOOL_NAMES = frozenset(t.name for t in AGENT_TOOLS)

# Global state
_engine = None
_config = None
_result = None
_rows: list[dict] = []
_id_to_idx: dict[int, int] = {}


def _initialize(file_paths: list[str], config_path: str | None = None) -> None:
    """Load data and run initial matching."""
    global _engine, _config, _result, _rows, _id_to_idx

    from goldenmatch.tui.engine import MatchEngine

    _engine = MatchEngine(file_paths)
    logger.info("Loaded %d records from %d files", _engine.row_count, len(file_paths))

    if config_path:
        from goldenmatch.config.loader import load_config
        _config = load_config(config_path)
    else:
        from goldenmatch.core.autoconfig import auto_configure
        parsed = [(f, Path(f).stem) for f in file_paths]
        _config = auto_configure(parsed)
        logger.info("Auto-configured matching rules")

    _result = _engine.run_full(_config)
    _rows = _engine.data.to_dicts()
    _id_to_idx = {row["__row_id__"]: i for i, row in enumerate(_rows)}
    logger.info(
        "Matching complete: %d clusters, %.1f%% match rate",
        _result.stats.total_clusters, _result.stats.match_rate,
    )


def create_server(file_paths: list[str], config_path: str | None = None) -> Server:
    """Create and configure the MCP server."""

    _initialize(file_paths, config_path)

    server = Server("goldenmatch")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return AGENT_TOOLS + [
            Tool(
                name="get_stats",
                description="Get dataset statistics: record count, cluster count, match rate, cluster sizes.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="find_duplicates",
                description="Find duplicate matches for a record. Provide field values to search against the loaded dataset.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "record": {
                            "type": "object",
                            "description": "Record fields to match (e.g. {\"name\": \"John Smith\", \"zip\": \"10001\"})",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results to return (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["record"],
                },
            ),
            Tool(
                name="explain_match",
                description="Explain why two records match or don't match. Shows per-field score breakdown.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "record_a": {
                            "type": "object",
                            "description": "First record fields",
                        },
                        "record_b": {
                            "type": "object",
                            "description": "Second record fields",
                        },
                    },
                    "required": ["record_a", "record_b"],
                },
            ),
            Tool(
                name="list_clusters",
                description="List duplicate clusters found in the dataset. Returns cluster IDs, sizes, and member counts.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "min_size": {
                            "type": "integer",
                            "description": "Minimum cluster size to include (default 2)",
                            "default": 2,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max clusters to return (default 20)",
                            "default": 20,
                        },
                    },
                },
            ),
            Tool(
                name="get_cluster",
                description="Get details of a specific cluster: all member records and their field values.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "integer",
                            "description": "Cluster ID to look up",
                        },
                    },
                    "required": ["cluster_id"],
                },
            ),
            Tool(
                name="get_golden_record",
                description="Get the merged golden (canonical) record for a cluster.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "integer",
                            "description": "Cluster ID",
                        },
                    },
                    "required": ["cluster_id"],
                },
            ),
            Tool(
                name="match_record",
                description=(
                    "Match a single record against the loaded dataset in real-time. "
                    "Paste a record's fields and instantly see if it matches any existing record. "
                    "Uses the configured matchkeys, scorers, and thresholds. "
                    "Example: {\"name\": \"John Smith\", \"email\": \"john@test.com\", \"zip\": \"10001\"}"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "record": {
                            "type": "object",
                            "description": "Record fields to match against the dataset",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum score to consider a match (default: use config threshold)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max matches to return (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["record"],
                },
            ),
            Tool(
                name="unmerge_record",
                description=(
                    "Remove a record from its cluster. The record becomes a singleton. "
                    "Remaining cluster members are re-clustered using stored pair scores. "
                    "Use this to fix bad merges."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "record_id": {
                            "type": "integer",
                            "description": "Row ID of the record to unmerge",
                        },
                    },
                    "required": ["record_id"],
                },
            ),
            Tool(
                name="shatter_cluster",
                description=(
                    "Break an entire cluster into individual records. "
                    "All members become singletons. Use when a cluster is completely wrong."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "integer",
                            "description": "Cluster ID to shatter",
                        },
                    },
                    "required": ["cluster_id"],
                },
            ),
            Tool(
                name="suggest_config",
                description=(
                    "Analyze bad merges and suggest config changes. "
                    "Provide examples of incorrect merges (pairs that should NOT have matched) "
                    "and GoldenMatch will identify which fields/thresholds to tighten. "
                    "Example: [{\"record_a\": {...}, \"record_b\": {...}, \"reason\": \"different people\"}]"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bad_merges": {
                            "type": "array",
                            "description": "List of bad merge examples with record_a, record_b, and optional reason",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "record_a": {"type": "object"},
                                    "record_b": {"type": "object"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["record_a", "record_b"],
                            },
                        },
                    },
                    "required": ["bad_merges"],
                },
            ),
            Tool(
                name="profile_data",
                description="Get data quality profile: column types, null rates, unique counts, sample values.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="export_results",
                description="Export matching results to a file (CSV or JSON).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_path": {
                            "type": "string",
                            "description": "File path to save results",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["csv", "json"],
                            "description": "Output format (default csv)",
                            "default": "csv",
                        },
                    },
                    "required": ["output_path"],
                },
            ),
            Tool(
                name="list_domains",
                description="List available domain extraction rulebooks (built-in + user-defined).",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="create_domain",
                description="Create a custom domain extraction rulebook. Define patterns for a specific data domain (medical devices, automotive parts, real estate, etc.).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Domain name (e.g. 'medical_devices', 'automotive_parts')",
                        },
                        "signals": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Column name keywords that trigger this domain (e.g. ['ndc', 'fda', 'implant'])",
                        },
                        "identifier_patterns": {
                            "type": "object",
                            "description": "Named regex patterns for domain identifiers (e.g. {'ndc': '\\\\b(\\\\d{5}-\\\\d{4}-\\\\d{2})\\\\b'})",
                        },
                        "brand_patterns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Brand/manufacturer names to extract (e.g. ['Medtronic', 'Abbott'])",
                        },
                        "attribute_patterns": {
                            "type": "object",
                            "description": "Named regex patterns for domain attributes (e.g. {'size': '\\\\b(\\\\d+mm)\\\\b'})",
                        },
                        "stop_words": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Words to strip during name normalization",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["local", "global"],
                            "description": "Save locally (.goldenmatch/domains/) or globally (~/.goldenmatch/domains/). Default: local.",
                            "default": "local",
                        },
                    },
                    "required": ["name", "signals"],
                },
            ),
            Tool(
                name="test_domain",
                description="Test a domain extraction rulebook against sample records. Shows what features would be extracted from the loaded data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "domain_name": {
                            "type": "string",
                            "description": "Name of the domain rulebook to test",
                        },
                        "sample_size": {
                            "type": "integer",
                            "description": "Number of records to test (default 10)",
                            "default": 10,
                        },
                    },
                    "required": ["domain_name"],
                },
            ),
            Tool(
                name="pprl_auto_config",
                description=(
                    "Analyze the loaded dataset and recommend optimal PPRL (privacy-preserving record linkage) configuration. "
                    "Returns recommended fields, bloom filter parameters, threshold, and explanation."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "security_level": {
                            "type": "string",
                            "enum": ["standard", "high", "paranoid"],
                            "description": "Security level (default: high)",
                            "default": "high",
                        },
                        "use_llm": {
                            "type": "boolean",
                            "description": "Use LLM for enhanced recommendations (requires API key)",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="pprl_link",
                description=(
                    "Run privacy-preserving record linkage between two parties' data. "
                    "Computes bloom filters, matches records without sharing raw data. "
                    "Specify fields, threshold, and security level."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_a": {
                            "type": "string",
                            "description": "Path to party A's CSV file",
                        },
                        "file_b": {
                            "type": "string",
                            "description": "Path to party B's CSV file",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Field names to match on (e.g. ['first_name', 'last_name', 'zip_code'])",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Match threshold (default: auto-detected)",
                        },
                        "security_level": {
                            "type": "string",
                            "enum": ["standard", "high", "paranoid"],
                            "default": "high",
                        },
                    },
                    "required": ["file_a", "file_b", "fields"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        # Delegate agent-level tools to the agent handler
        if name in _AGENT_TOOL_NAMES:
            return handle_agent_tool(name, arguments)
        try:
            result = _handle_tool(name, arguments)
            return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _handle_tool(name: str, args: dict) -> dict:
    """Dispatch tool calls."""
    if name == "get_stats":
        return _tool_get_stats()
    elif name == "find_duplicates":
        return _tool_find_duplicates(args.get("record", {}), args.get("top_k", 5))
    elif name == "explain_match":
        return _tool_explain_match(args.get("record_a", {}), args.get("record_b", {}))
    elif name == "list_clusters":
        return _tool_list_clusters(args.get("min_size", 2), args.get("limit", 20))
    elif name == "get_cluster":
        return _tool_get_cluster(args["cluster_id"])
    elif name == "get_golden_record":
        return _tool_get_golden_record(args["cluster_id"])
    elif name == "match_record":
        return _tool_match_record(args.get("record", {}), args.get("threshold"), args.get("top_k", 5))
    elif name == "unmerge_record":
        return _tool_unmerge_record(args["record_id"])
    elif name == "shatter_cluster":
        return _tool_shatter_cluster(args["cluster_id"])
    elif name == "suggest_config":
        return _tool_suggest_config(args.get("bad_merges", []))
    elif name == "profile_data":
        return _tool_profile_data()
    elif name == "export_results":
        return _tool_export_results(args["output_path"], args.get("format", "csv"))
    elif name == "list_domains":
        return _tool_list_domains()
    elif name == "create_domain":
        return _tool_create_domain(args)
    elif name == "test_domain":
        return _tool_test_domain(args.get("domain_name", ""), args.get("sample_size", 10))
    elif name == "pprl_auto_config":
        return _tool_pprl_auto_config(args.get("security_level", "high"), args.get("use_llm", False))
    elif name == "pprl_link":
        return _tool_pprl_link(args)
    else:
        return {"error": f"Unknown tool: {name}"}


def _tool_get_stats() -> dict:
    s = _result.stats
    return {
        "total_records": s.total_records,
        "total_clusters": s.total_clusters,
        "singleton_count": s.singleton_count,
        "match_rate": round(s.match_rate, 2),
        "avg_cluster_size": round(s.avg_cluster_size, 2),
        "max_cluster_size": s.max_cluster_size,
        "total_pairs": len(_result.scored_pairs),
    }


def _tool_find_duplicates(record: dict, top_k: int) -> dict:
    from goldenmatch.core.explainer import explain_pair

    matchkeys = _config.get_matchkeys()
    results = []

    for mk in matchkeys:
        if mk.type != "weighted":
            continue
        for row in _rows:
            exp = explain_pair(record, row, mk.fields, mk.threshold or 0.80)
            if exp.is_match:
                clean = {k: v for k, v in row.items() if not k.startswith("__")}
                results.append({
                    "record": clean,
                    "score": round(exp.total_score, 4),
                    "top_contributor": exp.top_contributor,
                })

    results.sort(key=lambda x: -x["score"])
    return {"matches": results[:top_k], "count": min(len(results), top_k)}


def _tool_explain_match(record_a: dict, record_b: dict) -> dict:
    from goldenmatch.core.explainer import explain_pair

    matchkeys = _config.get_matchkeys()
    fields = []
    threshold = 0.80
    for mk in matchkeys:
        if mk.type == "weighted":
            fields = mk.fields
            threshold = mk.threshold or 0.80
            break

    exp = explain_pair(record_a, record_b, fields, threshold)
    return {
        "total_score": round(exp.total_score, 4),
        "threshold": exp.threshold,
        "is_match": exp.is_match,
        "top_contributor": exp.top_contributor,
        "weakest_field": exp.weakest_field,
        "fields": [
            {
                "field": f.field_name,
                "scorer": f.scorer,
                "value_a": f.value_a,
                "value_b": f.value_b,
                "score": round(f.score, 4),
                "weight": f.weight,
                "contribution": round(f.contribution, 4),
                "diff_type": f.diff_type,
            }
            for f in exp.fields
        ],
    }


def _tool_list_clusters(min_size: int, limit: int) -> dict:
    clusters = []
    for cid, info in _result.clusters.items():
        if info["size"] >= min_size:
            clusters.append({
                "cluster_id": cid,
                "size": info["size"],
                "oversized": info.get("oversized", False),
            })
    clusters.sort(key=lambda x: -x["size"])
    return {"clusters": clusters[:limit], "total": len(clusters)}


def _tool_get_cluster(cluster_id: int) -> dict:
    info = _result.clusters.get(cluster_id)
    if not info:
        return {"error": f"Cluster {cluster_id} not found"}

    members = []
    for mid in info["members"]:
        idx = _id_to_idx.get(mid)
        if idx is not None:
            clean = {k: v for k, v in _rows[idx].items() if not k.startswith("__")}
            members.append(clean)

    return {"cluster_id": cluster_id, "size": info["size"], "members": members}


def _tool_get_golden_record(cluster_id: int) -> dict:
    if _result.golden is None:
        return {"error": "No golden records available"}

    golden_rows = _result.golden.filter(
        _result.golden["__cluster_id__"] == cluster_id
    ) if "__cluster_id__" in _result.golden.columns else None

    if golden_rows is None or golden_rows.height == 0:
        return {"error": f"No golden record for cluster {cluster_id}"}

    row = golden_rows.to_dicts()[0]
    clean = {k: v for k, v in row.items() if not k.startswith("__")}
    return {"cluster_id": cluster_id, "golden_record": clean}


def _tool_match_record(record: dict, threshold: float | None, top_k: int) -> dict:
    """Match a single record against the dataset using match_one."""
    from goldenmatch.core.match_one import match_one

    matchkeys = _config.get_matchkeys()
    all_matches = []

    for mk in matchkeys:
        if mk.type != "weighted":
            continue
        t = threshold if threshold is not None else (mk.threshold or 0.80)
        # Temporarily override threshold if user specified one
        import copy
        mk_copy = copy.deepcopy(mk)
        mk_copy.threshold = t

        matches = match_one(record, _engine.data, mk_copy)
        for row_id, score in matches:
            idx = _id_to_idx.get(row_id)
            if idx is not None:
                clean = {k: v for k, v in _rows[idx].items() if not k.startswith("__")}
                all_matches.append({
                    "row_id": row_id,
                    "score": round(score, 4),
                    "record": clean,
                })

    # Deduplicate by row_id, keep highest score
    seen = {}
    for m in all_matches:
        rid = m["row_id"]
        if rid not in seen or m["score"] > seen[rid]["score"]:
            seen[rid] = m
    deduped = sorted(seen.values(), key=lambda x: -x["score"])[:top_k]

    return {
        "matches": deduped,
        "count": len(deduped),
        "input_record": record,
    }


def _tool_unmerge_record(record_id: int) -> dict:
    """Remove a record from its cluster."""
    global _result

    updated = _engine.unmerge_record(record_id)
    if updated is None:
        return {"error": "No matching results. Run matching first."}

    _result = updated

    # Find the record's new cluster
    for cid, info in _result.clusters.items():
        if record_id in info["members"]:
            return {
                "status": "unmerged",
                "record_id": record_id,
                "new_cluster_id": cid,
                "new_cluster_size": info["size"],
                "total_clusters": _result.stats.total_clusters,
            }

    return {"status": "unmerged", "record_id": record_id}


def _tool_shatter_cluster(cluster_id: int) -> dict:
    """Break a cluster into singletons."""
    global _result

    info = _result.clusters.get(cluster_id)
    if info is None:
        return {"error": f"Cluster {cluster_id} not found"}

    member_count = info["size"]
    updated = _engine.unmerge_cluster(cluster_id)
    if updated is None:
        return {"error": "No matching results. Run matching first."}

    _result = updated

    return {
        "status": "shattered",
        "cluster_id": cluster_id,
        "records_freed": member_count,
        "total_clusters": _result.stats.total_clusters,
    }


def _tool_suggest_config(bad_merges: list[dict]) -> dict:
    """Analyze bad merges and suggest config changes."""
    from goldenmatch.core.explainer import explain_pair

    if not bad_merges:
        return {"error": "Provide at least one bad merge example."}

    matchkeys = _config.get_matchkeys()
    fields = []
    threshold = 0.80
    for mk in matchkeys:
        if mk.type == "weighted":
            fields = mk.fields
            threshold = mk.threshold or 0.80
            break

    # Analyze each bad merge
    analyses = []
    field_scores: dict[str, list[float]] = {}

    for merge in bad_merges:
        rec_a = merge.get("record_a", {})
        rec_b = merge.get("record_b", {})
        reason = merge.get("reason", "")

        exp = explain_pair(rec_a, rec_b, fields, threshold)

        analysis = {
            "total_score": round(exp.total_score, 4),
            "is_match": exp.is_match,
            "reason": reason,
            "guilty_fields": [],
        }

        for f in exp.fields:
            if f.score >= 0.7:  # This field contributed to the bad merge
                analysis["guilty_fields"].append({
                    "field": f.field_name,
                    "scorer": f.scorer,
                    "score": round(f.score, 4),
                    "value_a": f.value_a,
                    "value_b": f.value_b,
                })
            field_scores.setdefault(f.field_name, []).append(f.score)

        analyses.append(analysis)

    # Generate suggestions
    suggestions = []

    # Suggest raising threshold if bad merges have scores close to current threshold
    bad_scores = [a["total_score"] for a in analyses if a["is_match"]]
    if bad_scores:
        max_bad = max(bad_scores)
        if max_bad < 1.0:
            suggested_threshold = round(max_bad + 0.05, 2)
            suggestions.append({
                "type": "raise_threshold",
                "current": threshold,
                "suggested": suggested_threshold,
                "reason": f"Bad merges have scores up to {max_bad:.2f}. "
                         f"Raising threshold to {suggested_threshold} would reject them.",
            })

    # Identify fields that are too permissive
    for field_name, scores in field_scores.items():
        avg = sum(scores) / len(scores)
        if avg >= 0.7:
            suggestions.append({
                "type": "reduce_field_weight",
                "field": field_name,
                "avg_score_on_bad_merges": round(avg, 3),
                "reason": f"Field '{field_name}' scores high ({avg:.2f}) on bad merges. "
                         f"Consider reducing its weight or switching to a stricter scorer.",
            })

    return {
        "analyses": analyses,
        "suggestions": suggestions,
        "current_threshold": threshold,
        "bad_merges_analyzed": len(analyses),
    }


def _tool_profile_data() -> dict:
    profile = _engine.profile
    cols = []
    col_list = profile.get("columns", [])
    if isinstance(col_list, list):
        for info in col_list:
            if not isinstance(info, dict):
                continue
            cols.append({
                "column": info.get("name", ""),
                "type": info.get("suspected_type", info.get("dtype", "unknown")),
                "null_rate": round(info.get("null_rate", 0) * 100, 1),
                "unique_rate": round(info.get("unique_rate", 0) * 100, 1),
                "sample": [str(v) for v in info.get("sample_values", [])[:3]],
            })
    return {"columns": cols, "total_records": _engine.row_count}


def _tool_export_results(output_path: str, fmt: str) -> dict:
    import polars as pl

    path = Path(output_path)
    if fmt == "json":
        if _result.golden is not None:
            golden_dicts = _result.golden.to_dicts()
            clean = [{k: v for k, v in r.items() if not k.startswith("__")} for r in golden_dicts]
            path.write_text(json.dumps(clean, default=str, indent=2))
        else:
            path.write_text("[]")
    else:
        if _result.golden is not None:
            cols = [c for c in _result.golden.columns if not c.startswith("__")]
            _result.golden.select(cols).write_csv(str(path))
        else:
            path.write_text("")

    return {"exported": str(path), "format": fmt, "records": _result.golden.height if _result.golden is not None else 0}


def _tool_list_domains() -> dict:
    """List available domain extraction rulebooks."""
    from goldenmatch.core.domain_registry import discover_rulebooks
    rulebooks = discover_rulebooks()
    result = []
    for name, rb in rulebooks.items():
        result.append({
            "name": rb.name,
            "signals": rb.signals,
            "identifier_patterns": list(rb.identifier_patterns.keys()),
            "brand_count": len(rb.brand_patterns),
            "attribute_patterns": list(rb.attribute_patterns.keys()),
        })
    return {"domains": result, "count": len(result)}


def _tool_create_domain(args: dict) -> dict:
    """Create a custom domain extraction rulebook."""
    from goldenmatch.core.domain_registry import DomainRulebook, save_rulebook
    from pathlib import Path

    name = args["name"]
    scope = args.get("scope", "local")

    if scope == "global":
        save_dir = Path.home() / ".goldenmatch" / "domains"
    else:
        save_dir = Path(".goldenmatch/domains")

    rulebook = DomainRulebook(
        name=name,
        signals=args.get("signals", []),
        identifier_patterns=args.get("identifier_patterns", {}),
        brand_patterns=args.get("brand_patterns", []),
        attribute_patterns=args.get("attribute_patterns", {}),
        stop_words=args.get("stop_words", []),
    )

    path = save_rulebook(rulebook, save_dir / f"{name}.yaml")
    return {
        "status": "created",
        "name": name,
        "path": str(path),
        "scope": scope,
        "signals": rulebook.signals,
        "identifier_patterns": list(rulebook.identifier_patterns.keys()),
    }


def _tool_test_domain(domain_name: str, sample_size: int = 10) -> dict:
    """Test a domain rulebook against loaded data."""
    from goldenmatch.core.domain_registry import discover_rulebooks

    if not _rows:
        return {"error": "No data loaded. Start the MCP server with --file."}

    rulebooks = discover_rulebooks()
    if domain_name not in rulebooks:
        return {"error": f"Domain '{domain_name}' not found. Available: {list(rulebooks.keys())}"}

    rb = rulebooks[domain_name]
    # Get text columns
    sample_cols = [c for c in _rows[0].keys() if not c.startswith("__") and isinstance(_rows[0].get(c), str)]
    if not sample_cols:
        return {"error": "No text columns found in data."}

    text_col = sample_cols[0]
    results = []
    for row in _rows[:sample_size]:
        text = str(row.get(text_col, "") or "")
        extracted = rb.extract(text)
        results.append({
            "original": text[:100],
            "brand": extracted.get("brand"),
            "identifiers": extracted.get("identifiers", {}),
            "name_normalized": extracted.get("name_normalized"),
            "confidence": round(extracted.get("confidence", 0), 2),
        })

    return {
        "domain": domain_name,
        "text_column": text_col,
        "sample_size": len(results),
        "extractions": results,
    }


def _tool_pprl_auto_config(security_level: str = "high", use_llm: bool = False) -> dict:
    """Auto-configure PPRL parameters from loaded data."""
    if not _rows:
        return {"error": "No data loaded. Start the MCP server with --file."}

    import polars as pl
    from goldenmatch.pprl.autoconfig import auto_configure_pprl, auto_configure_pprl_llm

    df = pl.DataFrame(_rows)

    if use_llm:
        result = auto_configure_pprl_llm(df, security_level=security_level)
    else:
        result = auto_configure_pprl(df, security_level=security_level)

    return {
        "recommended_fields": result.recommended_fields,
        "threshold": result.recommended_config.threshold,
        "security_level": result.recommended_config.security_level,
        "ngram_size": result.recommended_config.ngram_size,
        "hash_functions": result.recommended_config.hash_functions,
        "bloom_filter_size": result.recommended_config.bloom_filter_size,
        "explanation": result.explanation,
        "field_profiles": [
            {
                "column": p.column,
                "field_type": p.field_type,
                "avg_length": round(p.avg_length, 1),
                "cardinality": p.cardinality,
                "usefulness_score": round(p.usefulness_score, 2),
            }
            for p in result.field_profiles
        ],
    }


def _tool_pprl_link(args: dict) -> dict:
    """Run PPRL linkage between two files."""
    import polars as pl
    from pathlib import Path
    from goldenmatch.pprl.protocol import PPRLConfig, run_pprl

    file_a = Path(args["file_a"])
    file_b = Path(args["file_b"])
    if not file_a.exists():
        return {"error": f"File not found: {file_a}"}
    if not file_b.exists():
        return {"error": f"File not found: {file_b}"}

    fields = args["fields"]
    threshold = args.get("threshold", 0.85)
    security_level = args.get("security_level", "high")

    _LEVELS = {"standard": (2, 20, 512), "high": (2, 30, 1024), "paranoid": (3, 40, 2048)}
    ngram, hashes, size = _LEVELS.get(security_level, (2, 30, 1024))

    config = PPRLConfig(
        fields=fields, threshold=threshold, security_level=security_level,
        ngram_size=ngram, hash_functions=hashes, bloom_filter_size=size,
    )

    df_a = pl.read_csv(file_a)
    df_b = pl.read_csv(file_b)

    result = run_pprl(df_a, df_b, config)

    cluster_summary = []
    for cid, members in sorted(result.clusters.items())[:20]:
        cluster_summary.append({
            "cluster_id": cid,
            "members": [{"party": pid, "record_id": rid} for pid, rid in members],
        })

    return {
        "clusters_found": len(result.clusters),
        "match_pairs": result.match_count,
        "total_comparisons": result.total_comparisons,
        "security_level": security_level,
        "threshold": threshold,
        "fields": fields,
        "clusters": cluster_summary,
    }


async def run_server(file_paths: list[str], config_path: str | None = None) -> None:
    """Run the MCP server over stdio."""
    server = create_server(file_paths, config_path)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
