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

logger = logging.getLogger(__name__)

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
        return [
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
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
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
    elif name == "profile_data":
        return _tool_profile_data()
    elif name == "export_results":
        return _tool_export_results(args["output_path"], args.get("format", "csv"))
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


async def run_server(file_paths: list[str], config_path: str | None = None) -> None:
    """Run the MCP server over stdio."""
    server = create_server(file_paths, config_path)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
