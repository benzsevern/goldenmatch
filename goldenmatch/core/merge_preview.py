"""Merge preview -- show exactly what will change before merging.

Displays a summary of proposed changes without writing anything:
- How many records will be affected
- Which fields will change and from what to what
- Confidence scores for each merge decision
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def generate_merge_preview(
    df: pl.DataFrame,
    clusters: dict[int, dict],
    golden_df: pl.DataFrame | None = None,
) -> dict:
    """Generate a preview of what merging would change.

    Returns:
        {
            "total_records": int,
            "records_affected": int,
            "records_unchanged": int,
            "clusters_to_merge": int,
            "field_changes": {col: {"changes": N, "examples": [...]}},
            "largest_clusters": [...],
            "risk_summary": str,
        }
    """
    cols = [c for c in df.columns if not c.startswith("__")]

    row_to_cluster: dict[int, int] = {}
    for cid, info in clusters.items():
        for mid in info.get("members", []):
            row_to_cluster[mid] = cid

    multi_clusters = {k: v for k, v in clusters.items() if v.get("size", 0) > 1}

    golden_lookup: dict[int, dict] = {}
    if golden_df is not None:
        for row in golden_df.to_dicts():
            cid = row.get("__cluster_id__")
            if cid is not None:
                golden_lookup[cid] = {k: v for k, v in row.items() if not k.startswith("__")}

    field_changes: dict[str, dict] = {}
    records_with_changes = set()
    total_affected = 0

    for row in df.to_dicts():
        rid = row.get("__row_id__", 0)
        cid = row_to_cluster.get(rid)
        if cid is None or cid not in multi_clusters:
            continue

        golden = golden_lookup.get(cid, {})
        if not golden:
            continue

        total_affected += 1
        row_changed = False

        for col in cols:
            orig = row.get(col)
            merged = golden.get(col)

            if orig is not None and merged is not None:
                if str(orig).strip() != str(merged).strip():
                    if col not in field_changes:
                        field_changes[col] = {"changes": 0, "examples": []}

                    field_changes[col]["changes"] += 1
                    row_changed = True

                    if len(field_changes[col]["examples"]) < 3:
                        field_changes[col]["examples"].append({
                            "from": str(orig)[:50],
                            "to": str(merged)[:50],
                            "cluster_id": cid,
                        })

        if row_changed:
            records_with_changes.add(rid)

    total_changes = sum(fc["changes"] for fc in field_changes.values())
    risk = "low"
    if total_changes > df.height * 0.5:
        risk = "high"
    elif total_changes > df.height * 0.2:
        risk = "medium"

    risk_summary = {
        "low": "Few changes expected. Safe to merge.",
        "medium": "Moderate changes. Review the diff before merging.",
        "high": "Many changes detected. Carefully review before proceeding.",
    }[risk]

    largest = sorted(multi_clusters.items(), key=lambda x: -x[1]["size"])[:5]
    largest_info = [
        {"cluster_id": cid, "size": info["size"], "oversized": info.get("oversized", False)}
        for cid, info in largest
    ]

    return {
        "total_records": df.height,
        "records_affected": total_affected,
        "records_with_changes": len(records_with_changes),
        "records_unchanged": df.height - len(records_with_changes),
        "clusters_to_merge": len(multi_clusters),
        "total_field_changes": total_changes,
        "field_changes": field_changes,
        "largest_clusters": largest_info,
        "risk_level": risk,
        "risk_summary": risk_summary,
    }


def format_preview_text(preview: dict) -> str:
    """Format merge preview as readable Rich text."""
    lines = []
    lines.append("[bold #d4a017]Merge Preview[/]")
    lines.append("")
    lines.append(f"  Records:           {preview['total_records']:,}")
    lines.append(f"  Will be affected:  [bold]{preview['records_affected']:,}[/]")
    lines.append(f"  With changes:      [bold]{preview['records_with_changes']:,}[/]")
    lines.append(f"  Unchanged:         {preview['records_unchanged']:,}")
    lines.append(f"  Clusters to merge: {preview['clusters_to_merge']:,}")
    lines.append(f"  Total field changes: [bold]{preview['total_field_changes']:,}[/]")
    lines.append("")

    risk = preview["risk_level"]
    risk_color = {"low": "#2ecc71", "medium": "#e67e22", "high": "#e74c3c"}[risk]
    lines.append(f"  Risk: [{risk_color}]{risk.upper()}[/] - {preview['risk_summary']}")
    lines.append("")

    if preview["field_changes"]:
        lines.append("[bold #d4a017]Field Changes:[/]")
        for col, info in sorted(preview["field_changes"].items(), key=lambda x: -x[1]["changes"]):
            lines.append(f"  [bold]{col}[/]: {info['changes']:,} changes")
            for ex in info["examples"]:
                lines.append(f"    [red]{ex['from']}[/] -> [green]{ex['to']}[/]")
        lines.append("")

    if preview["largest_clusters"]:
        lines.append("[bold #d4a017]Largest Clusters:[/]")
        for c in preview["largest_clusters"]:
            oversized = " [red](OVERSIZED)[/]" if c["oversized"] else ""
            lines.append(f"  Cluster #{c['cluster_id']}: {c['size']} records{oversized}")

    return "\n".join(lines)
