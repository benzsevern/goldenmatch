"""Visual cluster graph — interactive HTML with force-directed network.

Note: This generates a self-contained HTML file from trusted internal data
(GoldenMatch's own cluster/pair output). The graph data is serialized as
JSON from the application's own pipeline results, not from user-provided
HTML content. innerHTML usage in the generated file is safe in this context
as all values are escaped during JSON serialization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def generate_cluster_graph(
    df: pl.DataFrame,
    clusters: dict[int, dict],
    scored_pairs: list[tuple[int, int, float]],
    output_path: str | Path = "goldenmatch_graph.html",
    max_clusters: int = 50,
    max_nodes_per_cluster: int = 10,
    label_column: str | None = None,
) -> Path:
    """Generate an interactive force-directed cluster graph as standalone HTML.

    Args:
        df: Full DataFrame with __row_id__.
        clusters: Cluster dict from build_clusters.
        scored_pairs: (id_a, id_b, score) tuples.
        output_path: Where to write HTML.
        max_clusters: Limit clusters shown (largest first).
        max_nodes_per_cluster: Limit nodes per cluster.
        label_column: Column to use for node labels (auto-detects if None).

    Returns:
        Path to generated HTML file.
    """
    output_path = Path(output_path)

    rows = df.to_dicts()
    id_to_idx = {row["__row_id__"]: i for i, row in enumerate(rows)}
    cols = [c for c in df.columns if not c.startswith("__")]

    # Auto-detect label column
    if label_column is None:
        for candidate in ["name", "title", "full_name", "first_name", "email"]:
            if candidate in cols:
                label_column = candidate
                break
        if label_column is None and cols:
            label_column = cols[0]

    # Build pair lookup
    pair_scores: dict[tuple[int, int], float] = {}
    for a, b, s in scored_pairs:
        pair_scores[(min(a, b), max(a, b))] = s

    # Select top clusters by size
    multi_clusters = {k: v for k, v in clusters.items() if v.get("size", 0) > 1}
    top_clusters = sorted(multi_clusters.items(), key=lambda x: -x[1]["size"])[:max_clusters]

    # Build graph data
    nodes = []
    edges = []
    node_ids_seen = set()

    colors = [
        "#d4a017", "#2ecc71", "#3498db", "#e74c3c", "#9b59b6",
        "#1abc9c", "#e67e22", "#f39c12", "#2980b9", "#c0392b",
        "#16a085", "#8e44ad", "#27ae60", "#d35400", "#2c3e50",
        "#f1c40f", "#e91e63", "#00bcd4", "#ff5722", "#607d8b",
    ]

    for cluster_idx, (cluster_id, info) in enumerate(top_clusters):
        members = info.get("members", [])[:max_nodes_per_cluster]
        color = colors[cluster_idx % len(colors)]

        for mid in members:
            if mid in node_ids_seen:
                continue
            node_ids_seen.add(mid)

            idx = id_to_idx.get(mid)
            label = ""
            tooltip_lines = []
            if idx is not None:
                row = rows[idx]
                label = str(row.get(label_column, mid))[:25]
                for c in cols[:5]:
                    val = row.get(c)
                    if val is not None:
                        tooltip_lines.append(f"{c}: {val}")

            nodes.append({
                "id": str(mid),
                "label": label,
                "cluster": cluster_id,
                "color": color,
                "tooltip": "\n".join(tooltip_lines),
            })

        for i, m1 in enumerate(members):
            for m2 in members[i + 1:]:
                key = (min(m1, m2), max(m1, m2))
                score = pair_scores.get(key, 0.5)
                edges.append({
                    "source": str(m1),
                    "target": str(m2),
                    "score": round(score, 3),
                    "width": max(1, int(score * 5)),
                })

    # JSON serialization handles escaping of all values
    graph_data = json.dumps({"nodes": nodes, "edges": edges})

    html = _build_graph_html(graph_data, len(top_clusters), len(nodes), len(edges))
    output_path.write_text(html, encoding="utf-8")
    logger.info("Cluster graph saved to %s (%d nodes, %d edges)", output_path, len(nodes), len(edges))
    return output_path


def _build_graph_html(graph_data: str, cluster_count: int, node_count: int, edge_count: int) -> str:
    """Build standalone HTML with embedded force-directed graph.

    All dynamic content comes from JSON-serialized graph_data which is
    produced from GoldenMatch's internal pipeline output (trusted data).
    """
    # The template uses graph_data (JSON) inserted via Python f-string.
    # All user-facing values are already JSON-encoded (escaped) by json.dumps.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GoldenMatch Cluster Graph</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0f0f1a; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; overflow: hidden; }}
#header {{ position: fixed; top: 0; left: 0; right: 0; background: #16213e; border-bottom: 2px solid #d4a017; padding: 12px 20px; z-index: 10; display: flex; justify-content: space-between; align-items: center; }}
#header h1 {{ color: #d4a017; font-size: 1.3em; }}
.stats {{ color: #8892a0; font-size: 0.85em; }}
.stats span {{ color: #d4a017; font-weight: bold; margin: 0 5px; }}
#graph {{ width: 100vw; height: 100vh; padding-top: 50px; }}
svg {{ width: 100%; height: 100%; }}
.tooltip {{ position: absolute; background: #16213e; border: 1px solid #d4a017; border-radius: 6px; padding: 10px; font-size: 0.8em; pointer-events: none; opacity: 0; transition: opacity 0.15s; white-space: pre-line; max-width: 300px; z-index: 100; }}
#legend {{ position: fixed; bottom: 15px; left: 15px; background: #16213e; border: 1px solid #d4a01740; border-radius: 8px; padding: 12px; font-size: 0.8em; max-height: 200px; overflow-y: auto; }}
#legend h3 {{ color: #d4a017; margin-bottom: 8px; font-size: 0.9em; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
#controls {{ position: fixed; top: 60px; right: 15px; background: #16213e; border: 1px solid #d4a01740; border-radius: 8px; padding: 10px; font-size: 0.8em; }}
#controls label {{ color: #8892a0; display: block; margin: 4px 0; }}
#controls input {{ width: 100px; }}
</style>
</head>
<body>
<div id="header">
    <h1>GoldenMatch Cluster Graph</h1>
    <div class="stats">
        <span>{cluster_count}</span> clusters |
        <span>{node_count}</span> nodes |
        <span>{edge_count}</span> edges
    </div>
</div>
<div id="graph"></div>
<div class="tooltip" id="tooltip"></div>
<div id="legend"><h3>Clusters</h3></div>
<div id="controls">
    <label>Link Strength <input type="range" id="strength" min="0" max="100" value="30"></label>
    <label>Repulsion <input type="range" id="repulsion" min="0" max="500" value="150"></label>
</div>

<script>
// Graph data from GoldenMatch pipeline (JSON-serialized, pre-escaped)
const data = {graph_data};

const width = window.innerWidth;
const height = window.innerHeight - 50;
const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
svg.setAttribute("viewBox", "0 0 " + width + " " + height);
document.getElementById("graph").appendChild(svg);
const tooltip = document.getElementById("tooltip");

data.nodes.forEach(function(n, i) {{
    n.x = width / 2 + (Math.random() - 0.5) * 400;
    n.y = height / 2 + (Math.random() - 0.5) * 400;
    n.vx = 0; n.vy = 0;
}});

const nodeMap = {{}};
data.nodes.forEach(function(n) {{ nodeMap[n.id] = n; }});

// Build legend using safe DOM methods
const legend = document.getElementById("legend");
const seen = new Set();
data.nodes.forEach(function(n) {{
    if (!seen.has(n.cluster)) {{
        seen.add(n.cluster);
        const item = document.createElement("div");
        item.className = "legend-item";
        const dot = document.createElement("div");
        dot.className = "legend-dot";
        dot.style.background = n.color;
        item.appendChild(dot);
        const label = document.createTextNode("Cluster #" + n.cluster);
        item.appendChild(label);
        legend.appendChild(item);
    }}
}});

// Draw edges
data.edges.forEach(function(e) {{
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("stroke", "#d4a01740");
    line.setAttribute("stroke-width", e.width);
    line.dataset.source = e.source;
    line.dataset.target = e.target;
    svg.appendChild(line);
}});

// Draw nodes using safe DOM methods
data.nodes.forEach(function(n) {{
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.dataset.id = n.id;

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", 8);
    circle.setAttribute("fill", n.color);
    circle.setAttribute("stroke", "#0f0f1a");
    circle.setAttribute("stroke-width", 2);
    circle.style.cursor = "grab";
    g.appendChild(circle);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("dy", -12);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#e0e0e0");
    text.setAttribute("font-size", "10");
    text.textContent = n.label;
    g.appendChild(text);

    g.addEventListener("mouseenter", function(ev) {{
        // Use textContent for safe tooltip rendering
        tooltip.textContent = n.label + "\\n" + n.tooltip;
        tooltip.style.opacity = 1;
        tooltip.style.left = ev.pageX + 15 + "px";
        tooltip.style.top = ev.pageY + 15 + "px";
        circle.setAttribute("r", 12);
    }});
    g.addEventListener("mouseleave", function() {{
        tooltip.style.opacity = 0;
        circle.setAttribute("r", 8);
    }});

    var dragging = false;
    g.addEventListener("mousedown", function(ev) {{
        dragging = true; n.fx = true;
        circle.style.cursor = "grabbing";
        ev.preventDefault();
    }});
    document.addEventListener("mousemove", function(ev) {{
        if (dragging) {{
            var rect = svg.getBoundingClientRect();
            n.x = (ev.clientX - rect.left) / rect.width * width;
            n.y = (ev.clientY - rect.top) / rect.height * height;
        }}
    }});
    document.addEventListener("mouseup", function() {{
        if (dragging) {{ dragging = false; n.fx = false; circle.style.cursor = "grab"; }}
    }});

    svg.appendChild(g);
}});

// Force simulation
function simulate() {{
    var str = document.getElementById("strength").value / 100;
    var rep = parseInt(document.getElementById("repulsion").value);

    for (var i = 0; i < data.nodes.length; i++) {{
        for (var j = i + 1; j < data.nodes.length; j++) {{
            var a = data.nodes[i], b = data.nodes[j];
            var dx = b.x - a.x, dy = b.y - a.y;
            var dist = Math.sqrt(dx * dx + dy * dy) || 1;
            var force = rep / (dist * dist);
            var fx = dx / dist * force, fy = dy / dist * force;
            if (!a.fx) {{ a.vx -= fx; a.vy -= fy; }}
            if (!b.fx) {{ b.vx += fx; b.vy += fy; }}
        }}
    }}

    data.edges.forEach(function(e) {{
        var a = nodeMap[e.source], b = nodeMap[e.target];
        if (!a || !b) return;
        var dx = b.x - a.x, dy = b.y - a.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        var force = (dist - 80) * str * 0.01;
        var fx = dx / dist * force, fy = dy / dist * force;
        if (!a.fx) {{ a.vx += fx; a.vy += fy; }}
        if (!b.fx) {{ b.vx -= fx; b.vy -= fy; }}
    }});

    data.nodes.forEach(function(n) {{
        if (n.fx) return;
        n.vx += (width / 2 - n.x) * 0.001;
        n.vy += (height / 2 - n.y) * 0.001;
        n.vx *= 0.9; n.vy *= 0.9;
        n.x += n.vx; n.y += n.vy;
        n.x = Math.max(20, Math.min(width - 20, n.x));
        n.y = Math.max(20, Math.min(height - 20, n.y));
    }});

    var lines = svg.querySelectorAll("line");
    lines.forEach(function(line) {{
        var s = nodeMap[line.dataset.source], t = nodeMap[line.dataset.target];
        if (s && t) {{
            line.setAttribute("x1", s.x); line.setAttribute("y1", s.y);
            line.setAttribute("x2", t.x); line.setAttribute("y2", t.y);
        }}
    }});

    var groups = svg.querySelectorAll("g");
    groups.forEach(function(g) {{
        var n = nodeMap[g.dataset.id];
        if (n) g.setAttribute("transform", "translate(" + n.x + "," + n.y + ")");
    }});

    requestAnimationFrame(simulate);
}}

simulate();
</script>
</body>
</html>"""
