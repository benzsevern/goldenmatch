"""HTML report generator — standalone report with charts and match details."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import polars as pl

from goldenmatch.core.explainer import explain_pair, MatchExplanation

logger = logging.getLogger(__name__)


def generate_report(
    df: pl.DataFrame,
    clusters: dict[int, dict],
    scored_pairs: list[tuple[int, int, float]],
    golden_df: pl.DataFrame | None = None,
    matchkey_fields: list | None = None,
    threshold: float = 0.80,
    output_path: str | Path = "goldenmatch_report.html",
    title: str = "GoldenMatch Report",
) -> Path:
    """Generate a standalone HTML report with charts and match details.

    Args:
        df: The full DataFrame with __row_id__.
        clusters: Cluster dict from build_clusters.
        scored_pairs: List of (id_a, id_b, score) tuples.
        golden_df: Optional golden records DataFrame.
        matchkey_fields: Optional matchkey field configs for explanations.
        threshold: Match threshold used.
        output_path: Where to write the HTML file.
        title: Report title.

    Returns:
        Path to the generated HTML file.
    """
    output_path = Path(output_path)

    # Compute stats
    multi_clusters = {k: v for k, v in clusters.items() if v.get("size", 0) > 1}
    total_records = df.height
    total_clusters = len(multi_clusters)
    total_pairs = len(scored_pairs)
    singleton_count = sum(1 for v in clusters.values() if v.get("size", 0) == 1)
    match_rate = total_clusters / total_records * 100 if total_records > 0 else 0

    # Score distribution
    scores = [s for _, _, s in scored_pairs]
    score_histogram = _compute_histogram(scores, bins=20)

    # Cluster size distribution
    sizes = [v["size"] for v in multi_clusters.values()]
    size_counts = dict(Counter(sizes))

    # Top clusters (by size)
    top_clusters = sorted(multi_clusters.items(), key=lambda x: -x[1]["size"])[:10]

    # Sample match explanations
    explanations_html = ""
    if matchkey_fields and scored_pairs:
        rows = df.to_dicts()
        id_to_idx = {row["__row_id__"]: i for i, row in enumerate(rows)}

        sample_pairs = sorted(scored_pairs, key=lambda x: -x[2])[:5]
        for id_a, id_b, score in sample_pairs:
            idx_a = id_to_idx.get(id_a)
            idx_b = id_to_idx.get(id_b)
            if idx_a is not None and idx_b is not None:
                exp = explain_pair(rows[idx_a], rows[idx_b], matchkey_fields, threshold)
                explanations_html += _render_explanation(exp)

    # Build HTML
    html = _build_html(
        title=title,
        total_records=total_records,
        total_clusters=total_clusters,
        total_pairs=total_pairs,
        singleton_count=singleton_count,
        match_rate=match_rate,
        threshold=threshold,
        score_histogram=score_histogram,
        size_counts=size_counts,
        top_clusters=top_clusters,
        df=df,
        explanations_html=explanations_html,
        golden_count=golden_df.height if golden_df is not None else 0,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    return output_path


def _compute_histogram(values: list[float], bins: int = 20) -> list[dict]:
    """Compute histogram bins for chart rendering."""
    if not values:
        return []
    min_val, max_val = 0.0, 1.0
    bin_width = (max_val - min_val) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - min_val) / bin_width), bins - 1)
        counts[idx] += 1
    return [
        {"x": round(min_val + i * bin_width, 3), "y": counts[i]}
        for i in range(bins)
    ]


def _render_explanation(exp: MatchExplanation) -> str:
    """Render a single match explanation as HTML."""
    status_class = "match" if exp.is_match else "no-match"
    status_text = "MATCH" if exp.is_match else "NO MATCH"

    rows = ""
    for f in exp.fields:
        va = f.value_a or "—"
        vb = f.value_b or "—"
        diff_class = f.diff_type

        bar_width = int(f.score * 100)
        score_bar = f'<div class="score-bar"><div class="score-fill {diff_class}" style="width:{bar_width}%"></div></div>'

        rows += f"""
        <tr class="{diff_class}">
            <td>{f.field_name}</td>
            <td><code>{f.scorer}</code></td>
            <td class="val">{_escape(va[:30])}</td>
            <td class="val">{_escape(vb[:30])}</td>
            <td>{score_bar} {f.score:.3f}</td>
            <td>{f.weight:.1f}</td>
            <td><strong>{f.contribution:.3f}</strong></td>
        </tr>"""

    return f"""
    <div class="explanation {status_class}">
        <div class="exp-header">
            <span class="exp-score">Score: {exp.total_score:.3f}</span>
            <span class="exp-status {status_class}">{status_text}</span>
        </div>
        <table class="exp-table">
            <tr><th>Field</th><th>Scorer</th><th>Value A</th><th>Value B</th><th>Score</th><th>Weight</th><th>Contribution</th></tr>
            {rows}
        </table>
        <div class="exp-footer">
            Top contributor: <strong>{exp.top_contributor}</strong> |
            Weakest: <strong>{exp.weakest_field}</strong>
        </div>
    </div>"""


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_html(
    title, total_records, total_clusters, total_pairs, singleton_count,
    match_rate, threshold, score_histogram, size_counts, top_clusters,
    df, explanations_html, golden_count,
) -> str:
    """Build the complete standalone HTML report."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Top clusters HTML
    top_clusters_html = ""
    rows = df.to_dicts()
    id_to_idx = {row["__row_id__"]: i for i, row in enumerate(rows)}
    cols = [c for c in df.columns if not c.startswith("__")]

    for cluster_id, info in top_clusters[:5]:
        members = info.get("members", [])
        size = info.get("size", 0)
        top_clusters_html += f'<div class="cluster-card"><h4>Cluster #{cluster_id} ({size} records)</h4><table class="member-table"><tr>'
        for c in cols[:6]:
            top_clusters_html += f"<th>{_escape(c)}</th>"
        top_clusters_html += "</tr>"
        for mid in members[:5]:
            idx = id_to_idx.get(mid)
            if idx is not None:
                top_clusters_html += "<tr>"
                for c in cols[:6]:
                    val = str(rows[idx].get(c, ""))[:25]
                    top_clusters_html += f"<td>{_escape(val)}</td>"
                top_clusters_html += "</tr>"
        if size > 5:
            top_clusters_html += f'<tr><td colspan="{min(len(cols),6)}">... and {size - 5} more</td></tr>'
        top_clusters_html += "</table></div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_escape(title)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ color: #d4a017; font-size: 2em; margin-bottom: 5px; }}
h2 {{ color: #d4a017; font-size: 1.4em; margin: 30px 0 15px; border-bottom: 1px solid #d4a01740; padding-bottom: 8px; }}
h3 {{ color: #d4a017; font-size: 1.1em; margin: 20px 0 10px; }}
.subtitle {{ color: #8892a0; margin-bottom: 20px; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
.stat-card {{ background: #16213e; border: 1px solid #d4a01730; border-radius: 8px; padding: 20px; text-align: center; }}
.stat-value {{ font-size: 2em; font-weight: bold; color: #d4a017; }}
.stat-label {{ color: #8892a0; font-size: 0.85em; margin-top: 5px; }}
.chart-container {{ background: #16213e; border: 1px solid #d4a01730; border-radius: 8px; padding: 20px; margin: 15px 0; }}
.bar-chart {{ display: flex; align-items: flex-end; gap: 2px; height: 150px; }}
.bar {{ background: #d4a017; min-width: 8px; flex: 1; border-radius: 2px 2px 0 0; transition: opacity 0.2s; position: relative; }}
.bar:hover {{ opacity: 0.8; }}
.bar-label {{ position: absolute; bottom: -20px; left: 50%; transform: translateX(-50%); font-size: 0.65em; color: #8892a0; white-space: nowrap; }}
.chart-axis {{ display: flex; justify-content: space-between; color: #8892a0; font-size: 0.75em; margin-top: 25px; }}
.cluster-card {{ background: #16213e; border: 1px solid #d4a01730; border-radius: 8px; padding: 15px; margin: 10px 0; }}
.cluster-card h4 {{ color: #d4a017; margin-bottom: 10px; }}
.member-table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
.member-table th {{ background: #d4a01720; color: #d4a017; padding: 6px 10px; text-align: left; }}
.member-table td {{ padding: 6px 10px; border-bottom: 1px solid #ffffff10; }}
.explanation {{ background: #16213e; border: 1px solid #d4a01730; border-radius: 8px; padding: 15px; margin: 15px 0; }}
.exp-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; }}
.exp-score {{ font-size: 1.2em; font-weight: bold; color: #f0f0f0; }}
.exp-status.match {{ color: #2ecc71; font-weight: bold; }}
.exp-status.no-match {{ color: #e74c3c; font-weight: bold; }}
.exp-table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
.exp-table th {{ background: #d4a01720; color: #d4a017; padding: 6px 8px; text-align: left; }}
.exp-table td {{ padding: 6px 8px; border-bottom: 1px solid #ffffff10; }}
.exp-table .val {{ font-family: monospace; font-size: 0.9em; }}
.exp-table tr.identical td {{ background: #2ecc7110; }}
.exp-table tr.similar td {{ background: #e67e2210; }}
.exp-table tr.different td {{ background: #e74c3c10; }}
.exp-table tr.missing td {{ opacity: 0.5; }}
.score-bar {{ display: inline-block; width: 60px; height: 10px; background: #ffffff10; border-radius: 3px; margin-right: 5px; vertical-align: middle; }}
.score-fill {{ height: 100%; border-radius: 3px; }}
.score-fill.identical {{ background: #2ecc71; }}
.score-fill.similar {{ background: #e67e22; }}
.score-fill.different {{ background: #e74c3c; }}
.exp-footer {{ color: #8892a0; font-size: 0.85em; margin-top: 10px; }}
.size-dist {{ display: flex; gap: 10px; flex-wrap: wrap; }}
.size-badge {{ background: #d4a01720; border: 1px solid #d4a01740; border-radius: 20px; padding: 4px 12px; font-size: 0.85em; }}
.size-badge strong {{ color: #d4a017; }}
footer {{ text-align: center; color: #8892a0; font-size: 0.8em; margin-top: 40px; padding: 20px; border-top: 1px solid #ffffff10; }}
</style>
</head>
<body>
<div class="container">
    <h1>⚡ {_escape(title)}</h1>
    <p class="subtitle">Generated {generated}</p>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{total_records:,}</div>
            <div class="stat-label">Total Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_clusters:,}</div>
            <div class="stat-label">Duplicate Clusters</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_pairs:,}</div>
            <div class="stat-label">Matched Pairs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{match_rate:.1f}%</div>
            <div class="stat-label">Match Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{singleton_count:,}</div>
            <div class="stat-label">Unique Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{golden_count:,}</div>
            <div class="stat-label">Golden Records</div>
        </div>
    </div>

    <h2>Score Distribution</h2>
    <div class="chart-container">
        <div class="bar-chart">
            {"".join(f'<div class="bar" style="height:{max(2, int(b["y"] / max(1, max(h["y"] for h in score_histogram)) * 140))}px" title="{b["x"]:.2f}: {b["y"]} pairs"><div class="bar-label">{b["y"]}</div></div>' for b in score_histogram)}
        </div>
        <div class="chart-axis">
            <span>0.0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.0</span>
        </div>
        <p style="text-align:center;color:#8892a0;margin-top:5px;font-size:0.8em;">Match Score → (threshold: {threshold})</p>
    </div>

    <h2>Cluster Size Distribution</h2>
    <div class="size-dist">
        {"".join(f'<span class="size-badge"><strong>{size}</strong> records: {count} clusters</span>' for size, count in sorted(size_counts.items()))}
    </div>

    <h2>Top Clusters</h2>
    {top_clusters_html}

    {"<h2>Match Explanations (Top Pairs)</h2>" + explanations_html if explanations_html else ""}

    <footer>
        Generated by GoldenMatch | {total_records:,} records | {total_clusters:,} clusters | {total_pairs:,} pairs
    </footer>
</div>
</body>
</html>"""
