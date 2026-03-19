"""Before/After Data Quality Dashboard — visual transformation report.

Generates a standalone HTML page showing:
- Before: messy data with duplicates highlighted
- After: clean golden records with confidence scores
- The delta: records reduced, duplicates caught, quality improved
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def generate_dashboard(
    df: pl.DataFrame,
    clusters: dict[int, dict],
    scored_pairs: list[tuple[int, int, float]],
    golden_df: pl.DataFrame | None = None,
    output_path: str | Path = "goldenmatch_dashboard.html",
    title: str = "GoldenMatch",
) -> Path:
    """Generate a before/after data quality dashboard."""
    output_path = Path(output_path)
    cols = [c for c in df.columns if not c.startswith("__")]
    rows = df.to_dicts()

    total = df.height
    multi_clusters = {k: v for k, v in clusters.items() if v.get("size", 0) > 1}
    total_dupes = sum(v["size"] for v in multi_clusters.values())
    unique_entities = total - total_dupes + len(multi_clusters)
    reduction_pct = ((total - unique_entities) / total * 100) if total > 0 else 0
    golden_count = golden_df.height if golden_df is not None else len(multi_clusters)

    # Score distribution
    scores = [s for _, _, s in scored_pairs]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Cluster sizes
    sizes = [v["size"] for v in multi_clusters.values()]
    size_dist = dict(Counter(sizes))

    # Quality issues in original data
    quality_issues = _detect_quality_issues(df, cols)

    # Sample duplicate pairs for before/after
    id_to_idx = {row["__row_id__"]: i for i, row in enumerate(rows)}
    sample_pairs = []
    for cid, info in sorted(multi_clusters.items(), key=lambda x: -x[1]["size"])[:8]:
        members = info["members"]
        if len(members) >= 2:
            idx_a = id_to_idx.get(members[0])
            idx_b = id_to_idx.get(members[1])
            if idx_a is not None and idx_b is not None:
                pair_score = 0.0
                for a, b, s in scored_pairs:
                    if (a in members and b in members):
                        pair_score = max(pair_score, s)
                        break
                sample_pairs.append({
                    "a": {c: str(rows[idx_a].get(c, ""))[:40] for c in cols[:5]},
                    "b": {c: str(rows[idx_b].get(c, ""))[:40] for c in cols[:5]},
                    "score": round(pair_score, 3),
                    "size": info["size"],
                })

    # Golden record samples
    golden_samples = []
    if golden_df is not None:
        for row in golden_df.head(6).to_dicts():
            golden_samples.append({c: str(row.get(c, ""))[:40] for c in cols[:5]})

    html = _build_dashboard_html(
        title=title,
        total=total,
        unique_entities=unique_entities,
        total_dupes=total_dupes,
        reduction_pct=reduction_pct,
        golden_count=golden_count,
        cluster_count=len(multi_clusters),
        avg_score=avg_score,
        pair_count=len(scored_pairs),
        size_dist=size_dist,
        quality_issues=quality_issues,
        sample_pairs=sample_pairs,
        golden_samples=golden_samples,
        cols=cols[:5],
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard saved to %s", output_path)
    return output_path


def _detect_quality_issues(df: pl.DataFrame, cols: list[str]) -> list[dict]:
    """Detect data quality issues in the original data."""
    issues = []

    for col in cols:
        try:
            null_count = df[col].null_count()
            null_pct = null_count / df.height * 100 if df.height > 0 else 0
            if null_pct > 5:
                issues.append({"column": col, "issue": "missing_values", "detail": f"{null_pct:.0f}% null", "severity": "warning" if null_pct < 20 else "error"})

            if df[col].dtype in (pl.Utf8, pl.String):
                sample = df[col].drop_nulls().head(100).to_list()
                mixed_case = sum(1 for v in sample if str(v) != str(v).lower() and str(v) != str(v).upper())
                if mixed_case > 20:
                    issues.append({"column": col, "issue": "inconsistent_case", "detail": f"{mixed_case}% mixed case", "severity": "info"})

                with_spaces = sum(1 for v in sample if str(v) != str(v).strip())
                if with_spaces > 5:
                    issues.append({"column": col, "issue": "leading_trailing_spaces", "detail": f"{with_spaces} values", "severity": "warning"})
        except Exception:
            pass

    return issues


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_dashboard_html(
    title, total, unique_entities, total_dupes, reduction_pct,
    golden_count, cluster_count, avg_score, pair_count,
    size_dist, quality_issues, sample_pairs, golden_samples, cols,
) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Before/After delta cards
    before_after = f"""
    <div class="delta-grid">
        <div class="delta-card before">
            <div class="delta-label">BEFORE</div>
            <div class="delta-value">{total:,}</div>
            <div class="delta-sub">total records</div>
        </div>
        <div class="delta-arrow">&#10132;</div>
        <div class="delta-card after">
            <div class="delta-label">AFTER</div>
            <div class="delta-value">{unique_entities:,}</div>
            <div class="delta-sub">unique entities</div>
        </div>
        <div class="delta-card reduction">
            <div class="delta-label">REDUCTION</div>
            <div class="delta-value">{reduction_pct:.0f}%</div>
            <div class="delta-sub">{total_dupes:,} duplicates removed</div>
        </div>
    </div>"""

    # Quality issues
    issues_html = ""
    if quality_issues:
        issues_rows = ""
        for issue in quality_issues[:10]:
            sev_class = issue["severity"]
            icon = {"error": "&#10060;", "warning": "&#9888;", "info": "&#8505;"}.get(sev_class, "")
            issues_rows += f'<tr class="{sev_class}"><td>{icon}</td><td>{_esc(issue["column"])}</td><td>{_esc(issue["issue"].replace("_", " "))}</td><td>{_esc(issue["detail"])}</td></tr>'
        issues_html = f"""
        <h2>Data Quality Issues (Before)</h2>
        <table class="issues-table">
            <tr><th></th><th>Column</th><th>Issue</th><th>Detail</th></tr>
            {issues_rows}
        </table>"""

    # Sample duplicate pairs
    pairs_html = ""
    if sample_pairs:
        pair_cards = ""
        for i, pair in enumerate(sample_pairs):
            fields_a = " | ".join(f"{_esc(str(v))}" for v in pair["a"].values())
            fields_b = " | ".join(f"{_esc(str(v))}" for v in pair["b"].values())
            pair_cards += f"""
            <div class="pair-card">
                <div class="pair-header">Cluster ({pair['size']} records) — Score: {pair['score']}</div>
                <div class="pair-row before-row">{fields_a}</div>
                <div class="pair-row after-row">{fields_b}</div>
            </div>"""
        pairs_html = f"""
        <h2>Duplicate Pairs Found</h2>
        <div class="pair-list">{pair_cards}</div>"""

    # Golden records
    golden_html = ""
    if golden_samples:
        g_rows = ""
        for gs in golden_samples:
            cells = "".join(f"<td>{_esc(str(v))}</td>" for v in gs.values())
            g_rows += f"<tr>{cells}</tr>"
        g_headers = "".join(f"<th>{_esc(c)}</th>" for c in cols)
        golden_html = f"""
        <h2>Golden Records (After)</h2>
        <p class="section-sub">Merged canonical records — one per entity</p>
        <table class="golden-table">
            <tr>{g_headers}</tr>
            {g_rows}
        </table>"""

    # Cluster size distribution
    size_badges = " ".join(
        f'<span class="size-badge"><strong>{s}</strong> records: {c}</span>'
        for s, c in sorted(size_dist.items())
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_esc(title)} — Data Quality Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f1a; color: #e0e0e0; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 30px 20px; }}
h1 {{ color: #d4a017; font-size: 2.2em; text-align: center; margin-bottom: 5px; }}
.subtitle {{ color: #8892a0; text-align: center; margin-bottom: 30px; }}
h2 {{ color: #d4a017; font-size: 1.3em; margin: 35px 0 12px; border-bottom: 1px solid #d4a01730; padding-bottom: 8px; }}
.section-sub {{ color: #8892a0; font-size: 0.85em; margin-bottom: 12px; }}

/* Delta grid */
.delta-grid {{ display: flex; align-items: center; justify-content: center; gap: 20px; margin: 30px 0; flex-wrap: wrap; }}
.delta-card {{ background: #16213e; border-radius: 12px; padding: 25px 35px; text-align: center; min-width: 160px; }}
.delta-card.before {{ border: 2px solid #e74c3c40; }}
.delta-card.after {{ border: 2px solid #2ecc7140; }}
.delta-card.reduction {{ border: 2px solid #d4a01740; }}
.delta-label {{ font-size: 0.7em; font-weight: bold; letter-spacing: 2px; margin-bottom: 5px; }}
.before .delta-label {{ color: #e74c3c; }}
.after .delta-label {{ color: #2ecc71; }}
.reduction .delta-label {{ color: #d4a017; }}
.delta-value {{ font-size: 2.8em; font-weight: bold; color: #f0f0f0; }}
.delta-sub {{ color: #8892a0; font-size: 0.8em; margin-top: 4px; }}
.delta-arrow {{ font-size: 2em; color: #d4a017; }}

/* Stats row */
.stats-row {{ display: flex; gap: 12px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }}
.stat-pill {{ background: #16213e; border: 1px solid #d4a01730; border-radius: 20px; padding: 8px 18px; font-size: 0.85em; }}
.stat-pill strong {{ color: #d4a017; }}

/* Issues table */
.issues-table {{ width: 100%; border-collapse: collapse; background: #16213e; border-radius: 8px; overflow: hidden; }}
.issues-table th {{ background: #d4a01715; color: #d4a017; padding: 10px 12px; text-align: left; font-size: 0.85em; }}
.issues-table td {{ padding: 8px 12px; border-bottom: 1px solid #ffffff08; font-size: 0.85em; }}
.issues-table tr.error td {{ color: #e74c3c; }}
.issues-table tr.warning td {{ color: #e67e22; }}
.issues-table tr.info td {{ color: #8892a0; }}

/* Pair cards */
.pair-list {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }}
.pair-card {{ background: #16213e; border: 1px solid #d4a01720; border-radius: 8px; padding: 12px; }}
.pair-header {{ color: #d4a017; font-size: 0.8em; font-weight: bold; margin-bottom: 8px; }}
.pair-row {{ font-size: 0.8em; padding: 6px 8px; border-radius: 4px; margin: 3px 0; font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.before-row {{ background: #e74c3c10; border-left: 3px solid #e74c3c; }}
.after-row {{ background: #e67e2210; border-left: 3px solid #e67e22; }}

/* Golden table */
.golden-table {{ width: 100%; border-collapse: collapse; background: #16213e; border-radius: 8px; overflow: hidden; }}
.golden-table th {{ background: #2ecc7115; color: #2ecc71; padding: 10px 12px; text-align: left; }}
.golden-table td {{ padding: 8px 12px; border-bottom: 1px solid #ffffff08; }}

/* Size badges */
.size-dist {{ margin: 10px 0; }}
.size-badge {{ display: inline-block; background: #d4a01715; border: 1px solid #d4a01730; border-radius: 15px; padding: 4px 12px; font-size: 0.8em; margin: 3px; }}
.size-badge strong {{ color: #d4a017; }}

footer {{ text-align: center; color: #8892a0; font-size: 0.75em; margin-top: 40px; padding: 20px; border-top: 1px solid #ffffff10; }}
</style>
</head>
<body>
<div class="container">
    <h1>&#9889; {_esc(title)}</h1>
    <p class="subtitle">Data Quality Dashboard &mdash; {generated}</p>

    {before_after}

    <div class="stats-row">
        <span class="stat-pill"><strong>{cluster_count:,}</strong> duplicate clusters</span>
        <span class="stat-pill"><strong>{pair_count:,}</strong> matched pairs</span>
        <span class="stat-pill"><strong>{avg_score:.0%}</strong> avg confidence</span>
        <span class="stat-pill"><strong>{golden_count:,}</strong> golden records</span>
    </div>

    {issues_html}
    {pairs_html}

    <h2>Cluster Sizes</h2>
    <div class="size-dist">{size_badges}</div>

    {golden_html}

    <footer>Generated by GoldenMatch &mdash; {total:,} records &rarr; {unique_entities:,} entities</footer>
</div>
</body>
</html>"""
