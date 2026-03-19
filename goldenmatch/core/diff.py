"""CSV diff export — side-by-side before/after showing every change.

Generates a diff CSV/HTML showing:
- Original records
- Which cluster they belong to
- What the golden record value is for each field
- Which fields changed and how
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def generate_diff(
    df: pl.DataFrame,
    clusters: dict[int, dict],
    golden_df: pl.DataFrame | None = None,
    output_path: str | Path = "goldenmatch_diff.csv",
    format: str = "csv",
) -> Path:
    """Generate a before/after diff showing every change.

    Each row in the output represents one original record with:
    - All original fields
    - __cluster_id__: which cluster it belongs to (0 = unique)
    - __is_duplicate__: True if part of a multi-record cluster
    - __golden_*__: the merged golden value for each field
    - __changed_*__: True if this field differs from the golden value

    Args:
        df: Original DataFrame with __row_id__.
        clusters: Cluster dict from build_clusters.
        golden_df: Golden records DataFrame.
        output_path: Where to write.
        format: "csv" or "html".

    Returns:
        Path to the generated file.
    """
    output_path = Path(output_path)
    cols = [c for c in df.columns if not c.startswith("__")]

    # Build cluster membership lookup
    row_to_cluster: dict[int, int] = {}
    for cid, info in clusters.items():
        for mid in info.get("members", []):
            row_to_cluster[mid] = cid

    # Build golden record lookup by cluster
    golden_lookup: dict[int, dict] = {}
    if golden_df is not None:
        for row in golden_df.to_dicts():
            cid = row.get("__cluster_id__")
            if cid is not None:
                golden_lookup[cid] = {k: v for k, v in row.items() if not k.startswith("__")}

    # Build diff rows
    diff_rows = []
    for row in df.to_dicts():
        rid = row.get("__row_id__", 0)
        cid = row_to_cluster.get(rid, 0)
        cluster_info = clusters.get(cid, {})
        is_dupe = cluster_info.get("size", 1) > 1

        diff_row = {}

        # Original values
        for col in cols:
            diff_row[col] = row.get(col)

        # Cluster info
        diff_row["__cluster_id__"] = cid
        diff_row["__is_duplicate__"] = is_dupe
        diff_row["__cluster_size__"] = cluster_info.get("size", 1)

        # Golden values and change flags
        golden = golden_lookup.get(cid, {})
        changes = 0
        for col in cols:
            golden_val = golden.get(col)
            orig_val = row.get(col)
            diff_row[f"__golden_{col}__"] = golden_val

            # Determine if changed
            changed = False
            if golden_val is not None and orig_val is not None:
                changed = str(golden_val).strip() != str(orig_val).strip()
            diff_row[f"__changed_{col}__"] = changed
            if changed:
                changes += 1

        diff_row["__total_changes__"] = changes
        diff_rows.append(diff_row)

    diff_df = pl.DataFrame(diff_rows)

    # Sort: duplicates first, then by cluster and changes
    diff_df = diff_df.sort(
        ["__is_duplicate__", "__cluster_id__", "__total_changes__"],
        descending=[True, False, True],
    )

    if format == "html":
        _write_diff_html(diff_df, cols, output_path)
    else:
        diff_df.write_csv(str(output_path))

    total_changed = diff_df.filter(pl.col("__total_changes__") > 0).height
    logger.info("Diff saved to %s (%d records, %d with changes)", output_path, diff_df.height, total_changed)
    return output_path


def _write_diff_html(diff_df: pl.DataFrame, cols: list[str], path: Path) -> None:
    """Write diff as a styled HTML table with change highlighting."""
    rows_html = ""
    for row in diff_df.to_dicts():
        is_dupe = row.get("__is_duplicate__", False)
        changes = row.get("__total_changes__", 0)

        row_class = ""
        if is_dupe and changes > 0:
            row_class = "changed"
        elif is_dupe:
            row_class = "duplicate"

        cells = ""
        cells += f'<td>{row.get("__cluster_id__", "")}</td>'
        cells += f'<td>{row.get("__cluster_size__", 1)}</td>'

        for col in cols:
            orig = str(row.get(col, "") or "")
            golden = str(row.get(f"__golden_{col}__", "") or "")
            changed = row.get(f"__changed_{col}__", False)

            if changed:
                cell = f'<td class="changed-cell"><span class="orig">{_esc(orig)}</span><span class="arrow"> &rarr; </span><span class="golden">{_esc(golden)}</span></td>'
            else:
                cell = f'<td>{_esc(orig)}</td>'
            cells += cell

        rows_html += f'<tr class="{row_class}">{cells}</tr>\n'

    headers = '<th>Cluster</th><th>Size</th>' + "".join(f"<th>{_esc(c)}</th>" for c in cols)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>GoldenMatch Diff</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 20px; }}
h1 {{ color: #d4a017; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
th {{ background: #d4a01720; color: #d4a017; padding: 8px; text-align: left; position: sticky; top: 0; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #ffffff10; }}
tr.changed {{ background: #e67e2210; }}
tr.duplicate {{ background: #d4a01708; }}
.changed-cell {{ background: #e67e2220; }}
.orig {{ color: #e74c3c; text-decoration: line-through; }}
.arrow {{ color: #8892a0; }}
.golden {{ color: #2ecc71; font-weight: bold; }}
</style></head><body>
<h1>GoldenMatch Diff</h1>
<table><tr>{headers}</tr>{rows_html}</table>
</body></html>"""

    path.write_text(html, encoding="utf-8")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
