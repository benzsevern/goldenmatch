"""Lineage persistence -- save per-pair match explanations to a sidecar file.

Every merge decision gets a traceable explanation: which fields matched,
what scores they got, and why the pair was accepted. Enables post-hoc
auditing and "why did these merge?" queries without re-running the pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from goldenmatch.config.schemas import MatchkeyConfig

logger = logging.getLogger(__name__)


def build_lineage(
    scored_pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    matchkeys: list[MatchkeyConfig],
    clusters: dict[int, dict],
    max_pairs: int = 10000,
) -> list[dict]:
    """Build lineage records for scored pairs.

    Args:
        scored_pairs: All scored pairs from the pipeline.
        df: Full DataFrame with record data.
        matchkeys: Matchkey configs used for scoring.
        clusters: Cluster results with membership info.
        max_pairs: Cap on lineage records to prevent huge files.

    Returns:
        List of lineage dicts, one per scored pair.
    """
    from goldenmatch.core.explainer import explain_pair

    rows = df.to_dicts()
    row_ids = df["__row_id__"].to_list()
    id_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    # Map row_id to cluster_id
    row_to_cluster: dict[int, int] = {}
    for cid, cinfo in clusters.items():
        for mid in cinfo["members"]:
            row_to_cluster[mid] = cid

    # Find the weighted matchkey for explanations
    fields = []
    threshold = 0.80
    for mk in matchkeys:
        if mk.type == "weighted":
            fields = mk.fields
            threshold = mk.threshold or 0.80
            break

    lineage = []
    for a, b, score in scored_pairs[:max_pairs]:
        idx_a = id_to_idx.get(a)
        idx_b = id_to_idx.get(b)
        if idx_a is None or idx_b is None:
            continue

        row_a = rows[idx_a]
        row_b = rows[idx_b]

        # Get field-level explanation
        field_details = []
        if fields:
            exp = explain_pair(row_a, row_b, fields, threshold)
            for f in exp.fields:
                field_details.append({
                    "field": f.field_name,
                    "scorer": f.scorer,
                    "value_a": f.value_a,
                    "value_b": f.value_b,
                    "score": round(f.score, 4),
                    "weight": f.weight,
                    "diff_type": f.diff_type,
                })

        lineage.append({
            "row_id_a": a,
            "row_id_b": b,
            "score": round(score, 4),
            "cluster_id": row_to_cluster.get(a),
            "fields": field_details,
        })

    return lineage


def save_lineage(
    lineage: list[dict],
    output_dir: str | Path,
    run_name: str,
) -> Path:
    """Save lineage to a JSON sidecar file.

    Args:
        lineage: List of lineage dicts from build_lineage.
        output_dir: Directory to save the file.
        run_name: Run identifier for the filename.

    Returns:
        Path to the saved lineage file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{run_name}_lineage.json"
    data = {
        "generated_at": datetime.now().isoformat(),
        "run_name": run_name,
        "total_pairs": len(lineage),
        "pairs": lineage,
    }
    path.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")
    logger.info("Saved lineage for %d pairs to %s", len(lineage), path)
    return path


def load_lineage(path: str | Path) -> dict:
    """Load lineage from a JSON sidecar file."""
    path = Path(path)
    if not path.exists():
        return {"error": f"Lineage file not found: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))
