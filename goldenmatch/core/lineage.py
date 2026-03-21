"""Lineage persistence -- save per-pair match explanations to a sidecar file.

Every merge decision gets a traceable explanation: which fields matched,
what scores they got, and why the pair was accepted. Enables post-hoc
auditing and "why did these merge?" queries without re-running the pipeline.

Supports streaming output for large runs (no in-memory cap).
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
    natural_language: bool = False,
) -> list[dict]:
    """Build lineage records for scored pairs.

    Args:
        scored_pairs: All scored pairs from the pipeline.
        df: Full DataFrame with record data.
        matchkeys: Matchkey configs used for scoring.
        clusters: Cluster results with membership info.
        max_pairs: Cap on lineage records (0 or None = no cap).
        natural_language: Whether to include NL explanations.

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

    # Find the first weighted or probabilistic matchkey for explanations
    fields = []
    threshold = 0.80
    for mk in matchkeys:
        if mk.type in ("weighted", "probabilistic"):
            fields = mk.fields
            threshold = mk.threshold or 0.80
            break

    # Determine pair limit
    effective_max = max_pairs if max_pairs else len(scored_pairs)

    lineage = []
    for a, b, score in scored_pairs[:effective_max]:
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

        record = {
            "row_id_a": a,
            "row_id_b": b,
            "score": round(score, 4),
            "cluster_id": row_to_cluster.get(a),
            "fields": field_details,
        }

        # Add natural language explanation
        if natural_language and field_details:
            from goldenmatch.core.explain import explain_pair_nl
            record["explanation"] = explain_pair_nl(row_a, row_b, field_details, score)

        lineage.append(record)

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


def save_lineage_streaming(
    scored_pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    matchkeys: list[MatchkeyConfig],
    clusters: dict[int, dict],
    output_dir: str | Path,
    run_name: str,
    natural_language: bool = False,
) -> Path:
    """Save lineage with streaming -- writes pairs incrementally to disk.

    No in-memory cap. Handles arbitrarily large pair lists.

    Args:
        scored_pairs: All scored pairs.
        df: Full DataFrame.
        matchkeys: Matchkey configs.
        clusters: Cluster results.
        output_dir: Output directory.
        run_name: Run identifier.
        natural_language: Include NL explanations.

    Returns:
        Path to the saved lineage file.
    """
    from goldenmatch.core.explainer import explain_pair

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run_name}_lineage.json"

    rows = df.to_dicts()
    row_ids = df["__row_id__"].to_list()
    id_to_idx = {rid: i for i, rid in enumerate(row_ids)}

    row_to_cluster: dict[int, int] = {}
    for cid, cinfo in clusters.items():
        for mid in cinfo["members"]:
            row_to_cluster[mid] = cid

    fields = []
    threshold = 0.80
    for mk in matchkeys:
        if mk.type in ("weighted", "probabilistic"):
            fields = mk.fields
            threshold = mk.threshold or 0.80
            break

    # Stream write
    with open(path, "w", encoding="utf-8") as f:
        f.write('{\n')
        f.write(f'  "generated_at": "{datetime.now().isoformat()}",\n')
        f.write(f'  "run_name": "{run_name}",\n')
        f.write(f'  "total_pairs": {len(scored_pairs)},\n')
        f.write('  "pairs": [\n')

        written = 0
        for i, (a, b, score) in enumerate(scored_pairs):
            idx_a = id_to_idx.get(a)
            idx_b = id_to_idx.get(b)
            if idx_a is None or idx_b is None:
                continue

            row_a = rows[idx_a]
            row_b = rows[idx_b]

            field_details = []
            if fields:
                exp = explain_pair(row_a, row_b, fields, threshold)
                for fld in exp.fields:
                    field_details.append({
                        "field": fld.field_name,
                        "scorer": fld.scorer,
                        "value_a": fld.value_a,
                        "value_b": fld.value_b,
                        "score": round(fld.score, 4),
                        "weight": fld.weight,
                        "diff_type": fld.diff_type,
                    })

            record = {
                "row_id_a": a,
                "row_id_b": b,
                "score": round(score, 4),
                "cluster_id": row_to_cluster.get(a),
                "fields": field_details,
            }

            if natural_language and field_details:
                from goldenmatch.core.explain import explain_pair_nl
                record["explanation"] = explain_pair_nl(row_a, row_b, field_details, score)

            if written > 0:
                f.write(",\n")
            f.write("    " + json.dumps(record, default=str))
            written += 1

        f.write('\n  ]\n}\n')

    logger.info("Streamed lineage for %d pairs to %s", written, path)
    return path


def load_lineage(path: str | Path) -> dict:
    """Load lineage from a JSON sidecar file."""
    path = Path(path)
    if not path.exists():
        return {"error": f"Lineage file not found: {path}"}
    return json.loads(path.read_text(encoding="utf-8"))
