"""Undo/rollback -- revert a previous merge run.

Saves run snapshots so any merge can be undone.
Uses a simple JSON-based run log stored alongside output files.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RUN_LOG_FILE = ".goldenmatch_runs.json"


def save_run_snapshot(
    run_id: str,
    output_dir: str | Path,
    config_dict: dict,
    stats: dict,
    output_files: list[str],
    original_file: str | None = None,
) -> None:
    """Save a snapshot of a run for later rollback."""
    output_dir = Path(output_dir)
    log_path = output_dir / RUN_LOG_FILE

    runs = _load_run_log(log_path)

    snapshot = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config_dict,
        "stats": stats,
        "output_files": output_files,
        "original_file": original_file,
        "rolled_back": False,
    }

    runs.append(snapshot)

    # Keep only last 50 runs
    if len(runs) > 50:
        runs = runs[-50:]

    log_path.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")
    logger.info("Run snapshot saved: %s", run_id)


def list_runs(output_dir: str | Path = ".") -> list[dict]:
    """List all saved runs."""
    log_path = Path(output_dir) / RUN_LOG_FILE
    return _load_run_log(log_path)


def rollback_run(
    run_id: str,
    output_dir: str | Path = ".",
) -> dict:
    """Rollback a specific run by deleting its output files.

    Args:
        run_id: The run ID to rollback.
        output_dir: Directory containing the run log.

    Returns:
        Summary of what was rolled back.
    """
    output_dir = Path(output_dir)
    log_path = output_dir / RUN_LOG_FILE
    runs = _load_run_log(log_path)

    target = None
    for run in runs:
        if run["run_id"] == run_id:
            target = run
            break

    if target is None:
        return {"error": f"Run {run_id} not found", "available_runs": [r["run_id"] for r in runs]}

    if target.get("rolled_back"):
        return {"error": f"Run {run_id} was already rolled back"}

    # Delete output files
    deleted = []
    not_found = []
    for filepath in target.get("output_files", []):
        p = Path(filepath)
        if not p.is_absolute():
            p = output_dir / p
        if p.exists():
            p.unlink()
            deleted.append(str(p))
        else:
            not_found.append(str(p))

    # Mark as rolled back
    target["rolled_back"] = True
    target["rolled_back_at"] = datetime.now().isoformat()
    log_path.write_text(json.dumps(runs, indent=2, default=str), encoding="utf-8")

    logger.info("Rolled back run %s: deleted %d files", run_id, len(deleted))

    return {
        "run_id": run_id,
        "deleted": deleted,
        "not_found": not_found,
        "status": "rolled_back",
    }


def _load_run_log(path: Path) -> list[dict]:
    """Load the run log."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
