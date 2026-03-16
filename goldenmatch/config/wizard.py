"""Interactive config wizard with smart field suggestions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()

# ── Heuristic suggestions ──────────────────────────────────────────────────

_NAME_KEYWORDS = {"name", "first", "last", "fname", "lname", "given", "surname", "full_name"}
_EMAIL_KEYWORDS = {"email", "mail", "e_mail"}
_PHONE_KEYWORDS = {"phone", "tel", "mobile", "fax", "cell"}
_ZIP_KEYWORDS = {"zip", "postal", "postcode", "zip_code"}
_ADDRESS_KEYWORDS = {"address", "addr", "street", "city", "state", "country"}


def _matches_any(column: str, keywords: frozenset | set) -> bool:
    """Check if a column name matches any keyword (case-insensitive)."""
    col_lower = column.lower()
    for kw in keywords:
        if kw in col_lower:
            return True
    return False


def suggest_transforms(column_name: str) -> list[str]:
    """Suggest transforms based on column name heuristics."""
    col = column_name.lower()

    if _matches_any(col, _NAME_KEYWORDS):
        return ["lowercase", "strip", "normalize_whitespace"]
    if _matches_any(col, _EMAIL_KEYWORDS):
        return ["lowercase", "strip"]
    if _matches_any(col, _PHONE_KEYWORDS):
        return ["digits_only"]
    if _matches_any(col, _ZIP_KEYWORDS):
        return ["strip", "substring:0:5"]
    if _matches_any(col, _ADDRESS_KEYWORDS):
        return ["lowercase", "strip", "normalize_whitespace"]

    # Default: basic cleanup
    return ["strip"]


def suggest_scorer(column_name: str) -> str:
    """Suggest a scorer based on column name heuristics."""
    col = column_name.lower()

    if _matches_any(col, _NAME_KEYWORDS):
        return "jaro_winkler"
    if _matches_any(col, _EMAIL_KEYWORDS):
        return "levenshtein"
    if _matches_any(col, _PHONE_KEYWORDS):
        return "exact"
    if _matches_any(col, _ZIP_KEYWORDS):
        return "exact"
    if _matches_any(col, _ADDRESS_KEYWORDS):
        return "token_sort"

    return "jaro_winkler"


def run_wizard(output_path: str | Path | None = None) -> dict:
    """Run the interactive config wizard.

    Guides the user through building a GoldenMatch config via Rich prompts.

    Returns:
        The config as a dict (also saved to output_path if provided).
    """
    console.print("\n[bold cyan]GoldenMatch Config Wizard[/bold cyan]\n")

    # ── Mode selection ──
    mode = Prompt.ask(
        "What would you like to do?",
        choices=["dedupe", "match"],
        default="dedupe",
    )

    # ── File selection ──
    files: list[dict[str, str]] = []
    if mode == "dedupe":
        while True:
            path = Prompt.ask("Enter input file path")
            label = Prompt.ask("Source label", default=Path(path).stem)
            files.append({"path": path, "source_label": label})
            if not Confirm.ask("Add another file?", default=False):
                break
    else:
        target_path = Prompt.ask("Enter target file path")
        target_label = Prompt.ask("Target source label", default=Path(target_path).stem)
        files.append({"path": target_path, "source_label": target_label, "role": "target"})
        while True:
            ref_path = Prompt.ask("Enter reference file path")
            ref_label = Prompt.ask("Reference source label", default=Path(ref_path).stem)
            files.append({"path": ref_path, "source_label": ref_label, "role": "reference"})
            if not Confirm.ask("Add another reference file?", default=False):
                break

    # ── Matchkey building ──
    console.print("\n[bold]Matchkey Configuration[/bold]")
    matchkeys: list[dict[str, Any]] = []

    while True:
        mk_name = Prompt.ask("Matchkey name", default=f"mk_{len(matchkeys) + 1}")
        mk_type = Prompt.ask("Matchkey type", choices=["exact", "weighted"], default="exact")

        fields: list[dict[str, Any]] = []
        while True:
            field_name = Prompt.ask("  Field/column name")
            suggested_transforms = suggest_transforms(field_name)
            console.print(f"  [dim]Suggested transforms: {suggested_transforms}[/dim]")
            use_suggested = Confirm.ask("  Use suggested transforms?", default=True)
            transforms = suggested_transforms if use_suggested else []

            field_def: dict[str, Any] = {"field": field_name, "transforms": transforms}

            if mk_type == "weighted":
                scorer = suggest_scorer(field_name)
                console.print(f"  [dim]Suggested scorer: {scorer}[/dim]")
                use_scorer = Confirm.ask("  Use suggested scorer?", default=True)
                if not use_scorer:
                    scorer = Prompt.ask("  Scorer", choices=[
                        "exact", "jaro_winkler", "levenshtein", "token_sort", "soundex_match"
                    ])
                weight = float(Prompt.ask("  Weight", default="1.0"))
                field_def["scorer"] = scorer
                field_def["weight"] = weight

            fields.append(field_def)
            if not Confirm.ask("  Add another field to this matchkey?", default=False):
                break

        mk_def: dict[str, Any] = {"name": mk_name, "type": mk_type, "fields": fields}
        if mk_type == "weighted":
            threshold = float(Prompt.ask("  Match threshold (0.0-1.0)", default="0.8"))
            mk_def["threshold"] = threshold

        matchkeys.append(mk_def)
        if not Confirm.ask("Add another matchkey?", default=False):
            break

    # ── Blocking ──
    blocking: dict[str, Any] | None = None
    has_weighted = any(mk["type"] == "weighted" for mk in matchkeys)
    if has_weighted or Confirm.ask("\nConfigure blocking?", default=has_weighted):
        console.print("\n[bold]Blocking Configuration[/bold]")
        block_keys: list[dict[str, Any]] = []
        while True:
            block_fields_raw = Prompt.ask("  Blocking fields (comma-separated)")
            block_fields = [f.strip() for f in block_fields_raw.split(",") if f.strip()]
            block_keys.append({"fields": block_fields})
            if not Confirm.ask("  Add another blocking key?", default=False):
                break
        blocking = {"keys": block_keys}

    # ── Threshold (golden rules) ──
    golden_rules: dict[str, Any] | None = None
    if Confirm.ask("\nConfigure golden record rules?", default=False):
        strategy = Prompt.ask(
            "Default golden record strategy",
            choices=["most_complete", "most_recent", "source_priority", "majority_vote", "first_non_null"],
            default="most_complete",
        )
        golden_rules = {"default_strategy": strategy}

    # ── Output format ──
    console.print("\n[bold]Output Configuration[/bold]")
    out_format = Prompt.ask("Output format", choices=["csv", "parquet"], default="csv")
    out_dir = Prompt.ask("Output directory", default="./output")
    run_name = Prompt.ask("Run name (optional)", default="")

    # ── Build config dict ──
    config: dict[str, Any] = {"matchkeys": matchkeys}
    if blocking:
        config["blocking"] = blocking
    if golden_rules:
        config["golden_rules"] = golden_rules

    output_cfg: dict[str, str] = {"format": out_format, "directory": out_dir}
    if run_name:
        output_cfg["run_name"] = run_name
    config["output"] = output_cfg

    # ── Save ──
    if output_path is None:
        save = Confirm.ask("\nSave config to file?", default=True)
        if save:
            output_path = Prompt.ask("Output path", default="goldenmatch.yaml")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")
        console.print(f"\n[green]Config saved to {output_path}[/green]")

    return config
