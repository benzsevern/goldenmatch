"""Data quality profiler for GoldenMatch."""

from __future__ import annotations

import re
from typing import Any

import polars as pl
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── Heuristic type detection helpers ────────────────────────────────────────

_PHONE_STRIP_RE = re.compile(r"[()\-+.\s]")
_DATE_PATTERNS = [
    re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$"),
    re.compile(r"^\d{4}[/\-]\d{1,2}[/\-]\d{1,2}$"),
    re.compile(r"^\d{1,2}\s\w+\s\d{2,4}$"),
]
_ADDRESS_WORDS = re.compile(
    r"\b(st|street|ave|avenue|rd|road|dr|drive|blvd|boulevard|ln|lane|ct|court|way|pl|place|cir|circle)\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z \-']{0,28}[A-Za-z]$|^[A-Za-z]{2,3}$")


def _guess_type(values: list[str]) -> str:
    """Heuristic guess of what the string data looks like."""
    if not values:
        return "text"

    n = len(values)

    # email: >60% contain @ and a dot after @
    email_count = sum(1 for v in values if "@" in v and "." in v.split("@")[-1])
    if email_count / n > 0.6:
        return "email"

    # phone: >60% are mostly digits after stripping common phone chars
    phone_count = 0
    for v in values:
        stripped = _PHONE_STRIP_RE.sub("", v)
        if stripped.isdigit() and 7 <= len(stripped) <= 15:
            phone_count += 1
    if phone_count / n > 0.6:
        return "phone"

    # zip: >60% are 5 or 9-10 digit strings
    zip_count = 0
    for v in values:
        clean = v.replace("-", "")
        if clean.isdigit() and len(clean) in (5, 9):
            zip_count += 1
    if zip_count / n > 0.6:
        return "zip"

    # state: >60% are exactly 2 uppercase letters
    state_count = sum(1 for v in values if len(v) == 2 and v.isalpha() and v.isupper())
    if state_count / n > 0.6:
        return "state"

    # numeric: >60% parse as numbers
    numeric_count = 0
    for v in values:
        try:
            float(v.replace(",", ""))
            numeric_count += 1
        except ValueError:
            pass
    if numeric_count / n > 0.6:
        return "numeric"

    # name: >60% are 2-30 chars, alpha + spaces + hyphens only
    name_count = sum(1 for v in values if _NAME_RE.match(v.strip()))
    if name_count / n > 0.6:
        return "name"

    # address: >40% contain digits AND common address words
    addr_count = sum(
        1 for v in values
        if any(c.isdigit() for c in v) and _ADDRESS_WORDS.search(v)
    )
    if addr_count / n > 0.4:
        return "address"

    # date: >40% match common date patterns
    date_count = sum(
        1 for v in values
        if any(p.match(v.strip()) for p in _DATE_PATTERNS)
    )
    if date_count / n > 0.4:
        return "date"

    return "text"


# ── Column profiling ───────────────────────────────────────────────────────


def profile_column(series: pl.Series) -> dict[str, Any]:
    """Profile a single column and return a statistics dict."""
    name = series.name
    dtype = str(series.dtype)
    total = len(series)
    null_count = series.null_count()
    null_rate = null_count / total if total > 0 else 0.0

    non_null = series.drop_nulls()
    non_null_count = len(non_null)
    unique_count = non_null.n_unique() if non_null_count > 0 else 0
    unique_rate = unique_count / non_null_count if non_null_count > 0 else 0.0

    is_string = series.dtype in (pl.Utf8, pl.String)

    min_length: int | None = None
    max_length: int | None = None
    avg_length: float | None = None
    empty_string_count = 0
    suspected_type = "text"

    if is_string:
        lengths = non_null.str.len_chars()
        if non_null_count > 0:
            min_length = int(lengths.min())  # type: ignore[arg-type]
            max_length = int(lengths.max())  # type: ignore[arg-type]
            avg_length = float(lengths.mean())  # type: ignore[arg-type]

        # Count empty or whitespace-only strings (excluding actual nulls)
        if non_null_count > 0:
            empty_string_count = int((non_null.str.strip_chars() == "").sum())
        else:
            empty_string_count = 0

        # Heuristic type detection on non-null, non-empty values
        non_empty_vals = [
            v for v in non_null.to_list() if isinstance(v, str) and v.strip()
        ]
        suspected_type = _guess_type(non_empty_vals)
    else:
        empty_string_count = 0

    # Sample values
    sample_values: list[Any] = non_null.head(5).to_list() if non_null_count > 0 else []

    return {
        "name": name,
        "dtype": dtype,
        "total": total,
        "null_count": null_count,
        "null_rate": null_rate,
        "unique_count": unique_count,
        "unique_rate": unique_rate,
        "min_length": min_length,
        "max_length": max_length,
        "avg_length": avg_length,
        "sample_values": sample_values,
        "empty_string_count": empty_string_count,
        "suspected_type": suspected_type,
    }


# ── DataFrame profiling ───────────────────────────────────────────────────


def profile_dataframe(df: pl.DataFrame) -> dict[str, Any]:
    """Profile an entire DataFrame and return a comprehensive report dict."""
    total_rows = df.height
    total_columns = df.width
    columns = [profile_column(df[col]) for col in df.columns]

    # Duplicate rows: count of rows that appear more than once
    duplicate_row_count = total_rows - df.unique().height

    # Empty rows: all values are null or empty string
    empty_conditions = []
    for col in df.columns:
        if df[col].dtype in (pl.Utf8, pl.String):
            empty_conditions.append(
                pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")
            )
        else:
            empty_conditions.append(pl.col(col).is_null())

    if empty_conditions:
        all_empty_expr = empty_conditions[0]
        for cond in empty_conditions[1:]:
            all_empty_expr = all_empty_expr & cond
        empty_row_count = df.filter(all_empty_expr).height
    else:
        empty_row_count = 0

    # Issue detection
    issues: list[dict[str, Any]] = []

    # Name-to-type mapping for mismatch detection
    _name_type_hints = {
        "email": "email",
        "e_mail": "email",
        "phone": "phone",
        "telephone": "phone",
        "tel": "phone",
        "zip": "zip",
        "zipcode": "zip",
        "zip_code": "zip",
        "postal": "zip",
        "name": "name",
        "first_name": "name",
        "last_name": "name",
        "fname": "name",
        "lname": "name",
        "address": "address",
        "addr": "address",
        "street": "address",
        "state": "state",
    }

    for cp in columns:
        col_name = cp["name"]
        null_rate = cp["null_rate"]

        # ERROR: >95% nulls
        if null_rate > 0.95:
            issues.append({
                "severity": "error",
                "column": col_name,
                "message": f"Column '{col_name}' has {null_rate:.0%} null values (likely empty/wrong column).",
            })
        # WARNING: >50% nulls
        elif null_rate > 0.50:
            issues.append({
                "severity": "warning",
                "column": col_name,
                "message": f"Column '{col_name}' has {null_rate:.0%} null values.",
            })

        # WARNING: suspected type mismatch
        col_lower = col_name.lower().replace(" ", "_")
        expected_type = _name_type_hints.get(col_lower)
        if expected_type and cp["suspected_type"] != expected_type:
            issues.append({
                "severity": "warning",
                "column": col_name,
                "message": (
                    f"Column '{col_name}' is named like a {expected_type} column "
                    f"but data looks like '{cp['suspected_type']}'."
                ),
            })

        # WARNING: >20% empty strings
        if cp["total"] > 0 and cp["empty_string_count"] / cp["total"] > 0.20:
            issues.append({
                "severity": "warning",
                "column": col_name,
                "message": f"Column '{col_name}' has {cp['empty_string_count']} empty/whitespace-only values ({cp['empty_string_count'] / cp['total']:.0%}).",
            })

        # INFO: low cardinality
        non_null_count = cp["total"] - cp["null_count"]
        if cp["total"] > 100 and cp["unique_count"] < 5 and non_null_count > 0:
            issues.append({
                "severity": "info",
                "column": col_name,
                "message": f"Column '{col_name}' has very low cardinality ({cp['unique_count']} unique values).",
            })

        # INFO: appears to be an ID
        if cp["unique_rate"] == 1.0 and cp["null_count"] == 0:
            issues.append({
                "severity": "info",
                "column": col_name,
                "message": f"Column '{col_name}' appears to be a unique ID (100% unique, no nulls).",
            })

    # WARNING: duplicate rows
    if duplicate_row_count > 0:
        issues.append({
            "severity": "warning",
            "column": None,
            "message": f"Dataset contains {duplicate_row_count} duplicate row(s).",
        })

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "columns": columns,
        "duplicate_row_count": duplicate_row_count,
        "empty_row_count": empty_row_count,
        "issues": issues,
    }


# ── Report formatting ─────────────────────────────────────────────────────

_SEVERITY_STYLE = {
    "error": "bold red",
    "warning": "yellow",
    "info": "dim",
}

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def format_profile_report(profile: dict, df: pl.DataFrame | None = None) -> str:
    """Return a Rich-renderable string summary of a profile report.

    Args:
        profile: The output of profile_dataframe().
        df: Optional DataFrame to show a sample from.

    Returns:
        A string containing Rich markup for display.
    """
    from io import StringIO

    from rich.console import Console as _Console

    buf = StringIO()
    console = _Console(file=buf, force_terminal=True, width=120)

    # ── Overview panel ────────────────────────────────────────────────
    overview_lines = [
        f"[bold]Rows:[/bold] {profile['total_rows']}",
        f"[bold]Columns:[/bold] {profile['total_columns']}",
        f"[bold]Duplicate rows:[/bold] {profile['duplicate_row_count']}",
        f"[bold]Empty rows:[/bold] {profile['empty_row_count']}",
    ]
    console.print(Panel("\n".join(overview_lines), title="Overview", border_style="cyan"))

    # ── Per-column table ──────────────────────────────────────────────
    col_table = Table(title="Column Summary", show_lines=True)
    col_table.add_column("Name", style="bold")
    col_table.add_column("Type")
    col_table.add_column("Suspected")
    col_table.add_column("Null %", justify="right")
    col_table.add_column("Unique %", justify="right")
    col_table.add_column("Issues", justify="right")

    issue_counts: dict[str, int] = {}
    for iss in profile["issues"]:
        col = iss.get("column")
        if col:
            issue_counts[col] = issue_counts.get(col, 0) + 1

    for cp in profile["columns"]:
        null_pct = f"{cp['null_rate']:.1%}"
        unique_pct = f"{cp['unique_rate']:.1%}"
        n_issues = issue_counts.get(cp["name"], 0)
        issue_str = str(n_issues) if n_issues else "-"
        col_table.add_row(
            cp["name"],
            cp["dtype"],
            cp["suspected_type"],
            null_pct,
            unique_pct,
            issue_str,
        )

    console.print(col_table)

    # ── Issues list ───────────────────────────────────────────────────
    sorted_issues = sorted(
        profile["issues"],
        key=lambda i: _SEVERITY_ORDER.get(i["severity"], 99),
    )
    if sorted_issues:
        console.print()
        console.print(Panel("[bold]Issues[/bold]", border_style="yellow"))
        for iss in sorted_issues:
            sev = iss["severity"].upper()
            style = _SEVERITY_STYLE.get(iss["severity"], "")
            console.print(f"  [{style}][{sev}][/{style}] {iss['message']}")
    else:
        console.print()
        console.print("[green]No issues detected.[/green]")

    # ── Data sample ───────────────────────────────────────────────────
    if df is not None and df.height > 0:
        console.print()
        sample = df.head(5)
        sample_table = Table(title="Data Sample (first 5 rows)", show_lines=True)
        for col in sample.columns:
            sample_table.add_column(col)
        for row in sample.iter_rows():
            sample_table.add_row(*(str(v) if v is not None else "[dim]null[/dim]" for v in row))
        console.print(sample_table)

    return buf.getvalue()
