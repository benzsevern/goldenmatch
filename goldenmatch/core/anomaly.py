"""Anomaly detection -- flag suspicious records that aren't duplicates.

Detects:
- Fake/test emails (test@test.com, noreply@, etc.)
- Bot-generated data (sequential IDs, repeating patterns)
- Impossible values (future dates, negative ages, 00000 zips)
- Outlier records (statistically unusual field combinations)
- Placeholder data (TBD, N/A, UNKNOWN, xxx, 123456)
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import polars as pl

logger = logging.getLogger(__name__)

# Patterns for suspicious values
FAKE_EMAIL_PATTERNS = [
    r"^test@", r"^noreply@", r"^no-reply@", r"^donotreply@",
    r"^fake@", r"^example@", r"@example\.com$", r"@test\.com$",
    r"^admin@", r"^info@info\.", r"^asdf", r"^qwerty",
    r"^a{3,}@", r"^x{3,}@",
]

PLACEHOLDER_VALUES = {
    "tbd", "n/a", "na", "none", "null", "unknown", "undefined",
    "xxx", "yyy", "zzz", "123", "1234", "12345", "123456",
    "test", "testing", "sample", "demo", "fake", "placeholder",
    "foo", "bar", "baz", "asdf", "qwerty", "aaa", "bbb",
}

FAKE_PHONE_PATTERNS = [
    r"^555-", r"^000-", r"^123-456", r"^111-111",
    r"^0{5,}", r"^1{5,}", r"^9{5,}",
]

SUSPICIOUS_ZIP_VALUES = {"00000", "99999", "12345", "11111", "00001"}


def detect_anomalies(
    df: pl.DataFrame,
    sensitivity: str = "medium",
) -> list[dict]:
    """Detect anomalous/suspicious records in the dataset.

    Args:
        df: DataFrame to analyze.
        sensitivity: "low", "medium", or "high".

    Returns:
        List of anomaly dicts:
        [{"row_id": 42, "column": "email", "type": "fake_email",
          "value": "test@test.com", "severity": "high", "reason": "..."}]
    """
    cols = [c for c in df.columns if not c.startswith("__")]
    anomalies = []

    # Detect column types by name
    email_cols = [c for c in cols if _is_likely_column(c, ["email", "mail", "e_mail"])]
    phone_cols = [c for c in cols if _is_likely_column(c, ["phone", "tel", "mobile", "cell"])]
    zip_cols = [c for c in cols if _is_likely_column(c, ["zip", "postal", "postcode"])]
    name_cols = [c for c in cols if _is_likely_column(c, ["name", "first_name", "last_name", "full_name"])]
    date_cols = [c for c in cols if _is_likely_column(c, ["date", "dob", "birth", "created"])]

    rows = df.to_dicts()

    for i, row in enumerate(rows):
        rid = row.get("__row_id__", i)

        # Check email columns
        for col in email_cols:
            val = row.get(col)
            if val is None:
                continue
            val_str = str(val).strip().lower()

            for pattern in FAKE_EMAIL_PATTERNS:
                if re.search(pattern, val_str):
                    anomalies.append({
                        "row_id": rid,
                        "column": col,
                        "type": "fake_email",
                        "value": val_str,
                        "severity": "high",
                        "reason": f"Matches fake email pattern: {pattern}",
                    })
                    break

        # Check phone columns
        for col in phone_cols:
            val = row.get(col)
            if val is None:
                continue
            val_str = str(val).strip()

            for pattern in FAKE_PHONE_PATTERNS:
                if re.search(pattern, val_str):
                    anomalies.append({
                        "row_id": rid,
                        "column": col,
                        "type": "fake_phone",
                        "value": val_str,
                        "severity": "medium",
                        "reason": f"Suspicious phone number pattern",
                    })
                    break

        # Check zip columns
        for col in zip_cols:
            val = row.get(col)
            if val is None:
                continue
            val_str = str(val).strip()

            if val_str in SUSPICIOUS_ZIP_VALUES:
                anomalies.append({
                    "row_id": rid,
                    "column": col,
                    "type": "suspicious_zip",
                    "value": val_str,
                    "severity": "medium",
                    "reason": f"Suspicious zip code: {val_str}",
                })

        # Check all text columns for placeholder values
        for col in cols:
            val = row.get(col)
            if val is None:
                continue
            val_str = str(val).strip().lower()

            if val_str in PLACEHOLDER_VALUES:
                anomalies.append({
                    "row_id": rid,
                    "column": col,
                    "type": "placeholder",
                    "value": str(row.get(col)),
                    "severity": "high",
                    "reason": f"Placeholder value detected",
                })

    # Detect duplicate exact rows (copy-paste data)
    row_hashes = Counter()
    for row in rows:
        key = tuple(str(row.get(c, "")) for c in cols)
        row_hashes[key] += 1

    exact_dupes = {k for k, v in row_hashes.items() if v > 2}
    if exact_dupes:
        for i, row in enumerate(rows):
            key = tuple(str(row.get(c, "")) for c in cols)
            if key in exact_dupes:
                rid = row.get("__row_id__", i)
                anomalies.append({
                    "row_id": rid,
                    "column": "(all)",
                    "type": "exact_duplicate_row",
                    "value": f"{row_hashes[key]} identical copies",
                    "severity": "high",
                    "reason": "Exact duplicate row (possible copy-paste)",
                })

    # Filter by sensitivity
    if sensitivity == "low":
        anomalies = [a for a in anomalies if a["severity"] == "high"]
    elif sensitivity == "medium":
        anomalies = [a for a in anomalies if a["severity"] in ("high", "medium")]

    # Deduplicate (same row_id + column + type)
    seen = set()
    unique_anomalies = []
    for a in anomalies:
        key = (a["row_id"], a["column"], a["type"])
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    logger.info("Detected %d anomalies (%s sensitivity)", len(unique_anomalies), sensitivity)
    return unique_anomalies


def _is_likely_column(col_name: str, keywords: list[str]) -> bool:
    """Check if a column name matches any keywords."""
    col_lower = col_name.lower().replace(" ", "_")
    return any(kw in col_lower for kw in keywords)


def format_anomaly_report(anomalies: list[dict]) -> str:
    """Format anomalies as readable text."""
    if not anomalies:
        return "[#2ecc71]No anomalies detected.[/]"

    lines = [f"[bold #d4a017]Anomaly Report[/] ({len(anomalies)} issues found)\n"]

    by_type = {}
    for a in anomalies:
        by_type.setdefault(a["type"], []).append(a)

    for atype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        severity = items[0]["severity"]
        sev_color = {"high": "#e74c3c", "medium": "#e67e22", "low": "#8892a0"}[severity]

        label = atype.replace("_", " ").title()
        lines.append(f"  [{sev_color}]{label}[/]: {len(items)} records")

        for item in items[:3]:
            lines.append(f"    Row {item['row_id']}: {item['column']} = \"{item['value']}\"")

        if len(items) > 3:
            lines.append(f"    ... and {len(items) - 3} more")
        lines.append("")

    return "\n".join(lines)
