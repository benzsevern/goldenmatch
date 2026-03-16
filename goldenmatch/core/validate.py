"""Column-level validation rules with quarantine support for GoldenMatch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl


@dataclass
class ValidationRule:
    """A single column validation rule.

    Attributes:
        column: Column name to validate.
        rule_type: One of: regex, min_length, max_length, not_null, in_set, format.
        params: Rule-specific parameters.
        action: "flag" (add flag column), "null" (set to null), "quarantine" (move row).
    """
    column: str
    rule_type: str
    params: dict = field(default_factory=dict)
    action: str = "flag"


def _check_format_email(value: str | None) -> bool:
    """Check if value looks like a valid email."""
    if value is None:
        return False
    value = str(value).strip().lower()
    if not value or "@" not in value:
        return False
    parts = value.split("@")
    if len(parts) != 2:
        return False
    return "." in parts[1] and len(parts[0]) > 0 and len(parts[1]) > 2


def _check_format_phone(value: str | None) -> bool:
    """Check if value looks like a valid phone number."""
    if value is None:
        return False
    digits = re.sub(r"\D", "", str(value))
    return len(digits) >= 7 and digits.isdigit()


def _check_format_zip5(value: str | None) -> bool:
    """Check if value looks like a valid 5-digit ZIP code."""
    if value is None:
        return False
    clean = str(value).strip().split("-")[0].split(" ")[0]
    digits = re.sub(r"\D", "", clean)
    return len(digits) >= 5 and digits[:5].isdigit()


_FORMAT_CHECKERS = {
    "email": _check_format_email,
    "phone": _check_format_phone,
    "zip5": _check_format_zip5,
}


def _evaluate_rule(series: pl.Series, rule: ValidationRule) -> pl.Series:
    """Evaluate a rule against a series, returning a boolean Series (True=passed)."""
    n = len(series)

    if rule.rule_type == "not_null":
        return series.is_not_null()

    if rule.rule_type == "regex":
        pattern = rule.params["pattern"]
        # Null values fail regex
        return series.is_not_null() & series.fill_null("").str.contains(pattern) & series.is_not_null()

    if rule.rule_type == "min_length":
        length = rule.params["length"]
        return series.is_not_null() & (series.fill_null("").str.len_chars() >= length) & series.is_not_null()

    if rule.rule_type == "max_length":
        length = rule.params["length"]
        return series.is_not_null() & (series.fill_null("").str.len_chars() <= length) & series.is_not_null()

    if rule.rule_type == "in_set":
        values = rule.params["values"]
        return series.is_not_null() & series.is_in(values)

    if rule.rule_type == "format":
        fmt_type = rule.params["type"]
        checker = _FORMAT_CHECKERS.get(fmt_type)
        if checker is None:
            raise ValueError(f"Unknown format type: {fmt_type!r}. Available: {sorted(_FORMAT_CHECKERS)}")
        # Use map_elements for format checking
        return series.map_elements(checker, return_dtype=pl.Boolean)

    raise ValueError(f"Unknown rule_type: {rule.rule_type!r}")


def validate_dataframe(
    df: pl.DataFrame,
    rules: list[ValidationRule],
) -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    """Validate a DataFrame against a list of rules.

    Returns:
        (valid_df, quarantine_df, validation_report)

        - valid_df: rows that passed all rules (with flags/nulls applied)
        - quarantine_df: rows that failed quarantine rules (with __quarantine_reason__)
        - validation_report: list of dicts with rule evaluation stats
    """
    validation_report: list[dict] = []
    quarantine_mask = pl.Series("__q__", [False] * df.height)
    quarantine_reasons: list[list[str]] = [[] for _ in range(df.height)]

    working_df = df.clone()

    for rule in rules:
        if rule.column not in working_df.columns:
            raise ValueError(f"Column {rule.column!r} not found in DataFrame")

        series = working_df[rule.column]
        passed = _evaluate_rule(series, rule)
        failed = ~passed

        total_checked = len(series)
        passed_count = int(passed.sum())
        failed_count = total_checked - passed_count
        fail_rate = failed_count / total_checked if total_checked > 0 else 0.0

        validation_report.append({
            "rule": rule.rule_type,
            "column": rule.column,
            "total_checked": total_checked,
            "passed": passed_count,
            "failed": failed_count,
            "fail_rate": fail_rate,
        })

        if rule.action == "flag":
            flag_col = f"__vf_{rule.column}_{rule.rule_type}__"
            working_df = working_df.with_columns(passed.alias(flag_col))

        elif rule.action == "null":
            # Set failing values to null
            working_df = working_df.with_columns(
                pl.when(passed)
                .then(pl.col(rule.column))
                .otherwise(None)
                .alias(rule.column)
            )

        elif rule.action == "quarantine":
            # Mark rows for quarantine
            quarantine_mask = quarantine_mask | failed
            failed_list = failed.to_list()
            for i, is_failed in enumerate(failed_list):
                if is_failed:
                    quarantine_reasons[i].append(
                        f"{rule.column}:{rule.rule_type}"
                    )

    # Split into valid and quarantine DataFrames
    quarantine_df = working_df.filter(quarantine_mask)
    valid_df = working_df.filter(~quarantine_mask)

    # Add quarantine reason column
    if quarantine_df.height > 0:
        # Build reason strings for quarantined rows
        reason_series_data = []
        quarantine_indices = quarantine_mask.to_list()
        for i, is_q in enumerate(quarantine_indices):
            if is_q:
                reason_series_data.append("; ".join(quarantine_reasons[i]))
        quarantine_df = quarantine_df.with_columns(
            pl.Series("__quarantine_reason__", reason_series_data)
        )
    else:
        quarantine_df = quarantine_df.with_columns(
            pl.Series("__quarantine_reason__", [], dtype=pl.Utf8)
        )

    return valid_df, quarantine_df, validation_report
