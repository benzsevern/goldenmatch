"""Database-side blocking — translate blocking keys into SQL WHERE clauses."""

from __future__ import annotations

import logging
import re

from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig
from goldenmatch.db.connector import _quote_ident

logger = logging.getLogger(__name__)


def build_blocking_query(
    table: str,
    record: dict,
    blocking_config: BlockingConfig,
    exclude_id: int | None = None,
    id_column: str = "id",
    limit: int = 1000,
) -> str:
    """Build SQL WHERE clause from blocking config and a new record.

    Translates blocking key transforms into Postgres functions:
    - soundex → soundex()
    - lowercase → lower()
    - substring:0:5 → substring(col, 1, 5)
    - strip → trim()
    """
    conditions = []

    for key_config in blocking_config.keys:
        field_conditions = []
        for field_name in key_config.fields:
            value = record.get(field_name)
            if value is None:
                continue

            col_expr = _quote_ident(field_name)
            val_expr = _escape_value(str(value))

            # Apply transforms to both column and value
            for transform in key_config.transforms:
                col_expr, val_expr = _apply_sql_transform(col_expr, val_expr, transform)

            field_conditions.append(f"{col_expr} = {val_expr}")

        if field_conditions:
            conditions.append("(" + " AND ".join(field_conditions) + ")")

    if not conditions:
        return ""

    where = " OR ".join(conditions)

    # Exclude the record itself
    exclude = ""
    if exclude_id is not None:
        exclude = f" AND {_quote_ident(id_column)} != {exclude_id}"

    return (
        f"SELECT * FROM {_quote_ident(table)} "
        f"WHERE ({where}){exclude} "
        f"LIMIT {limit}"
    )


def _apply_sql_transform(col_expr: str, val_expr: str, transform: str) -> tuple[str, str]:
    """Apply a transform to both column expression and value expression."""
    if transform == "lowercase":
        return f"lower({col_expr})", f"lower({val_expr})"
    elif transform == "uppercase":
        return f"upper({col_expr})", f"upper({val_expr})"
    elif transform == "strip":
        return f"trim({col_expr})", f"trim({val_expr})"
    elif transform == "soundex":
        return f"soundex({col_expr})", f"soundex({val_expr})"
    elif transform.startswith("substring:"):
        parts = transform.split(":")
        start = int(parts[1]) + 1  # SQL is 1-indexed
        length = int(parts[2]) - int(parts[1])
        return (
            f"substring({col_expr}, {start}, {length})",
            f"substring({val_expr}, {start}, {length})",
        )
    else:
        # Unknown transform — skip SQL-side, will be handled in Python
        logger.debug("Transform '%s' not supported in SQL, skipping", transform)
        return col_expr, val_expr


def _escape_value(value: str) -> str:
    """Escape a string value for SQL."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"
