"""Output writer for GoldenMatch results."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def write_output(
    df: pl.DataFrame,
    directory: str | Path,
    run_name: str,
    output_type: str,
    fmt: str,
) -> Path:
    """Write a DataFrame to the specified format.

    Supports csv, parquet, and xlsx formats.
    Returns the Path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{run_name}_{output_type}.{fmt}"
    path = directory / filename

    if fmt == "csv":
        df.write_csv(path)
    elif fmt == "parquet":
        df.write_parquet(path)
    elif fmt == "xlsx":
        df.write_excel(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return path
