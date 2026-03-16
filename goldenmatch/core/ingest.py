"""File ingest utilities for GoldenMatch."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_file(
    path: Path | str,
    delimiter: str = ",",
    encoding: str = "utf8",
    sheet: str | None = None,
) -> pl.LazyFrame:
    """Load a data file as a Polars LazyFrame.

    Supports CSV (.csv), Parquet (.parquet), and Excel (.xlsx) formats.

    Args:
        path: Path to the file.
        delimiter: Column delimiter for CSV files.
        encoding: Text encoding for CSV files.
        sheet: Sheet name for Excel files (optional).

    Returns:
        A Polars LazyFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pl.scan_csv(path, separator=delimiter, encoding=encoding)
    elif suffix == ".parquet":
        return pl.scan_parquet(path)
    elif suffix == ".xlsx":
        kwargs = {"engine": "openpyxl"}
        if sheet is not None:
            kwargs["sheet_name"] = sheet
        df = pl.read_excel(path, **kwargs)
        return df.lazy()
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}")


def load_files(file_specs: list[tuple[Path | str, str]]) -> list[pl.LazyFrame]:
    """Load multiple files, adding a __source__ column to each.

    Args:
        file_specs: List of (path, source_name) tuples.

    Returns:
        List of LazyFrames, each with a __source__ column.
    """
    frames = []
    for path, source_name in file_specs:
        lf = load_file(path)
        lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
        frames.append(lf)
    return frames


def validate_columns(lf: pl.LazyFrame, required: list[str]) -> None:
    """Check that required columns exist in a LazyFrame schema.

    Args:
        lf: The LazyFrame to validate.
        required: List of column names that must be present.

    Raises:
        ValueError: If any required columns are missing.
    """
    available = list(lf.collect_schema().names())
    missing = [c for c in required if c not in available]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {available}"
        )
