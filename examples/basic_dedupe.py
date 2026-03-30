#!/usr/bin/env python
"""Basic deduplication -- find and merge duplicate records.

Generates sample customer data with duplicates (different casing,
phone formats, etc.), deduplicates it, and prints the results.

Usage:
    pip install goldenmatch
    python examples/basic_dedupe.py
"""
from __future__ import annotations

import csv
import sys
import os
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def create_sample_data() -> Path:
    """Create a CSV with realistic duplicates."""
    rows = [
        ["first_name", "last_name", "email", "phone", "city"],
        ["John", "Smith", "john@acme.com", "(555) 123-4567", "New York"],
        ["john", "smith", "john@acme.com", "555.123.4567", "New York"],
        ["JOHN", "SMITH", "JOHN@ACME.COM", "5551234567", "new york"],
        ["Jane", "Doe", "jane@corp.com", "555-987-6543", "Chicago"],
        ["Bob", "Wilson", "bob@test.com", "555-111-2222", "Boston"],
        ["Robert", "Wilson", "bob@test.com", "(555) 111-2222", "Boston"],
    ]
    path = Path(tempfile.mktemp(suffix=".csv"))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


if __name__ == "__main__":
    import goldenmatch as gm
    import polars as pl

    path = create_sample_data()
    print("=" * 60)
    print("GoldenMatch -- Basic Deduplication")
    print("=" * 60)

    df = pl.read_csv(path)
    print(f"\nInput: {df.shape[0]} records")
    print(df)

    # Deduplicate with exact email matching
    result = gm.dedupe(str(path), exact=["email"])

    print(f"\nResults:")
    print(f"  Total records: {result.total_records}")
    print(f"  Clusters found: {result.total_clusters}")
    print(f"  Match rate: {result.match_rate:.1%}")
    print(f"  Duplicates: {result.dupes.shape[0] if result.dupes is not None else 0}")
    print(f"  Unique records: {result.unique.shape[0] if result.unique is not None else 0}")
    print(f"  Golden records: {result.golden.shape[0] if result.golden is not None else 0}")

    if result.golden is not None:
        print("\nGolden Records (canonical):")
        display_cols = [c for c in result.golden.columns if not c.startswith("__")]
        print(result.golden.select(display_cols))

    path.unlink()
