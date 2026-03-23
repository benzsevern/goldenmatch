#!/usr/bin/env python
"""Basic deduplication -- the simplest possible GoldenMatch workflow.

Generates sample customer data with duplicates, deduplicates it,
and prints the results. No config file needed.
"""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import goldenmatch as gm
import polars as pl
import tempfile
from pathlib import Path

# Generate sample data with duplicates
data = pl.DataFrame({
    "first_name": ["John", "john", "Jane", "JOHN", "Bob", "jane", "Robert"],
    "last_name": ["Smith", "Smith", "Doe", "Smyth", "Jones", "Doe", "Jones"],
    "email": ["john@example.com", "john@example.com", "jane@test.com",
              "john@example.com", "bob@work.com", "jane@test.com", "bob@work.com"],
    "phone": ["555-1234", "555-1234", "555-5678", "555-1234",
              "555-9999", "555-5678", "555-9999"],
})

# Save to temp file
tmp = Path(tempfile.mkdtemp()) / "customers.csv"
data.write_csv(tmp)
print(f"Input: {data.height} records")
print(data)

# Deduplicate with exact email matching
result = gm.dedupe(str(tmp), exact=["email"])

print(f"\nResult: {result}")
print(f"Golden records: {result.golden.height if result.golden is not None else 0}")
print(f"Duplicates found: {result.dupes.height if result.dupes is not None else 0}")

if result.golden is not None:
    print("\nGolden records:")
    print(result.golden)
