#!/usr/bin/env python
"""Privacy-preserving patient matching across hospitals.

Two hospitals need to find shared patients without sharing raw data.
GoldenMatch encrypts records into bloom filters and matches on the
encrypted representations.
"""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import goldenmatch as gm
import polars as pl
import tempfile
from pathlib import Path

# Hospital A patient records
hospital_a = pl.DataFrame({
    "first_name": ["John", "Jane", "Robert", "Alice", "Michael"],
    "last_name": ["Smith", "Doe", "Johnson", "Williams", "Brown"],
    "dob": ["1990-01-15", "1985-03-22", "1978-11-08", "1992-07-30", "1970-05-12"],
    "zip": ["10001", "20002", "30003", "10001", "40004"],
})

# Hospital B patient records (some overlap, with typos)
hospital_b = pl.DataFrame({
    "first_name": ["john", "Janet", "Robert", "Charlie", "michael"],
    "last_name": ["Smith", "Doe", "Johnson", "Wilson", "Brown"],
    "dob": ["1990-01-15", "1985-03-22", "1978-11-08", "1995-12-01", "1970-05-12"],
    "zip": ["10001", "20002", "30003", "50005", "40004"],
})

tmp = Path(tempfile.mkdtemp())
a_path = tmp / "hospital_a.csv"
b_path = tmp / "hospital_b.csv"
hospital_a.write_csv(a_path)
hospital_b.write_csv(b_path)

print(f"Hospital A: {hospital_a.height} patients")
print(f"Hospital B: {hospital_b.height} patients")

# Auto-configured PPRL (picks fields and threshold automatically)
print("\n--- Auto-configured PPRL ---")
result = gm.pprl_link(str(a_path), str(b_path))
print(f"Matches found: {result['match_count']}")
print(f"Clusters: {len(result['clusters'])}")
print(f"Config used: {result['config']}")

# Manual PPRL with specific fields
print("\n--- Manual PPRL (high security) ---")
result = gm.pprl_link(
    str(a_path), str(b_path),
    fields=["first_name", "last_name", "dob"],
    threshold=0.80,
    security_level="high",
)
print(f"Matches found: {result['match_count']}")
for cid, members in result["clusters"].items():
    a_members = [f"A:{rid}" for pid, rid in members if pid == "party_a"]
    b_members = [f"B:{rid}" for pid, rid in members if pid == "party_b"]
    print(f"  Cluster {cid}: {a_members} <-> {b_members}")
