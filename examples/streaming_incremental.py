#!/usr/bin/env python
"""Streaming / incremental matching -- process new records one at a time.

Shows how to match incoming records against an existing dataset
without re-running the full pipeline.
"""
import goldenmatch as gm
import polars as pl

# Existing customer database
existing = pl.DataFrame({
    "__row_id__": [0, 1, 2, 3, 4],
    "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
    "last_name": ["Smith", "Doe", "Jones", "Brown", "Wilson"],
    "email": ["john@x.com", "jane@y.com", "bob@z.com", "alice@w.com", "charlie@v.com"],
}).with_columns(pl.col("__row_id__").cast(pl.Int64))

# Define matchkey
mk = gm.MatchkeyConfig(
    name="fuzzy_name",
    type="weighted",
    threshold=0.80,
    fields=[
        gm.MatchkeyField(field="first_name", scorer="jaro_winkler", weight=0.5,
                         transforms=["lowercase"]),
        gm.MatchkeyField(field="last_name", scorer="jaro_winkler", weight=0.5,
                         transforms=["lowercase"]),
    ],
)

# New records arriving one at a time
new_records = [
    {"first_name": "john", "last_name": "Smith", "email": "jsmith@new.com"},
    {"first_name": "Janet", "last_name": "Doe", "email": "janet@new.com"},
    {"first_name": "Xavier", "last_name": "Zhang", "email": "xavier@new.com"},
    {"first_name": "Bobby", "last_name": "Jones", "email": "bobby@new.com"},
]

print(f"Existing database: {existing.height} records")
print(f"New records to match: {len(new_records)}\n")

for i, record in enumerate(new_records):
    matches = gm.match_one(record, existing, mk)
    name = f"{record['first_name']} {record['last_name']}"
    if matches:
        best_id, best_score = matches[0]
        matched_row = existing.filter(pl.col("__row_id__") == best_id).to_dicts()[0]
        matched_name = f"{matched_row['first_name']} {matched_row['last_name']}"
        print(f"  {name} -> MATCH: {matched_name} (score={best_score:.2f})")
    else:
        print(f"  {name} -> NEW ENTITY (no match found)")
