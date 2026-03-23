#!/usr/bin/env python
"""Build a matching config programmatically -- no YAML needed.

Shows how to construct matchkeys, blocking, and golden rules entirely
in Python using the GoldenMatchConfig Pydantic model.
"""
import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import goldenmatch as gm
import polars as pl
import tempfile
from pathlib import Path

# Sample data
data = pl.DataFrame({
    "first_name": ["John", "Jon", "Jane", "Jonathan", "Bob", "Bobby"],
    "last_name": ["Smith", "Smith", "Doe", "Smith", "Jones", "Jones"],
    "zip": ["10001", "10001", "20002", "10001", "30003", "30003"],
    "email": ["john@x.com", "jon@x.com", "jane@y.com", "jsmith@x.com", "bob@z.com", "bobby@z.com"],
})

tmp = Path(tempfile.mkdtemp()) / "data.csv"
data.write_csv(tmp)

# Build config programmatically
config = gm.GoldenMatchConfig(
    matchkeys=[
        # Exact email match
        gm.MatchkeyConfig(
            name="exact_email",
            type="exact",
            fields=[gm.MatchkeyField(field="email", transforms=["lowercase", "strip"])],
        ),
        # Fuzzy name + zip
        gm.MatchkeyConfig(
            name="fuzzy_name_zip",
            type="weighted",
            threshold=0.80,
            fields=[
                gm.MatchkeyField(field="first_name", scorer="jaro_winkler", weight=0.4,
                                 transforms=["lowercase", "strip"]),
                gm.MatchkeyField(field="last_name", scorer="jaro_winkler", weight=0.4,
                                 transforms=["lowercase", "strip"]),
                gm.MatchkeyField(field="zip", scorer="exact", weight=0.2),
            ],
        ),
    ],
    blocking=gm.BlockingConfig(
        keys=[gm.BlockingKeyConfig(fields=["zip"])],
    ),
    golden_rules=gm.GoldenRulesConfig(
        default_strategy="most_complete",
    ),
)

print("Config built programmatically:")
print(f"  Matchkeys: {[mk.name for mk in config.get_matchkeys()]}")
print(f"  Blocking: {config.blocking.keys[0].fields}")
print(f"  Golden strategy: {config.golden_rules.default_strategy}")

# Run pipeline
result = gm.dedupe(str(tmp), config=config)
print(f"\nResult: {result}")

# Inspect clusters
for cid, cinfo in result.clusters.items():
    members = cinfo["members"]
    confidence = cinfo["confidence"]
    print(f"  Cluster {cid}: {len(members)} records, confidence={confidence:.2f}")
