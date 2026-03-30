#!/usr/bin/env python
"""Advanced configuration -- multi-pass blocking, ensemble scoring.

Shows how to configure GoldenMatch for maximum accuracy using
standardization, multi-pass blocking, and weighted matchkeys
with multiple scorers.

Usage:
    pip install goldenmatch
    python examples/advanced_config.py
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


def create_tricky_data() -> Path:
    """Create data with subtle duplicates that need advanced config."""
    rows = [
        ["first_name", "last_name", "email", "phone", "address", "city"],
        ["John", "Smith", "john@acme.com", "(555) 123-4567", "123 Main St", "New York"],
        ["john", "smith", "john@acme.com", "555.123.4567", "123 Main Street", "new york"],
        ["Jon", "Smyth", "jon.smyth@acme.com", "(555) 123-4567", "123 Main St", "New York"],
        ["Jane", "Doe", "jane@corp.com", "555-987-6543", "456 Oak Ave", "Chicago"],
        ["  JANE  ", "DOE", "JANE@CORP.COM", "(555)987-6543", "456 Oak Avenue", "  chicago  "],
        ["Bob", "Wilson", "bob@test.com", "555-111-2222", "789 Elm St", "Boston"],
    ]
    path = Path(tempfile.mktemp(suffix=".csv"))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


if __name__ == "__main__":
    import goldenmatch as gm
    import polars as pl

    path = create_tricky_data()
    df = pl.read_csv(path)
    print("=" * 60)
    print("GoldenMatch -- Advanced Configuration")
    print("=" * 60)
    print(f"\nInput: {df.shape[0]} records with tricky duplicates")
    print(df)

    # Build a full config programmatically
    config = gm.GoldenMatchConfig(
        # Standardize fields before matching
        standardization=gm.StandardizationConfig(
            rules={
                "first_name": ["strip", "name_proper"],
                "last_name": ["strip", "name_proper"],
                "email": ["email"],
                "phone": ["phone"],
                "address": ["address"],
                "city": ["strip", "name_proper"],
            }
        ),
        # Multi-pass blocking: try multiple keys, union candidates
        blocking=gm.BlockingConfig(
            strategy="multi_pass",
            keys=[gm.BlockingKeyConfig(fields=["email"], transforms=["lowercase", "strip"])],
            passes=[
                gm.BlockingKeyConfig(fields=["email"], transforms=["lowercase", "strip"]),
                gm.BlockingKeyConfig(fields=["last_name"], transforms=["lowercase", "strip"]),
            ],
        ),
        # Weighted matchkey with per-field scorers
        matchkeys=[
            gm.MatchkeyConfig(
                name="identity",
                type="weighted",
                threshold=0.75,
                fields=[
                    gm.MatchkeyField(
                        field="first_name", scorer="jaro_winkler", weight=1.0,
                        transforms=["lowercase", "strip"],
                    ),
                    gm.MatchkeyField(
                        field="last_name", scorer="jaro_winkler", weight=1.0,
                        transforms=["lowercase", "strip"],
                    ),
                    gm.MatchkeyField(
                        field="email", scorer="jaro_winkler", weight=0.8,
                        transforms=["lowercase", "strip"],
                    ),
                    gm.MatchkeyField(
                        field="address", scorer="token_sort", weight=0.6,
                        transforms=["lowercase", "strip"],
                    ),
                ],
            ),
        ],
        # Golden record selection strategy
        golden_rules=gm.GoldenRulesConfig(default_strategy="most_complete"),
    )

    print(f"\nConfig:")
    print(f"  Matchkeys: {[mk.name for mk in config.get_matchkeys()]}")
    print(f"  Blocking: multi_pass ({len(config.blocking.passes)} passes)")
    print(f"  Standardization: {list(config.standardization.rules.keys())}")
    print(f"  Golden strategy: {config.golden_rules.default_strategy}")

    # Run pipeline
    result = gm.dedupe_df(df, config=config)

    print(f"\nResults:")
    print(f"  Clusters: {result.total_clusters}")
    print(f"  Match rate: {result.match_rate:.1%}")

    # Inspect clusters
    for cid, cinfo in result.clusters.items():
        members = cinfo["members"]
        confidence = cinfo["confidence"]
        print(f"  Cluster {cid}: {len(members)} records, confidence={confidence:.2f}")

    if result.golden is not None:
        display_cols = [c for c in result.golden.columns if not c.startswith("__")]
        print("\nGolden Records:")
        print(result.golden.select(display_cols))

    path.unlink()
