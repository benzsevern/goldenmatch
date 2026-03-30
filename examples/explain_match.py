#!/usr/bin/env python
"""Explain matching decisions -- see per-field score breakdowns.

Uses explain_pair_df to produce a natural language explanation
of why two records match (or don't).

Usage:
    pip install goldenmatch
    python examples/explain_match.py
"""
from __future__ import annotations

import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    import goldenmatch as gm

    print("=" * 60)
    print("GoldenMatch -- Explain Match")
    print("=" * 60)

    # --- Pair 1: same person, different formatting ---
    a = {"first_name": "John", "last_name": "Smith", "email": "john@test.com"}
    b = {"first_name": "john", "last_name": "SMITH", "email": "john@test.com"}

    print("\nPair 1 (same person, casing differences):")
    print(f"  A: {a}")
    print(f"  B: {b}")
    explanation = gm.explain_pair_df(
        a, b,
        fuzzy={"first_name": 1.0, "last_name": 1.0},
        exact=["email"],
    )
    print(f"\n  {explanation}")

    # --- Pair 2: different people ---
    c = {"first_name": "Jane", "last_name": "Doe", "email": "jane@other.com"}

    print("\n" + "-" * 60)
    print("\nPair 2 (different people):")
    print(f"  A: {a}")
    print(f"  C: {c}")
    explanation = gm.explain_pair_df(
        a, c,
        fuzzy={"first_name": 1.0, "last_name": 1.0},
        exact=["email"],
    )
    print(f"\n  {explanation}")

    # --- Pair 3: fuzzy name match, same email ---
    d = {"first_name": "Jon", "last_name": "Smyth", "email": "john@test.com"}

    print("\n" + "-" * 60)
    print("\nPair 3 (fuzzy name, same email):")
    print(f"  A: {a}")
    print(f"  D: {d}")
    explanation = gm.explain_pair_df(
        a, d,
        fuzzy={"first_name": 1.0, "last_name": 1.0},
        exact=["email"],
        scorer="jaro_winkler",
    )
    print(f"\n  {explanation}")
