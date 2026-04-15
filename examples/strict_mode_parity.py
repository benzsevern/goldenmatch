#!/usr/bin/env python
"""Strict mode -- deterministic parity runs (v1.5.0).

``auto_configure_df(df, strict=True)`` still runs preflight + postflight
and collects advisories, but suppresses auto-adjustments (threshold nudges
and similar) so the config is reproducible across runs.

Use strict=True for:
  * DQBench / regression testing
  * Comparing two GoldenMatch versions on identical data
  * Anything where you need byte-for-byte reproducible outputs

This example runs ``auto_configure_df`` twice on the same DataFrame --
once with defaults, once with ``strict=True`` -- and prints both
configs' thresholds plus the resulting pipelines' postflight adjustments.

Usage:
    pip install goldenmatch
    python examples/strict_mode_parity.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def build_sample_df():
    import polars as pl

    rows = [
        ("Alice",  "Johnson", "alice@uni.edu", 2019, "Uni X"),
        ("alice",  "johnson", "ALICE@UNI.EDU", 2019, "Uni X"),
        ("Bob",    "Smith",   "bob@corp.com",  2020, "Corp"),
        ("Robert", "Smith",   "bob@corp.com",  2020, "Corp"),
        ("Carol",  "Nguyen",  "carol@org.net", 2018, "Org"),
        ("Carol",  "Nguyen",  "carol@org.net", 2018, "Org"),
        ("David",  "Park",    "dpark@lab.io",  2021, "Lab"),
        ("Eve",    "Kim",     "eve@firm.com",  2022, "Firm"),
        ("Frank",  "Olsen",   "frank@lab.net", 2023, "Lab"),
        ("Grace",  "Hopper",  "grace@navy.mil", 1985, "USN"),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "first_name": pl.Utf8, "last_name": pl.Utf8,
            "email": pl.Utf8, "year": pl.Int64, "affiliation": pl.Utf8,
        },
        orient="row",
    )


def summarize(label: str, result) -> None:
    cfg = result.config
    mk = cfg.get_matchkeys()
    thresholds = [getattr(m, "threshold", None) for m in mk]
    post = result.postflight_report
    print(f"[{label}]")
    print(f"  matchkey thresholds: {thresholds}")
    if post is None:
        print("  postflight_report: (none)")
        return
    print(f"  postflight advisories: {len(post.advisories)}")
    for adv in post.advisories:
        print(f"    - {adv}")
    print(f"  postflight adjustments: {len(post.adjustments)}")
    for adj in post.adjustments:
        print(
            f"    - {adj.field}: {adj.from_value!r} -> {adj.to_value!r} "
            f"(signal={adj.signal})"
        )


def main() -> None:
    import goldenmatch as gm

    df = build_sample_df()

    print("=" * 72)
    print("GoldenMatch v1.5.0 -- strict-mode parity")
    print("=" * 72)

    # Default: postflight may auto-nudge the threshold on clear bimodality.
    cfg_default = gm.auto_configure_df(df)
    result_default = gm.dedupe_df(df, config=cfg_default)

    # Strict: signals and advisories still computed, but no adjustments.
    cfg_strict = gm.auto_configure_df(df, strict=True)
    result_strict = gm.dedupe_df(df, config=cfg_strict)

    summarize("default", result_default)
    print()
    summarize("strict=True", result_strict)

    print()
    print("Takeaway:")
    print(
        "  strict=True is the right choice for DQBench / regression / "
        "reproducibility. You still see advisories so you know what the "
        "signals said -- you just don't let auto-config rewrite thresholds "
        "out from under you."
    )


if __name__ == "__main__":
    main()
