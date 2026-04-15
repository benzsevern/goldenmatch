#!/usr/bin/env python
"""Auto-config verification -- preflight + postflight walkthrough (v1.5.0).

Runs a zero-config dedupe on a small bibliography-style DataFrame and
inspects the verification layer that auto-config and the pipeline attach
to the result:

  * ``result.config._preflight_report.findings`` -- what the 6 preflight
    checks found (repairs, warnings, errors).
  * ``result.postflight_report.signals`` -- the 8-key TypedDict schema
    of scoring signals (histogram, cluster sizes, overlap, etc.).
  * ``result.postflight_report.advisories`` -- human-readable hints.
  * ``result.postflight_report.adjustments`` -- any auto-applied
    threshold / weight nudges.

Usage:
    pip install goldenmatch
    python examples/verification_inspection.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def build_sample_df():
    """Realistic-ish bibliographic data with duplicates and misleading cols."""
    import polars as pl

    rows = [
        # (first_name, last_name, email, year, affiliation, record_id)
        ("Alice",   "Johnson",  "alice.johnson@uni.edu",  2019, "Uni of X", "R0001"),
        ("alice",   "johnson",  "ALICE.JOHNSON@UNI.EDU",  2019, "Uni of X", "R0001"),
        ("A.",      "Johnson",  "alice.johnson@uni.edu",  2019, "University of X", "R0001"),
        ("Bob",     "Smith",    "bob.smith@corp.com",     2020, "Corp Labs", "R0002"),
        ("Robert",  "Smith",    "bob.smith@corp.com",     2020, "Corp Labs", "R0002"),
        ("Carol",   "Nguyen",   "carol.n@research.org",   2018, "Research Org", "R0003"),
        ("Carol",   "Nguyen",   "carol.n@research.org",   2018, "Research Org", "R0003"),
        ("David",   "Park",     "dpark@startup.io",       2021, "Startup",   "R0004"),
        ("D.",      "Park",     "dpark@startup.io",       2021, "Startup Inc", "R0004"),
        ("Eve",     "Martinez", "eve.m@hospital.gov",     2017, "Hospital",  "R0005"),
        ("Frank",   "Olsen",    "frank@lab.net",          2022, "Lab",       "R0006"),
        ("Grace",   "Hopper",   "ghopper@navy.mil",       1985, "USN",       "R0007"),
        ("Henry",   "Kim",      "hk@firm.com",            2023, "Firm",      "R0008"),
        ("Isabel",  "Ortega",   "iortega@org.es",         2019, "Org ES",    "R0009"),
        ("Jackie",  "Wu",       "jwu@co.jp",              2020, "Co JP",     "R0010"),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "first_name": pl.Utf8,
            "last_name": pl.Utf8,
            "email": pl.Utf8,
            "year": pl.Int64,
            "affiliation": pl.Utf8,
            "record_id": pl.Utf8,
        },
        orient="row",
    )


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    import goldenmatch as gm

    df = build_sample_df()

    print_header("GoldenMatch v1.5.0 -- Verification inspection")
    print(f"Input: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Columns: {df.columns}")

    # Zero-config: preflight runs at the end of auto_configure_df;
    # postflight runs in the pipeline after scoring.
    result = gm.dedupe_df(df)

    print_header("Preflight findings (config._preflight_report)")
    report = getattr(result.config, "_preflight_report", None)
    if report is None:
        print("(no preflight report attached -- config was user-supplied?)")
    else:
        print(f"config_was_modified: {report.config_was_modified}")
        print(f"total findings: {len(report.findings)}")
        if not report.findings:
            print("(no findings -- config passed cleanly)")
        for f in report.findings:
            status = "repaired" if f.repaired else "open"
            print(
                f"  [{f.severity:7}] {f.check:20} subject={f.subject!r:24} "
                f"status={status}"
            )
            print(f"            {f.message}")
            if f.repair_note:
                print(f"            repair: {f.repair_note}")

    print_header("Postflight signals (result.postflight_report.signals)")
    post = result.postflight_report
    if post is None:
        print("(no postflight report -- pipeline must have bailed early)")
        return
    sig = post.signals
    # Iterate the 8 stable TypedDict keys.
    for key in (
        "total_pairs_scored",
        "current_threshold",
        "threshold_overlap_pct",
        "blocking_recall",
        "score_histogram",
        "block_size_percentiles",
        "preliminary_cluster_sizes",
        "oversized_clusters",
    ):
        value = sig.get(key)
        if key == "score_histogram" and isinstance(value, dict):
            bins = value.get("bins", [])
            counts = value.get("counts", [])
            print(f"  {key:30} bins={len(bins)} counts_sum={sum(counts)}")
        elif key == "oversized_clusters":
            print(f"  {key:30} count={len(value) if value else 0}")
            for oc in value or []:
                print(
                    f"      cluster_id={oc.get('cluster_id')} "
                    f"size={oc.get('size')} "
                    f"bottleneck={oc.get('bottleneck_pair')}"
                )
        elif key == "threshold_overlap_pct" and isinstance(value, (int, float)):
            print(f"  {key:30} {value:.2%}")
        else:
            print(f"  {key:30} {value!r}")

    print_header("Postflight advisories")
    if not post.advisories:
        print("(none)")
    for adv in post.advisories:
        print(f"  - {adv}")

    print_header("Postflight adjustments")
    if not post.adjustments:
        print("(none -- threshold and weights unchanged)")
    for adj in post.adjustments:
        print(
            f"  {adj.field}: {adj.from_value!r} -> {adj.to_value!r} "
            f"(signal={adj.signal}) -- {adj.reason}"
        )

    print_header("Summary")
    print(
        f"Dedupe result: {result.total_records} records -> "
        f"{result.total_clusters} clusters "
        f"(match rate {result.match_rate:.1%})"
    )


if __name__ == "__main__":
    main()
