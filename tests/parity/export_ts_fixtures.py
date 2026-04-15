"""Generate fixture JSON for the TS port's autoconfigVerify-parity tests.

Writes `packages/goldenmatch-js/tests/parity/autoconfig-verify-fixtures.json`.
Each fixture records the input rows plus the PreflightFindings Python produced,
so the TS parity harness can assert its preflight surfaces the same findings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl  # noqa: E402

from goldenmatch.core.autoconfig import auto_configure_df  # noqa: E402

FIXTURES = [
    (
        "small_synthetic_names",
        lambda: pl.DataFrame(
            {
                "name": [f"person {i}" for i in range(30)],
                "email": [f"p{i}@example.com" for i in range(30)],
            }
        ),
    ),
    (
        "near_unique_id",
        lambda: pl.DataFrame(
            {
                "voter_reg_num": [f"REG{i}" for i in range(100)],
                "last_name": [f"name{i % 20}" for i in range(100)],
            }
        ),
    ),
    (
        "all_same_state",
        lambda: pl.DataFrame(
            {
                "state": ["NC"] * 100,
                "last_name": [f"name{i}" for i in range(100)],
            }
        ),
    ),
    (
        "multi_author_biblio",
        lambda: pl.DataFrame(
            {
                "title": [f"paper {i}" for i in range(30)],
                "authors": [
                    f"Alice Smith{i}, Bob Jones{i}, Carol White{i}, Dave Brown{i}"
                    for i in range(30)
                ],
                "year": [str(2000 + (i % 20)) for i in range(30)],
            }
        ),
    ),
]


def serialize_fixture(name: str, df: pl.DataFrame) -> dict:
    try:
        cfg = auto_configure_df(df)
        findings = cfg._preflight_report.findings
        pre = [
            dict(
                check=f.check,
                severity=f.severity,
                subject=f.subject,
                message=f.message,
                repaired=f.repaired,
                repair_note=f.repair_note,
            )
            for f in findings
        ]
        error = None
    except Exception as exc:  # noqa: BLE001
        pre = []
        error = f"{type(exc).__name__}: {exc}"
    return {
        "name": name,
        "rows_count": df.height,
        "rows_json": df.to_dicts(),
        "preflight_findings": pre,
        "error": error,
    }


def main() -> None:
    out = (
        Path(__file__).parent.parent.parent
        / "packages"
        / "goldenmatch-js"
        / "tests"
        / "parity"
        / "autoconfig-verify-fixtures.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fixtures = [serialize_fixture(n, b()) for n, b in FIXTURES]
    out.write_text(json.dumps(fixtures, indent=2, default=str))
    print(f"wrote {out} with {len(fixtures)} fixtures")


if __name__ == "__main__":
    main()
