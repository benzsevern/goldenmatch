"""Synthetic dupe set built from NCVR 10K sample for F1 measurement.

Selects 500 records, creates a perturbed copy of each (typo in last_name,
transposed zip digits, dropped middle_name). Returns the combined DataFrame
and the ground-truth set of (original_row_id, duplicate_row_id) pairs.
"""
from __future__ import annotations
import random
from pathlib import Path
import polars as pl


REPO_ROOT = Path(__file__).parent.parent.parent
NCVR_SAMPLE = REPO_ROOT / "tests" / "benchmarks" / "datasets" / "NCVR" / "ncvoter_sample_10k.txt"


def _typo_last_name(name: str) -> str:
    """Swap two adjacent characters in a name. Stable for short names."""
    if not name or len(name) < 3:
        return name
    i = len(name) // 2
    chars = list(name)
    chars[i], chars[i - 1] = chars[i - 1], chars[i]
    return "".join(chars)


def _transpose_zip(zip_code: str) -> str:
    """Swap the last two digits of a zip code."""
    if not zip_code or len(zip_code) < 2:
        return zip_code
    return zip_code[:-2] + zip_code[-1] + zip_code[-2]


def build_ncvr_synth_df(seed: int = 42, n_dupes: int = 500) -> tuple[pl.DataFrame, set[tuple[int, int]]]:
    """Return (df, gt_pairs).

    df has 2*n_dupes rows: the first n_dupes are originals (sampled from
    the NCVR 10K sample), the next n_dupes are perturbed copies.
    gt_pairs is the set of (orig_row_id, dup_row_id) tuples where row_id
    is the position in df.
    """
    rng = random.Random(seed)
    df = pl.read_csv(NCVR_SAMPLE, separator="\t", encoding="utf8-lossy", ignore_errors=True)
    keep = ["county_desc", "voter_reg_num", "last_name", "first_name", "middle_name",
            "res_street_address", "res_city_desc", "state_cd", "zip_code",
            "full_phone_number", "birth_year", "gender_code", "race_code"]
    df = df.select([c for c in keep if c in df.columns])
    # Sample n_dupes rows
    indices = rng.sample(range(df.height), n_dupes)
    originals = df[indices]
    # Build perturbed copies
    rows = originals.to_dicts()
    perturbed_rows = []
    for r in rows:
        pr = dict(r)
        if pr.get("last_name"):
            pr["last_name"] = _typo_last_name(str(pr["last_name"]))
        if pr.get("zip_code"):
            pr["zip_code"] = _transpose_zip(str(pr["zip_code"]))
        pr["middle_name"] = ""  # drop
        perturbed_rows.append(pr)
    perturbed = pl.DataFrame(perturbed_rows, schema=originals.schema)
    combined = pl.concat([originals, perturbed], how="vertical")
    gt = {(i, i + n_dupes) for i in range(n_dupes)}
    return combined, gt


if __name__ == "__main__":
    df, gt = build_ncvr_synth_df()
    print(f"df: {df.height} rows, {len(df.columns)} cols. GT pairs: {len(gt)}")
