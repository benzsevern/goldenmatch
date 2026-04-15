"""Capture current auto-config output for three benchmarks as parity pins.
Run ONCE before the AutoConfigDecisions refactor.
Output lives in autoconfig-classification.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import polars as pl
from goldenmatch.core.autoconfig import auto_configure_df

DATASETS = Path(__file__).parent.parent / "benchmarks" / "datasets"

def pin_config(name: str, df: pl.DataFrame) -> dict:
    cfg = auto_configure_df(df)
    mks = cfg.get_matchkeys()
    return {
        "name": name,
        "rows": df.height,
        "blocking": {
            "strategy": cfg.blocking.strategy,
            "keys": [{"fields": k.fields, "transforms": k.transforms} for k in (cfg.blocking.keys or [])],
            "passes": [{"fields": k.fields, "transforms": k.transforms} for k in (cfg.blocking.passes or [])],
        },
        "matchkeys": [
            {
                "name": mk.name,
                "type": mk.type,
                "threshold": mk.threshold,
                "fields": [
                    {"field": f.field, "scorer": f.scorer, "weight": f.weight,
                     "transforms": f.transforms}
                    for f in mk.fields
                ],
            }
            for mk in mks
        ],
    }

if __name__ == "__main__":
    pins = []
    # DBLP-ACM combined
    d = DATASETS / "DBLP-ACM"
    dblp = pl.read_csv(d / "DBLP2.csv", encoding="utf8-lossy", ignore_errors=True)
    acm  = pl.read_csv(d / "ACM.csv", encoding="utf8-lossy", ignore_errors=True)
    pins.append(pin_config("dblp_acm", pl.concat([dblp, acm], how="diagonal_relaxed")))
    # NCVR 10K
    ncvr_path = DATASETS / "NCVR" / "ncvoter_sample_10k.txt"
    df_ncvr = pl.read_csv(ncvr_path, separator="\t", encoding="utf8-lossy", ignore_errors=True)
    keep = ["county_desc","voter_reg_num","last_name","first_name","middle_name",
            "res_street_address","res_city_desc","state_cd","zip_code",
            "full_phone_number","birth_year","gender_code","race_code"]
    pins.append(pin_config("ncvr_10k", df_ncvr.select([c for c in keep if c in df_ncvr.columns])))
    # Abt-Buy
    d = DATASETS / "Abt-Buy"
    abt = pl.read_csv(d / "Abt.csv", encoding="utf8-lossy", ignore_errors=True)
    buy = pl.read_csv(d / "Buy.csv", encoding="utf8-lossy", ignore_errors=True)
    pins.append(pin_config("abt_buy", pl.concat([abt, buy], how="diagonal_relaxed")))

    out = Path(__file__).parent / "autoconfig-classification.json"
    out.write_text(json.dumps(pins, indent=2, default=str))
    print(f"wrote {out} with {len(pins)} pins")
