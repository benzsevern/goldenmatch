"""Sweep strategies for product matching datasets (Abt-Buy, Amazon-Google)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.config.schemas import (
    MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
)

DATASETS = Path(__file__).parent / "datasets"


def run_and_evaluate(name, df_a, df_b, gt, matchkeys, blocking=None, std=None):
    df_a2 = df_a.cast({c: pl.Utf8 for c in df_a.columns}).with_columns(pl.lit("a").alias("__source__"))
    df_b2 = df_b.cast({c: pl.Utf8 for c in df_b.columns}).with_columns(pl.lit("b").alias("__source__"))
    combined = pl.concat([df_a2, df_b2], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    combined, _ = auto_fix_dataframe(combined)
    if std:
        combined = apply_standardization(combined.lazy(), std).collect()
    combined = compute_matchkeys(combined.lazy(), matchkeys).collect()

    row_to_src = {}
    row_to_id = {}
    for r in combined.select("__row_id__", "__source__", "id").to_dicts():
        row_to_src[r["__row_id__"]] = r["__source__"]
        row_to_id[r["__row_id__"]] = str(r["id"]).strip()

    t0 = time.perf_counter()
    all_pairs = []
    for mk in matchkeys:
        if mk.type == "exact":
            all_pairs.extend(find_exact_matches(combined.lazy(), mk))
        elif mk.type == "weighted" and blocking:
            blocks = build_blocks(combined.lazy(), blocking)
            for b in blocks:
                bdf = b.df.collect()
                all_pairs.extend(find_fuzzy_matches(bdf, mk, pre_scored_pairs=b.pre_scored_pairs))

    found = set()
    for a, b, s in all_pairs:
        if row_to_src.get(a) != row_to_src.get(b):
            if row_to_src.get(a) == "a":
                found.add((row_to_id[a], row_to_id[b]))
            else:
                found.add((row_to_id[b], row_to_id[a]))

    elapsed = time.perf_counter() - t0
    tp = found & gt
    prec = len(tp) / len(found) if found else 0
    rec = len(tp) / len(gt) if gt else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
    print(f"  {name:<50} P={prec:.1%} R={rec:.1%} F1={f1:.1%} ({len(found):>5} pairs, {elapsed:.1f}s)")
    return f1


def make_mp_blocking(col):
    return BlockingConfig(
        keys=[BlockingKeyConfig(fields=[col], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=[col], transforms=["lowercase", "substring:0:3"]),
            BlockingKeyConfig(fields=[col], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=[col], transforms=["lowercase", "token_sort", "substring:0:5"]),
            BlockingKeyConfig(fields=[col], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=[col], transforms=["lowercase", "first_token"]),
            BlockingKeyConfig(fields=[col], transforms=["lowercase", "last_token"]),
        ],
        max_block_size=500,
    )


def sweep_abt_buy():
    print("=" * 70)
    print("ABT-BUY STRATEGY SWEEP")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS / "Abt-Buy/Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS / "Abt-Buy/Buy.csv", encoding="utf8-lossy")
    gt = set(
        (str(r["idAbt"]).strip(), str(r["idBuy"]).strip())
        for r in pl.read_csv(DATASETS / "Abt-Buy/abt_buy_perfectMapping.csv").to_dicts()
    )
    blk = make_mp_blocking("name")

    for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        run_and_evaluate(f"token_sort (t={t})", df_a, df_b, gt, matchkeys=[
            MatchkeyConfig(name="ts", comparison="weighted", threshold=t, fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="token_sort", weight=1.0)])],
            blocking=blk)

    for t in [0.45, 0.50, 0.55, 0.60, 0.65]:
        run_and_evaluate(f"ensemble (t={t})", df_a, df_b, gt, matchkeys=[
            MatchkeyConfig(name="ens", comparison="weighted", threshold=t, fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="ensemble", weight=1.0)])],
            blocking=blk)

    for t in [0.50, 0.55, 0.60, 0.65]:
        run_and_evaluate(f"jw(0.3)+ts(0.7) (t={t})", df_a, df_b, gt, matchkeys=[
            MatchkeyConfig(name="jts", comparison="weighted", threshold=t, fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="jaro_winkler", weight=0.3),
                MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.7)])],
            blocking=blk)

    for t in [0.50, 0.55, 0.60]:
        run_and_evaluate(f"levenshtein (t={t})", df_a, df_b, gt, matchkeys=[
            MatchkeyConfig(name="lev", comparison="weighted", threshold=t, fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="levenshtein", weight=1.0)])],
            blocking=blk)

    try:
        for t in [0.55, 0.60, 0.65, 0.70, 0.75]:
            run_and_evaluate(f"emb ann_pairs k=30 (t={t})", df_a, df_b, gt, matchkeys=[
                MatchkeyConfig(name="emb", comparison="weighted", threshold=t, fields=[
                    MatchkeyField(column="name", transforms=[], scorer="embedding", weight=1.0, model="all-MiniLM-L6-v2")])],
                blocking=BlockingConfig(
                    keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
                    strategy="ann_pairs", ann_column="name", ann_model="all-MiniLM-L6-v2", ann_top_k=30))

        for t in [0.50, 0.55, 0.60]:
            run_and_evaluate(f"emb(0.5)+ts(0.5) ann (t={t})", df_a, df_b, gt, matchkeys=[
                MatchkeyConfig(name="hyb", comparison="weighted", threshold=t, fields=[
                    MatchkeyField(column="name", transforms=[], scorer="embedding", weight=0.5, model="all-MiniLM-L6-v2"),
                    MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.5)])],
                blocking=BlockingConfig(
                    keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
                    strategy="ann_pairs", ann_column="name", ann_model="all-MiniLM-L6-v2", ann_top_k=30))

        for t in [0.55, 0.60, 0.65]:
            run_and_evaluate(f"emb(0.7)+ts(0.3) ann (t={t})", df_a, df_b, gt, matchkeys=[
                MatchkeyConfig(name="hyb2", comparison="weighted", threshold=t, fields=[
                    MatchkeyField(column="name", transforms=[], scorer="embedding", weight=0.7, model="all-MiniLM-L6-v2"),
                    MatchkeyField(column="name", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.3)])],
                blocking=BlockingConfig(
                    keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
                    strategy="ann_pairs", ann_column="name", ann_model="all-MiniLM-L6-v2", ann_top_k=30))
    except ImportError as e:
        print(f"  [SKIPPED embedding: {e}]")


def sweep_amazon_google():
    print("\n" + "=" * 70)
    print("AMAZON-GOOGLE STRATEGY SWEEP")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS / "Amazon-GoogleProducts/Amazon.csv", encoding="utf8-lossy",
                       infer_schema_length=10000, ignore_errors=True)
    df_b = pl.read_csv(DATASETS / "Amazon-GoogleProducts/GoogleProducts.csv", encoding="utf8-lossy",
                       infer_schema_length=10000, ignore_errors=True)
    df_b = df_b.rename({"name": "title"})

    gt = set(
        (str(r["idAmazon"]).strip(), str(r["idGoogleBase"]).strip())
        for r in pl.read_csv(DATASETS / "Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv").to_dicts()
    )

    blk = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "token_sort", "substring:0:5"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "first_token"]),
            BlockingKeyConfig(fields=["manufacturer"], transforms=["lowercase", "substring:0:3"]),
        ],
        max_block_size=500,
    )

    for t in [0.45, 0.50, 0.55, 0.60, 0.65]:
        run_and_evaluate(f"ts(title) (t={t})", df_a, df_b, gt, matchkeys=[
            MatchkeyConfig(name="ts", comparison="weighted", threshold=t, fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"], scorer="token_sort", weight=1.0)])],
            blocking=blk)

    for t in [0.45, 0.50, 0.55, 0.60]:
        run_and_evaluate(f"ts(0.7)+jw_mfr(0.3) (t={t})", df_a, df_b, gt, matchkeys=[
            MatchkeyConfig(name="tm", comparison="weighted", threshold=t, fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.7),
                MatchkeyField(column="manufacturer", transforms=["lowercase", "strip"], scorer="jaro_winkler", weight=0.3)])],
            blocking=blk)

    for t in [0.45, 0.50, 0.55, 0.60]:
        run_and_evaluate(f"ensemble(title) (t={t})", df_a, df_b, gt, matchkeys=[
            MatchkeyConfig(name="ens", comparison="weighted", threshold=t, fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"], scorer="ensemble", weight=1.0)])],
            blocking=blk)

    try:
        for t in [0.55, 0.60, 0.65, 0.70, 0.75]:
            run_and_evaluate(f"emb(title) ann k=30 (t={t})", df_a, df_b, gt, matchkeys=[
                MatchkeyConfig(name="emb", comparison="weighted", threshold=t, fields=[
                    MatchkeyField(column="title", transforms=[], scorer="embedding", weight=1.0, model="all-MiniLM-L6-v2")])],
                blocking=BlockingConfig(
                    keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
                    strategy="ann_pairs", ann_column="title", ann_model="all-MiniLM-L6-v2", ann_top_k=30))

        for t in [0.50, 0.55, 0.60]:
            run_and_evaluate(f"emb(0.5)+ts(0.5) ann (t={t})", df_a, df_b, gt, matchkeys=[
                MatchkeyConfig(name="hyb", comparison="weighted", threshold=t, fields=[
                    MatchkeyField(column="title", transforms=[], scorer="embedding", weight=0.5, model="all-MiniLM-L6-v2"),
                    MatchkeyField(column="title", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.5)])],
                blocking=BlockingConfig(
                    keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
                    strategy="ann_pairs", ann_column="title", ann_model="all-MiniLM-L6-v2", ann_top_k=30))

        for t in [0.55, 0.60, 0.65]:
            run_and_evaluate(f"emb(0.7)+ts(0.3) ann (t={t})", df_a, df_b, gt, matchkeys=[
                MatchkeyConfig(name="hyb2", comparison="weighted", threshold=t, fields=[
                    MatchkeyField(column="title", transforms=[], scorer="embedding", weight=0.7, model="all-MiniLM-L6-v2"),
                    MatchkeyField(column="title", transforms=["lowercase", "strip"], scorer="token_sort", weight=0.3)])],
                blocking=BlockingConfig(
                    keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
                    strategy="ann_pairs", ann_column="title", ann_model="all-MiniLM-L6-v2", ann_top_k=30))
    except ImportError as e:
        print(f"  [SKIPPED embedding: {e}]")


if __name__ == "__main__":
    sweep_abt_buy()
    sweep_amazon_google()
