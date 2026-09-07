#!/usr/bin/env python3
"""bench-biblio-scale-2633: ad hoc AT-SCALE validation for issue #2633's exit
condition.

#2633 shipped (PR #2875/#2882) a fix that ANDs `year` onto the exact-key
blocking pick (`__title_key__`) on bibliographic-domain data, cutting DBLP-ACM
(4,910 rows) candidate pairs 33,563 -> 5,749 at identical recall. The issue's
own exit condition is explicit that this needs validating "on
`bench-quality-scale` / QIS rather than the suggest-quality panel" -- DBLP-ACM
and every other suggest-panel dataset are small enough that a 22x comparison
cost is invisible there; only a real dataset at real scale exercises the
`_is_scale_safe` projection + the `_typed_projected_block` domain-cap logic
this compound depends on.

Why this is a STANDALONE script, not a QIS shape
--------------------------------------------------
`scripts/quality_invariant_scale.py::generate_with_gt` only has two shapes
("phase5", "realistic"), both person-shaped -- no bibliographic generator
exists. Building one into QIS (a new shape + a permanent bench-quality-scale
gate) is a bigger, separate investment; this script is scoped to answer the
ONE question #2633 left open: does the shipped title+year compound hold up
(fewer-or-equal candidate pairs, unchanged recall) as the biblio dataset
scales from a few thousand rows to hundreds of thousands, or does it degrade
the way the token-blocking / recall-floor designs the issue explored and
rejected did (measured 22x comparison-cost blowup at scale)?

What it measures, per row-count rung
-------------------------------------
1. EXACT candidate pairs + max block size from the REAL committed zero-config
   blocking config, via `goldenmatch.core.blocker.build_blocks` (not an
   estimate -- the same approach `scripts/suggest_quality/oracle.py`'s
   `_record_candidate_metrics` uses, materializing every block and counting).
2. The blocking key(s) actually chosen (to confirm the title+year compound is
   what fires, not some other path).
3. Real F1/precision/recall via `dedupe_df` + `score_quality` (the same O(N)
   streaming scorer `quality_invariant_scale.py` uses -- reused, not
   reimplemented) against synthetic ground truth.
4. Wall time for the auto-config + blocking-measurement step and the full
   dedupe step, separately.

Dataset generator
------------------
Two rows per true cluster (DBLP-ACM's actual shape: a two-source linkage,
every true match is exactly one duplicate pair). `title_key`'s vocabulary
grows SUBLINEARLY with N (Heaps'-law-ish, exponent 0.6) and is Zipfian-
weighted -- a small number of very common first-title-words dominate a large
share of papers, which is what makes the MOST COMMON title bucket the
interesting scale-safety case (a uniform/modulo vocabulary, as used in the
small unit-test fixtures, would understate this). `year` is drawn from a
FIXED 71-year window regardless of N (1950-2020) -- matching the real-world
invariant that publication year does not grow with corpus size, which is
exactly the property the #2633 compound exploits ("year is free selectivity
... all ground-truth pairs agree on it").

Both rows in a cluster share the identical extracted `title_key` and `year`
(by construction: same first title word, same year) -- domain extraction of
the same real paper indexed by two sources always agrees on both, the same
trust `_is_bibliographic_dataset` routing already relies on.

Usage
-----
    uv run python scripts/bench_biblio_scale_2633.py \
        --tiers 5000,50000,200000,1000000 --seed 0 --out-json biblio_scale.json

Row counts are rounded down to a multiple of 2 (ROWS_PER_CLUSTER).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROWS_PER_CLUSTER = 2

# Fixed regardless of N -- the real-world invariant #2633's fix exploits.
YEAR_MIN, YEAR_MAX = 1950, 2020  # inclusive
N_VENUES = 40
N_AUTHOR_TOKENS = 500


def _zipf_weighted_index(rng: np.random.Generator, vocab_size: int, n: int) -> np.ndarray:
    """n draws from {0..vocab_size-1}, Zipfian-weighted (rank i has weight
    1/(i+1)) so a small number of common words dominate -- realistic title-word
    frequency, and the case that actually stresses the most-common block."""
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    weights = 1.0 / ranks
    weights /= weights.sum()
    return rng.choice(vocab_size, size=n, p=weights)


def generate_biblio_with_gt(n_rows: int, seed: int = 0) -> tuple[Any, np.ndarray]:
    """Synthetic bibliographic dedupe dataset (DBLP-ACM shape) + ground-truth
    cluster ids (one per row, aligned to row order)."""
    import polars as pl

    n_rows = (n_rows // ROWS_PER_CLUSTER) * ROWS_PER_CLUSTER
    n_clusters = n_rows // ROWS_PER_CLUSTER

    # Heaps'-law-ish sublinear vocab growth: more papers -> more distinct
    # first-words, but not proportionally (real academic-title vocabulary is
    # bounded by the language, not by corpus size).
    title_vocab = max(200, int(n_clusters**0.6))

    # Independent RNG streams per field (prefix-stability discipline, same as
    # quality_invariant_scale.py::_generate_realistic): the first k values of
    # an n-sized draw equal a k-sized draw, so results are comparable/
    # reproducible across rungs, not just across seeds.
    rng_title = np.random.default_rng((seed, 1))
    rng_year = np.random.default_rng((seed, 2))
    rng_venue = np.random.default_rng((seed, 3))
    rng_author = np.random.default_rng((seed, 4))

    title_word_idx = _zipf_weighted_index(rng_title, title_vocab, n_clusters)
    years = rng_year.integers(YEAR_MIN, YEAR_MAX + 1, size=n_clusters)
    venue_idx = rng_venue.integers(0, N_VENUES, size=n_clusters)
    # 2-3 author tokens per cluster, joined -- a fuzzy (not exact) scoring
    # signal so dedupe has more than one field to work with, matching real
    # bibliographic matchkeys.
    author_a = rng_author.integers(0, N_AUTHOR_TOKENS, size=n_clusters)
    author_b = rng_author.integers(0, N_AUTHOR_TOKENS, size=n_clusters)

    cids = np.repeat(np.arange(n_clusters, dtype=np.int64), ROWS_PER_CLUSTER)
    title_word_idx_r = np.repeat(title_word_idx, ROWS_PER_CLUSTER)
    years_r = np.repeat(years, ROWS_PER_CLUSTER)
    venue_idx_r = np.repeat(venue_idx, ROWS_PER_CLUSTER)
    author_a_r = np.repeat(author_a, ROWS_PER_CLUSTER)
    author_b_r = np.repeat(author_b, ROWS_PER_CLUSTER)
    # 0 for the first row of each cluster (source A), 1 for the second
    # (source B) -- gives the two duplicate rows slightly different title
    # suffixes (real near-duplicate formatting) while sharing the identical
    # extracted title_key + year.
    source_r = np.tile(np.arange(ROWS_PER_CLUSTER, dtype=np.int64), n_clusters)

    title_words = np.array([f"word{w}" for w in range(title_vocab)])
    title_word_str = title_words[title_word_idx_r]
    suffix = np.where(
        source_r == 0,
        np.char.add(np.char.add("study of topic ", cids.astype("U")), " a"),
        np.char.add(np.char.add("analysis for subject ", cids.astype("U")), " b"),
    )
    titles = np.char.add(np.char.add(title_word_str, " "), suffix)

    authors = np.char.add(
        np.char.add(np.char.add("Author", author_a_r.astype("U")), ", Author"),
        author_b_r.astype("U"),
    )
    venues = np.char.add("Venue", venue_idx_r.astype("U"))

    df = pl.DataFrame(
        {
            "doi": [f"10.{1000 + i}/synth.{i}" for i in range(n_rows)],
            "title": titles,
            "authors": authors,
            "venue": venues,
            "year": years_r.astype(str),
        }
    )
    return df, cids


def _exact_block_metrics(df: Any, cfg: Any) -> dict:
    """EXACT candidate-pair count + max block size from the committed
    blocking config, by actually materializing every block (not an
    estimate). Mirrors suggest_quality/oracle.py::_record_candidate_metrics,
    but on the PREPARED frame (post auto_fix/standardize/domain/matchkeys --
    exactly what the real pipeline blocks on), which is the gap that helper's
    own KNOWN LIMITATION comment documents for a config keyed on a derived
    column like `__title_key__`. Fixed here by actually running domain
    extraction first -- the same `_apply_domain_extraction` step
    `dedupe_df`/`_run_dedupe_pipeline` runs internally -- so `__title_key__`
    exists before `build_blocks` looks for it."""
    from goldenmatch.core.blocker import build_blocks
    from goldenmatch.core.pipeline import _apply_domain_extraction

    blocking = getattr(cfg, "blocking", None)
    if blocking is None:
        return {"error": "config has no blocking section"}
    try:
        prepared = _apply_domain_extraction(df, cfg)
        total_pairs = 0
        max_block = 0
        n_blocks = 0
        for block in build_blocks(prepared, blocking):
            frame = block.materialize()
            sz = frame.height
            if sz > 1:
                total_pairs += sz * (sz - 1) // 2
                max_block = max(max_block, sz)
                n_blocks += 1
        keys = [list(k.fields) for k in (blocking.keys or [])]
        return {
            "candidate_pairs": total_pairs,
            "max_block_size": max_block,
            "n_nonsingleton_blocks": n_blocks,
            "blocking_strategy": blocking.strategy,
            "blocking_keys": keys,
        }
    except Exception as exc:  # noqa: BLE001 -- advisory measurement, never fatal
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_rung(n_rows: int, seed: int = 0) -> dict:
    import goldenmatch
    from goldenmatch.core.bench import bench_capture, stage

    os.environ.setdefault("GOLDENMATCH_AUTOCONFIG_MEMORY", "0")

    t0 = time.time()
    df, gt = generate_biblio_with_gt(n_rows, seed=seed)
    t_gen = time.time() - t0
    n_rows_actual = df.height

    record: dict = {"n_rows": n_rows_actual, "seed": seed, "t_gen_s": round(t_gen, 3)}

    with bench_capture():
        t1 = time.time()
        with stage("qis_biblio_autoconfig"):
            # Same measurement-mode idiom as quality_invariant_scale.py::run_rung:
            # allow_red_config + confidence_required=False so a RED-but-usable
            # config still gets measured rather than raising; _skip_finalize
            # avoids a redundant full-df verification pass.
            cfg = goldenmatch.auto_configure_df(
                df,
                confidence_required=False,
                allow_red_config=True,
                _skip_finalize=True,
            )
            for mk in cfg.matchkeys or []:
                if getattr(mk, "type", None) == "weighted" and getattr(mk, "rerank", False):
                    mk.rerank = False
                if getattr(mk, "type", None) == "exact" and getattr(mk, "negative_evidence", None):
                    mk.negative_evidence = []
        t_autoconfig = time.time() - t1

        block_metrics = _exact_block_metrics(df, cfg)
        record["t_autoconfig_s"] = round(t_autoconfig, 3)
        record.update(block_metrics)

        t2 = time.time()
        with stage("qis_biblio_dedupe"):
            result = goldenmatch.dedupe_df(df, config=cfg)
        t_dedupe = time.time() - t2
        record["t_dedupe_s"] = round(t_dedupe, 3)

    sys.path.insert(0, str(REPO / "scripts"))
    from quality_invariant_scale import score_quality  # noqa: PLC0415

    predicted: dict[int, list[int]] = {}
    for cid, c in (result.clusters or {}).items():
        members = c.get("members") or []
        if len(members) > 1:
            predicted[int(cid)] = list(members)
    metrics = score_quality(predicted, gt)
    record["quality"] = metrics
    record["committed_red"] = bool(getattr(cfg, "_committed_red", False))
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--tiers", default="5000,50000,200000,1000000", help="Comma-separated row counts."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default="biblio_scale_2633.json")
    args = p.parse_args(argv)

    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    records = []
    for n in tiers:
        print(f"[bench-biblio-2633] rung n={n} ...", flush=True)
        rec = run_rung(n, seed=args.seed)
        print(f"[bench-biblio-2633]   -> {json.dumps(rec, default=str)[:400]}", flush=True)
        records.append(rec)

    out = {"tiers": records, "meta": {"seed": args.seed, "tiers_requested": tiers}}
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[bench-biblio-2633] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
