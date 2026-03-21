"""v0.3.0 Feature Benchmarks — Fellegi-Sunter, Learned Blocking, LLM Budget.

Compares new v0.3.0 features against the existing weighted/static baselines
on Leipzig benchmark datasets.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
from goldenmatch.core.ingest import load_file
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches, score_blocks_parallel
from goldenmatch.core.probabilistic import train_em, score_probabilistic
from goldenmatch.core.learned_blocking import learn_blocking_rules, apply_learned_blocks
from goldenmatch.core.cluster import build_clusters
from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    BlockingConfig, BlockingKeyConfig, OutputConfig,
)

DATASETS_DIR = Path(__file__).parent / "datasets"


@dataclass
class BenchResult:
    dataset: str
    strategy: str
    found_pairs: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    time_seconds: float
    extra: str = ""


def load_ground_truth(mapping_path: Path, id_col_a: str, id_col_b: str) -> set[tuple[str, str]]:
    df = pl.read_csv(mapping_path)
    pairs = set()
    for row in df.to_dicts():
        a = str(row[id_col_a]).strip()
        b = str(row[id_col_b]).strip()
        pairs.add((a, b))
    return pairs


def evaluate(
    found_pairs: list[tuple[int, int, float]],
    df: pl.DataFrame,
    ground_truth: set[tuple[str, str]],
    id_col_a: str,
    id_col_b: str,
    source_a_label: str,
    source_b_label: str,
) -> tuple[int, int, int]:
    """Returns (tp, fp, fn)."""
    rows = df.to_dicts()
    id_to_idx = {r["__row_id__"]: i for i, r in enumerate(rows)}
    source_col = "__source__"

    tp = fp = 0
    for a, b, s in found_pairs:
        idx_a = id_to_idx.get(a)
        idx_b = id_to_idx.get(b)
        if idx_a is None or idx_b is None:
            continue
        row_a = rows[idx_a]
        row_b = rows[idx_b]

        # Ensure cross-source
        if row_a.get(source_col) == row_b.get(source_col):
            continue

        # Get IDs in correct order (source_a first)
        if row_a.get(source_col) == source_a_label:
            pair = (str(row_a.get(id_col_a, row_a.get("id", a))).strip(),
                    str(row_b.get(id_col_b, row_b.get("id", b))).strip())
        else:
            pair = (str(row_b.get(id_col_a, row_b.get("id", b))).strip(),
                    str(row_a.get(id_col_b, row_a.get("id", a))).strip())

        if pair in ground_truth:
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - tp
    return tp, fp, fn


def _load_dblp_acm():
    """Load DBLP-ACM dataset."""
    ds_dir = DATASETS_DIR / "DBLP-ACM"
    df_a = load_file(ds_dir / "DBLP2.csv").collect()
    df_b = load_file(ds_dir / "ACM.csv").collect()

    df_a = df_a.with_columns(pl.lit("dblp").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("acm").alias("__source__"))

    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__")

    gt = load_ground_truth(ds_dir / "DBLP-ACM_perfectMapping.csv", "idDBLP", "idACM")
    return combined, gt


def _load_abt_buy():
    """Load Abt-Buy dataset."""
    ds_dir = DATASETS_DIR / "Abt-Buy"
    df_a = load_file(ds_dir / "Abt.csv").collect()
    df_b = load_file(ds_dir / "Buy.csv").collect()

    df_a = df_a.with_columns(pl.lit("abt").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("buy").alias("__source__"))

    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__")

    gt = load_ground_truth(ds_dir / "abt_buy_perfectMapping.csv", "idAbt", "idBuy")
    return combined, gt


# ══════════════════════════════════════════════════════════════════════════
# Benchmark 1: Fellegi-Sunter vs Weighted on DBLP-ACM
# ══════════════════════════════════════════════════════════════════════════


def bench_weighted_dblp_acm(df, gt):
    """Baseline: weighted fuzzy matching on DBLP-ACM."""
    t0 = time.perf_counter()

    mk = MatchkeyConfig(
        name="fuzzy",
        type="weighted",
        threshold=0.85,
        fields=[
            MatchkeyField(field="title", scorer="token_sort", weight=1.0, transforms=["lowercase"]),
            MatchkeyField(field="authors", scorer="token_sort", weight=0.5, transforms=["lowercase"]),
            MatchkeyField(field="year", scorer="exact", weight=0.3),
        ],
    )
    blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:8"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
        ],
        max_block_size=500,
    )

    lf = df.lazy()
    lf = compute_matchkeys(lf, [mk])
    collected = lf.collect()
    blocks = build_blocks(collected.lazy(), blocking)
    matched_pairs: set[tuple[int, int]] = set()
    pairs = score_blocks_parallel(blocks, mk, matched_pairs)

    elapsed = time.perf_counter() - t0
    tp, fp, fn = evaluate(pairs, df, gt, "id", "id", "dblp", "acm")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return BenchResult(
        dataset="DBLP-ACM", strategy="weighted(0.85)",
        found_pairs=len(pairs), true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=prec, recall=rec, f1=f1, time_seconds=elapsed,
    )


def bench_probabilistic_dblp_acm(df, gt):
    """NEW: Fellegi-Sunter probabilistic matching on DBLP-ACM."""
    t0 = time.perf_counter()

    mk = MatchkeyConfig(
        name="fs",
        type="probabilistic",
        fields=[
            MatchkeyField(field="title", scorer="token_sort", levels=3, partial_threshold=0.8, transforms=["lowercase"]),
            MatchkeyField(field="authors", scorer="token_sort", levels=3, partial_threshold=0.7, transforms=["lowercase"]),
            MatchkeyField(field="year", scorer="exact", levels=2),
        ],
    )
    blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:8"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
        ],
        max_block_size=500,
    )

    lf = df.lazy()
    collected = lf.collect()

    # Train EM
    em_result = train_em(collected, mk, n_sample_pairs=20000, max_iterations=30)

    # Build blocks and score
    blocks = build_blocks(collected.lazy(), blocking)
    all_pairs = []
    for block in blocks:
        block_df = block.df.collect() if hasattr(block.df, 'collect') else block.df
        pairs = score_probabilistic(block_df, mk, em_result)
        all_pairs.extend(pairs)

    elapsed = time.perf_counter() - t0
    tp, fp, fn = evaluate(all_pairs, df, gt, "id", "id", "dblp", "acm")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return BenchResult(
        dataset="DBLP-ACM", strategy="probabilistic(F-S EM)",
        found_pairs=len(all_pairs), true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=prec, recall=rec, f1=f1, time_seconds=elapsed,
        extra=f"EM: converged={em_result.converged}, iters={em_result.iterations}, match_rate={em_result.proportion_matched:.4f}",
    )


# ══════════════════════════════════════════════════════════════════════════
# Benchmark 2: Learned Blocking vs Static on DBLP-ACM
# ══════════════════════════════════════════════════════════════════════════


def bench_static_blocking_dblp_acm(df, gt):
    """Baseline: static blocking."""
    t0 = time.perf_counter()

    mk = MatchkeyConfig(
        name="fuzzy",
        type="weighted",
        threshold=0.85,
        fields=[
            MatchkeyField(field="title", scorer="token_sort", weight=1.0, transforms=["lowercase"]),
            MatchkeyField(field="authors", scorer="token_sort", weight=0.5, transforms=["lowercase"]),
            MatchkeyField(field="year", scorer="exact", weight=0.3),
        ],
    )
    blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:8"])],
        strategy="static",
        max_block_size=500,
    )

    lf = df.lazy()
    lf = compute_matchkeys(lf, [mk])
    collected = lf.collect()
    blocks = build_blocks(collected.lazy(), blocking)
    n_blocks = len(blocks)
    total_block_records = sum(b.df.collect().height for b in blocks)

    matched_pairs: set[tuple[int, int]] = set()
    pairs = score_blocks_parallel(blocks, mk, matched_pairs)

    elapsed = time.perf_counter() - t0
    tp, fp, fn = evaluate(pairs, df, gt, "id", "id", "dblp", "acm")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return BenchResult(
        dataset="DBLP-ACM", strategy="static_blocking",
        found_pairs=len(pairs), true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=prec, recall=rec, f1=f1, time_seconds=elapsed,
        extra=f"blocks={n_blocks}, block_records={total_block_records}",
    )


def bench_learned_blocking_dblp_acm(df, gt):
    """NEW: Learned blocking predicate selection."""
    t0 = time.perf_counter()

    mk = MatchkeyConfig(
        name="fuzzy",
        type="weighted",
        threshold=0.85,
        fields=[
            MatchkeyField(field="title", scorer="token_sort", weight=1.0, transforms=["lowercase"]),
            MatchkeyField(field="authors", scorer="token_sort", weight=0.5, transforms=["lowercase"]),
            MatchkeyField(field="year", scorer="exact", weight=0.3),
        ],
    )

    # Phase 1: static blocking on sample to get training pairs
    sample_blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:8"])],
        strategy="static",
        max_block_size=500,
    )
    lf = df.lazy()
    lf = compute_matchkeys(lf, [mk])
    collected = lf.collect()

    # Sample — keep small to avoid combinatorial explosion in predicate evaluation
    sample_size = min(500, collected.height)
    sample_df = collected.sample(sample_size, seed=42)
    sample_blocks = build_blocks(sample_df.lazy(), sample_blocking)
    matched_pairs: set[tuple[int, int]] = set()
    sample_pairs = score_blocks_parallel(sample_blocks, mk, matched_pairs)

    # Phase 2: Learn blocking rules — limit columns to avoid huge search space
    cols = ["title", "authors", "year"]
    cols = [c for c in cols if c in collected.columns]
    rules = learn_blocking_rules(
        sample_df, sample_pairs,
        columns=cols,
        min_recall=0.60,
        min_reduction=0.30,
        predicate_depth=1,  # depth-1 only for speed
    )

    # Phase 3: Apply learned blocks to full dataset
    learned_blocks = apply_learned_blocks(collected.lazy(), rules, max_block_size=500)
    n_blocks = len(learned_blocks)
    total_block_records = sum(b.df.collect().height for b in learned_blocks)

    matched_pairs2: set[tuple[int, int]] = set()
    pairs = score_blocks_parallel(learned_blocks, mk, matched_pairs2)

    elapsed = time.perf_counter() - t0
    tp, fp, fn = evaluate(pairs, df, gt, "id", "id", "dblp", "acm")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    rule_desc = "; ".join(r.key()[:50] for r in rules[:3])

    return BenchResult(
        dataset="DBLP-ACM", strategy="learned_blocking",
        found_pairs=len(pairs), true_positives=tp, false_positives=fp, false_negatives=fn,
        precision=prec, recall=rec, f1=f1, time_seconds=elapsed,
        extra=f"blocks={n_blocks}, block_records={total_block_records}, rules=[{rule_desc}]",
    )


# ══════════════════════════════════════════════════════════════════════════
# Benchmark 3: LLM Budget on Abt-Buy (simulated — no real API calls)
# ══════════════════════════════════════════════════════════════════════════


def bench_llm_budget_simulation(df, gt):
    """Simulate LLM budget behavior on product matching.

    Since we can't make real API calls in benchmarks, we simulate:
    - Score all pairs with fuzzy matching
    - Apply the three-tier logic with different budget caps
    - Measure what fraction of pairs would go to LLM at each budget
    """
    mk = MatchkeyConfig(
        name="fuzzy",
        type="weighted",
        threshold=0.50,
        fields=[
            MatchkeyField(field="name", scorer="token_sort", weight=1.0, transforms=["lowercase"]),
        ],
    )
    blocking = BlockingConfig(
        keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"])],
        max_block_size=200,
    )

    lf = df.lazy()
    lf = compute_matchkeys(lf, [mk])
    collected = lf.collect()
    blocks = build_blocks(collected.lazy(), blocking)
    matched_pairs: set[tuple[int, int]] = set()
    pairs = score_blocks_parallel(blocks, mk, matched_pairs)

    # Three-tier classification
    auto_accept = [p for p in pairs if p[2] >= 0.95]
    candidates = [p for p in pairs if 0.75 <= p[2] < 0.95]
    below = [p for p in pairs if p[2] < 0.75]

    # Budget simulation: estimate tokens per pair
    tokens_per_pair = 80
    cost_per_1k_tokens = 0.00015  # gpt-4o-mini input

    results = []
    for budget_usd in [0.0, 0.10, 0.50, 1.00, 999.0]:
        max_pairs_by_budget = int(budget_usd / (tokens_per_pair * cost_per_1k_tokens / 1000)) if budget_usd > 0 else 0
        llm_scored = min(len(candidates), max_pairs_by_budget)
        # Simulate: assume LLM would accept 60% of candidates (typical for product data)
        simulated_accepts = int(llm_scored * 0.60)
        total_matches = len(auto_accept) + simulated_accepts

        tp, fp, fn = evaluate(
            auto_accept + candidates[:simulated_accepts],
            df, gt, "id", "id", "abt", "buy",
        )
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        results.append(BenchResult(
            dataset="Abt-Buy", strategy=f"budget=${budget_usd:.2f}",
            found_pairs=total_matches, true_positives=tp, false_positives=fp, false_negatives=fn,
            precision=prec, recall=rec, f1=f1, time_seconds=0,
            extra=f"auto_accept={len(auto_accept)}, candidates={len(candidates)}, llm_scored={llm_scored}, below={len(below)}",
        ))

    return results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def print_result(r: BenchResult):
    print(f"  {r.strategy:40s}  P={r.precision:5.1%}  R={r.recall:5.1%}  F1={r.f1:5.1%}  {r.time_seconds:6.1f}s")
    if r.extra:
        print(f"    {r.extra}")


if __name__ == "__main__":
    print("=" * 70)
    print("GoldenMatch v0.3.0 Feature Benchmarks")
    print("=" * 70)

    # ── DBLP-ACM ──
    print(f"\n{'=' * 70}")
    print("Benchmark 1: Fellegi-Sunter vs Weighted (DBLP-ACM)")
    print("=" * 70)

    df_dblp, gt_dblp = _load_dblp_acm()
    print(f"  Loaded {df_dblp.height} records, {len(gt_dblp)} ground truth pairs\n")

    r1 = bench_weighted_dblp_acm(df_dblp, gt_dblp)
    print_result(r1)

    r2 = bench_probabilistic_dblp_acm(df_dblp, gt_dblp)
    print_result(r2)

    delta_f1 = r2.f1 - r1.f1
    print(f"\n  F-S vs Weighted: {'+' if delta_f1 >= 0 else ''}{delta_f1:.1%} F1 difference")

    # ── Blocking ──
    print(f"\n{'=' * 70}")
    print("Benchmark 2: Learned Blocking vs Static (DBLP-ACM)")
    print("=" * 70)

    r3 = bench_static_blocking_dblp_acm(df_dblp, gt_dblp)
    print_result(r3)

    r4 = bench_learned_blocking_dblp_acm(df_dblp, gt_dblp)
    print_result(r4)

    delta_f1_b = r4.f1 - r3.f1
    print(f"\n  Learned vs Static: {'+' if delta_f1_b >= 0 else ''}{delta_f1_b:.1%} F1, "
          f"{r4.time_seconds - r3.time_seconds:+.1f}s time")

    # ── LLM Budget ──
    print(f"\n{'=' * 70}")
    print("Benchmark 3: LLM Budget Simulation (Abt-Buy)")
    print("=" * 70)

    df_abt, gt_abt = _load_abt_buy()
    print(f"  Loaded {df_abt.height} records, {len(gt_abt)} ground truth pairs\n")

    budget_results = bench_llm_budget_simulation(df_abt, gt_abt)
    for r in budget_results:
        print_result(r)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Strategy':<42s}  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Time':>6s}")
    print(f"  {'-'*42}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in [r1, r2, r3, r4] + budget_results:
        print(f"  {r.dataset + ': ' + r.strategy:<42s}  {r.precision:5.1%}  {r.recall:5.1%}  {r.f1:5.1%}  {r.time_seconds:5.1f}s")
