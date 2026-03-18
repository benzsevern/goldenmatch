"""Run GoldenMatch against Leipzig benchmark datasets.

Computes precision, recall, F1 for each dataset using the provided ground truth.
Tests both exact and fuzzy matching strategies.
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
from goldenmatch.core.ingest import load_file
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.standardize import apply_standardization
from goldenmatch.core.matchkey import compute_matchkeys
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    BlockingConfig, BlockingKeyConfig, OutputConfig,
    GoldenRulesConfig, GoldenFieldRule, StandardizationConfig,
)

DATASETS_DIR = Path(__file__).parent / "datasets"


@dataclass
class BenchmarkResult:
    dataset: str
    strategy: str
    source_a_rows: int
    source_b_rows: int
    ground_truth_pairs: int
    found_pairs: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    time_seconds: float


def load_ground_truth(mapping_path: Path, id_col_a: str, id_col_b: str) -> set[tuple[str, str]]:
    """Load ground truth mapping as a set of (id_a, id_b) pairs."""
    df = pl.read_csv(mapping_path)
    pairs = set()
    for row in df.to_dicts():
        a = str(row[id_col_a]).strip()
        b = str(row[id_col_b]).strip()
        pairs.add((a, b))
    return pairs


def run_matching(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    matchkeys: list[MatchkeyConfig],
    blocking: BlockingConfig | None = None,
    standardization: dict | None = None,
) -> tuple[set[tuple[str, str]], float]:
    """Run matching and return found pairs + time."""
    t0 = time.perf_counter()

    # Cast all columns to string for safe concatenation
    df_a = df_a.cast({col: pl.Utf8 for col in df_a.columns})
    df_b = df_b.cast({col: pl.Utf8 for col in df_b.columns})

    # Combine with source tags and row IDs
    df_a = df_a.with_columns(pl.lit("source_a").alias("__source__"))
    df_b = df_b.with_columns(pl.lit("source_b").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(
        pl.col("__row_id__").cast(pl.Int64)
    )

    # Auto-fix
    combined, _ = auto_fix_dataframe(combined)

    # Standardize
    if standardization:
        lf = combined.lazy()
        lf = apply_standardization(lf, standardization)
        combined = lf.collect()

    # Compute matchkeys
    lf = combined.lazy()
    lf = compute_matchkeys(lf, matchkeys)
    combined = lf.collect()

    # Build source/id lookups
    row_to_source = {}
    row_to_id = {}
    for row in combined.select("__row_id__", "__source__", "id").to_dicts():
        row_to_source[row["__row_id__"]] = row["__source__"]
        row_to_id[row["__row_id__"]] = str(row["id"]).strip()

    # Score pairs
    all_pairs = []
    for mk in matchkeys:
        if mk.type == "exact":
            pairs = find_exact_matches(combined.lazy(), mk)
            all_pairs.extend(pairs)
        elif mk.type == "weighted" and blocking:
            blocks = build_blocks(combined.lazy(), blocking)
            for block in blocks:
                bdf = block.df.collect()
                pairs = find_fuzzy_matches(bdf, mk, pre_scored_pairs=block.pre_scored_pairs)
                all_pairs.extend(pairs)

    # Filter to cross-source pairs only, convert to original IDs
    found = set()
    for a, b, score in all_pairs:
        src_a = row_to_source.get(a)
        src_b = row_to_source.get(b)
        if src_a != src_b:
            id_a = row_to_id.get(a)
            id_b = row_to_id.get(b)
            # Normalize: source_a ID first
            if src_a == "source_a":
                found.add((id_a, id_b))
            else:
                found.add((id_b, id_a))

    elapsed = time.perf_counter() - t0
    return found, elapsed


def evaluate(
    dataset_name: str,
    strategy_name: str,
    found_pairs: set[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
    source_a_rows: int,
    source_b_rows: int,
    elapsed: float,
) -> BenchmarkResult:
    """Compute precision, recall, F1."""
    tp = found_pairs & ground_truth
    fp = found_pairs - ground_truth
    fn = ground_truth - found_pairs

    precision = len(tp) / len(found_pairs) if found_pairs else 0.0
    recall = len(tp) / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return BenchmarkResult(
        dataset=dataset_name,
        strategy=strategy_name,
        source_a_rows=source_a_rows,
        source_b_rows=source_b_rows,
        ground_truth_pairs=len(ground_truth),
        found_pairs=len(found_pairs),
        true_positives=len(tp),
        false_positives=len(fp),
        false_negatives=len(fn),
        precision=precision,
        recall=recall,
        f1=f1,
        time_seconds=elapsed,
    )


def print_result(r: BenchmarkResult):
    print(f"\n  {'-'*60}")
    print(f"  {r.dataset} — {r.strategy}")
    print(f"  {'-'*60}")
    print(f"  Source A: {r.source_a_rows:,}  |  Source B: {r.source_b_rows:,}")
    print(f"  Ground truth pairs: {r.ground_truth_pairs:,}")
    print(f"  Found pairs:        {r.found_pairs:,}")
    print(f"  True positives:     {r.true_positives:,}")
    print(f"  False positives:    {r.false_positives:,}")
    print(f"  False negatives:    {r.false_negatives:,}")
    print(f"  Precision:          {r.precision:.2%}")
    print(f"  Recall:             {r.recall:.2%}")
    print(f"  F1 Score:           {r.f1:.2%}")
    print(f"  Time:               {r.time_seconds:.2f}s")


# -- Dataset Configurations --------------------------------------------------


def run_dblp_acm():
    """DBLP-ACM: bibliographic matching on title+authors+venue+year."""
    print("\n" + "=" * 70)
    print("DBLP-ACM (2,614 vs 2,294 — bibliographic)")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS_DIR / "DBLP-ACM" / "DBLP2.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS_DIR / "DBLP-ACM" / "ACM.csv", encoding="utf8-lossy")
    gt = load_ground_truth(
        DATASETS_DIR / "DBLP-ACM" / "DBLP-ACM_perfectMapping.csv",
        "idDBLP", "idACM"
    )

    results = []

    # Strategy 1: Exact title match
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="title_exact",
            fields=[MatchkeyField(column="title", transforms=["lowercase", "strip"])],
            comparison="exact",
        ),
    ], standardization={"title": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-ACM", "exact_title", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Strategy 2: Fuzzy title + authors
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_title_authors",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="jaro_winkler", weight=0.5),
                MatchkeyField(column="authors", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=0.3),
                MatchkeyField(column="year", transforms=["strip"],
                              scorer="exact", weight=0.2),
            ],
            comparison="weighted",
            threshold=0.85,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"])],
        strategy="adaptive",
        sub_block_keys=[BlockingKeyConfig(fields=["year"], transforms=[])],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "authors": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-ACM", "fuzzy_title+authors+year", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Strategy 3: Exact title + fuzzy title (cascading)
    found_exact, elapsed_exact = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="title_exact",
            fields=[MatchkeyField(column="title", transforms=["lowercase", "strip"])],
            comparison="exact",
        ),
    ], standardization={"title": ["strip", "trim_whitespace"]})

    found_fuzzy, elapsed_fuzzy = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_title_authors",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="jaro_winkler", weight=0.5),
                MatchkeyField(column="authors", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=0.3),
                MatchkeyField(column="year", transforms=["strip"],
                              scorer="exact", weight=0.2),
            ],
            comparison="weighted",
            threshold=0.80,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:4"])],
        strategy="adaptive",
        sub_block_keys=[BlockingKeyConfig(fields=["year"], transforms=[])],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "authors": ["strip", "trim_whitespace"]})

    combined_found = found_exact | found_fuzzy
    r = evaluate("DBLP-ACM", "cascaded_exact+fuzzy(0.80)", combined_found, gt, len(df_a), len(df_b), elapsed_exact + elapsed_fuzzy)
    print_result(r)
    results.append(r)

    # Multi-pass blocking + fuzzy
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_mp",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="jaro_winkler", weight=0.5),
                MatchkeyField(column="authors", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=0.3),
                MatchkeyField(column="year", transforms=["strip"],
                              scorer="exact", weight=0.2),
            ],
            comparison="weighted",
            threshold=0.85,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["authors"], transforms=["lowercase", "first_token"]),
        ],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "authors": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-ACM", "multi_pass+fuzzy(0.85)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Multi-pass + ensemble scorer
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="ensemble_mp",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="ensemble", weight=0.5),
                MatchkeyField(column="authors", transforms=["lowercase", "strip"],
                              scorer="ensemble", weight=0.3),
                MatchkeyField(column="year", transforms=["strip"],
                              scorer="exact", weight=0.2),
            ],
            comparison="weighted",
            threshold=0.85,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["authors"], transforms=["lowercase", "first_token"]),
        ],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "authors": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-ACM", "multi_pass+ensemble(0.85)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    return results


def run_dblp_scholar():
    """DBLP-Scholar: larger bibliographic matching (2,616 vs 64,263)."""
    print("\n" + "=" * 70)
    print("DBLP-Scholar (2,616 vs 64,263 — bibliographic, large)")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS_DIR / "DBLP-Scholar" / "DBLP1.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS_DIR / "DBLP-Scholar" / "Scholar.csv", encoding="utf8-lossy")
    gt = load_ground_truth(
        DATASETS_DIR / "DBLP-Scholar" / "DBLP-Scholar_perfectMapping.csv",
        "idDBLP", "idScholar"
    )

    results = []

    # Exact title
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="title_exact",
            fields=[MatchkeyField(column="title", transforms=["lowercase", "strip"])],
            comparison="exact",
        ),
    ], standardization={"title": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-Scholar", "exact_title", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Fuzzy title + year with blocking
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_title",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="jaro_winkler", weight=0.7),
                MatchkeyField(column="year", transforms=["strip"],
                              scorer="exact", weight=0.3),
            ],
            comparison="weighted",
            threshold=0.85,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:6"])],
        strategy="adaptive",
        sub_block_keys=[BlockingKeyConfig(fields=["year"], transforms=[])],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-Scholar", "fuzzy_title+year(0.85)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Multi-pass blocking (wider recall)
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_mp",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="jaro_winkler", weight=0.6),
                MatchkeyField(column="authors", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=0.2),
                MatchkeyField(column="year", transforms=["strip"],
                              scorer="exact", weight=0.2),
            ],
            comparison="weighted",
            threshold=0.80,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:6"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["authors"], transforms=["lowercase", "first_token"]),
        ],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "authors": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-Scholar", "multi_pass+fuzzy(0.80)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Cascaded exact + multi-pass fuzzy
    found_exact_ds = found  # reuse exact from above
    found_exact_ds2, _ = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="title_exact",
            fields=[MatchkeyField(column="title", transforms=["lowercase", "strip"])],
            comparison="exact",
        ),
    ], standardization={"title": ["strip", "trim_whitespace"]})
    combined_found = found_exact_ds2 | found
    r = evaluate("DBLP-Scholar", "cascaded_exact+mp(0.80)", combined_found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Multi-pass + ensemble scorer
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="ensemble_mp",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="ensemble", weight=0.6),
                MatchkeyField(column="authors", transforms=["lowercase", "strip"],
                              scorer="ensemble", weight=0.2),
                MatchkeyField(column="year", transforms=["strip"],
                              scorer="exact", weight=0.2),
            ],
            comparison="weighted",
            threshold=0.80,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:6"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["authors"], transforms=["lowercase", "first_token"]),
        ],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "authors": ["strip", "trim_whitespace"]})
    r = evaluate("DBLP-Scholar", "multi_pass+ensemble(0.80)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    return results


def run_abt_buy():
    """Abt-Buy: e-commerce product matching on name/description."""
    print("\n" + "=" * 70)
    print("Abt-Buy (1,081 vs 1,092 — e-commerce)")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS_DIR / "Abt-Buy" / "Buy.csv", encoding="utf8-lossy")
    gt = load_ground_truth(
        DATASETS_DIR / "Abt-Buy" / "abt_buy_perfectMapping.csv",
        "idAbt", "idBuy"
    )

    results = []

    # Exact name
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="name_exact",
            fields=[MatchkeyField(column="name", transforms=["lowercase", "strip"])],
            comparison="exact",
        ),
    ], standardization={"name": ["strip", "trim_whitespace"]})
    r = evaluate("Abt-Buy", "exact_name", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Fuzzy name
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_name",
            fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=1.0),
            ],
            comparison="weighted",
            threshold=0.75,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"])],
        strategy="adaptive",
        sub_block_keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:8"])],
        max_block_size=200,
    ), standardization={"name": ["strip", "trim_whitespace"]})
    r = evaluate("Abt-Buy", "fuzzy_name(token_sort,0.75)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Ensemble scorer + multi-pass
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="ensemble_name",
            fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"],
                              scorer="ensemble", weight=1.0),
            ],
            comparison="weighted",
            threshold=0.75,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "soundex"]),
        ],
        max_block_size=500,
    ), standardization={"name": ["strip", "trim_whitespace"]})
    r = evaluate("Abt-Buy", "ensemble+multi_pass(0.75)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Multi-pass fuzzy name (noisy blocking)
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_name_mp",
            fields=[
                MatchkeyField(column="name", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=1.0),
            ],
            comparison="weighted",
            threshold=0.65,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["name"], transforms=["lowercase", "soundex"]),
        ],
        max_block_size=500,
    ), standardization={"name": ["strip", "trim_whitespace"]})
    r = evaluate("Abt-Buy", "multi_pass+token_sort(0.65)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Embedding scorer with ANN blocking — sweep thresholds
    try:
        for thresh in [0.75, 0.80, 0.85]:
            found, elapsed = run_matching(df_a, df_b, matchkeys=[
                MatchkeyConfig(
                    name="emb_name",
                    fields=[
                        MatchkeyField(column="name", transforms=[],
                                      scorer="embedding", weight=1.0,
                                      model="all-MiniLM-L6-v2"),
                    ],
                    comparison="weighted",
                    threshold=thresh,
                ),
            ], blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
                strategy="ann",
                ann_column="name",
                ann_model="all-MiniLM-L6-v2",
                ann_top_k=20,
            ), standardization={"name": ["strip", "trim_whitespace"]})
            r = evaluate("Abt-Buy", f"embedding+ANN({thresh})", found, gt, len(df_a), len(df_b), elapsed)
            print_result(r)
            results.append(r)

        # Hybrid: embedding + token_sort (weighted)
        found, elapsed = run_matching(df_a, df_b, matchkeys=[
            MatchkeyConfig(
                name="hybrid_name",
                fields=[
                    MatchkeyField(column="name", transforms=[],
                                  scorer="embedding", weight=0.6,
                                  model="all-MiniLM-L6-v2"),
                    MatchkeyField(column="name", transforms=["lowercase", "strip"],
                                  scorer="token_sort", weight=0.4),
                ],
                comparison="weighted",
                threshold=0.70,
            ),
        ], blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
            strategy="ann",
            ann_column="name",
            ann_model="all-MiniLM-L6-v2",
            ann_top_k=20,
        ), standardization={"name": ["strip", "trim_whitespace"]})
        r = evaluate("Abt-Buy", "hybrid_emb+ts(0.70)", found, gt, len(df_a), len(df_b), elapsed)
        print_result(r)
        results.append(r)
        # Record-level embedding + ann_pairs (no Union-Find)
        for thresh in [0.75, 0.80, 0.85]:
            found, elapsed = run_matching(df_a, df_b, matchkeys=[
                MatchkeyConfig(
                    name="rec_emb_name",
                    fields=[
                        MatchkeyField(
                            scorer="record_embedding",
                            columns=["name"],
                            weight=1.0,
                            model="all-MiniLM-L6-v2",
                        ),
                    ],
                    comparison="weighted",
                    threshold=thresh,
                ),
            ], blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase", "substring:0:3"])],
                strategy="ann_pairs",
                ann_column="name",
                ann_model="all-MiniLM-L6-v2",
                ann_top_k=20,
            ), standardization={"name": ["strip", "trim_whitespace"]})
            r = evaluate("Abt-Buy", f"rec_emb+ann_pairs({thresh})", found, gt, len(df_a), len(df_b), elapsed)
            print_result(r)
            results.append(r)
    except ImportError as e:
        print(f"\n  [SKIPPED embedding strategies: {e}]")

    return results


def run_amazon_google():
    """Amazon-GoogleProducts: e-commerce matching."""
    print("\n" + "=" * 70)
    print("Amazon-GoogleProducts (1,363 vs 3,226 — e-commerce)")
    print("=" * 70)

    df_a = pl.read_csv(DATASETS_DIR / "Amazon-GoogleProducts" / "Amazon.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b = pl.read_csv(DATASETS_DIR / "Amazon-GoogleProducts" / "GoogleProducts.csv", encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)

    # Rename columns to align
    df_b = df_b.rename({"name": "title"})

    gt = load_ground_truth(
        DATASETS_DIR / "Amazon-GoogleProducts" / "Amzon_GoogleProducts_perfectMapping.csv",
        "idAmazon", "idGoogleBase"
    )

    results = []

    # Exact title
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="title_exact",
            fields=[MatchkeyField(column="title", transforms=["lowercase", "strip"])],
            comparison="exact",
        ),
    ], standardization={"title": ["strip", "trim_whitespace"]})
    r = evaluate("Amazon-Google", "exact_title", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Ensemble + multi-pass
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="ensemble_title",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="ensemble", weight=0.7),
                MatchkeyField(column="manufacturer", transforms=["lowercase", "strip"],
                              scorer="ensemble", weight=0.3),
            ],
            comparison="weighted",
            threshold=0.65,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["manufacturer"], transforms=["lowercase", "substring:0:3"]),
        ],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "manufacturer": ["strip", "trim_whitespace"]})
    r = evaluate("Amazon-Google", "ensemble+multi_pass(0.65)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Fuzzy title + manufacturer
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_title_mfr",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=0.7),
                MatchkeyField(column="manufacturer", transforms=["lowercase", "strip"],
                              scorer="jaro_winkler", weight=0.3),
            ],
            comparison="weighted",
            threshold=0.70,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"])],
        strategy="adaptive",
        sub_block_keys=[BlockingKeyConfig(fields=["manufacturer"], transforms=["lowercase", "substring:0:3"])],
        max_block_size=300,
    ), standardization={"title": ["strip", "trim_whitespace"], "manufacturer": ["strip", "trim_whitespace"]})
    r = evaluate("Amazon-Google", "fuzzy_title+mfr(0.70)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Multi-pass fuzzy
    found, elapsed = run_matching(df_a, df_b, matchkeys=[
        MatchkeyConfig(
            name="fuzzy_title_mp",
            fields=[
                MatchkeyField(column="title", transforms=["lowercase", "strip"],
                              scorer="token_sort", weight=0.7),
                MatchkeyField(column="manufacturer", transforms=["lowercase", "strip"],
                              scorer="jaro_winkler", weight=0.3),
            ],
            comparison="weighted",
            threshold=0.60,
        ),
    ], blocking=BlockingConfig(
        keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
        strategy="multi_pass",
        passes=[
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:5"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "token_sort", "substring:0:8"]),
            BlockingKeyConfig(fields=["title"], transforms=["lowercase", "soundex"]),
            BlockingKeyConfig(fields=["manufacturer"], transforms=["lowercase", "substring:0:3"]),
        ],
        max_block_size=500,
    ), standardization={"title": ["strip", "trim_whitespace"], "manufacturer": ["strip", "trim_whitespace"]})
    r = evaluate("Amazon-Google", "multi_pass+fuzzy(0.60)", found, gt, len(df_a), len(df_b), elapsed)
    print_result(r)
    results.append(r)

    # Embedding scorer with ANN blocking — sweep thresholds
    try:
        for thresh in [0.75, 0.80, 0.85]:
            found, elapsed = run_matching(df_a, df_b, matchkeys=[
                MatchkeyConfig(
                    name="emb_title",
                    fields=[
                        MatchkeyField(column="title", transforms=[],
                                      scorer="embedding", weight=1.0,
                                      model="all-MiniLM-L6-v2"),
                    ],
                    comparison="weighted",
                    threshold=thresh,
                ),
            ], blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
                strategy="ann",
                ann_column="title",
                ann_model="all-MiniLM-L6-v2",
                ann_top_k=20,
            ), standardization={"title": ["strip", "trim_whitespace"]})
            r = evaluate("Amazon-Google", f"embedding+ANN({thresh})", found, gt, len(df_a), len(df_b), elapsed)
            print_result(r)
            results.append(r)

        # Hybrid embedding + manufacturer
        found, elapsed = run_matching(df_a, df_b, matchkeys=[
            MatchkeyConfig(
                name="emb_title_mfr",
                fields=[
                    MatchkeyField(column="title", transforms=[],
                                  scorer="embedding", weight=0.7,
                                  model="all-MiniLM-L6-v2"),
                    MatchkeyField(column="manufacturer", transforms=["lowercase", "strip"],
                                  scorer="jaro_winkler", weight=0.3),
                ],
                comparison="weighted",
                threshold=0.70,
            ),
        ], blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
            strategy="ann",
            ann_column="title",
            ann_model="all-MiniLM-L6-v2",
            ann_top_k=30,
        ), standardization={"title": ["strip", "trim_whitespace"], "manufacturer": ["strip", "trim_whitespace"]})
        r = evaluate("Amazon-Google", "emb+mfr_hybrid(0.70)", found, gt, len(df_a), len(df_b), elapsed)
        print_result(r)
        results.append(r)

        # Record-level embedding + ann_pairs
        for thresh in [0.75, 0.80, 0.85]:
            found, elapsed = run_matching(df_a, df_b, matchkeys=[
                MatchkeyConfig(
                    name="rec_emb_title",
                    fields=[
                        MatchkeyField(
                            scorer="record_embedding",
                            columns=["title", "manufacturer"],
                            weight=1.0,
                            model="all-MiniLM-L6-v2",
                        ),
                    ],
                    comparison="weighted",
                    threshold=thresh,
                ),
            ], blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:3"])],
                strategy="ann_pairs",
                ann_column="title",
                ann_model="all-MiniLM-L6-v2",
                ann_top_k=20,
            ), standardization={"title": ["strip", "trim_whitespace"], "manufacturer": ["strip", "trim_whitespace"]})
            r = evaluate("Amazon-Google", f"rec_emb+ann_pairs({thresh})", found, gt, len(df_a), len(df_b), elapsed)
            print_result(r)
            results.append(r)
    except ImportError as e:
        print(f"\n  [SKIPPED embedding strategies: {e}]")

    return results


def main():
    print("=" * 70)
    print("GOLDENMATCH — Leipzig Benchmark Suite")
    print("=" * 70)

    all_results = []
    all_results.extend(run_dblp_acm())
    all_results.extend(run_dblp_scholar())
    all_results.extend(run_abt_buy())
    all_results.extend(run_amazon_google())

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'Dataset':<20} {'Strategy':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Time':>6}")
    print(f"  {'-'*20} {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for r in all_results:
        print(f"  {r.dataset:<20} {r.strategy:<30} {r.precision:>5.1%} {r.recall:>5.1%} {r.f1:>5.1%} {r.time_seconds:>5.1f}s")


if __name__ == "__main__":
    main()
