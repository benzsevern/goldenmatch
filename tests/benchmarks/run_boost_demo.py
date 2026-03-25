"""Demo: Vertex AI embeddings + Active Learning Boost on Abt-Buy and Amazon-Google.

Simulates the TUI Boost tab flow:
1. Score all pairs with Vertex AI embeddings
2. Active sampling selects the 20 hardest borderline pairs
3. Label them using ground truth (simulates human y/n)
4. Train logistic regression on the 20 labels
5. Re-score all pairs with the classifier
6. Measure before/after F1
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0692108803")

import numpy as np
import polars as pl
from goldenmatch.core.autofix import auto_fix_dataframe
from goldenmatch.core.vertex_embedder import VertexEmbedder
from goldenmatch.core.active_sampling import select_active_pairs
from goldenmatch.core.boost import extract_feature_matrix

DATASETS = Path(__file__).parent / "datasets"


def run_boost_demo(
    name, file_a, file_b, gt_file, id_col_a, id_col_b,
    embed_col, n_labels=20,
):
    print(f"\n{'='*70}")
    print(f"  {name} -- Vertex AI + Active Learning Boost ({n_labels} labels)")
    print(f"{'='*70}")

    # Load data
    df_a = pl.read_csv(file_a, encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b = pl.read_csv(file_b, encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    gt_df = pl.read_csv(gt_file)
    gt = set(
        (str(r[id_col_a]).strip(), str(r[id_col_b]).strip())
        for r in gt_df.to_dicts()
    )

    # Combine
    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns}).with_columns(pl.lit("a").alias("__source__"))
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns}).with_columns(pl.lit("b").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    combined, _ = auto_fix_dataframe(combined)

    ids = combined["id"].to_list()
    srcs = combined["__source__"].to_list()
    n = len(ids)

    # Step 1: Vertex AI embeddings
    print("\n  Step 1: Computing Vertex AI embeddings...")
    t0 = time.perf_counter()
    emb = VertexEmbedder()
    values = [str(v).strip() if v is not None else "" for v in combined[embed_col].to_list()]
    embeddings = emb.embed_column(values, cache_key=f"boost_{name}_{embed_col}")
    sim = embeddings @ embeddings.T
    t_embed = time.perf_counter() - t0
    print(f"    Embedded {n} records in {t_embed:.1f}s (dim={embeddings.shape[1]})")

    # Build cross-source pairs with scores (brute force for small datasets)
    # Use a smarter threshold: take top-K per source_a record to avoid flooding with junk
    print("\n  Step 2: Building cross-source pairs (top-20 per record)...")
    scored_pairs = []
    a_indices = [i for i in range(n) if srcs[i] == "a"]
    b_indices = [i for i in range(n) if srcs[i] == "b"]

    for i in a_indices:
        # Get top-20 matches for this source_a record
        sims = [(j, float(sim[i][j])) for j in b_indices]
        sims.sort(key=lambda x: x[1], reverse=True)
        for j, s in sims[:20]:
            scored_pairs.append((i, j, s))

    # Deduplicate
    seen = set()
    deduped = []
    for a, b, s in scored_pairs:
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            deduped.append((a, b, s))
    scored_pairs = deduped

    print(f"    {len(scored_pairs)} candidate pairs (top-20 per record)")

    # Baseline: evaluate at optimal threshold
    def evaluate_at_threshold(pairs, thresh, label=""):
        found = set()
        for a, b, s in pairs:
            if s >= thresh:
                found.add((str(ids[a]).strip(), str(ids[b]).strip()))
        tp = found & gt
        prec = len(tp) / len(found) if found else 0
        rec = len(tp) / len(gt) if gt else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        if label:
            print(f"    {label}: P={prec:.1%} R={rec:.1%} F1={f1:.1%} ({len(found)} pairs)")
        return f1, prec, rec

    # Find optimal threshold for baseline
    print("\n  Step 3: Baseline (Vertex embeddings only)...")
    best_f1 = 0
    best_thresh = 0.7
    for t in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        f1, _, _ = evaluate_at_threshold(scored_pairs, t)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    evaluate_at_threshold(scored_pairs, best_thresh, f"Best baseline (t={best_thresh})")

    # Step 4: Stratified sampling -- heavy weight on decision boundary
    print(f"\n  Step 4: Stratified sampling -- selecting {n_labels} pairs across score range...")
    scores_arr = np.array([s for _, _, s in scored_pairs])
    # Heavy weight on 0.85-1.0 where matches and non-matches are mixed
    buckets = [
        (0.95, 1.01, 4),   # very high -- likely matches
        (0.90, 0.95, 4),   # high -- mix
        (0.85, 0.90, 4),   # borderline high
        (0.80, 0.85, 3),   # borderline
        (0.70, 0.80, 3),   # lower borderline
        (0.00, 0.70, 2),   # low -- likely non-matches
    ]
    selected = []
    for lo, hi, count in buckets:
        bucket_idx = [i for i, s in enumerate(scores_arr) if lo <= s < hi]
        if bucket_idx:
            np.random.seed(42 + len(selected))
            chosen = list(np.random.choice(bucket_idx, size=min(count, len(bucket_idx)), replace=False))
            selected.extend(chosen)
    selected = selected[:n_labels]
    print(f"    Selected {len(selected)} pairs from {len(buckets)} score buckets")

    # Step 5: Label using ground truth (simulates human y/n in TUI)
    print(f"\n  Step 5: Labeling {len(selected)} pairs (using ground truth)...")
    labels = {}
    match_count = 0
    nonmatch_count = 0
    for idx in selected:
        a, b, s = scored_pairs[idx]
        id_a = str(ids[a]).strip()
        id_b = str(ids[b]).strip()
        is_match = (id_a, id_b) in gt
        labels[idx] = is_match
        if is_match:
            match_count += 1
        else:
            nonmatch_count += 1

    print(f"    Labeled: {match_count} matches, {nonmatch_count} non-matches")

    # Show sample labels
    print(f"\n    Sample labeled pairs:")
    for i, idx in enumerate(selected[:5]):
        a, b, s = scored_pairs[idx]
        label = "MATCH" if labels[idx] else "NON-MATCH"
        val_a = str(values[a])[:40]
        val_b = str(values[b])[:40]
        print(f"      [{label:>9}] score={s:.3f}")
        print(f"                A: {val_a}")
        print(f"                B: {val_b}")

    # Step 6: Extract features and train classifier
    print(f"\n  Step 6: Training classifier on {len(labels)} labels...")
    matchable_cols = [c for c in combined.columns if not c.startswith("__")]
    labeled_indices = sorted(labels.keys())
    labeled_pairs = [scored_pairs[i] for i in labeled_indices]
    y = np.array([1.0 if labels[i] else 0.0 for i in labeled_indices])

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"    Classes: {n_pos} matches, {n_neg} non-matches")

    if n_pos == 0 or n_neg == 0:
        print(f"    [SKIP] Need both classes to train. Got {n_pos} matches, {n_neg} non-matches.")
        return

    X_labeled = extract_feature_matrix(labeled_pairs, combined, matchable_cols)

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_labeled, y)

    # Step 7: Rerank ALL pairs (blend, not replace)
    print("\n  Step 7: Reranking all pairs (blend classifier + original)...")
    X_all = extract_feature_matrix(scored_pairs, combined, matchable_cols)
    probs = clf.predict_proba(X_all)[:, 1]

    # Blend: classifier confidence determines blend weight
    boosted_pairs = []
    for i, (a, b, original_score) in enumerate(scored_pairs):
        clf_prob = float(probs[i])
        clf_confidence = abs(clf_prob - 0.5) * 2.0
        alpha = min(clf_confidence, 0.7)
        blended = (1 - alpha) * original_score + alpha * clf_prob
        boosted_pairs.append((a, b, blended))

    # Evaluate boosted results
    print("\n  Step 8: Results comparison...")
    best_boost_f1 = 0
    best_boost_thresh = 0.5
    for t in np.arange(0.5, 0.96, 0.05):
        f1, _, _ = evaluate_at_threshold(boosted_pairs, float(t))
        if f1 > best_boost_f1:
            best_boost_f1 = f1
            best_boost_thresh = float(t)

    print(f"\n    BEFORE (Vertex embeddings only):")
    evaluate_at_threshold(scored_pairs, best_thresh, f"  t={best_thresh}")

    print(f"\n    AFTER (Vertex + {n_labels} labels reranked):")
    evaluate_at_threshold(boosted_pairs, best_boost_thresh, f"  t={best_boost_thresh}")

    # Also try different label counts
    print(f"\n  Bonus: Sweeping label counts...")
    for n_lab in [10, 20, 30, 50, 100]:
        if n_lab > len(scored_pairs):
            break
        sel = []
        for lo, hi, count in buckets:
            scale = n_lab / 20.0
            cnt = max(int(count * scale), 1)
            bidx = [i for i, s in enumerate(scores_arr) if lo <= s < hi]
            if bidx:
                np.random.seed(42 + n_lab)
                sel.extend(list(np.random.choice(bidx, size=min(cnt, len(bidx)), replace=False)))
        sel = sel[:n_lab]
        y_sub = np.array([1.0 if (str(ids[scored_pairs[i][0]]).strip(), str(ids[scored_pairs[i][1]]).strip()) in gt else 0.0 for i in sel])
        X_sub = extract_feature_matrix([scored_pairs[i] for i in sel], combined, matchable_cols)
        if len(set(y_sub)) < 2:
            continue
        clf_sub = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf_sub.fit(X_sub, y_sub)
        probs_sub = clf_sub.predict_proba(X_all)[:, 1]
        boosted_sub = []
        for ii, (a, b, orig) in enumerate(scored_pairs):
            cp = float(probs_sub[ii])
            cc = abs(cp - 0.5) * 2.0
            al = min(cc, 0.7)
            boosted_sub.append((a, b, (1 - al) * orig + al * cp))
        best_sub = 0
        best_sub_t = 0.5
        for t in np.arange(0.5, 0.96, 0.05):
            f1, _, _ = evaluate_at_threshold(boosted_sub, float(t))
            if f1 > best_sub:
                best_sub = f1
                best_sub_t = float(t)
        n_pos_sub = int(y_sub.sum())
        evaluate_at_threshold(boosted_sub, best_sub_t, f"  {n_lab:>3} labels ({n_pos_sub} match, t={best_sub_t:.2f})")


if __name__ == "__main__":
    run_boost_demo(
        "Abt-Buy",
        DATASETS / "Abt-Buy/Abt.csv",
        DATASETS / "Abt-Buy/Buy.csv",
        DATASETS / "Abt-Buy/abt_buy_perfectMapping.csv",
        "idAbt", "idBuy",
        embed_col="name",
        n_labels=20,
    )

    # Amazon-Google needs column rename
    df_b_ag = pl.read_csv(DATASETS / "Amazon-GoogleProducts/GoogleProducts.csv",
                          encoding="utf8-lossy", infer_schema_length=10000, ignore_errors=True)
    df_b_ag = df_b_ag.rename({"name": "title"})
    tmp_path = DATASETS / "Amazon-GoogleProducts/_GoogleProducts_renamed.csv"
    df_b_ag.write_csv(tmp_path)

    run_boost_demo(
        "Amazon-Google",
        DATASETS / "Amazon-GoogleProducts/Amazon.csv",
        tmp_path,
        DATASETS / "Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv",
        "idAmazon", "idGoogleBase",
        embed_col="title",
        n_labels=20,
    )
