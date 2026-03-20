"""Benchmark: Active sampling vs random sampling on Abt-Buy with Vertex AI."""

import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0692108803")

import numpy as np
import polars as pl

from goldenmatch.core.active_sampling import select_active_pairs
from goldenmatch.core.vertex_embedder import VertexEmbedder

DATASETS = Path(__file__).parent / "datasets"


def main():
    df_a = pl.read_csv(DATASETS / "Abt-Buy/Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS / "Abt-Buy/Buy.csv", encoding="utf8-lossy")
    gt_df = pl.read_csv(DATASETS / "Abt-Buy/abt_buy_perfectMapping.csv")
    gt = {(str(r["idAbt"]).strip(), str(r["idBuy"]).strip()) for r in gt_df.to_dicts()}

    ids_a = [str(row["id"]).strip() for row in df_a.to_dicts()]
    ids_b = [str(row["id"]).strip() for row in df_b.to_dicts()]
    texts_a = [str(row.get("name", ""))[:200] for row in df_a.to_dicts()]
    texts_b = [str(row.get("name", ""))[:200] for row in df_b.to_dicts()]

    print(f"Abt: {len(texts_a)}, Buy: {len(texts_b)}, GT: {len(gt)} pairs")

    embedder = VertexEmbedder()
    emb_a = embedder.embed_column(texts_a, cache_key="abt_active2")
    emb_b = embedder.embed_column(texts_b, cache_key="buy_active2")
    sim = emb_a @ emb_b.T

    # Build candidate pairs
    candidate_pairs = []
    for i in range(len(ids_a)):
        top_k = np.argsort(-sim[i])[:10]
        for j in top_k:
            score = float(sim[i][j])
            if score >= 0.4:
                candidate_pairs.append((i, j, score))

    print(f"Candidate pairs: {len(candidate_pairs)}")

    rng = np.random.default_rng(42)

    def simulate_label(pair_idx, noise=0.05):
        i, j, s = candidate_pairs[pair_idx]
        is_match = (ids_a[i], ids_b[j]) in gt
        if rng.random() < noise:
            is_match = not is_match
        return is_match

    def learn_threshold(indices, labels):
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.40, 0.95, 0.02):
            tp = sum(1 for idx, lbl in zip(indices, labels) if lbl and candidate_pairs[idx][2] >= t)
            fp = sum(1 for idx, lbl in zip(indices, labels) if not lbl and candidate_pairs[idx][2] >= t)
            fn = sum(1 for idx, lbl in zip(indices, labels) if lbl and candidate_pairs[idx][2] < t)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t

    def evaluate_threshold(t):
        found = set()
        for i in range(len(ids_a)):
            j = np.argmax(sim[i])
            if sim[i][j] >= t:
                found.add((ids_a[i], ids_b[j]))
        tp = found & gt
        p = len(tp) / len(found) if found else 0
        r = len(tp) / len(gt) if gt else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    # === Random 300 ===
    print("\n=== Random Sampling (300 labels) ===")
    rand_idx = rng.choice(len(candidate_pairs), min(300, len(candidate_pairs)), replace=False).tolist()
    rand_labels = [simulate_label(i) for i in rand_idx]
    t = learn_threshold(rand_idx, rand_labels)
    p, r, f1 = evaluate_threshold(t)
    print(f"  Threshold: {t:.2f}")
    print(f"  P={p:.1%} R={r:.1%} F1={f1:.1%}")

    # === Active 300 (3 rounds) ===
    print("\n=== Active Sampling (300 labels, 3 rounds) ===")
    scores_arr = np.array([s for _, _, s in candidate_pairs])
    labeled = set()
    all_idx, all_lbl = [], []

    for rd in range(3):
        probs = None
        if all_idx:
            buckets = defaultdict(lambda: [0, 0])
            for idx, lbl in zip(all_idx, all_lbl):
                b = int(candidate_pairs[idx][2] * 10)
                buckets[b][0] += 1
                if lbl:
                    buckets[b][1] += 1
            probs = np.zeros(len(candidate_pairs))
            for pi, (_, _, s) in enumerate(candidate_pairs):
                b = int(s * 10)
                tot, m = buckets.get(b, [1, 0])
                probs[pi] = m / tot if tot > 0 else 0.5

        batch = select_active_pairs(
            candidate_pairs, current_probs=probs, labeled_indices=labeled,
            strategy="combined", n=100,
        )
        batch_labels = [simulate_label(i) for i in batch]
        labeled.update(batch)
        all_idx.extend(batch)
        all_lbl.extend(batch_labels)
        print(f"  Round {rd+1}: {len(batch)} labels ({sum(batch_labels)} match)")

    t = learn_threshold(all_idx, all_lbl)
    p, r, f1 = evaluate_threshold(t)
    print(f"  Threshold: {t:.2f}")
    print(f"  P={p:.1%} R={r:.1%} F1={f1:.1%}")

    # === Active 150 (half budget) ===
    print("\n=== Active Sampling (150 labels) ===")
    active_150 = select_active_pairs(candidate_pairs, strategy="combined", n=150)
    labels_150 = [simulate_label(i) for i in active_150]
    t = learn_threshold(active_150, labels_150)
    p, r, f1 = evaluate_threshold(t)
    print(f"  Threshold: {t:.2f}")
    print(f"  P={p:.1%} R={r:.1%} F1={f1:.1%}")

    # === Active 100 ===
    print("\n=== Active Sampling (100 labels) ===")
    active_100 = select_active_pairs(candidate_pairs, strategy="combined", n=100)
    labels_100 = [simulate_label(i) for i in active_100]
    t = learn_threshold(active_100, labels_100)
    p, r, f1 = evaluate_threshold(t)
    print(f"  Threshold: {t:.2f}")
    print(f"  P={p:.1%} R={r:.1%} F1={f1:.1%}")

    # === Summary ===
    print("\n" + "=" * 55)
    print("SUMMARY (Abt-Buy with Vertex AI embeddings)")
    print("=" * 55)
    print(f"  Zero-shot (no labels):   F1=84.8%  cost=$0")
    print(f"  Random 300 labels:       shown above  cost=~$0.30")
    print(f"  Active 300 (3 rounds):   shown above  cost=~$0.30")
    print(f"  Active 150 (half):       shown above  cost=~$0.15")
    print(f"  Active 100 (third):      shown above  cost=~$0.10")


if __name__ == "__main__":
    main()
