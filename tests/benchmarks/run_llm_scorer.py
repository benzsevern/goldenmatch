"""LLM-as-scorer: Use GPT-4o-mini to score borderline Abt-Buy pairs."""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import polars as pl
from goldenmatch.core.vertex_embedder import VertexEmbedder
from goldenmatch.core.autofix import auto_fix_dataframe

DATASETS = Path(__file__).parent / "datasets"
API_KEY = os.environ.get("OPENAI_API_KEY", "")


def call_gpt_batch(pairs_text, batch_size=20):
    """Score pairs with GPT-4o-mini. Returns list of bools (match/non-match)."""
    results = []
    for i in range(0, len(pairs_text), batch_size):
        batch = pairs_text[i:i + batch_size]

        # Build a single prompt with multiple pairs for efficiency
        prompt_parts = ["For each numbered pair, answer YES if they are the same product, NO if different. Answer with just the numbers and YES/NO, one per line.\n"]
        for j, (text_a, text_b) in enumerate(batch):
            prompt_parts.append(f"{j+1}. Product A: {text_a[:150]}\n   Product B: {text_b[:150]}")

        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "\n".join(prompt_parts)}],
            "temperature": 0,
            "max_tokens": len(batch) * 10,
        }).encode()

        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        })

        for attempt in range(3):
            try:
                resp = urllib.request.urlopen(req, timeout=30)
                result = json.loads(resp.read())
                answer = result["choices"][0]["message"]["content"].strip()

                # Parse YES/NO from each line
                batch_results = []
                for line in answer.split("\n"):
                    line = line.strip().upper()
                    if "YES" in line:
                        batch_results.append(True)
                    elif "NO" in line:
                        batch_results.append(False)

                # Pad if parsing missed some
                while len(batch_results) < len(batch):
                    batch_results.append(False)

                results.extend(batch_results[:len(batch)])
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    time.sleep((attempt + 1) * 5)
                    continue
                raise

        if (i // batch_size) % 5 == 0 and i > 0:
            print(f"      Scored {i + len(batch)}/{len(pairs_text)}...")

    return results


def run():
    print("=" * 70)
    print("ABT-BUY: GPT-4o-mini as scorer on Vertex AI candidates")
    print("=" * 70)

    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0692108803")

    df_a = pl.read_csv(DATASETS / "Abt-Buy/Abt.csv", encoding="utf8-lossy")
    df_b = pl.read_csv(DATASETS / "Abt-Buy/Buy.csv", encoding="utf8-lossy")
    gt = set(
        (str(r["idAbt"]).strip(), str(r["idBuy"]).strip())
        for r in pl.read_csv(DATASETS / "Abt-Buy/abt_buy_perfectMapping.csv").to_dicts()
    )

    df_a = df_a.cast({c: pl.Utf8 for c in df_a.columns}).with_columns(pl.lit("a").alias("__source__"))
    df_b = df_b.cast({c: pl.Utf8 for c in df_b.columns}).with_columns(pl.lit("b").alias("__source__"))
    combined = pl.concat([df_a, df_b], how="diagonal")
    combined = combined.with_row_index("__row_id__").with_columns(pl.col("__row_id__").cast(pl.Int64))
    combined, _ = auto_fix_dataframe(combined)

    ids = combined["id"].to_list()
    srcs = combined["__source__"].to_list()
    n = len(ids)

    # Step 1: Vertex embeddings for candidate generation
    print("\n  Step 1: Vertex AI embeddings for candidates...")
    emb = VertexEmbedder()
    texts_nd = []
    for row in combined.iter_rows(named=True):
        name = str(row.get("name", "") or "").strip()
        desc = str(row.get("description", "") or "").strip()
        texts_nd.append(f"{name} {desc}" if desc else name)
    embeddings = emb.embed_column(texts_nd, cache_key="abt_buy_name_desc")
    sim = embeddings @ embeddings.T

    # Step 2: Select borderline pairs (Vertex score 0.75-0.95)
    print("\n  Step 2: Selecting borderline pairs (Vertex score 0.75-0.95)...")
    a_idx = [i for i in range(n) if srcs[i] == "a"]
    b_idx = [i for i in range(n) if srcs[i] == "b"]

    borderline = []
    for i in a_idx:
        for j in b_idx:
            s = float(sim[i][j])
            if 0.75 <= s <= 0.95:
                borderline.append((i, j, s))

    print(f"    {len(borderline)} borderline pairs")

    # Also collect high-confidence pairs (>0.95) -- accept without LLM
    high_conf = set()
    for i in a_idx:
        for j in b_idx:
            if sim[i][j] >= 0.95:
                high_conf.add((str(ids[i]).strip(), str(ids[j]).strip()))
    print(f"    {len(high_conf)} high-confidence pairs (>0.95, auto-accept)")

    # Step 3: Score borderline pairs with GPT-4o-mini
    print(f"\n  Step 3: Scoring {len(borderline)} pairs with GPT-4o-mini...")
    pairs_text = []
    for i, j, s in borderline:
        name_a = str(combined[i, "name"] or "")
        name_b = str(combined[j, "name"] or "")
        desc_a = str(combined[i, "description"] or "")[:100]
        desc_b = str(combined[j, "description"] or "")[:100]
        text_a = f"{name_a} -- {desc_a}" if desc_a else name_a
        text_b = f"{name_b} -- {desc_b}" if desc_b else name_b
        pairs_text.append((text_a, text_b))

    t0 = time.perf_counter()
    llm_labels = call_gpt_batch(pairs_text, batch_size=20)
    t_llm = time.perf_counter() - t0

    n_match = sum(llm_labels)
    print(f"    GPT scored {len(llm_labels)} pairs in {t_llm:.1f}s")
    print(f"    Matches: {n_match}, Non-matches: {len(llm_labels) - n_match}")

    # Step 4: Build final pairs
    llm_found = set()
    for idx, (i, j, s) in enumerate(borderline):
        if idx < len(llm_labels) and llm_labels[idx]:
            llm_found.add((str(ids[i]).strip(), str(ids[j]).strip()))

    # Combine: high-confidence (auto-accept) + LLM-approved borderline
    combined_found = high_conf | llm_found

    # Step 5: Evaluate
    print("\n  Step 4: Results...")
    tp = combined_found & gt
    fp = combined_found - gt
    fn = gt - combined_found
    prec = len(tp) / len(combined_found) if combined_found else 0
    rec = len(tp) / len(gt) if gt else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    print(f"\n    Vertex baseline (name+desc, t=0.88):  F1=62.8%")
    print(f"    GPT-4o-mini scorer:                   P={prec:.1%} R={rec:.1%} F1={f1:.1%}")
    print(f"    Pairs: {len(combined_found)} ({len(high_conf)} auto + {len(llm_found)} LLM-approved)")
    print(f"    True positives: {len(tp)}")
    print(f"    False positives: {len(fp)}")
    print(f"    False negatives: {len(fn)}")
    print(f"    LLM cost: ~${len(borderline) * 0.0003:.2f}")

    # Also try with different high-confidence thresholds
    print("\n  Bonus: Sweeping auto-accept threshold...")
    for auto_thresh in [0.90, 0.92, 0.95, 0.98]:
        hc = set()
        for i in a_idx:
            for j in b_idx:
                if sim[i][j] >= auto_thresh:
                    hc.add((str(ids[i]).strip(), str(ids[j]).strip()))
        cf = hc | llm_found
        tp2 = cf & gt
        prec2 = len(tp2) / len(cf) if cf else 0
        rec2 = len(tp2) / len(gt) if gt else 0
        f12 = 2 * prec2 * rec2 / (prec2 + rec2) if prec2 + rec2 > 0 else 0
        print(f"    auto>={auto_thresh} + LLM: P={prec2:.1%} R={rec2:.1%} F1={f12:.1%} ({len(cf)} pairs)")


if __name__ == "__main__":
    run()
