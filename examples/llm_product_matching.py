#!/usr/bin/env python
"""LLM-enhanced product matching.

Shows how to configure LLM clustering for product data where
fuzzy string matching alone isn't enough. The LLM understands
that "Sony WH-1000XM5" and "Sony Noise Cancelling Headphones XM5"
are the same product.

Requires OPENAI_API_KEY environment variable.
"""
import os
import goldenmatch as gm
import polars as pl
import tempfile
from pathlib import Path

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY to run this example with real LLM calls.")
    print("Showing config setup only.\n")

# Sample product data from two sources
data = pl.DataFrame({
    "title": [
        "Sony WH-1000XM5 Wireless Headphones Black",
        "Sony Noise Cancelling Headphones WH1000XM5",
        "Apple AirPods Pro 2nd Generation",
        "AirPods Pro (2nd Gen) with MagSafe",
        "Samsung Galaxy S24 Ultra 256GB",
        "Galaxy S24 Ultra SM-S928B 256GB Titanium",
        "Bose QuietComfort 45 Headphones",
        "Logitech MX Master 3S Mouse",
    ],
    "price": ["348", "350", "249", "249", "1299", "1300", "329", "99"],
    "source": ["store_a", "store_b", "store_a", "store_b",
               "store_a", "store_b", "store_a", "store_a"],
})

tmp = Path(tempfile.mkdtemp()) / "products.csv"
data.write_csv(tmp)

# Config with LLM clustering mode
config = gm.GoldenMatchConfig(
    matchkeys=[
        gm.MatchkeyConfig(
            name="fuzzy_title",
            type="weighted",
            threshold=0.70,
            fields=[
                gm.MatchkeyField(field="title", scorer="token_sort", weight=0.8,
                                 transforms=["lowercase", "strip"]),
                gm.MatchkeyField(field="price", scorer="exact", weight=0.2),
            ],
        ),
    ],
    blocking=gm.BlockingConfig(
        keys=[gm.BlockingKeyConfig(fields=["title"], transforms=["lowercase", "substring:0:4"])],
    ),
    llm_scorer=gm.LLMScorerConfig(
        enabled=True,
        mode="cluster",          # Send blocks to LLM for group clustering
        cluster_max_size=50,     # Max records per LLM block
        cluster_min_size=3,      # Below this, fall back to pairwise
        budget=gm.BudgetConfig(
            max_cost_usd=0.50,   # Cap total LLM spend
            max_calls=100,
        ),
    ),
)

print("LLM Product Matching Config:")
print(f"  Mode: {config.llm_scorer.mode}")
print(f"  Budget: ${config.llm_scorer.budget.max_cost_usd:.2f} max")
print(f"  Matchkey: {config.get_matchkeys()[0].name}")
print(f"  Threshold: {config.get_matchkeys()[0].threshold}")

if os.environ.get("OPENAI_API_KEY"):
    result = gm.dedupe(str(tmp), config=config)
    print(f"\nResult: {result}")
    for cid, cinfo in result.clusters.items():
        print(f"  Cluster {cid}: {cinfo['size']} records, confidence={cinfo['confidence']:.2f}")
else:
    # Demo without LLM -- just fuzzy matching
    config.llm_scorer.enabled = False
    result = gm.dedupe(str(tmp), config=config)
    print(f"\nFuzzy-only result (no LLM): {result}")
    print("Enable OPENAI_API_KEY for LLM-enhanced clustering.")
