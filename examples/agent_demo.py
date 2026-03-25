#!/usr/bin/env python3
"""GoldenMatch ER Agent Demo

Demonstrates the autonomous entity resolution agent:
1. Profiles data and detects domain
2. Selects optimal matching strategy with reasoning
3. Runs the pipeline with confidence gating
4. Shows review queue for borderline matches
5. Explains individual match decisions

Run:
    python examples/agent_demo.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from goldenmatch import AgentSession, gate_pairs


def create_demo_data(path: Path) -> None:
    """Create a realistic customer dataset with duplicates."""
    pl.DataFrame({
        "name": [
            "John Smith", "Jon Smith", "JOHN SMITH",
            "Jane Doe", "Janet Doe",
            "Robert Johnson", "Bob Johnson", "Robert B. Johnson",
            "Alice Williams", "alice williams",
            "Michael Brown", "Mike Brown",
            "Emily Davis",
            "David Wilson",
            "Sarah Miller",
            "James Taylor",
        ],
        "email": [
            "john.smith@gmail.com", "john.smith@gmail.com", "john.smith@gmail.com",
            "jane.doe@yahoo.com", "janet.doe@yahoo.com",
            "rjohnson@company.com", "bob.j@company.com", "rjohnson@company.com",
            "alice.w@hotmail.com", "alice.w@hotmail.com",
            "mbrown@work.org", "mbrown@work.org",
            "emily.d@email.com",
            "dwilson@email.com",
            "smiller@email.com",
            "jtaylor@email.com",
        ],
        "phone": [
            "555-0101", "555-0101", "555-0101",
            "555-0202", "555-0203",
            "555-0303", "555-0304", "555-0303",
            "555-0404", "555-0404",
            "555-0505", "555-0505",
            "555-0606",
            "555-0707",
            "555-0808",
            "555-0909",
        ],
        "zip": [
            "10001", "10001", "10001",
            "20002", "20002",
            "30003", "30003", "30003",
            "40004", "40004",
            "50005", "50005",
            "60006",
            "70007",
            "80008",
            "90009",
        ],
    }).write_csv(path)


def main():
    # Suppress noisy library output
    import logging
    logging.disable(logging.WARNING)
    import os
    os.environ["GOLDENCHECK_QUIET"] = "1"

    print()
    print("=" * 60)
    print("  GoldenMatch ER Agent Demo")
    print("=" * 60)
    print()

    # Create demo data
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "customers.csv"
        create_demo_data(csv_path)

        df = pl.read_csv(csv_path)
        print(f"Input: {df.height} customer records, {df.width} columns")
        print(f"Columns: {', '.join(df.columns)}")
        print()

        # --- Step 1: Analyze ---
        print("-" * 60)
        print("Step 1: ANALYZE DATA")
        print("-" * 60)

        session = AgentSession()
        analysis = session.analyze(str(csv_path))

        print(f"  Domain detected: {analysis.get('domain_detected', 'general')}")
        print(f"  Strategy: {analysis['strategy']}")
        print(f"  Why: {analysis['why']}")
        if analysis.get("strong_ids"):
            print(f"  Strong ID fields: {analysis['strong_ids']}")
        if analysis.get("fuzzy_fields"):
            print(f"  Fuzzy-matchable fields: {analysis['fuzzy_fields']}")
        print(f"  Storage tier: {session.review_queue.storage_tier}")
        print()

        # --- Step 2: Deduplicate ---
        print("-" * 60)
        print("Step 2: DEDUPLICATE (with confidence gating)")
        print("-" * 60)

        result = session.deduplicate(str(csv_path))

        dedupe_result = result["results"]
        reasoning = result["reasoning"]
        dist = result.get("confidence_distribution", {})

        print(f"  Clusters found: {dedupe_result.total_clusters}")
        print(f"  Match rate: {dedupe_result.match_rate:.1%}")
        print()
        print("  Confidence gating:")
        print(f"    Auto-merged (>0.95):  {dist.get('auto_merged', 0)}")
        print(f"    Review queue (0.75-0.95): {dist.get('review_queue', 0)}")
        print(f"    Auto-rejected (<0.75): {dist.get('auto_rejected', 0)}")
        print()

        # --- Step 3: Reasoning ---
        print("-" * 60)
        print("Step 3: REASONING")
        print("-" * 60)

        strategy_name = reasoning.get('strategy_chosen') or reasoning.get('strategy') or analysis.get('strategy', 'N/A')
        why_text = reasoning.get('why') or analysis.get('why', 'N/A')
        print(f"  Strategy chosen: {strategy_name}")
        print(f"  Why: {why_text}")
        print()

        alts = reasoning.get("alternatives_considered", [])
        if alts:
            print("  Alternatives considered:")
            for alt in alts:
                print(f"    - {alt['strategy']}: {alt['why_not']}")
            print()

        # --- Step 4: Review Queue ---
        print("-" * 60)
        print("Step 4: REVIEW QUEUE")
        print("-" * 60)

        pending = session.review_queue.list_pending("agent")
        if pending:
            print(f"  {len(pending)} pairs awaiting review:")
            for item in pending[:5]:
                print(f"    ({item.id_a}, {item.id_b}) score={item.score:.3f}")
                if item.explanation:
                    print(f"      {item.explanation[:80]}")
        else:
            print("  No borderline pairs -- all matches were high confidence or rejected.")
        print()

        stats = session.review_queue.stats("agent")
        print(f"  Queue stats: {stats}")
        print()

        # --- Step 5: Compare Strategies ---
        print("-" * 60)
        print("Step 5: COMPARE STRATEGIES")
        print("-" * 60)

        comparison = session.compare_strategies(str(csv_path))
        strategies = comparison.get("strategies", comparison)
        if isinstance(strategies, str):
            strategies = {}
        recommended = comparison.get("recommended", "")
        print(f"  {'Strategy':<25} {'Clusters':>10} {'Match Rate':>12} {'Recommended':>12}")
        print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12}")
        for strategy, metrics in strategies.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                rec = " <--" if strategy == recommended else ""
                print(
                    f"  {strategy:<25} "
                    f"{metrics.get('clusters', 0):>10} "
                    f"{metrics.get('match_rate', 0):>11.1%} "
                    f"{rec:>12}"
                )
            elif isinstance(metrics, dict):
                print(f"  {strategy:<25} {'error':>10}")
        print()

        # --- Summary ---
        print("=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  Input: {df.height} records")
        print(f"  Output: {dedupe_result.total_clusters} clusters")
        print(f"  Strategy: {strategy_name}")
        print(f"  Storage: {result.get('storage', 'memory')}")
        print()
        print("  Agent card: goldenmatch agent-serve --port 8200")
        print("  Discovery:  GET http://localhost:8200/.well-known/agent.json")
        print()


if __name__ == "__main__":
    main()
