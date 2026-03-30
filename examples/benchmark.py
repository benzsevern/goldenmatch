#!/usr/bin/env python
"""Benchmark GoldenMatch with DQBench ER.

Runs the standard DQBench entity resolution benchmark and prints
a rich scorecard. Measures precision, recall, F1, and throughput
across multiple ER datasets.

Usage:
    pip install goldenmatch dqbench
    python examples/benchmark.py

For best results, set OPENAI_API_KEY for LLM scoring (~$0.25).
Without LLM: ~77.  With LLM: ~95.
"""
from __future__ import annotations

import sys
import os

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    from dqbench.adapters.goldenmatch_adapter import GoldenMatchAdapter
    from dqbench.runner import run_er_benchmark
    from dqbench.report import report_er_rich

    print("=" * 60)
    print("GoldenMatch -- DQBench ER Benchmark")
    print("=" * 60)
    print()

    adapter = GoldenMatchAdapter()
    sc = run_er_benchmark(adapter)
    report_er_rich(sc)
    print(f"\nDQBench ER Score: {sc.dqbench_er_score:.2f} / 100")
