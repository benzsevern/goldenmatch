# GoldenMatch Examples

Self-contained, runnable scripts demonstrating every major feature. Each generates its own sample data and runs in under 10 seconds.

## Quick Start

```bash
pip install goldenmatch
cd examples
python basic_dedupe.py
```

## Examples

| Script | What it shows | Deps |
|--------|---------------|------|
| `basic_dedupe.py` | Find and merge duplicate records with fuzzy matching | goldenmatch |
| `explain_match.py` | Per-field score breakdown explaining why two records match | goldenmatch |
| `advanced_config.py` | Standardization, multi-pass blocking, weighted matchkeys | goldenmatch |
| `custom_config.py` | Build matchkeys, blocking, golden rules in Python | goldenmatch |
| `evaluate_and_tune.py` | Measure accuracy, evaluate against ground truth | goldenmatch |
| `streaming_incremental.py` | Match new records one at a time against existing data | goldenmatch |
| `multi_source_pipeline.py` | CRM + marketing + vendor dedupe with golden records | goldenmatch |
| `pprl_healthcare.py` | Cross-hospital patient matching without sharing data | goldenmatch |
| `llm_product_matching.py` | LLM clustering for product data | goldenmatch, OPENAI_API_KEY |
| `agent_demo.py` | Autonomous ER agent with confidence gating and review queue | goldenmatch |
| `benchmark.py` | DQBench ER benchmark (precision, recall, F1, throughput) | goldenmatch, dqbench |

## For Coding AIs

These examples are designed to be copy-pasted as starting points. Every example uses `import goldenmatch as gm` and demonstrates the pattern you need. Adapt the data loading and field names to your use case.

## Running the Benchmark

```bash
pip install goldenmatch dqbench
python benchmark.py
# Without LLM: ~77 / 100
# With OPENAI_API_KEY set: ~95 / 100
```
