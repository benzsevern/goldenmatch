# GoldenMatch Examples

Self-contained, runnable scripts demonstrating every major feature. Each generates its own sample data and runs in under 10 seconds.

## Quick Start

```bash
pip install goldenmatch
cd examples
python basic_dedupe.py
```

## Examples

| Script | What it shows |
|--------|---------------|
| `basic_dedupe.py` | Simplest dedupe -- exact email matching, zero config |
| `custom_config.py` | Build matchkeys, blocking, golden rules in Python |
| `pprl_healthcare.py` | Cross-hospital patient matching without sharing data |
| `evaluate_and_tune.py` | Measure accuracy, evaluate against ground truth |
| `streaming_incremental.py` | Match new records one at a time against existing data |
| `llm_product_matching.py` | LLM clustering for product data (requires OPENAI_API_KEY) |
| `multi_source_pipeline.py` | CRM + marketing + vendor dedupe with golden records |

## For Coding AIs

These examples are designed to be copy-pasted as starting points. Every example uses `import goldenmatch as gm` and demonstrates the pattern you need. Adapt the data loading and field names to your use case.
