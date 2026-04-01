# LLM Boost

GoldenMatch can use LLMs (Claude or GPT-4) to improve accuracy on harder datasets by labeling training pairs and fine-tuning a local embedding model. The LLM is only needed once — the trained model is saved and reused for free on subsequent runs.

## Setup

```bash
pip install goldenmatch[llm]

# Set API key (Anthropic or OpenAI)
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# First run: labels pairs, trains model (~$0.30, ~7 min)
goldenmatch dedupe products.csv --llm-boost

# Subsequent runs: loads saved model ($0, instant)
goldenmatch dedupe products.csv --llm-boost

# Force re-training
goldenmatch dedupe products.csv --llm-retrain
```

## How It Works

### Tiered Auto-Escalation

GoldenMatch auto-escalates through three levels based on measured accuracy:

**Level 1: Zero-Shot (free, instant)**
Standard scorers (jaro_winkler, embedding, etc.) with no training.

**Level 2: Bi-Encoder Fine-Tuning (~$0.20, ~2 min)**
- LLM labels 200 pairs with yes/no classification
- Fine-tunes sentence-transformer (MiniLM) on labeled pairs
- Re-embeds all records with fine-tuned model
- Cosine similarity scoring

**Level 3: Cross-Encoder (Ditto-style) (~$0.50, ~5 min)**
Activated if Level 2 F1 < 60%:
- Labels 300 more pairs (reuses Level 2 labels)
- Data augmentation: span deletion, span shuffling, column dropping
- Trains cross-encoder classifier (both records as one sequence)
- Bi-encoder filters to ~2K uncertain pairs, cross-encoder re-scores

### Optimal Strategy

For product matching, the **LLM scorer** (not LLM boost) is the best approach:

| Approach | Abt-Buy F1 | Cost |
|----------|-----------|------|
| Zero-shot (MiniLM) | 44.5% | $0 |
| Zero-shot (Vertex AI) | 62.8% | ~$0.05 |
| Bi-encoder fine-tune (Level 2) | 58.7% | ~$0.30 |
| Cross-encoder (Level 3) | 65.5% | ~$0.50 |
| **LLM scorer (Vertex + GPT-4o-mini)** | **81.7%** | **~$0.74** |

The LLM scorer (`llm_scorer: {enabled: true}` in config) dramatically outperforms fine-tuning because GPT-4o-mini understands product semantics that local models cannot learn from 300 labels.

For structured data (names, addresses), LLM boost fine-tuning remains effective:

## Model Persistence

- **Bi-encoder**: saved to `.goldenmatch_finetuned_model/`
- **Cross-encoder**: saved to `.goldenmatch_cross_encoder/`
- **Classifier**: saved to `.goldenmatch_model.json` (JSON format, no serialization security risks)

Saved models are reused automatically on subsequent `--llm-boost` runs. Column hash detects data schema changes and triggers retraining.

## Cost

| Labels | LLM Cost | Subsequent Runs |
|--------|----------|-----------------|
| 200 (Level 2) | ~$0.20 | $0 |
| 500 (Level 3) | ~$0.50 | $0 |

Uses Claude Haiku or GPT-4o-mini by default (~$0.001/pair).

## CLI Flags

| Flag | Description |
|------|-------------|
| `--llm-boost` | Enable LLM-powered accuracy boost |
| `--llm-retrain` | Force re-labeling and retraining |
| `--llm-provider` | Override provider (auto, anthropic, openai) |
| `--llm-max-labels` | Max pairs to label (default 500) |

## LLM Scorer: Iterative Calibration (v1.2.6)

The LLM scorer now uses iterative calibration for large candidate sets. Instead of scoring every borderline pair:

1. **Sample** 100 pairs stratified across the score range
2. **Ask the LLM** to classify them (match/non-match)
3. **Learn the threshold** via grid search over LLM labels
4. **Refine** with focused sampling near the threshold
5. **Apply** the learned threshold to all remaining candidates

This reduced the Bulldozer benchmark (401K rows) from 37,500 LLM-scored pairs (~$0.50, 25 min) to 200 pairs (~$0.01, 42s).

```yaml
llm_scorer:
  enabled: true
  batch_size: 75
  max_workers: 3
  calibration_sample_size: 100
  calibration_max_rounds: 5
  calibration_convergence_delta: 0.01
  budget:
    max_cost_usd: 1.00
    max_calls: 500
```

## ANN Hybrid Blocking (v1.2.6)

Multi-pass blocking now supports ANN fallback for oversized blocks. Set `ann_column` on your blocking config:

```yaml
blocking:
  strategy: multi_pass
  passes:
    - fields: [model, state]
    - fields: [model, category]
  max_block_size: 1000
  skip_oversized: true
  ann_column: description_text   # enables ANN fallback
  ann_top_k: 20
```

Oversized blocks are embedded (unique text values only) and sub-blocked via FAISS instead of being skipped. See `examples/equipment_dedup.py` for a complete example.
