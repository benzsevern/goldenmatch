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

The best results come from training on multi-pass blocking pairs (clean training data) and scoring on ANN blocking pairs (high recall):

| Approach | Abt-Buy F1 | Cost |
|----------|-----------|------|
| Zero-shot | 44.5% | $0 |
| Bi-encoder fine-tune | 52.7% | ~$0.30 |
| **Optimal (train multi-pass, score ANN)** | **59.5%** | **~$0.30** |

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
