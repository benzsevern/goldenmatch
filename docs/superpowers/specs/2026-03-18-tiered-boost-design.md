# Tiered LLM Boost — Design Specification

## Overview

Replace the single-level LLM boost with a tiered system that auto-escalates from cheap/fast to expensive/accurate based on measured results. Level 1 is zero-shot (free), Level 2 is bi-encoder fine-tuning (~$0.20), Level 3 is Ditto-style cross-encoder fine-tuning with data augmentation (~$0.50 total). The user runs `--llm-boost` and the system handles escalation automatically. Target: 70%+ F1 on Abt-Buy (up from 52.7%).

## Problem Statement

Level 2 (bi-encoder fine-tuning) achieves 52.7% F1 on Abt-Buy — an improvement over zero-shot (44.5%) but far behind Ditto's 89.3%. The root cause: bi-encoders embed each record independently and compare with cosine similarity. They can't learn cross-record attention patterns like "PSLX350H appears in both records." Ditto uses a cross-encoder that sees both records simultaneously, combined with data augmentation to maximize learning from small training sets.

## Goals

- Add cross-encoder fine-tuning (Level 3) as an auto-escalation from bi-encoder (Level 2)
- Replicate Ditto's key techniques: cross-encoder classification, data augmentation
- Reuse Level 2's labels to minimize additional LLM cost
- Use bi-encoder as fast filter, cross-encoder only on uncertain pairs (~2K instead of all)
- No new CLI flags — `--llm-boost` handles all levels
- Target: 70%+ F1 on Abt-Buy, 55%+ F1 on Amazon-Google

## Non-Goals

- Matching Ditto's 89% F1 (they use more data, larger models, GPU training)
- GPU as a requirement (CPU training in 5-10 min is acceptable)
- Knowledge distillation or model compression
- Replacing Levels 1-2 — they remain the fast path for easy datasets

---

## Feature 1: Cross-Encoder Training

### Record Serialization (Ditto-style)

Both records concatenated as a single sequence pair:

```
[CLS] name: Sony PSLX350H Turntable | price: 149.00 [SEP] name: Sony Belt-Drive Stereo Turntable | price: 149.99 [SEP]
```

The transformer attends across both records — it learns that shared tokens (like model numbers) across records signal a match.

### Model

Base: `cross-encoder/ms-marco-MiniLM-L-6-v2` (small, fast, good starting point for fine-tuning).

The `sentence-transformers` library provides `CrossEncoder` class that handles:
- Classification head on `[CLS]` token
- `BCEWithLogitsLoss` for binary classification
- Training loop with evaluation

### Training Configuration

- Epochs: 10 with early stopping (patience=3 on validation loss)
- Batch size: 16
- Learning rate: 2e-5
- Training data: 500 labeled pairs augmented to ~2,000 examples
- Validation split: 10% held out for early stopping

### New File: `goldenmatch/core/cross_encoder.py`

```python
def serialize_record(row: dict, columns: list[str]) -> str:
    """Serialize record fields into Ditto-style text."""

def train_cross_encoder(
    train_pairs: list[tuple[str, str, bool]],
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 10,
    save_dir: Path | None = None,
) -> CrossEncoder:
    """Fine-tune cross-encoder on labeled pairs."""

def score_pairs(
    model: CrossEncoder,
    pairs: list[tuple[str, str]],
) -> list[float]:
    """Score pairs with cross-encoder. Returns match probabilities."""

def load_cross_encoder(model_dir: Path) -> CrossEncoder | None:
    """Load saved cross-encoder model."""
```

---

## Feature 2: Data Augmentation

Three augmentation techniques from Ditto, applied to training pairs only:

### Span Deletion (40% probability)
Randomly remove 1-3 consecutive tokens. Teaches the model not to rely on any single token.
```
Before: "Sony PSLX350H Belt Drive Turntable"
After:  "Sony Belt Drive Turntable"
```

### Span Shuffling (30% probability)
Randomly reorder a span of 2-4 tokens. Teaches order invariance.
```
Before: "Sony PSLX350H Belt Drive Turntable"
After:  "Sony PSLX350H Drive Belt Turntable"
```

### Column Dropping (30% probability)
Randomly drop one column from the serialized record. Teaches matching with incomplete data.
```
Before: "name: Sony Turntable | price: 149.00 | manufacturer: Sony"
After:  "name: Sony Turntable | manufacturer: Sony"
```

### Application

Each training pair is augmented 3x → 500 pairs become 2,000 training examples. Augmentation applied randomly to record A, record B, or both.

```python
def augment_pair(text_a: str, text_b: str, label: bool) -> list[tuple[str, str, bool]]:
    """Generate 3 augmented versions of a training pair."""
```

---

## Feature 3: Tiered Orchestration

### Level Flow

```
--llm-boost triggered
    │
    ├─ Level 2: bi-encoder fine-tune (existing)
    │   ├─ Label 200 pairs with LLM (~$0.20)
    │   ├─ Fine-tune sentence-transformer
    │   ├─ Re-score all pairs with cosine sim
    │   └─ Measure cross-val F1
    │
    ├─ F1 >= 60%? → Done, return Level 2 results
    │
    ├─ Escalation check:
    │   ├─ TUI mode: prompt user "Upgrade to Deep Matching? (~$0.30, ~5 min)"
    │   └─ --no-tui mode: auto-escalate if F1 < 60%
    │
    └─ Level 3: cross-encoder fine-tune
        ├─ Reuse Level 2's 200 labels
        ├─ Label 300 more pairs with LLM (~$0.30)
        ├─ Augment 500 → 2,000 training examples
        ├─ Train cross-encoder (10 epochs, ~5 min CPU)
        ├─ Bi-encoder filters to ~2,000 uncertain pairs (0.3 < score < 0.8)
        ├─ Cross-encoder re-scores uncertain pairs (~20s)
        ├─ Merge: >0.8 from bi-encoder = match, <0.3 = non-match, middle = cross-encoder score
        └─ Return merged results
```

### Score Merging

After cross-encoder scores the uncertain pairs:
- Pairs with bi-encoder score > 0.8: keep as matches (high confidence)
- Pairs with bi-encoder score < 0.3: reject (high confidence non-match)
- Pairs with 0.3 ≤ bi-encoder score ≤ 0.8: use cross-encoder probability as final score

### Modifications to `boost.py`

The existing `boost_accuracy` function becomes the orchestrator:

```python
def boost_accuracy(...):
    # Level 2 (existing code)
    bi_encoder_result = finetune_and_rescore(...)
    cv_f1 = measure_cross_val_f1(...)

    if cv_f1 >= 0.60:
        return bi_encoder_result

    if not should_escalate(cv_f1):
        return bi_encoder_result

    # Level 3
    additional_labels = label_pairs(...)  # 300 more
    all_labels = level2_labels + additional_labels
    augmented = augment_training_data(all_labels)  # 500 → 2,000
    cross_model = train_cross_encoder(augmented)

    # Score uncertain pairs only
    uncertain = [(a, b, s) for a, b, s in bi_encoder_result if 0.3 <= s <= 0.8]
    cross_scores = score_pairs(cross_model, uncertain)

    return merge_scores(bi_encoder_result, uncertain, cross_scores)
```

### `should_escalate` Function

- TUI mode: display prompt with estimated cost and time, return user's choice
- `--no-tui` mode: auto-escalate if F1 < 0.60
- If no API key budget remaining (labels would exceed `max_labels`): skip

### Model Persistence

- Cross-encoder saved to `.goldenmatch_cross_encoder/` directory
- On subsequent runs with `--llm-boost`, load saved model and skip training
- `--llm-retrain` forces re-training at whatever level was last used
- Staleness detection: column hash comparison (same as bi-encoder)

---

## Rollout Plan

1. **Phase 1: Cross-Encoder Module**
   - `cross_encoder.py` with serialization, training, scoring, loading
   - Data augmentation functions
   - Tests with mocked models

2. **Phase 2: Tiered Orchestration**
   - Modify `boost.py` to add escalation logic
   - Score merging
   - TUI escalation prompt
   - Tests for escalation flow

3. **Phase 3: Benchmark**
   - Simulate Level 3 on Abt-Buy and Amazon-Google
   - Measure F1 at different label counts
   - Compare against Level 2 results
   - Update README

## Testing Strategy

### Unit Tests

- `test_cross_encoder.py`:
  - Record serialization format
  - Data augmentation produces correct number of examples
  - Span deletion removes tokens
  - Span shuffling reorders tokens
  - Column dropping removes columns
  - Cross-encoder training on synthetic data (mocked model)
  - Score merging logic (high/low/uncertain thresholds)

### Integration Tests

- Full escalation flow: Level 2 → check F1 → Level 3
- Model persistence: train once, load on second run
- Bi-encoder filter → cross-encoder re-score pipeline

### Benchmark Validation

- Target: Abt-Buy > 70% F1 with Level 3
- Amazon-Google > 55% F1 with Level 3
- DBLP-ACM should not regress (Level 2 sufficient, never escalates)

## Dependencies

Uses existing packages:
- `sentence-transformers` (already in `[embeddings]` extra) — provides `CrossEncoder` class
- `torch` (transitive via sentence-transformers)
- No new dependencies needed

## Cost Analysis

| Level | Labels | Augmented | LLM Cost | Training Time (CPU) | Inference |
|-------|--------|-----------|----------|---------------------|-----------|
| 1 | 0 | 0 | $0 | 0 | Instant |
| 2 | 200 | 200 | ~$0.20 | ~2 min | Instant (cosine sim) |
| 3 | 500 (200 reused) | 2,000 | ~$0.50 total | ~5-10 min | ~20s (2K pairs) |

Subsequent runs with saved model: $0 at any level.
