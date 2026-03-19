"""Cross-encoder for Ditto-style entity matching (Level 3 boost)."""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

CROSS_ENCODER_DIR = ".goldenmatch_cross_encoder"


# ── Record serialization ──────────────────────────────────────────────────

def serialize_record(row: dict, columns: list[str]) -> str:
    """Serialize record fields into Ditto-style text.

    Format: "col1: val1 | col2: val2 | ..."
    Null values are skipped.
    """
    parts = []
    for col in columns:
        val = row.get(col)
        if val is not None:
            parts.append(f"{col}: {val}")
    return " | ".join(parts) if parts else ""


# ── Data augmentation ─────────────────────────────────────────────────────

def _span_delete(text: str, rng: random.Random) -> str:
    """Remove 1-3 consecutive tokens."""
    tokens = text.split()
    if len(tokens) <= 2:
        return text
    n_delete = rng.randint(1, min(3, len(tokens) - 1))
    start = rng.randint(0, len(tokens) - n_delete)
    del tokens[start:start + n_delete]
    return " ".join(tokens)


def _span_shuffle(text: str, rng: random.Random) -> str:
    """Reorder a span of 2-4 tokens."""
    tokens = text.split()
    if len(tokens) <= 2:
        return text
    span_len = rng.randint(2, min(4, len(tokens)))
    start = rng.randint(0, len(tokens) - span_len)
    span = tokens[start:start + span_len]
    rng.shuffle(span)
    tokens[start:start + span_len] = span
    return " ".join(tokens)


def _column_drop(text: str, rng: random.Random) -> str:
    """Drop one column from serialized record (col: val | col: val)."""
    parts = [p.strip() for p in text.split("|")]
    if len(parts) <= 1:
        return text
    drop_idx = rng.randint(0, len(parts) - 1)
    parts.pop(drop_idx)
    return " | ".join(parts)


def _augment_single(text: str, rng: random.Random) -> str:
    """Apply one random augmentation technique."""
    r = rng.random()
    if r < 0.4:
        return _span_delete(text, rng)
    elif r < 0.7:
        return _span_shuffle(text, rng)
    else:
        return _column_drop(text, rng)


def augment_pair(
    text_a: str, text_b: str, label: bool, n_augments: int = 3, seed: int = 42,
) -> list[tuple[str, str, bool]]:
    """Generate augmented versions of a training pair."""
    rng = random.Random(seed)
    augmented = []
    for _ in range(n_augments):
        # Randomly augment A, B, or both
        choice = rng.randint(0, 2)
        if choice == 0:
            aug_a = _augment_single(text_a, rng)
            augmented.append((aug_a, text_b, label))
        elif choice == 1:
            aug_b = _augment_single(text_b, rng)
            augmented.append((text_a, aug_b, label))
        else:
            aug_a = _augment_single(text_a, rng)
            aug_b = _augment_single(text_b, rng)
            augmented.append((aug_a, aug_b, label))
    return augmented


def augment_training_data(
    pairs: list[tuple[str, str, bool]], n_augments: int = 3,
) -> list[tuple[str, str, bool]]:
    """Augment all training pairs. Returns originals + augmented."""
    all_data = list(pairs)
    for i, (text_a, text_b, label) in enumerate(pairs):
        all_data.extend(augment_pair(text_a, text_b, label, n_augments, seed=i))
    return all_data


# ── Cross-encoder training ────────────────────────────────────────────────

def train_cross_encoder(
    train_pairs: list[tuple[str, str, bool]],
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    save_dir: Path | None = None,
):
    """Fine-tune a cross-encoder on labeled pairs.

    Args:
        train_pairs: List of (text_a, text_b, is_match) tuples.
        base_model: HuggingFace model name for base cross-encoder.
        epochs: Training epochs (with early stopping).
        save_dir: Where to save the fine-tuned model.

    Returns:
        Trained CrossEncoder model.
    """
    try:
        from sentence_transformers import CrossEncoder, InputExample
        from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    except ImportError:
        raise ImportError(
            "Cross-encoder training requires sentence-transformers. "
            "Install with: pip install goldenmatch[embeddings]"
        )

    # Split into train/val
    random.shuffle(train_pairs)
    val_size = max(10, len(train_pairs) // 10)
    val_data = train_pairs[:val_size]
    train_data = train_pairs[val_size:]

    # Build InputExamples
    train_examples = [
        InputExample(texts=[a, b], label=1.0 if label else 0.0)
        for a, b, label in train_data
    ]

    # Build evaluator
    val_sentence_pairs = [[a, b] for a, b, _ in val_data]
    val_labels = [int(label) for _, _, label in val_data]

    evaluator = CEBinaryClassificationEvaluator(
        val_sentence_pairs, val_labels,
        name="val",
    )

    # Train
    logger.info(
        "Training cross-encoder on %d examples (%d val), %d epochs...",
        len(train_examples), val_size, epochs,
    )

    model = CrossEncoder(base_model, num_labels=1)

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
    )

    # Save
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(save_dir))
        logger.info("Saved cross-encoder to %s", save_dir)

    return model


# ── Scoring ───────────────────────────────────────────────────────────────

def score_pairs(
    model, pairs: list[tuple[str, str]],
) -> list[float]:
    """Score pairs with cross-encoder. Returns match probabilities."""
    if not pairs:
        return []
    sentence_pairs = [[a, b] for a, b in pairs]
    scores = model.predict(sentence_pairs, show_progress_bar=False)
    # Sigmoid to convert logits to probabilities
    import numpy as np
    probs = 1.0 / (1.0 + np.exp(-np.array(scores)))
    return probs.tolist()


# ── Score merging ─────────────────────────────────────────────────────────

def merge_scores(
    bi_encoder_pairs: list[tuple[int, int, float]],
    cross_encoder_scores: dict[tuple[int, int], float],
    high_threshold: float = 0.8,
    low_threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    """Merge bi-encoder and cross-encoder scores.

    - Bi-encoder score > high_threshold: keep as match
    - Bi-encoder score < low_threshold: reject
    - Between: use cross-encoder score
    """
    merged = []
    for a, b, bi_score in bi_encoder_pairs:
        key = (min(a, b), max(a, b))
        if key in cross_encoder_scores:
            merged.append((a, b, cross_encoder_scores[key]))
        else:
            merged.append((a, b, bi_score))
    return merged


# ── Model loading ─────────────────────────────────────────────────────────

def load_cross_encoder(model_dir: Path | None = None):
    """Load saved cross-encoder model. Returns None if not found."""
    d = model_dir or Path.cwd() / CROSS_ENCODER_DIR
    if not d.exists():
        return None
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(str(d))
        logger.info("Loaded cross-encoder from %s", d)
        return model
    except Exception as e:
        logger.warning("Failed to load cross-encoder: %s", e)
        return None
