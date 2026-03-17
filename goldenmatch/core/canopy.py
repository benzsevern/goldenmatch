"""TF-IDF canopy clustering for GoldenMatch blocking."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_canopies(
    values: list[str],
    loose_threshold: float = 0.3,
    tight_threshold: float = 0.7,
    max_canopy_size: int = 500,
) -> list[list[int]]:
    """Build overlapping canopies using TF-IDF cosine similarity.

    Args:
        values: Text values to cluster (one per record).
        loose_threshold: Minimum cosine similarity for canopy membership.
        tight_threshold: Similarity above which records are removed from
            future canopy centers.
        max_canopy_size: Maximum number of records per canopy.

    Returns:
        List of canopies, each a list of record indices.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise ImportError(
            "Canopy clustering requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        )

    if not values:
        return []

    # Build TF-IDF matrix
    clean_values = [v if v else "" for v in values]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    tfidf_matrix = vectorizer.fit_transform(clean_values)

    n = tfidf_matrix.shape[0]
    available = set(range(n))
    canopies: list[list[int]] = []

    # Process records in random order for diversity
    rng = np.random.default_rng(42)
    order = rng.permutation(n)

    for center_idx in order:
        center_idx = int(center_idx)
        if center_idx not in available:
            continue

        # Compute cosine similarity between center and all records
        sims = cosine_similarity(tfidf_matrix[center_idx], tfidf_matrix).flatten()

        # Find canopy members: similarity > loose_threshold
        members = [i for i in range(n) if sims[i] >= loose_threshold]

        if not members:
            # At minimum, include the center itself
            members = [center_idx]

        # Cap canopy size — keep highest-similarity members
        if len(members) > max_canopy_size:
            member_sims = [(i, sims[i]) for i in members]
            member_sims.sort(key=lambda x: x[1], reverse=True)
            members = [i for i, _ in member_sims[:max_canopy_size]]

        canopies.append(sorted(members))

        # Remove tightly-bound records from future centers
        for i in members:
            if sims[i] >= tight_threshold:
                available.discard(i)

    logger.info(
        "Built %d canopies from %d records (loose=%.2f, tight=%.2f)",
        len(canopies),
        n,
        loose_threshold,
        tight_threshold,
    )

    return canopies
