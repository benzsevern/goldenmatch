"""Multi-table / graph entity resolution.

Matches within entity types, then propagates evidence across relationships
to boost scores between related entities. Iterates until convergence.

Usage in config:
    entities:
      - name: customers
        sources: [{path: crm.csv, source_name: crm}]
        matchkeys: [{name: cust_fuzzy, type: weighted, ...}]

      - name: orders
        sources: [{path: orders.csv, source_name: orders}]
        matchkeys: [{name: order_exact, type: exact, ...}]

    relationships:
      - from: orders
        to: customers
        join_key: customer_id
        evidence_weight: 0.3

    graph:
      max_iterations: 5
      convergence_threshold: 0.01
      propagation_mode: additive
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig
from goldenmatch.core.cluster import build_clusters
from goldenmatch.core.pipeline import run_dedupe

logger = logging.getLogger(__name__)


@dataclass
class EntityType:
    """An entity type in the graph."""

    name: str
    sources: list[tuple[str, str]]  # (path, source_name)
    config: GoldenMatchConfig
    df: pl.DataFrame | None = None
    clusters: dict[int, dict] = field(default_factory=dict)
    scored_pairs: list[tuple[int, int, float]] = field(default_factory=list)


@dataclass
class Relationship:
    """A relationship between two entity types."""

    from_entity: str
    to_entity: str
    join_key: str  # foreign key in from_entity
    evidence_weight: float = 0.3


@dataclass
class GraphERResult:
    """Result of graph entity resolution."""

    entities: dict[str, EntityType]
    iterations: int
    converged: bool
    evidence_propagated: int  # number of pair score boosts applied


def run_graph_er(
    entities: list[EntityType],
    relationships: list[Relationship],
    max_iterations: int = 5,
    convergence_threshold: float = 0.01,
    propagation_mode: str = "additive",
) -> GraphERResult:
    """Run multi-table entity resolution with evidence propagation.

    Algorithm:
    1. Match within each entity type independently
    2. For each relationship, find linked records across entity types
    3. If linked records in entity A are matched, boost scores in entity B
    4. Re-cluster entity B with boosted scores
    5. Repeat until no scores change more than convergence_threshold

    Args:
        entities: List of EntityType configs.
        relationships: Cross-entity relationships.
        max_iterations: Max propagation iterations.
        convergence_threshold: Stop when max score change < this.
        propagation_mode: "additive" (add weight) or "multiplicative" (multiply).

    Returns:
        GraphERResult with final entity clusters and stats.
    """
    entity_map = {e.name: e for e in entities}

    # Step 1: Initial matching within each entity type
    for entity in entities:
        logger.info("Graph ER: matching entity '%s'", entity.name)
        result = run_dedupe(entity.sources, entity.config)
        entity.clusters = result.get("clusters", {})
        entity.scored_pairs = result.get("scored_pairs", [])

        # Load data for relationship lookups
        from goldenmatch.core.ingest import load_file
        frames = []
        for path, source_name in entity.sources:
            lf = load_file(path)
            lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
            frames.append(lf.collect())
        entity.df = pl.concat(frames) if frames else pl.DataFrame()
        if "__row_id__" not in entity.df.columns:
            entity.df = entity.df.with_row_index("__row_id__")

    # Step 2-5: Iterative evidence propagation
    converged = False
    total_propagated = 0

    for iteration in range(max_iterations):
        max_delta = 0.0
        iteration_propagated = 0

        for rel in relationships:
            from_entity = entity_map.get(rel.from_entity)
            to_entity = entity_map.get(rel.to_entity)

            if from_entity is None or to_entity is None:
                logger.warning("Relationship references unknown entity: %s -> %s", rel.from_entity, rel.to_entity)
                continue

            if from_entity.df is None or to_entity.df is None:
                continue

            # Find evidence: which "to" records should have boosted scores
            # based on matched "from" records sharing the same join_key
            delta, n_boosted = _propagate_evidence(
                from_entity, to_entity, rel,
                propagation_mode=propagation_mode,
            )
            max_delta = max(max_delta, delta)
            iteration_propagated += n_boosted

        total_propagated += iteration_propagated

        logger.info(
            "Graph ER iteration %d: max_delta=%.4f, boosted=%d pairs",
            iteration + 1, max_delta, iteration_propagated,
        )

        if max_delta < convergence_threshold or iteration_propagated == 0:
            converged = True
            break

        # Re-cluster entities that received evidence
        for entity in entities:
            if entity.scored_pairs:
                all_ids = list(range(entity.df.height)) if entity.df is not None else []
                if entity.df is not None and "__row_id__" in entity.df.columns:
                    all_ids = entity.df["__row_id__"].to_list()
                max_cluster = 100
                if entity.config.golden_rules:
                    max_cluster = entity.config.golden_rules.max_cluster_size
                entity.clusters = build_clusters(entity.scored_pairs, all_ids, max_cluster)

    return GraphERResult(
        entities=entity_map,
        iterations=iteration + 1 if not converged else iteration + 1,
        converged=converged,
        evidence_propagated=total_propagated,
    )


def _propagate_evidence(
    from_entity: EntityType,
    to_entity: EntityType,
    rel: Relationship,
    propagation_mode: str = "additive",
) -> tuple[float, int]:
    """Propagate match evidence from one entity to another.

    If records A1 and A2 in from_entity are in the same cluster,
    and A1.join_key = B1 and A2.join_key = B2 in to_entity,
    then boost the score between B1 and B2.

    Returns (max_score_delta, n_boosted_pairs).
    """
    if from_entity.df is None or to_entity.df is None:
        return 0.0, 0

    join_key = rel.join_key
    if join_key not in from_entity.df.columns:
        logger.warning("Join key '%s' not found in entity '%s'", join_key, from_entity.name)
        return 0.0, 0

    # Build mapping: from_row_id -> join_key_value
    from_rows = from_entity.df.select(["__row_id__", join_key]).to_dicts()
    from_id_to_key = {r["__row_id__"]: r[join_key] for r in from_rows}

    # Build mapping: join_key_value -> to_row_ids
    if join_key in to_entity.df.columns:
        # Direct key in to_entity
        to_rows = to_entity.df.select(["__row_id__", join_key]).to_dicts()
    elif "id" in to_entity.df.columns:
        # Map via to_entity's id column
        to_rows = to_entity.df.select(["__row_id__", "id"]).to_dicts()
        join_key = "id"
    else:
        return 0.0, 0

    key_to_to_ids: dict = {}
    for r in to_rows:
        val = r[join_key]
        if val is not None:
            key_to_to_ids.setdefault(val, []).append(r["__row_id__"])

    # For each cluster in from_entity, find linked to_entity pairs
    max_delta = 0.0
    n_boosted = 0
    existing_scores = {(min(a, b), max(a, b)): s for a, b, s in to_entity.scored_pairs}

    for cid, cinfo in from_entity.clusters.items():
        if cinfo["size"] < 2:
            continue

        # Find to_entity records linked to this cluster's members
        linked_to_ids = set()
        for member_id in cinfo["members"]:
            fk_value = from_id_to_key.get(member_id)
            if fk_value is not None:
                for to_id in key_to_to_ids.get(fk_value, []):
                    linked_to_ids.add(to_id)

        if len(linked_to_ids) < 2:
            continue

        # Boost scores between all linked to_entity records
        linked_list = sorted(linked_to_ids)
        for i in range(len(linked_list)):
            for j in range(i + 1, len(linked_list)):
                pair_key = (linked_list[i], linked_list[j])
                old_score = existing_scores.get(pair_key, 0.0)

                if propagation_mode == "multiplicative":
                    new_score = min(1.0, old_score * (1 + rel.evidence_weight))
                else:  # additive
                    new_score = min(1.0, old_score + rel.evidence_weight)

                delta = abs(new_score - old_score)
                if delta > 0:
                    max_delta = max(max_delta, delta)
                    existing_scores[pair_key] = new_score
                    n_boosted += 1

    # Update to_entity scored_pairs
    to_entity.scored_pairs = [(a, b, s) for (a, b), s in existing_scores.items()]

    return max_delta, n_boosted
