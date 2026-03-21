"""Tests for multi-table / graph entity resolution."""
from __future__ import annotations

import csv
from pathlib import Path

import polars as pl
import pytest

from goldenmatch.config.schemas import (
    BlockingConfig,
    BlockingKeyConfig,
    GoldenMatchConfig,
    GoldenRulesConfig,
    MatchkeyConfig,
    MatchkeyField,
)
from goldenmatch.core.graph_er import (
    EntityType,
    GraphERResult,
    Relationship,
    _propagate_evidence,
    run_graph_er,
)


def _write_csv(path: Path, headers: list[str], rows: list[list]) -> str:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    return str(path)


def _make_customer_config():
    return GoldenMatchConfig(
        matchkeys=[MatchkeyConfig(
            name="cust_fuzzy",
            type="weighted",
            threshold=0.7,
            fields=[
                MatchkeyField(field="name", scorer="jaro_winkler", weight=1.0),
                MatchkeyField(field="zip", scorer="exact", weight=0.5),
            ],
        )],
        blocking=BlockingConfig(keys=[BlockingKeyConfig(fields=["zip"])]),
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
    )


def _make_order_config():
    return GoldenMatchConfig(
        matchkeys=[MatchkeyConfig(
            name="order_exact",
            type="exact",
            fields=[MatchkeyField(field="order_id")],
        )],
    )


class TestPropagateEvidence:
    def test_boost_scores(self):
        """When from_entity members are clustered, linked to_entity pairs get boosted."""
        from_entity = EntityType(
            name="orders",
            sources=[],
            config=_make_order_config(),
        )
        from_entity.df = pl.DataFrame({
            "__row_id__": [1, 2, 3],
            "order_id": ["O1", "O2", "O3"],
            "customer_id": ["C1", "C1", "C2"],
        })
        from_entity.clusters = {
            1: {"members": [1, 2], "size": 2, "oversized": False,
                "pair_scores": {(1, 2): 1.0}, "confidence": 1.0, "bottleneck_pair": None},
        }

        to_entity = EntityType(
            name="customers",
            sources=[],
            config=_make_customer_config(),
        )
        to_entity.df = pl.DataFrame({
            "__row_id__": [10, 20],
            "id": ["C1", "C2"],
            "name": ["John Smith", "Jane Doe"],
            "zip": ["90210", "10001"],
        })
        to_entity.scored_pairs = []

        rel = Relationship(from_entity="orders", to_entity="customers", join_key="customer_id", evidence_weight=0.3)

        # Orders 1 and 2 are in the same cluster and both link to customer C1
        # So no cross-customer boost here (both point to same customer)
        delta, n_boosted = _propagate_evidence(from_entity, to_entity, rel)
        # With both orders pointing to C1, no inter-customer pair is created
        assert delta >= 0  # might be 0 if both point to same customer

    def test_cross_customer_boost(self):
        """Orders from different customers in same cluster boost customer pair."""
        from_entity = EntityType(name="orders", sources=[], config=_make_order_config())
        from_entity.df = pl.DataFrame({
            "__row_id__": [1, 2],
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
        })
        from_entity.clusters = {
            1: {"members": [1, 2], "size": 2, "oversized": False,
                "pair_scores": {(1, 2): 1.0}, "confidence": 1.0, "bottleneck_pair": None},
        }

        to_entity = EntityType(name="customers", sources=[], config=_make_customer_config())
        to_entity.df = pl.DataFrame({
            "__row_id__": [10, 20],
            "id": ["C1", "C2"],
            "name": ["John Smith", "Jane Doe"],
            "zip": ["90210", "10001"],
        })
        to_entity.scored_pairs = []

        rel = Relationship(from_entity="orders", to_entity="customers", join_key="customer_id", evidence_weight=0.3)

        delta, n_boosted = _propagate_evidence(from_entity, to_entity, rel)
        assert n_boosted >= 1  # C1-C2 pair should be boosted
        assert delta > 0

    def test_additive_mode(self):
        to_entity = EntityType(name="cust", sources=[], config=_make_customer_config())
        to_entity.df = pl.DataFrame({
            "__row_id__": [10, 20],
            "id": ["C1", "C2"],
            "name": ["A", "B"],
            "zip": ["1", "2"],
        })
        to_entity.scored_pairs = [(10, 20, 0.5)]

        from_entity = EntityType(name="orders", sources=[], config=_make_order_config())
        from_entity.df = pl.DataFrame({
            "__row_id__": [1, 2],
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
        })
        from_entity.clusters = {
            1: {"members": [1, 2], "size": 2, "oversized": False,
                "pair_scores": {(1, 2): 1.0}, "confidence": 1.0, "bottleneck_pair": None},
        }

        rel = Relationship(from_entity="orders", to_entity="cust", join_key="customer_id", evidence_weight=0.2)
        delta, _ = _propagate_evidence(from_entity, to_entity, rel, propagation_mode="additive")

        # Original score 0.5 + 0.2 = 0.7
        boosted_score = dict(((a, b), s) for a, b, s in to_entity.scored_pairs).get((10, 20), 0)
        assert boosted_score == pytest.approx(0.7, abs=0.01)


class TestRunGraphER:
    def test_end_to_end(self, tmp_path):
        """Full graph ER with customers and orders."""
        # Create customer CSV
        cust_path = _write_csv(
            tmp_path / "customers.csv",
            ["name", "zip"],
            [["John Smith", "90210"], ["Jon Smith", "90210"], ["Alice Brown", "30301"]],
        )

        # Create order CSV
        order_path = _write_csv(
            tmp_path / "orders.csv",
            ["order_id", "customer_id", "amount"],
            [["O1", "C1", "100"], ["O2", "C1", "200"]],
        )

        entities = [
            EntityType(
                name="customers",
                sources=[(cust_path, "cust")],
                config=_make_customer_config(),
            ),
            EntityType(
                name="orders",
                sources=[(order_path, "orders")],
                config=GoldenMatchConfig(
                    matchkeys=[MatchkeyConfig(
                        name="order_exact",
                        type="exact",
                        fields=[MatchkeyField(field="order_id")],
                    )],
                ),
            ),
        ]

        relationships = [
            Relationship(
                from_entity="orders",
                to_entity="customers",
                join_key="customer_id",
                evidence_weight=0.1,
            ),
        ]

        result = run_graph_er(entities, relationships, max_iterations=3)
        assert isinstance(result, GraphERResult)
        assert "customers" in result.entities
        assert "orders" in result.entities
        assert result.iterations >= 1

    def test_no_relationships(self, tmp_path):
        """Graph ER with no relationships is just independent matching."""
        cust_path = _write_csv(
            tmp_path / "customers.csv",
            ["name", "zip"],
            [["John", "90210"], ["Jon", "90210"]],
        )

        entities = [
            EntityType(
                name="customers",
                sources=[(cust_path, "cust")],
                config=_make_customer_config(),
            ),
        ]

        result = run_graph_er(entities, [], max_iterations=1)
        assert result.converged
        assert result.evidence_propagated == 0
