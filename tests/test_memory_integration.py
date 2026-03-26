"""End-to-end integration tests for Learning Memory."""
import pytest
from datetime import datetime
import polars as pl
from goldenmatch.core.memory.store import MemoryStore, Correction
from goldenmatch.core.memory.corrections import (
    apply_corrections, build_row_lookup, compute_field_hash, compute_record_hash,
)
from goldenmatch.core.memory.learner import MemoryLearner


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "integration.db"))


@pytest.fixture
def df():
    return pl.DataFrame({
        "__row_id__": [1, 2, 3, 4, 5],
        "name": ["John Smith", "John Smith", "Jane Doe", "Bob Jones", "John Smith"],
        "zip": ["12345", "12345", "67890", "11111", "12345"],
    })


class TestCorrectThenRerun:
    def test_approved_pair_gets_1_on_rerun(self, store, df):
        fields = ["name", "zip"]
        lookup = build_row_lookup(df, fields)
        fh = compute_field_hash(lookup[1], lookup[2])
        rh = f"{compute_record_hash(df, 1)}:{compute_record_hash(df, 2)}"

        store.add_correction(Correction(
            id="c1", id_a=1, id_b=2, decision="approve",
            source="steward", trust=1.0, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))

        pairs = [(1, 2, 0.83), (1, 5, 0.91), (3, 4, 0.40)]
        result, stats = apply_corrections(pairs, store, df, fields, dataset="test")

        assert result[0] == (1, 2, 1.0)
        assert result[1] == (1, 5, 0.91)
        assert result[2] == (3, 4, 0.40)
        assert stats.applied == 1

    def test_rejected_pair_gets_0_on_rerun(self, store, df):
        fields = ["name", "zip"]
        lookup = build_row_lookup(df, fields)
        fh = compute_field_hash(lookup[1], lookup[2])
        rh = f"{compute_record_hash(df, 1)}:{compute_record_hash(df, 2)}"

        store.add_correction(Correction(
            id="c1", id_a=1, id_b=2, decision="reject",
            source="steward", trust=1.0, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))

        pairs = [(1, 2, 0.83)]
        result, stats = apply_corrections(pairs, store, df, fields, dataset="test")
        assert result[0] == (1, 2, 0.0)


class TestCorrectThenLearn:
    def test_threshold_improves_after_corrections(self, store):
        for i in range(7):
            store.add_correction(Correction(
                id=f"a{i}", id_a=i*2, id_b=i*2+1, decision="approve",
                source="steward", trust=1.0, field_hash=f"fh{i}", record_hash=f"rh{i}",
                original_score=0.85 + i * 0.02, dataset="mk1",
                created_at=datetime.now(),
            ))
        for i in range(5):
            store.add_correction(Correction(
                id=f"r{i}", id_a=100+i*2, id_b=100+i*2+1, decision="reject",
                source="steward", trust=1.0, field_hash=f"rfh{i}", record_hash=f"rrh{i}",
                original_score=0.60 + i * 0.02, dataset="mk1",
                created_at=datetime.now(),
            ))

        learner = MemoryLearner(store, threshold_min=10)
        adjustments = learner.learn()
        assert len(adjustments) == 1
        assert 0.68 < adjustments[0].threshold < 0.85

    def test_no_new_corrections_after_learn(self, store):
        for i in range(12):
            store.add_correction(Correction(
                id=f"c{i}", id_a=i*2, id_b=i*2+1,
                decision="approve" if i < 7 else "reject",
                source="steward", trust=1.0,
                field_hash=f"fh{i}", record_hash=f"rh{i}",
                original_score=0.85 if i < 7 else 0.50,
                dataset="mk1", created_at=datetime.now(),
            ))

        learner = MemoryLearner(store, threshold_min=10)
        assert learner.has_new_corrections()
        learner.learn()
        assert not learner.has_new_corrections()


class TestHumanOverridesAgent:
    def test_human_beats_agent(self, store, df):
        fields = ["name", "zip"]
        lookup = build_row_lookup(df, fields)
        fh = compute_field_hash(lookup[1], lookup[2])
        rh = f"{compute_record_hash(df, 1)}:{compute_record_hash(df, 2)}"

        store.add_correction(Correction(
            id="agent-1", id_a=1, id_b=2, decision="approve",
            source="llm", trust=0.5, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))
        store.add_correction(Correction(
            id="human-1", id_a=1, id_b=2, decision="reject",
            source="steward", trust=1.0, field_hash=fh, record_hash=rh,
            original_score=0.82, dataset="test", created_at=datetime.now(),
        ))

        pairs = [(1, 2, 0.82)]
        result, stats = apply_corrections(pairs, store, df, fields, dataset="test")
        assert result[0][2] == 0.0
