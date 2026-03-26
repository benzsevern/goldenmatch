"""Tests for MemoryLearner -- threshold tuning and weight adjustment."""
import pytest
from datetime import datetime
from goldenmatch.core.memory.store import MemoryStore, Correction, LearnedAdjustment
from goldenmatch.core.memory.learner import MemoryLearner


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "test.db"))


def _add_corrections(store, matchkey, approved_scores, rejected_scores):
    for i, score in enumerate(approved_scores):
        store.add_correction(Correction(
            id=f"a-{matchkey}-{i}", id_a=i * 2, id_b=i * 2 + 1,
            decision="approve", source="steward", trust=1.0,
            field_hash=f"fh-{i}", record_hash=f"rh-{i}",
            original_score=score, matchkey_name=matchkey, dataset=matchkey,
            created_at=datetime.now(),
        ))
    for i, score in enumerate(rejected_scores):
        store.add_correction(Correction(
            id=f"r-{matchkey}-{i}", id_a=1000 + i * 2, id_b=1000 + i * 2 + 1,
            decision="reject", source="steward", trust=1.0,
            field_hash=f"fh-r-{i}", record_hash=f"rh-r-{i}",
            original_score=score, matchkey_name=matchkey, dataset=matchkey,
            created_at=datetime.now(),
        ))


class TestHasNewCorrections:
    def test_no_corrections(self, store):
        learner = MemoryLearner(store)
        assert not learner.has_new_corrections()

    def test_with_corrections_no_learning(self, store):
        _add_corrections(store, "mk1", [0.9], [0.3])
        learner = MemoryLearner(store)
        assert learner.has_new_corrections()

    def test_after_learning(self, store):
        _add_corrections(store, "mk1", [0.9] * 6, [0.3] * 5)
        learner = MemoryLearner(store, threshold_min=10)
        learner.learn()
        assert not learner.has_new_corrections()


class TestThresholdTuning:
    def test_below_minimum_no_learning(self, store):
        _add_corrections(store, "mk1", [0.9] * 5, [0.3] * 3)
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 0

    def test_threshold_clean_separation(self, store):
        _add_corrections(store, "mk1",
                         [0.85, 0.88, 0.90, 0.92, 0.95],
                         [0.50, 0.55, 0.60, 0.65, 0.70])
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 1
        adj = result[0]
        assert adj.matchkey_name == "mk1"
        assert 0.70 < adj.threshold < 0.85

    def test_threshold_overlapping(self, store):
        _add_corrections(store, "mk1",
                         [0.80, 0.82, 0.85, 0.88, 0.90],
                         [0.78, 0.81, 0.60, 0.55, 0.50])
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 1
        assert 0.50 < result[0].threshold < 0.95


class TestWeightAdjustment:
    def test_below_minimum_no_weights(self, store):
        _add_corrections(store, "mk1", [0.9] * 20, [0.3] * 10)
        learner = MemoryLearner(store, threshold_min=10, weights_min=50)
        result = learner.learn()
        assert len(result) == 1
        assert result[0].field_weights is None

    def test_threshold_still_produced_at_50(self, store):
        _add_corrections(store, "mk1", [0.9] * 30, [0.3] * 25)
        learner = MemoryLearner(store, threshold_min=10, weights_min=50)
        result = learner.learn()
        assert len(result) == 1
        assert result[0].threshold is not None


class TestEdgeCases:
    def test_all_approved_no_learning(self, store):
        _add_corrections(store, "mk1", [0.9] * 15, [])
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 0

    def test_all_rejected_no_learning(self, store):
        _add_corrections(store, "mk1", [], [0.3] * 15)
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        assert len(result) == 0

    def test_learn_filters_by_matchkey(self, store):
        _add_corrections(store, "mk1", [0.9] * 6, [0.3] * 5)
        _add_corrections(store, "mk2", [0.8] * 7, [0.4] * 4)
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn(matchkey_name="mk1")
        assert len(result) == 1
        assert result[0].matchkey_name == "mk1"


class TestMultipleMatchkeys:
    def test_learns_per_matchkey(self, store):
        _add_corrections(store, "mk1", [0.9] * 6, [0.3] * 5)
        _add_corrections(store, "mk2", [0.8] * 7, [0.4] * 4)
        learner = MemoryLearner(store, threshold_min=10)
        result = learner.learn()
        names = {r.matchkey_name for r in result}
        assert "mk1" in names
        assert "mk2" in names
