"""Tests for MemoryStore CRUD operations."""
import pytest
from datetime import datetime
from goldenmatch.core.memory.store import MemoryStore, Correction, LearnedAdjustment


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "test_memory.db"))


def _make_correction(**kwargs) -> Correction:
    defaults = dict(
        id="test-1", id_a=1, id_b=2, decision="approve",
        source="steward", trust=1.0, field_hash="abc123",
        record_hash="def456", original_score=0.85,
        matchkey_name=None, reason=None, dataset="test",
        created_at=datetime.now(),
    )
    defaults.update(kwargs)
    return Correction(**defaults)


class TestAddAndGet:
    def test_add_and_get(self, store):
        c = _make_correction()
        store.add_correction(c)
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result is not None
        assert result.decision == "approve"
        assert result.trust == 1.0

    def test_get_missing_returns_none(self, store):
        assert store.get_pair_correction(99, 100) is None

    def test_get_corrections_list(self, store):
        store.add_correction(_make_correction(id="c1", id_a=1, id_b=2))
        store.add_correction(_make_correction(id="c2", id_a=3, id_b=4))
        result = store.get_corrections(dataset="test")
        assert len(result) == 2

    def test_count_corrections(self, store):
        store.add_correction(_make_correction(id="c1", id_a=1, id_b=2))
        store.add_correction(_make_correction(id="c2", id_a=3, id_b=4))
        assert store.count_corrections(dataset="test") == 2
        assert store.count_corrections(dataset="other") == 0


class TestUpsertAndTrust:
    def test_upsert_higher_trust_wins(self, store):
        store.add_correction(_make_correction(
            id="c1", id_a=1, id_b=2, decision="approve", trust=0.5, source="llm",
        ))
        store.add_correction(_make_correction(
            id="c2", id_a=1, id_b=2, decision="reject", trust=1.0, source="steward",
        ))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result.decision == "reject"
        assert result.trust == 1.0

    def test_upsert_lower_trust_ignored(self, store):
        store.add_correction(_make_correction(
            id="c1", id_a=1, id_b=2, decision="approve", trust=1.0, source="steward",
        ))
        store.add_correction(_make_correction(
            id="c2", id_a=1, id_b=2, decision="reject", trust=0.5, source="llm",
        ))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result.decision == "approve"

    def test_upsert_same_trust_latest_wins(self, store):
        store.add_correction(_make_correction(
            id="c1", id_a=1, id_b=2, decision="approve", trust=1.0,
            created_at=datetime(2026, 1, 1),
        ))
        store.add_correction(_make_correction(
            id="c2", id_a=1, id_b=2, decision="reject", trust=1.0,
            created_at=datetime(2026, 3, 1),
        ))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result.decision == "reject"


class TestBulkLookup:
    def test_bulk_lookup(self, store):
        store.add_correction(_make_correction(id="c1", id_a=1, id_b=2))
        store.add_correction(_make_correction(id="c2", id_a=3, id_b=4))
        result = store.get_pair_corrections_bulk(
            [(1, 2), (3, 4), (5, 6)], dataset="test",
        )
        assert (1, 2) in result
        assert (3, 4) in result
        assert (5, 6) not in result


class TestAdjustments:
    def test_save_and_get_adjustment(self, store):
        adj = LearnedAdjustment(
            matchkey_name="mk1", threshold=0.82,
            field_weights=None, sample_size=15,
            learned_at=datetime.now(),
        )
        store.save_adjustment(adj)
        result = store.get_adjustment("mk1")
        assert result is not None
        assert result.threshold == 0.82

    def test_get_all_adjustments(self, store):
        store.save_adjustment(LearnedAdjustment(
            matchkey_name="mk1", threshold=0.8, field_weights=None,
            sample_size=10, learned_at=datetime.now(),
        ))
        store.save_adjustment(LearnedAdjustment(
            matchkey_name="mk2", threshold=0.9,
            field_weights={"name": 0.6, "zip": 0.4},
            sample_size=55, learned_at=datetime.now(),
        ))
        result = store.get_all_adjustments()
        assert len(result) == 2


class TestCorrectionsSince:
    def test_corrections_since(self, store):
        old = _make_correction(id="c1", id_a=1, id_b=2, created_at=datetime(2026, 1, 1))
        new = _make_correction(id="c2", id_a=3, id_b=4, created_at=datetime(2026, 3, 25))
        store.add_correction(old)
        store.add_correction(new)
        result = store.corrections_since(datetime(2026, 3, 1))
        assert len(result) == 1
        assert result[0].id_a == 3


class TestPairCanonicalization:
    def test_reversed_pair_finds_correction(self, store):
        """Correction stored as (2,1) should be found when looking up (1,2)."""
        store.add_correction(_make_correction(id="c1", id_a=2, id_b=1))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result is not None
        assert result.decision == "approve"

    def test_reversed_pair_upsert(self, store):
        """Storing (1,2) then (2,1) should upsert the same logical pair."""
        store.add_correction(_make_correction(id="c1", id_a=1, id_b=2, decision="approve"))
        store.add_correction(_make_correction(id="c2", id_a=2, id_b=1, decision="reject"))
        result = store.get_pair_correction(1, 2, dataset="test")
        assert result.decision == "reject"
        assert store.count_corrections(dataset="test") == 1

    def test_bulk_lookup_reversed(self, store):
        store.add_correction(_make_correction(id="c1", id_a=2, id_b=1))
        result = store.get_pair_corrections_bulk([(1, 2)], dataset="test")
        assert (1, 2) in result


class TestUnsupportedBackend:
    def test_raises_not_implemented(self, tmp_path):
        import pytest
        with pytest.raises(NotImplementedError, match="postgres"):
            MemoryStore(backend="postgres", path=str(tmp_path / "x.db"))


class TestContextManager:
    def test_context_manager(self, tmp_path):
        with MemoryStore(backend="sqlite", path=str(tmp_path / "ctx.db")) as store:
            store.add_correction(_make_correction())
            assert store.count_corrections(dataset="test") == 1
        # Connection closed after with block


class TestLastLearnTime:
    def test_no_adjustments(self, store):
        assert store.last_learn_time() is None

    def test_with_adjustment(self, store):
        now = datetime.now()
        store.save_adjustment(LearnedAdjustment(
            matchkey_name="mk1", threshold=0.8, field_weights=None,
            sample_size=10, learned_at=now,
        ))
        result = store.last_learn_time()
        assert result is not None
