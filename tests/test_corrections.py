"""Tests for apply_corrections and hash functions."""
import pytest
from datetime import datetime
import polars as pl
from goldenmatch.core.memory.corrections import (
    apply_corrections, compute_field_hash, compute_record_hash,
    build_row_lookup, CorrectionStats,
)
from goldenmatch.core.memory.store import MemoryStore, Correction


@pytest.fixture
def store(tmp_path):
    return MemoryStore(backend="sqlite", path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "__row_id__": [1, 2, 3, 4],
        "name": ["John Smith", "John Smith", "Jane Doe", "Bob Jones"],
        "zip": ["12345", "12345", "67890", "11111"],
        "email": ["john@x.com", "jsmith@x.com", "jane@x.com", "bob@x.com"],
    })


def _get_hashes(df, id_a, id_b, fields):
    lookup = build_row_lookup(df, fields)
    fh = compute_field_hash(lookup[id_a], lookup[id_b])
    rh_a = compute_record_hash(df, id_a)
    rh_b = compute_record_hash(df, id_b)
    return fh, f"{rh_a}:{rh_b}"


def _make_correction(store, id_a, id_b, decision, field_hash="", record_hash="", **kw):
    defaults = dict(
        source="steward", trust=1.0, original_score=0.85, dataset="test",
        created_at=datetime.now(),
    )
    defaults.update(kw)
    c = Correction(
        id=f"c-{id_a}-{id_b}", id_a=id_a, id_b=id_b,
        decision=decision, field_hash=field_hash, record_hash=record_hash,
        **defaults,
    )
    store.add_correction(c)


class TestBuildRowLookup:
    def test_basic(self, sample_df):
        lookup = build_row_lookup(sample_df, ["name", "zip"])
        assert lookup[1] == ("John Smith", "12345")
        assert lookup[3] == ("Jane Doe", "67890")

    def test_all_rows(self, sample_df):
        lookup = build_row_lookup(sample_df, ["name"])
        assert len(lookup) == 4


class TestComputeFieldHash:
    def test_deterministic(self):
        h1 = compute_field_hash(("John", "12345"), ("John", "12345"))
        h2 = compute_field_hash(("John", "12345"), ("John", "12345"))
        assert h1 == h2

    def test_different_values(self):
        h1 = compute_field_hash(("John", "12345"), ("Jane", "67890"))
        h2 = compute_field_hash(("John", "12345"), ("John", "12345"))
        assert h1 != h2

    def test_length(self):
        h = compute_field_hash(("a",), ("b",))
        assert len(h) == 16


class TestComputeRecordHash:
    def test_deterministic(self, sample_df):
        h1 = compute_record_hash(sample_df, 1)
        h2 = compute_record_hash(sample_df, 1)
        assert h1 == h2

    def test_different_records(self, sample_df):
        h1 = compute_record_hash(sample_df, 1)
        h2 = compute_record_hash(sample_df, 3)
        assert h1 != h2


class TestApplyCorrections:
    def test_no_corrections(self, store, sample_df):
        pairs = [(1, 2, 0.85), (3, 4, 0.60)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result == pairs
        assert stats.applied == 0

    def test_approved_override_with_hashes(self, store, sample_df):
        fh, rh = _get_hashes(sample_df, 1, 2, ["name", "zip"])
        _make_correction(store, 1, 2, "approve", fh, rh)
        pairs = [(1, 2, 0.85)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 1.0
        assert stats.applied == 1

    def test_rejected_override_with_hashes(self, store, sample_df):
        fh, rh = _get_hashes(sample_df, 1, 2, ["name", "zip"])
        _make_correction(store, 1, 2, "reject", fh, rh)
        pairs = [(1, 2, 0.85)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 0.0
        assert stats.applied == 1

    def test_approved_override_empty_hashes(self, store, sample_df):
        """Empty hashes = skip staleness check, always apply."""
        _make_correction(store, 1, 2, "approve")  # empty hashes
        pairs = [(1, 2, 0.85)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 1.0
        assert stats.applied == 1

    def test_stale_field_hash(self, store, sample_df):
        _make_correction(store, 1, 2, "approve", "wrong_hash", "wrong_record")
        pairs = [(1, 2, 0.85)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 0.85
        assert stats.stale == 1
        assert (1, 2) in stats.stale_pairs

    def test_mixed_corrections(self, store, sample_df):
        fh, rh = _get_hashes(sample_df, 1, 2, ["name", "zip"])
        _make_correction(store, 1, 2, "approve", fh, rh)
        pairs = [(1, 2, 0.85), (3, 4, 0.60)]
        result, stats = apply_corrections(pairs, store, sample_df, ["name", "zip"], dataset="test")
        assert result[0][2] == 1.0
        assert result[1][2] == 0.60
        assert stats.applied == 1
        assert stats.total_pairs == 2
