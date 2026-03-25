"""Tests for confidence-gated review queue."""

import os

import pytest

from goldenmatch.core.review_queue import ReviewItem, ReviewQueue, gate_pairs


# ── ReviewItem ──────────────────────────────────────────────────────────

class TestReviewItem:
    def test_creation_defaults(self):
        item = ReviewItem(job_name="j1", id_a=1, id_b=2, score=0.85, explanation="close match")
        assert item.status == "pending"
        assert item.decided_by is None
        assert item.decided_at is None
        assert item.reason is None

    def test_approve(self):
        item = ReviewItem(job_name="j1", id_a=1, id_b=2, score=0.85, explanation="x")
        item.approve("alice")
        assert item.status == "approved"
        assert item.decided_by == "alice"
        assert item.decided_at is not None

    def test_reject(self):
        item = ReviewItem(job_name="j1", id_a=1, id_b=2, score=0.80, explanation="x")
        item.reject("bob", reason="different entities")
        assert item.status == "rejected"
        assert item.decided_by == "bob"
        assert item.reason == "different entities"
        assert item.decided_at is not None

    def test_reject_default_reason(self):
        item = ReviewItem(job_name="j1", id_a=1, id_b=2, score=0.80, explanation="x")
        item.reject("bob")
        assert item.reason == ""


# ── gate_pairs ──────────────────────────────────────────────────────────

class TestGatePairs:
    def test_normal_split(self):
        pairs = [
            (1, 2, 0.99),  # auto-merge
            (3, 4, 0.85),  # review
            (5, 6, 0.50),  # auto-reject
            (7, 8, 0.95),  # boundary: review (not > 0.95)
            (9, 10, 0.75), # boundary: review (>= 0.75)
        ]
        merged, review, rejected = gate_pairs(pairs)
        assert len(merged) == 1
        assert merged[0] == (1, 2, 0.99)
        assert len(review) == 3
        assert len(rejected) == 1
        assert rejected[0] == (5, 6, 0.50)

    def test_custom_thresholds(self):
        pairs = [(1, 2, 0.90), (3, 4, 0.60), (5, 6, 0.40)]
        merged, review, rejected = gate_pairs(pairs, merge_threshold=0.85, review_threshold=0.50)
        assert len(merged) == 1
        assert len(review) == 1
        assert len(rejected) == 1

    def test_empty_input(self):
        merged, review, rejected = gate_pairs([])
        assert merged == []
        assert review == []
        assert rejected == []

    def test_all_auto_merged(self):
        pairs = [(1, 2, 0.99), (3, 4, 0.97)]
        merged, review, rejected = gate_pairs(pairs)
        assert len(merged) == 2
        assert len(review) == 0
        assert len(rejected) == 0

    def test_all_auto_rejected(self):
        pairs = [(1, 2, 0.10), (3, 4, 0.20)]
        merged, review, rejected = gate_pairs(pairs)
        assert len(merged) == 0
        assert len(review) == 0
        assert len(rejected) == 2


# ── Memory backend ──────────────────────────────────────────────────────

class TestMemoryBackend:
    def test_add_and_list_pending(self):
        q = ReviewQueue(backend="memory")
        q.add("job1", 1, 2, 0.85, "close match")
        q.add("job1", 3, 4, 0.80, "partial match")
        pending = q.list_pending("job1")
        assert len(pending) == 2
        assert all(isinstance(it, ReviewItem) for it in pending)

    def test_approve_removes_from_pending(self):
        q = ReviewQueue(backend="memory")
        q.add("job1", 1, 2, 0.85, "close match")
        q.approve("job1", 1, 2, "alice")
        assert len(q.list_pending("job1")) == 0

    def test_reject_removes_from_pending(self):
        q = ReviewQueue(backend="memory")
        q.add("job1", 1, 2, 0.85, "close match")
        q.reject("job1", 1, 2, "bob", reason="wrong")
        assert len(q.list_pending("job1")) == 0

    def test_stats(self):
        q = ReviewQueue(backend="memory")
        q.add("job1", 1, 2, 0.85, "x")
        q.add("job1", 3, 4, 0.80, "y")
        q.add("job1", 5, 6, 0.78, "z")
        q.approve("job1", 1, 2, "alice")
        q.reject("job1", 3, 4, "bob")
        s = q.stats("job1")
        assert s == {"pending": 1, "approved": 1, "rejected": 1}

    def test_empty_job(self):
        q = ReviewQueue(backend="memory")
        assert q.list_pending("nonexistent") == []
        assert q.stats("nonexistent") == {"pending": 0, "approved": 0, "rejected": 0}

    def test_storage_tier(self):
        q = ReviewQueue(backend="memory")
        assert q.storage_tier == "memory"

    def test_multiple_jobs_isolated(self):
        q = ReviewQueue(backend="memory")
        q.add("job1", 1, 2, 0.85, "x")
        q.add("job2", 3, 4, 0.80, "y")
        assert len(q.list_pending("job1")) == 1
        assert len(q.list_pending("job2")) == 1


# ── SQLite backend ──────────────────────────────────────────────────────

class TestSQLiteBackend:
    def test_add_and_list(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        q = ReviewQueue(backend="sqlite")
        assert q.storage_tier == "sqlite"
        q.add("job1", 1, 2, 0.85, "close match")
        q.add("job1", 3, 4, 0.80, "partial match")
        pending = q.list_pending("job1")
        assert len(pending) == 2

    def test_approve_and_reject(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        q = ReviewQueue(backend="sqlite")
        q.add("job1", 1, 2, 0.85, "x")
        q.add("job1", 3, 4, 0.80, "y")
        q.approve("job1", 1, 2, "alice")
        q.reject("job1", 3, 4, "bob", reason="bad")
        assert len(q.list_pending("job1")) == 0
        s = q.stats("job1")
        assert s == {"pending": 0, "approved": 1, "rejected": 1}

    def test_persistence_across_instances(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        q1 = ReviewQueue(backend="sqlite")
        q1.add("job1", 1, 2, 0.85, "close match")
        q1.approve("job1", 1, 2, "alice")
        q1.add("job1", 3, 4, 0.80, "partial match")

        # New instance reads same DB
        q2 = ReviewQueue(backend="sqlite")
        pending = q2.list_pending("job1")
        assert len(pending) == 1
        assert pending[0].id_a == 3
        s = q2.stats("job1")
        assert s == {"pending": 1, "approved": 1, "rejected": 0}

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            ReviewQueue(backend="redis")
