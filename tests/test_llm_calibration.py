"""Tests for iterative LLM calibration."""
from __future__ import annotations

import pytest


class TestComputeThreshold:
    """Test the grid-search threshold computation."""

    def test_clean_separation(self):
        """When matches and non-matches are cleanly separated, threshold lands between."""
        from goldenmatch.core.llm_scorer import _compute_threshold
        # All NO below 0.85, all YES above
        labels = {
            0: (0.80, False),
            1: (0.82, False),
            2: (0.84, False),
            3: (0.86, True),
            4: (0.88, True),
            5: (0.90, True),
            6: (0.92, True),
        }
        threshold = _compute_threshold(labels)
        assert 0.84 < threshold < 0.86

    def test_clean_separation_exact(self):
        """Exercises the max(no) < min(yes) fast path."""
        from goldenmatch.core.llm_scorer import _compute_threshold
        labels = {
            0: (0.80, False),
            1: (0.82, False),
            2: (0.90, True),
            3: (0.92, True),
        }
        threshold = _compute_threshold(labels)
        # Should be midpoint of gap: (0.82 + 0.90) / 2 = 0.86
        assert 0.85 <= threshold <= 0.87

    def test_all_yes(self):
        """When all labels are YES, threshold should be at or below minimum score."""
        from goldenmatch.core.llm_scorer import _compute_threshold
        labels = {
            0: (0.80, True),
            1: (0.85, True),
            2: (0.90, True),
        }
        threshold = _compute_threshold(labels)
        assert threshold <= 0.80

    def test_all_no(self):
        """When all labels are NO, threshold should be above the max score."""
        from goldenmatch.core.llm_scorer import _compute_threshold
        labels = {
            0: (0.80, False),
            1: (0.85, False),
            2: (0.90, False),
        }
        threshold = _compute_threshold(labels)
        assert threshold > 0.90

    def test_overlapping_scores(self):
        """When there is overlap, threshold minimizes misclassifications."""
        from goldenmatch.core.llm_scorer import _compute_threshold
        labels = {
            0: (0.80, False),
            1: (0.82, False),
            2: (0.84, True),   # overlap
            3: (0.85, False),  # overlap
            4: (0.87, True),
            5: (0.90, True),
            6: (0.92, True),
        }
        threshold = _compute_threshold(labels)
        assert 0.83 <= threshold <= 0.88


import random


class TestStratifiedSample:
    """Test round 1 stratified sampling."""

    def test_covers_full_range(self):
        from goldenmatch.core.llm_scorer import _stratified_sample
        pairs = [(i, i+1, 0.80 + 0.15 * (i / 999)) for i in range(1000)]
        candidate_indices = list(range(1000))
        sample = _stratified_sample(
            candidate_indices, pairs, sample_size=100,
            score_lo=0.80, score_hi=0.95, already_scored=set(),
        )
        assert len(sample) == 100
        sample_scores = sorted(pairs[i][2] for i in sample)
        assert sample_scores[0] < 0.83
        assert sample_scores[-1] > 0.92

    def test_respects_already_scored(self):
        from goldenmatch.core.llm_scorer import _stratified_sample
        pairs = [(i, i+1, 0.80 + 0.15 * (i / 99)) for i in range(100)]
        candidate_indices = list(range(100))
        already_scored = set(range(50))
        sample = _stratified_sample(
            candidate_indices, pairs, sample_size=100,
            score_lo=0.80, score_hi=0.95, already_scored=already_scored,
        )
        assert len(sample) == 50
        assert not (set(sample) & already_scored)

    def test_fewer_candidates_than_sample_size(self):
        from goldenmatch.core.llm_scorer import _stratified_sample
        pairs = [(i, i+1, 0.85) for i in range(10)]
        candidate_indices = list(range(10))
        sample = _stratified_sample(
            candidate_indices, pairs, sample_size=100,
            score_lo=0.80, score_hi=0.95, already_scored=set(),
        )
        assert len(sample) == 10


class TestFocusedSample:
    """Test round 2+ focused sampling near threshold."""

    def test_samples_near_threshold(self):
        from goldenmatch.core.llm_scorer import _focused_sample
        pairs = [(i, i+1, 0.80 + 0.15 * (i / 999)) for i in range(1000)]
        candidate_indices = list(range(1000))
        sample = _focused_sample(
            candidate_indices, pairs, sample_size=100,
            threshold=0.87, band_width=0.03,
            already_scored=set(),
        )
        for i in sample:
            assert 0.84 - 0.001 <= pairs[i][2] <= 0.90 + 0.001

    def test_respects_already_scored(self):
        from goldenmatch.core.llm_scorer import _focused_sample
        pairs = [(i, i+1, 0.87) for i in range(100)]
        candidate_indices = list(range(100))
        already_scored = set(range(80))
        sample = _focused_sample(
            candidate_indices, pairs, sample_size=100,
            threshold=0.87, band_width=0.03,
            already_scored=already_scored,
        )
        assert len(sample) == 20
        assert not (set(sample) & already_scored)


from unittest.mock import patch, MagicMock


class TestIterativeCalibrate:
    """Test the full calibration loop with mocked LLM."""

    def _make_pairs_and_lookup(self, n=500):
        pairs = []
        for i in range(n):
            score = 0.80 + 0.15 * (i / (n - 1))
            pairs.append((i * 2, i * 2 + 1, score))
        row_lookup = {r: {"name": f"record_{r}"} for r in range(n * 2 + 2)}
        candidate_indices = list(range(n))
        return pairs, row_lookup, candidate_indices

    def test_converges_in_few_rounds(self):
        from goldenmatch.core.llm_scorer import _iterative_calibrate

        pairs, row_lookup, candidates = self._make_pairs_and_lookup(500)
        real_threshold = 0.87

        def mock_batch_score(candidate_indices, pairs, row_lookup, cols,
                             provider, api_key, model, batch_size,
                             budget=None, max_workers=1):
            return {i: pairs[i][2] >= real_threshold for i in candidate_indices}

        with patch("goldenmatch.core.llm_scorer._batch_score", side_effect=mock_batch_score):
            threshold, llm_results = _iterative_calibrate(
                candidates, pairs, row_lookup, ["name"],
                "openai", "fake-key", "gpt-4o-mini", 75,
                sample_size=100, max_rounds=5,
                convergence_delta=0.01,
                candidate_lo=0.80, candidate_hi=0.95,
            )

        assert 0.85 <= threshold <= 0.89
        assert 100 <= len(llm_results) <= 400

    def test_accumulates_labels_across_rounds(self):
        """Labels from all rounds should be preserved in llm_results."""
        from goldenmatch.core.llm_scorer import _iterative_calibrate

        pairs, row_lookup, candidates = self._make_pairs_and_lookup(500)

        call_count = 0
        def mock_batch_score(candidate_indices, pairs, row_lookup, cols,
                             provider, api_key, model, batch_size,
                             budget=None, max_workers=1):
            nonlocal call_count
            call_count += 1
            return {i: pairs[i][2] >= 0.87 for i in candidate_indices}

        with patch("goldenmatch.core.llm_scorer._batch_score", side_effect=mock_batch_score):
            threshold, llm_results = _iterative_calibrate(
                candidates, pairs, row_lookup, ["name"],
                "openai", "fake-key", "gpt-4o-mini", 75,
                sample_size=50, max_rounds=5,
                convergence_delta=0.01,
                candidate_lo=0.80, candidate_hi=0.95,
            )

        # Should have results from multiple rounds
        assert len(llm_results) >= 50  # at least round 1
        # If converged in 2 rounds, should have ~100 labels
        if call_count >= 2:
            assert len(llm_results) >= 80  # accumulated from both rounds

    def test_budget_exhaustion_mid_calibration(self):
        from goldenmatch.core.llm_scorer import _iterative_calibrate
        from goldenmatch.core.llm_budget import BudgetTracker
        from goldenmatch.config.schemas import BudgetConfig

        pairs, row_lookup, candidates = self._make_pairs_and_lookup(500)
        budget = BudgetTracker(BudgetConfig(max_calls=1))

        def mock_batch_score(candidate_indices, pairs, row_lookup, cols,
                             provider, api_key, model, batch_size,
                             budget=None, max_workers=1):
            if budget:
                budget.record_usage(100, 50, model)
            return {i: pairs[i][2] >= 0.87 for i in candidate_indices}

        with patch("goldenmatch.core.llm_scorer._batch_score", side_effect=mock_batch_score):
            threshold, llm_results = _iterative_calibrate(
                candidates, pairs, row_lookup, ["name"],
                "openai", "fake-key", "gpt-4o-mini", 75,
                budget=budget,
                sample_size=100, max_rounds=5,
                convergence_delta=0.01,
                candidate_lo=0.80, candidate_hi=0.95,
            )

        assert isinstance(threshold, float)
        assert len(llm_results) > 0

    def test_immediate_convergence(self):
        """With convergence_delta=1.0, should stop after round 1."""
        from goldenmatch.core.llm_scorer import _iterative_calibrate

        pairs, row_lookup, candidates = self._make_pairs_and_lookup(500)

        call_count = 0
        def mock_batch_score(candidate_indices, pairs, row_lookup, cols,
                             provider, api_key, model, batch_size,
                             budget=None, max_workers=1):
            nonlocal call_count
            call_count += 1
            return {i: pairs[i][2] >= 0.87 for i in candidate_indices}

        with patch("goldenmatch.core.llm_scorer._batch_score", side_effect=mock_batch_score):
            threshold, llm_results = _iterative_calibrate(
                candidates, pairs, row_lookup, ["name"],
                "openai", "fake-key", "gpt-4o-mini", 75,
                sample_size=100, max_rounds=5,
                convergence_delta=1.0,  # always converges
                candidate_lo=0.80, candidate_hi=0.95,
            )

        # Round 1 runs, then round 2 runs and convergence check passes
        # (delta from midpoint 0.875 to ~0.87 is < 1.0)
        assert len(llm_results) <= 200  # at most 2 rounds


import polars as pl


class TestLLMScorePairsCalibration:
    """End-to-end test of llm_score_pairs with calibration path."""

    def test_large_candidate_set_uses_calibration(self):
        """When candidates > sample_size, calibration is used."""
        from goldenmatch.core.llm_scorer import llm_score_pairs
        from goldenmatch.config.schemas import LLMScorerConfig

        n = 500
        pairs = [(i * 2, i * 2 + 1, 0.80 + 0.15 * (i / (n - 1))) for i in range(n)]
        pairs.extend([(1001, 1002, 0.98), (1003, 1004, 0.99)])

        df = pl.DataFrame({
            "__row_id__": list(range(n * 2 + 5)),
            "name": [f"record_{i}" for i in range(n * 2 + 5)],
        })

        config = LLMScorerConfig(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            auto_threshold=0.95,
            candidate_lo=0.80,
            candidate_hi=0.95,
            calibration_sample_size=100,
        )

        real_threshold = 0.87

        def mock_batch_score(candidate_indices, pairs, row_lookup, cols,
                             provider, api_key, model, batch_size,
                             budget=None, max_workers=5):
            return {i: pairs[i][2] >= real_threshold for i in candidate_indices}

        with patch("goldenmatch.core.llm_scorer._batch_score", side_effect=mock_batch_score):
            with patch("goldenmatch.core.llm_scorer._detect_provider", return_value=("openai", "fake-key")):
                result = llm_score_pairs(pairs, df, config=config)

        # Auto-accept pairs should be 1.0
        assert result[500][2] == 1.0
        assert result[501][2] == 1.0

        # Pairs above learned threshold should be promoted
        promoted = [r for r in result[:500] if r[2] == 1.0]
        assert len(promoted) > 0

        # Pairs below learned threshold keep original score (NOT 0.0)
        kept = [r for r in result[:500] if r[2] != 1.0]
        for a, b, s in kept:
            assert s > 0

    def test_small_candidate_set_uses_direct_scoring(self):
        """When candidates <= sample_size, all are scored directly."""
        from goldenmatch.core.llm_scorer import llm_score_pairs
        from goldenmatch.config.schemas import LLMScorerConfig

        n = 50
        pairs = [(i * 2, i * 2 + 1, 0.85) for i in range(n)]

        df = pl.DataFrame({
            "__row_id__": list(range(n * 2 + 1)),
            "name": [f"record_{i}" for i in range(n * 2 + 1)],
        })

        config = LLMScorerConfig(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            candidate_lo=0.80,
            calibration_sample_size=100,
        )

        def mock_batch_score(candidate_indices, pairs, row_lookup, cols,
                             provider, api_key, model, batch_size,
                             budget=None, max_workers=5):
            return {i: (i % 2 == 0) for i in candidate_indices}

        with patch("goldenmatch.core.llm_scorer._batch_score", side_effect=mock_batch_score):
            with patch("goldenmatch.core.llm_scorer._detect_provider", return_value=("openai", "fake-key")):
                result = llm_score_pairs(pairs, df, config=config)

        for i, (a, b, s) in enumerate(result):
            if i % 2 == 0:
                assert s == 1.0  # LLM said YES
            else:
                assert s == 0.85  # LLM said NO -> keep original (never demote)


import threading
from unittest.mock import patch, MagicMock


class TestConcurrentBatchScoring:
    """Test the ThreadPoolExecutor path in _batch_score."""

    def test_concurrent_path_collects_all_results(self):
        """With max_workers=3 and >2 batches, concurrent path should collect all pair results."""
        from goldenmatch.core.llm_scorer import _batch_score

        # Create enough pairs for >2 batches at batch_size=5
        n = 20
        pairs = [(i * 2, i * 2 + 1, 0.85) for i in range(n)]
        row_lookup = {r: {"name": f"rec_{r}"} for r in range(n * 2 + 2)}
        candidate_indices = list(range(n))

        call_count = 0
        lock = threading.Lock()

        def mock_openai(prompt, api_key, model, max_tokens=100):
            nonlocal call_count
            with lock:
                call_count += 1
            # Count pairs in prompt by counting "A:"
            n_pairs = prompt.count("A:")
            lines = "\n".join(f"{i+1}. YES" for i in range(n_pairs))
            return lines, 100, 50

        with patch("goldenmatch.core.llm_scorer._call_openai", side_effect=mock_openai):
            results = _batch_score(
                candidate_indices, pairs, row_lookup, ["name"],
                "openai", "fake-key", "gpt-4o-mini", 5,  # batch_size=5 -> 4 batches
                max_workers=3,
            )

        # All 20 pairs should have results
        assert len(results) == 20
        # All should be True (we returned YES for all)
        assert all(v is True for v in results.values())
        # Should have used multiple batches
        assert call_count == 4  # 20 pairs / 5 per batch

    def test_sequential_path_for_small_batches(self):
        """With <=2 batches, should use sequential path."""
        from goldenmatch.core.llm_scorer import _batch_score

        pairs = [(0, 1, 0.85), (2, 3, 0.85)]
        row_lookup = {r: {"name": f"rec_{r}"} for r in range(5)}

        def mock_openai(prompt, api_key, model, max_tokens=100):
            return "1. YES\n2. YES", 50, 20

        with patch("goldenmatch.core.llm_scorer._call_openai", side_effect=mock_openai):
            results = _batch_score(
                [0, 1], pairs, row_lookup, ["name"],
                "openai", "fake-key", "gpt-4o-mini", 10,  # batch_size=10 -> 1 batch
                max_workers=5,
            )

        assert len(results) == 2


class TestBudgetTrackerThreadSafety:
    """Test that BudgetTracker is thread-safe under concurrent access."""

    def test_concurrent_record_usage(self):
        """Multiple threads calling record_usage should produce correct totals."""
        from goldenmatch.core.llm_budget import BudgetTracker
        from goldenmatch.config.schemas import BudgetConfig

        budget = BudgetTracker(BudgetConfig(max_cost_usd=100.0))
        n_threads = 10
        calls_per_thread = 100

        def worker():
            for _ in range(calls_per_thread):
                budget.record_usage(100, 50, "gpt-4o-mini")

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert budget.total_calls == n_threads * calls_per_thread

    def test_budget_exhaustion_under_concurrency(self):
        """Budget should eventually stop allowing sends under concurrent load."""
        from goldenmatch.core.llm_budget import BudgetTracker
        from goldenmatch.config.schemas import BudgetConfig

        budget = BudgetTracker(BudgetConfig(max_calls=10))

        sent_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal sent_count
            for _ in range(100):
                if budget.can_send(100):
                    budget.record_usage(100, 50, "gpt-4o-mini")
                    with lock:
                        sent_count += 1

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not exceed max_calls by much (race window is small)
        assert budget.total_calls <= 15  # some overshoot acceptable due to race
        assert budget.budget_exhausted


class TestGetEmbedderRouting:
    """Test that get_embedder returns the right backend based on GPU mode."""

    def test_vertex_mode_returns_vertex_embedder(self):
        """When GPU mode is VERTEX, get_embedder should return VertexEmbedder."""
        from goldenmatch.core.embedder import get_embedder, _embedders
        from goldenmatch.core.gpu import GPUMode

        # Clear cache
        _embedders.clear()

        mock_vertex = MagicMock()
        mock_vertex_cls = MagicMock(return_value=mock_vertex)

        # detect_gpu_mode is imported inside get_embedder, so patch at source
        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.VERTEX):
            # Patch vertex_embedder module so the inner import succeeds
            with patch.dict("sys.modules", {"goldenmatch.core.vertex_embedder": MagicMock(VertexEmbedder=mock_vertex_cls)}):
                result = get_embedder("test-model")

        assert result is mock_vertex
        _embedders.clear()

    def test_cpu_mode_returns_local_embedder(self):
        """When GPU mode is CPU_SAFE, get_embedder should return local Embedder."""
        from goldenmatch.core.embedder import get_embedder, _embedders, Embedder
        from goldenmatch.core.gpu import GPUMode

        _embedders.clear()

        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.CPU_SAFE):
            result = get_embedder("test-model")

        assert isinstance(result, Embedder)
        assert result.model_name == "test-model"
        _embedders.clear()

    def test_gpu_detection_failure_falls_back_to_local(self):
        """If detect_gpu_mode raises, should fall back to local Embedder."""
        from goldenmatch.core.embedder import get_embedder, _embedders, Embedder

        _embedders.clear()

        with patch("goldenmatch.core.gpu.detect_gpu_mode", side_effect=RuntimeError("torch hang")):
            result = get_embedder("test-model")

        assert isinstance(result, Embedder)
        _embedders.clear()

    def test_vertex_import_failure_falls_back_to_local(self):
        """If VertexEmbedder can't be imported, should fall back to local."""
        from goldenmatch.core.embedder import get_embedder, _embedders, Embedder
        from goldenmatch.core.gpu import GPUMode

        _embedders.clear()

        with patch("goldenmatch.core.gpu.detect_gpu_mode", return_value=GPUMode.VERTEX):
            # Make the vertex_embedder import fail
            import sys
            original = sys.modules.get("goldenmatch.core.vertex_embedder")
            sys.modules["goldenmatch.core.vertex_embedder"] = None  # force ImportError
            try:
                result = get_embedder("test-model")
            finally:
                if original is not None:
                    sys.modules["goldenmatch.core.vertex_embedder"] = original
                else:
                    sys.modules.pop("goldenmatch.core.vertex_embedder", None)

        assert isinstance(result, Embedder)
        _embedders.clear()
