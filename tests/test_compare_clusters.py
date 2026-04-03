"""Tests for CCMS cluster comparison."""
import pytest

from goldenmatch.core.compare_clusters import compare_clusters, CompareResult


def test_identical_clusters_all_unchanged():
    """Two identical cluster dicts should produce all unchanged, TWI=1.0."""
    clusters_a = {
        1: {"members": [1, 2, 3], "size": 3, "pair_scores": {}, "oversized": False},
        2: {"members": [4, 5], "size": 2, "pair_scores": {}, "oversized": False},
        3: {"members": [6], "size": 1, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        10: {"members": [1, 2, 3], "size": 3, "pair_scores": {}, "oversized": False},
        20: {"members": [4, 5], "size": 2, "pair_scores": {}, "oversized": False},
        30: {"members": [6], "size": 1, "pair_scores": {}, "oversized": False},
    }
    result = compare_clusters(clusters_a, clusters_b)
    assert isinstance(result, CompareResult)
    assert result.unchanged == 3
    assert result.merged == 0
    assert result.partitioned == 0
    assert result.overlapping == 0
    assert result.cc1 == 3
    assert result.cc2 == 3
    assert result.sc1 == 1
    assert result.sc2 == 1
    assert result.twi == 1.0
    assert result.rc == 6


def test_merged_case():
    """ER1 cluster is proper subset of an ER2 cluster."""
    clusters_a = {
        1: {"members": [1, 2], "size": 2, "pair_scores": {}, "oversized": False},
        2: {"members": [3], "size": 1, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        10: {"members": [1, 2, 3], "size": 3, "pair_scores": {}, "oversized": False},
    }
    result = compare_clusters(clusters_a, clusters_b)
    assert result.merged == 2
    assert result.unchanged == 0


def test_partitioned_case():
    """ER1 cluster split into multiple ER2 clusters, no external members added."""
    clusters_a = {
        1: {"members": [1, 2, 3, 4], "size": 4, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        10: {"members": [1, 2], "size": 2, "pair_scores": {}, "oversized": False},
        20: {"members": [3, 4], "size": 2, "pair_scores": {}, "oversized": False},
    }
    result = compare_clusters(clusters_a, clusters_b)
    assert result.partitioned == 1
    assert result.unchanged == 0
    assert result.cc1 == 1
    assert result.cc2 == 2


def test_overlapping_case():
    """ER1 cluster members redistributed with members from other ER1 clusters."""
    clusters_a = {
        1: {"members": [1, 2, 3], "size": 3, "pair_scores": {}, "oversized": False},
        2: {"members": [4, 5, 6], "size": 3, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        10: {"members": [1, 2, 4], "size": 3, "pair_scores": {}, "oversized": False},
        20: {"members": [3, 5, 6], "size": 3, "pair_scores": {}, "oversized": False},
    }
    result = compare_clusters(clusters_a, clusters_b)
    assert result.overlapping == 2
    assert result.unchanged == 0


def test_mixed_cases_paper_example():
    """Table 1 from the CCMS paper: 16 references, 7 ER1 clusters, 8 ER2 clusters."""
    clusters_a = {
        1: {"members": [1], "size": 1, "pair_scores": {}, "oversized": False},
        2: {"members": [2, 3], "size": 2, "pair_scores": {}, "oversized": False},
        3: {"members": [4, 5, 6], "size": 3, "pair_scores": {}, "oversized": False},
        4: {"members": [7], "size": 1, "pair_scores": {}, "oversized": False},
        5: {"members": [8, 9, 10], "size": 3, "pair_scores": {}, "oversized": False},
        6: {"members": [11, 12, 13], "size": 3, "pair_scores": {}, "oversized": False},
        7: {"members": [14, 15, 16], "size": 3, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        10: {"members": [1], "size": 1, "pair_scores": {}, "oversized": False},
        20: {"members": [2, 3], "size": 2, "pair_scores": {}, "oversized": False},
        30: {"members": [4, 5, 6, 7], "size": 4, "pair_scores": {}, "oversized": False},
        40: {"members": [8, 9], "size": 2, "pair_scores": {}, "oversized": False},
        50: {"members": [10], "size": 1, "pair_scores": {}, "oversized": False},
        60: {"members": [11, 12, 14, 15], "size": 4, "pair_scores": {}, "oversized": False},
        70: {"members": [13], "size": 1, "pair_scores": {}, "oversized": False},
        80: {"members": [16], "size": 1, "pair_scores": {}, "oversized": False},
    }
    result = compare_clusters(clusters_a, clusters_b)
    assert result.unchanged == 2
    assert result.merged == 2
    assert result.partitioned == 1
    assert result.overlapping == 2
    assert result.cc1 == 7
    assert result.cc2 == 8
    assert result.rc == 16


def test_mismatched_row_ids_raises():
    """Different row ID sets should raise ValueError."""
    clusters_a = {
        1: {"members": [1, 2], "size": 2, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        1: {"members": [1, 3], "size": 2, "pair_scores": {}, "oversized": False},
    }
    with pytest.raises(ValueError, match="different row IDs"):
        compare_clusters(clusters_a, clusters_b)


def test_singleton_counts():
    """Singleton counts should be computed correctly."""
    clusters_a = {
        1: {"members": [1], "size": 1, "pair_scores": {}, "oversized": False},
        2: {"members": [2], "size": 1, "pair_scores": {}, "oversized": False},
        3: {"members": [3, 4], "size": 2, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        10: {"members": [1], "size": 1, "pair_scores": {}, "oversized": False},
        20: {"members": [2, 3, 4], "size": 3, "pair_scores": {}, "oversized": False},
    }
    result = compare_clusters(clusters_a, clusters_b)
    assert result.sc1 == 2
    assert result.sc2 == 1


def test_asymmetry_uc_same_mc_pc_swap():
    """compare(A,B) vs compare(B,A): UC same, MC/PC may swap, TWI symmetric."""
    clusters_a = {
        1: {"members": [1, 2, 3, 4], "size": 4, "pair_scores": {}, "oversized": False},
    }
    clusters_b = {
        10: {"members": [1, 2], "size": 2, "pair_scores": {}, "oversized": False},
        20: {"members": [3, 4], "size": 2, "pair_scores": {}, "oversized": False},
    }
    ab = compare_clusters(clusters_a, clusters_b)
    ba = compare_clusters(clusters_b, clusters_a)
    assert ab.unchanged == ba.unchanged
    assert ab.partitioned == 1
    assert ba.merged == 2
    assert abs(ab.twi - ba.twi) < 1e-9


def test_empty_clusters():
    """Both empty cluster dicts should produce zero counts, TWI=1.0."""
    result = compare_clusters({}, {})
    assert result.unchanged == 0
    assert result.rc == 0
    assert result.twi == 1.0


def test_summary_dict():
    """summary() should return all expected keys with correct types."""
    clusters = {
        1: {"members": [1, 2], "size": 2, "pair_scores": {}, "oversized": False},
    }
    result = compare_clusters(clusters, clusters)
    s = result.summary()
    expected_keys = {
        "unchanged", "merged", "partitioned", "overlapping",
        "rc", "cc1", "cc2", "sc1", "sc2", "twi",
        "unchanged_pct", "merged_pct", "partitioned_pct", "overlapping_pct",
    }
    assert set(s.keys()) == expected_keys
    assert s["unchanged_pct"] == 1.0
