"""Tests for parameter sensitivity analysis."""
import pytest
from unittest.mock import patch, MagicMock

from goldenmatch.core.sensitivity import (
    SweepParam, SweepPoint, SensitivityResult, run_sensitivity,
    _validate_field, _generate_values,
)
from goldenmatch.core.compare_clusters import CompareResult


def _make_clusters(cluster_map: dict[int, list[int]]) -> dict[int, dict]:
    """Helper to build cluster dicts from {cid: [members]} shorthand."""
    return {
        cid: {"members": sorted(members), "size": len(members), "pair_scores": {}, "oversized": False}
        for cid, members in cluster_map.items()
    }


def test_sweep_baseline_unchanged():
    """When sweep value equals baseline, comparison should show all unchanged."""
    baseline_clusters = _make_clusters({1: [1, 2], 2: [3, 4]})
    sweep = SweepParam(field="threshold", start=0.85, stop=0.85, step=0.05)

    mock_config = MagicMock()
    mock_config.get_matchkeys.return_value = [
        MagicMock(name="fuzzy", threshold=0.85, type="weighted"),
    ]

    with patch("goldenmatch.core.sensitivity.run_dedupe") as mock_run:
        mock_run.return_value = {"clusters": baseline_clusters}
        results = run_sensitivity(
            file_specs=[("fake.csv", "test")],
            config=mock_config,
            sweep_params=[sweep],
        )

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SensitivityResult)
    assert r.baseline_value == 0.85
    assert len(r.points) == 1
    assert r.points[0].comparison.unchanged == 2
    assert r.points[0].comparison.twi == 1.0


def test_invalid_sweep_field_raises():
    """Invalid field name should raise ValueError."""
    config = MagicMock()
    config.get_matchkeys.return_value = []
    with pytest.raises(ValueError, match="Unsupported sweep field"):
        _validate_field("nonexistent_field", config)


def test_generate_values_single_point():
    """start == stop should produce a single value."""
    param = SweepParam(field="threshold", start=0.85, stop=0.85, step=0.05)
    values = _generate_values(param)
    assert values == [0.85]


def test_generate_values_range():
    """Should generate correct range of values."""
    param = SweepParam(field="threshold", start=0.70, stop=0.90, step=0.10)
    values = _generate_values(param)
    assert values == [0.7, 0.8, 0.9]


def test_sweep_multiple_values():
    """Sweeping across 3 values should produce 3 SweepPoints."""
    clusters_1 = _make_clusters({1: [1, 2], 2: [3, 4]})
    clusters_2 = _make_clusters({1: [1, 2, 3, 4]})  # merged
    clusters_3 = _make_clusters({1: [1], 2: [2], 3: [3], 4: [4]})  # all singletons

    sweep = SweepParam(field="threshold", start=0.70, stop=0.90, step=0.10)
    call_count = [0]
    return_values = [
        {"clusters": clusters_1},  # baseline
        {"clusters": clusters_2},  # 0.70
        {"clusters": clusters_1},  # 0.80 (same as baseline)
        {"clusters": clusters_3},  # 0.90
    ]

    def mock_run(*args, **kwargs):
        result = return_values[call_count[0]]
        call_count[0] += 1
        return result

    with patch("goldenmatch.core.sensitivity.run_dedupe", side_effect=mock_run):
        results = run_sensitivity(
            file_specs=[("fake.csv", "test")],
            config=MagicMock(),
            sweep_params=[sweep],
        )

    assert len(results) == 1
    assert len(results[0].points) == 3
    assert results[0].points[1].comparison.unchanged == 2  # 0.80 = baseline


def test_stability_report():
    """stability_report() should identify best unchanged value."""
    clusters = _make_clusters({1: [1, 2], 2: [3, 4]})
    sweep = SweepParam(field="threshold", start=0.85, stop=0.85, step=0.05)

    with patch("goldenmatch.core.sensitivity.run_dedupe") as mock_run:
        mock_run.return_value = {"clusters": clusters}
        results = run_sensitivity(
            file_specs=[("fake.csv", "test")],
            config=MagicMock(),
            sweep_params=[sweep],
        )

    report = results[0].stability_report()
    assert report["best_value"] == 0.85
    assert report["best_unchanged_pct"] == 1.0
    assert len(report["points"]) == 1
