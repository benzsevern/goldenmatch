"""Tests for evaluation engine."""
from __future__ import annotations

import textwrap
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from goldenmatch.core.evaluate import evaluate_clusters, evaluate_pairs, EvalResult


class TestEvaluatePairs:
    def test_perfect_pairs(self):
        """All predicted pairs are in ground truth."""
        predicted = [(1, 2, 0.9), (3, 4, 0.85)]
        ground_truth = {(1, 2), (3, 4)}
        result = evaluate_pairs(predicted, ground_truth)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_partial_match(self):
        predicted = [(1, 2, 0.9), (5, 6, 0.8)]  # (5,6) is FP
        ground_truth = {(1, 2), (3, 4)}  # (3,4) is FN
        result = evaluate_pairs(predicted, ground_truth)
        assert result.tp == 1
        assert result.fp == 1
        assert result.fn == 1
        assert result.precision == 0.5
        assert result.recall == 0.5

    def test_empty_predicted(self):
        result = evaluate_pairs([], {(1, 2)})
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.fn == 1

    def test_symmetric_pairs(self):
        """(1,2) should match (2,1) in ground truth."""
        predicted = [(2, 1, 0.9)]
        ground_truth = {(1, 2)}
        result = evaluate_pairs(predicted, ground_truth)
        assert result.tp == 1

    def test_empty_ground_truth(self):
        result = evaluate_pairs([(1, 2, 0.9)], set())
        assert result.precision == 0.0
        assert result.recall == 0.0


class TestEvaluateClusters:
    def test_cluster_to_pairs(self):
        """Clusters with >1 member generate pairs for evaluation."""
        clusters = {
            1: {"members": [1, 2, 3], "size": 3},
            2: {"members": [4], "size": 1},
        }
        ground_truth = {(1, 2), (1, 3), (2, 3)}
        result = evaluate_clusters(clusters, ground_truth)
        assert result.tp == 3
        assert result.precision == 1.0
        assert result.recall == 1.0


class TestEvalResult:
    def test_summary_dict(self):
        result = EvalResult(tp=8, fp=2, fn=1)
        d = result.summary()
        assert d["precision"] == pytest.approx(0.8, abs=1e-3)
        assert d["recall"] == pytest.approx(8 / 9, abs=1e-3)
        assert "f1" in d


# ── CLI tests ────────────────────────────────────────────────────────────

from goldenmatch.cli.main import app

runner = CliRunner()


class TestEvaluateCLI:
    @pytest.fixture
    def sample_data(self, tmp_path):
        # Create input CSV
        data_path = tmp_path / "data.csv"
        pl.DataFrame({
            "first_name": ["John", "john", "Jane", "Bob"],
            "last_name": ["Smith", "Smith", "Doe", "Jones"],
            "email": ["j@x.com", "j@x.com", "jane@t.com", "bob@t.com"],
        }).write_csv(data_path)

        # Create ground truth CSV (row 0 and 1 are dupes)
        gt_path = tmp_path / "ground_truth.csv"
        pl.DataFrame({"id_a": [0], "id_b": [1]}).write_csv(gt_path)

        # Create config
        config_path = tmp_path / "config.yaml"
        config_path.write_text(textwrap.dedent("""\
            matchkeys:
              - name: exact_email
                type: exact
                fields:
                  - field: email
                    transforms: [lowercase, strip]
        """))
        return data_path, gt_path, config_path

    def test_evaluate_basic(self, sample_data):
        data_path, gt_path, config_path = sample_data
        result = runner.invoke(app, [
            "evaluate",
            str(data_path),
            "--config", str(config_path),
            "--ground-truth", str(gt_path),
        ])
        assert result.exit_code == 0
        assert "Precision" in result.stdout or "precision" in result.stdout.lower()
        assert "Recall" in result.stdout or "recall" in result.stdout.lower()

    def test_evaluate_missing_gt(self, sample_data, tmp_path):
        data_path, _, config_path = sample_data
        result = runner.invoke(app, [
            "evaluate",
            str(data_path),
            "--config", str(config_path),
            "--ground-truth", str(tmp_path / "does_not_exist.csv"),
        ])
        assert result.exit_code != 0
