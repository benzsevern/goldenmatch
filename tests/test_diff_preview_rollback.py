"""Tests for CSV diff, merge preview, and undo/rollback."""

from __future__ import annotations

import json

import polars as pl
import pytest


class TestCSVDiff:
    def test_generate_diff_csv(self, tmp_path):
        from goldenmatch.core.diff import generate_diff

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["John Smith", "Jon Smith", "Jane Doe"],
            "email": ["john@t.com", "jon@t.com", "jane@t.com"],
        })
        clusters = {1: {"size": 2, "members": [0, 1]}, 2: {"size": 1, "members": [2]}}
        golden = pl.DataFrame({
            "__cluster_id__": [1],
            "name": ["John Smith"],
            "email": ["john@t.com"],
        })

        output = tmp_path / "diff.csv"
        result = generate_diff(df, clusters, golden, output)

        assert result.exists()
        diff_df = pl.read_csv(str(result))
        assert diff_df.height == 3
        assert "__is_duplicate__" in diff_df.columns
        assert "__golden_name__" in diff_df.columns
        assert "__changed_name__" in diff_df.columns

    def test_diff_html(self, tmp_path):
        from goldenmatch.core.diff import generate_diff

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John", "Jon"],
        })
        clusters = {1: {"size": 2, "members": [0, 1]}}
        golden = pl.DataFrame({"__cluster_id__": [1], "name": ["John"]})

        output = tmp_path / "diff.html"
        result = generate_diff(df, clusters, golden, output, format="html")

        content = result.read_text()
        assert "GoldenMatch Diff" in content
        assert "John" in content

    def test_diff_no_duplicates(self, tmp_path):
        from goldenmatch.core.diff import generate_diff

        df = pl.DataFrame({"__row_id__": [0], "name": ["John"]})
        output = tmp_path / "diff.csv"
        result = generate_diff(df, {}, None, output)
        assert result.exists()


class TestMergePreview:
    def test_generate_preview(self):
        from goldenmatch.core.merge_preview import generate_merge_preview

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "name": ["John Smith", "Jon Smith", "Jane Doe"],
            "email": ["john@t.com", "jon@t.com", "jane@t.com"],
        })
        clusters = {1: {"size": 2, "members": [0, 1]}, 2: {"size": 1, "members": [2]}}
        golden = pl.DataFrame({
            "__cluster_id__": [1],
            "name": ["John Smith"],
            "email": ["john@t.com"],
        })

        preview = generate_merge_preview(df, clusters, golden)

        assert preview["total_records"] == 3
        assert preview["records_affected"] == 2
        assert preview["clusters_to_merge"] == 1
        assert preview["risk_level"] in ("low", "medium", "high")
        assert "name" in preview["field_changes"]

    def test_preview_no_changes(self):
        from goldenmatch.core.merge_preview import generate_merge_preview

        df = pl.DataFrame({"__row_id__": [0], "name": ["John"]})
        preview = generate_merge_preview(df, {}, None)
        assert preview["records_affected"] == 0
        assert preview["total_field_changes"] == 0

    def test_format_preview(self):
        from goldenmatch.core.merge_preview import generate_merge_preview, format_preview_text

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John", "Jon"],
        })
        clusters = {1: {"size": 2, "members": [0, 1]}}
        golden = pl.DataFrame({"__cluster_id__": [1], "name": ["John"]})

        preview = generate_merge_preview(df, clusters, golden)
        text = format_preview_text(preview)
        assert "Merge Preview" in text
        assert "name" in text


class TestRollback:
    def test_save_and_list_runs(self, tmp_path):
        from goldenmatch.core.rollback import save_run_snapshot, list_runs

        save_run_snapshot(
            run_id="test-run-1",
            output_dir=tmp_path,
            config_dict={"matchkeys": []},
            stats={"total": 100},
            output_files=["golden.csv", "clusters.csv"],
        )

        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test-run-1"

    def test_rollback_deletes_files(self, tmp_path):
        from goldenmatch.core.rollback import save_run_snapshot, rollback_run

        # Create fake output files
        (tmp_path / "golden.csv").write_text("data")
        (tmp_path / "clusters.csv").write_text("data")

        save_run_snapshot(
            run_id="run-to-undo",
            output_dir=tmp_path,
            config_dict={},
            stats={},
            output_files=["golden.csv", "clusters.csv"],
        )

        result = rollback_run("run-to-undo", tmp_path)
        assert result["status"] == "rolled_back"
        assert len(result["deleted"]) == 2
        assert not (tmp_path / "golden.csv").exists()
        assert not (tmp_path / "clusters.csv").exists()

    def test_rollback_nonexistent(self, tmp_path):
        from goldenmatch.core.rollback import rollback_run

        result = rollback_run("fake-id", tmp_path)
        assert "error" in result

    def test_double_rollback(self, tmp_path):
        from goldenmatch.core.rollback import save_run_snapshot, rollback_run

        (tmp_path / "out.csv").write_text("data")
        save_run_snapshot("r1", tmp_path, {}, {}, ["out.csv"])
        rollback_run("r1", tmp_path)

        result = rollback_run("r1", tmp_path)
        assert "error" in result
        assert "already" in result["error"]
