"""Tests for schema-free matching and before/after dashboard."""

from __future__ import annotations

import polars as pl
import pytest


class TestSchemaMatch:
    def test_exact_name_match(self):
        from goldenmatch.core.schema_match import auto_map_columns

        df_a = pl.DataFrame({"name": ["John"], "email": ["j@t.com"]})
        df_b = pl.DataFrame({"name": ["Jane"], "email": ["jane@t.com"]})

        mappings = auto_map_columns(df_a, df_b)
        mapped_cols = {m["col_a"] for m in mappings}
        assert "name" in mapped_cols
        assert "email" in mapped_cols

    def test_synonym_match(self):
        from goldenmatch.core.schema_match import auto_map_columns

        df_a = pl.DataFrame({"email": ["j@t.com"], "phone": ["555"]})
        df_b = pl.DataFrame({"contact_email": ["jane@t.com"], "telephone": ["555"]})

        mappings = auto_map_columns(df_a, df_b)
        col_pairs = {(m["col_a"], m["col_b"]) for m in mappings}
        assert ("email", "contact_email") in col_pairs
        assert ("phone", "telephone") in col_pairs

    def test_composite_name(self):
        from goldenmatch.core.schema_match import auto_map_columns

        df_a = pl.DataFrame({"full_name": ["John Smith"], "zip": ["10001"]})
        df_b = pl.DataFrame({"first_name": ["John"], "last_name": ["Smith"], "zip": ["10001"]})

        mappings = auto_map_columns(df_a, df_b)
        methods = {m["method"] for m in mappings}
        assert "composite" in methods or "exact_name" in methods

    def test_no_match_different_schemas(self):
        from goldenmatch.core.schema_match import auto_map_columns

        df_a = pl.DataFrame({"product_code": ["ABC"], "warehouse": ["W1"]})
        df_b = pl.DataFrame({"color": ["red"], "size": ["large"]})

        mappings = auto_map_columns(df_a, df_b)
        # Should find few or no mappings for completely unrelated schemas
        assert len(mappings) <= 1

    def test_apply_column_mapping(self):
        from goldenmatch.core.schema_match import auto_map_columns, apply_column_mapping

        df_a = pl.DataFrame({"email": ["j@t.com"]})
        df_b = pl.DataFrame({"contact_email": ["jane@t.com"]})

        mappings = auto_map_columns(df_a, df_b)
        df_b_renamed = apply_column_mapping(df_b, mappings, df_a.columns, side="b")

        assert "email" in df_b_renamed.columns

    def test_composite_apply(self):
        from goldenmatch.core.schema_match import auto_map_columns, apply_column_mapping

        df_a = pl.DataFrame({"name": ["John Smith"]})
        df_b = pl.DataFrame({"first_name": ["Jane"], "last_name": ["Doe"]})

        mappings = auto_map_columns(df_a, df_b)
        composite_maps = [m for m in mappings if m.get("method") == "composite"]

        if composite_maps:
            df_b_merged = apply_column_mapping(df_b, mappings, df_a.columns, side="b")
            assert "name" in df_b_merged.columns
            assert df_b_merged["name"][0] == "Jane Doe"


class TestDashboard:
    def test_generate_dashboard(self, tmp_path):
        from goldenmatch.core.dashboard import generate_dashboard

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "name": ["John Smith", "Jon Smith", "Jane Doe", "Bob"],
            "email": ["john@t.com", "jon@t.com", "jane@t.com", "bob@t.com"],
            "zip": ["10001", "10001", "90210", "30301"],
        })

        clusters = {
            1: {"size": 2, "members": [0, 1], "oversized": False},
            2: {"size": 1, "members": [2], "oversized": False},
            3: {"size": 1, "members": [3], "oversized": False},
        }

        pairs = [(0, 1, 0.95)]

        golden = pl.DataFrame({
            "__cluster_id__": [1],
            "name": ["John Smith"],
            "email": ["john@t.com"],
            "zip": ["10001"],
        })

        output = tmp_path / "dashboard.html"
        result = generate_dashboard(
            df, clusters, pairs, golden_df=golden,
            output_path=output, title="Test Dashboard",
        )

        assert result.exists()
        content = result.read_text()
        assert "BEFORE" in content
        assert "AFTER" in content
        assert "4" in content  # total records
        assert "John Smith" in content
        assert "duplicate" in content.lower()

    def test_dashboard_with_quality_issues(self, tmp_path):
        from goldenmatch.core.dashboard import generate_dashboard

        df = pl.DataFrame({
            "__row_id__": list(range(10)),
            "name": ["John", None, "Jane", None, None, "Bob", "Alice", None, "Eve", "Tom"],
            "email": [f"u{i}@t.com" for i in range(10)],
        })

        output = tmp_path / "dash2.html"
        result = generate_dashboard(df, {}, [], output_path=output)

        content = result.read_text()
        assert "missing" in content.lower() or "null" in content.lower()

    def test_empty_dashboard(self, tmp_path):
        from goldenmatch.core.dashboard import generate_dashboard

        df = pl.DataFrame({"__row_id__": [0], "name": ["John"]})
        output = tmp_path / "empty.html"
        result = generate_dashboard(df, {}, [], output_path=output)
        assert result.exists()
