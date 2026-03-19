"""Tests for match explainer and HTML report."""

from __future__ import annotations

import pytest

from goldenmatch.config.schemas import MatchkeyField
from goldenmatch.core.explainer import explain_pair, format_explanation_text


class TestExplainPair:
    def test_identical_records(self):
        record = {"name": "John Smith", "email": "john@test.com", "zip": "10001"}
        fields = [
            MatchkeyField(field="name", scorer="jaro_winkler", weight=0.5, transforms=["lowercase"]),
            MatchkeyField(field="email", scorer="exact", weight=0.3, transforms=["lowercase"]),
            MatchkeyField(field="zip", scorer="exact", weight=0.2),
        ]
        exp = explain_pair(record, record, fields, threshold=0.80)

        assert exp.is_match is True
        assert exp.total_score == pytest.approx(1.0)
        assert all(f.diff_type == "identical" for f in exp.fields)

    def test_similar_names(self):
        record_a = {"name": "John Smith", "zip": "10001"}
        record_b = {"name": "Jon Smith", "zip": "10001"}
        fields = [
            MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7, transforms=["lowercase"]),
            MatchkeyField(field="zip", scorer="exact", weight=0.3),
        ]
        exp = explain_pair(record_a, record_b, fields, threshold=0.80)

        assert exp.is_match is True
        assert exp.total_score > 0.80
        # name should be similar, zip identical
        name_field = next(f for f in exp.fields if f.field_name == "name")
        zip_field = next(f for f in exp.fields if f.field_name == "zip")
        assert name_field.diff_type == "similar"
        assert zip_field.diff_type == "identical"

    def test_different_records(self):
        record_a = {"name": "John Smith", "zip": "10001"}
        record_b = {"name": "Jane Doe", "zip": "90210"}
        fields = [
            MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7),
            MatchkeyField(field="zip", scorer="exact", weight=0.3),
        ]
        exp = explain_pair(record_a, record_b, fields, threshold=0.80)

        assert exp.is_match is False
        assert exp.total_score < 0.80

    def test_missing_field(self):
        record_a = {"name": "John", "email": "john@test.com"}
        record_b = {"name": "John", "email": None}
        fields = [
            MatchkeyField(field="name", scorer="exact", weight=0.5),
            MatchkeyField(field="email", scorer="exact", weight=0.5),
        ]
        exp = explain_pair(record_a, record_b, fields, threshold=0.80)

        email_field = next(f for f in exp.fields if f.field_name == "email")
        assert email_field.diff_type == "missing"

    def test_top_contributor(self):
        record_a = {"name": "John Smith", "zip": "10001"}
        record_b = {"name": "Jon Smith", "zip": "10001"}
        fields = [
            MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7),
            MatchkeyField(field="zip", scorer="exact", weight=0.3),
        ]
        exp = explain_pair(record_a, record_b, fields)

        # name has higher weight and high score — should be top contributor
        assert exp.top_contributor == "name"

    def test_ensemble_scorer(self):
        record_a = {"name": "John Smith"}
        record_b = {"name": "Smith John"}  # reordered
        fields = [MatchkeyField(field="name", scorer="ensemble", weight=1.0)]
        exp = explain_pair(record_a, record_b, fields)

        name_field = exp.fields[0]
        assert name_field.score > 0.8  # ensemble catches reordering

    def test_transforms_applied(self):
        record_a = {"name": "  JOHN SMITH  "}
        record_b = {"name": "john smith"}
        fields = [MatchkeyField(field="name", scorer="exact", weight=1.0, transforms=["lowercase", "strip"])]
        exp = explain_pair(record_a, record_b, fields)

        name_field = exp.fields[0]
        assert name_field.transformed_a == "john smith"
        assert name_field.transformed_b == "john smith"
        assert name_field.score == 1.0


class TestFormatExplanation:
    def test_format_produces_text(self):
        record_a = {"name": "John", "zip": "10001"}
        record_b = {"name": "Jon", "zip": "10001"}
        fields = [
            MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7),
            MatchkeyField(field="zip", scorer="exact", weight=0.3),
        ]
        exp = explain_pair(record_a, record_b, fields)
        text = format_explanation_text(exp)

        assert "name" in text
        assert "zip" in text
        assert "Score:" in text


class TestHTMLReport:
    def test_generate_report(self, tmp_path):
        import polars as pl
        from goldenmatch.core.report import generate_report

        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "name": ["John Smith", "Jon Smith", "Jane Doe", "Bob Johnson"],
            "zip": ["10001", "10001", "90210", "30301"],
        })

        clusters = {
            1: {"size": 2, "members": [0, 1], "oversized": False},
            2: {"size": 1, "members": [2], "oversized": False},
            3: {"size": 1, "members": [3], "oversized": False},
        }

        pairs = [(0, 1, 0.95)]

        output = tmp_path / "report.html"
        result = generate_report(
            df, clusters, pairs,
            output_path=output,
            title="Test Report",
        )

        assert result.exists()
        content = result.read_text()
        assert "Test Report" in content
        assert "4" in content  # total records
        assert "John Smith" in content
        assert "score" in content.lower()

    def test_report_with_explanations(self, tmp_path):
        import polars as pl
        from goldenmatch.core.report import generate_report

        df = pl.DataFrame({
            "__row_id__": [0, 1],
            "name": ["John Smith", "Jon Smith"],
            "zip": ["10001", "10001"],
        })

        clusters = {1: {"size": 2, "members": [0, 1], "oversized": False}}
        pairs = [(0, 1, 0.95)]
        fields = [
            MatchkeyField(field="name", scorer="jaro_winkler", weight=0.7),
            MatchkeyField(field="zip", scorer="exact", weight=0.3),
        ]

        output = tmp_path / "report_exp.html"
        result = generate_report(
            df, clusters, pairs,
            matchkey_fields=fields,
            output_path=output,
        )

        content = result.read_text()
        assert "Match Explanations" in content
        assert "jaro_winkler" in content
        assert "contribution" in content.lower()

    def test_empty_report(self, tmp_path):
        import polars as pl
        from goldenmatch.core.report import generate_report

        df = pl.DataFrame({"__row_id__": [0], "name": ["John"]})
        output = tmp_path / "empty.html"
        result = generate_report(df, {}, [], output_path=output)

        assert result.exists()
        content = result.read_text()
        assert "0" in content  # 0 clusters
