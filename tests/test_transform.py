"""Tests for GoldenFlow integration."""
from __future__ import annotations

from unittest.mock import patch
from dataclasses import dataclass

import polars as pl

from goldenmatch.core.transform import run_transform


@dataclass
class _MockTransformRecord:
    transform: str
    column: str
    affected_rows: int
    sample_before: list[str]
    sample_after: list[str]


@dataclass
class _MockManifest:
    records: list[_MockTransformRecord]


@dataclass
class _MockTransformResult:
    df: pl.DataFrame
    manifest: _MockManifest


def _sample_df():
    return pl.DataFrame({
        "name": ["John", "Jane"],
        "phone": ["(555) 123-4567", "555.987.6543"],
    })


def test_goldenflow_not_installed():
    """When goldenflow is not installed, returns df unchanged."""
    df = _sample_df()
    with patch("goldenmatch.core.transform._goldenflow_available", return_value=False):
        result_df, fixes = run_transform(df)
    assert result_df.equals(df)
    assert fixes == []


def test_disabled_via_config():
    """mode='disabled' skips transform."""
    from goldenmatch.config.schemas import TransformConfig
    df = _sample_df()
    config = TransformConfig(mode="disabled")
    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True):
        result_df, fixes = run_transform(df, config)
    assert result_df.equals(df)
    assert fixes == []


def test_enabled_false():
    """enabled=False skips transform."""
    from goldenmatch.config.schemas import TransformConfig
    df = _sample_df()
    config = TransformConfig(enabled=False)
    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True):
        result_df, fixes = run_transform(df, config)
    assert result_df.equals(df)
    assert fixes == []


def test_transform_applied_announced(capsys):
    """Announced mode applies transforms and prints summary."""
    df = _sample_df()
    transformed_df = df.with_columns(pl.lit("+15551234567").alias("phone"))

    mock_result = _MockTransformResult(
        df=transformed_df,
        manifest=_MockManifest(records=[
            _MockTransformRecord(
                transform="phone_e164",
                column="phone",
                affected_rows=2,
                sample_before=["(555) 123-4567"],
                sample_after=["+15551234567"],
            ),
        ]),
    )

    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True), \
         patch("goldenmatch.core.transform._do_transform", return_value=mock_result):
        result_df, fixes = run_transform(df)

    assert result_df.equals(transformed_df)
    assert len(fixes) == 1
    assert fixes[0]["fix"] == "goldenflow:phone_e164"
    assert fixes[0]["column"] == "phone"
    assert fixes[0]["rows_affected"] == 2
    assert "(555) 123-4567" in fixes[0]["detail"]
    assert "+15551234567" in fixes[0]["detail"]

    captured = capsys.readouterr()
    assert "GoldenFlow" in captured.out
    assert "phone_e164" in captured.out


def test_transform_applied_silent(capsys):
    """Silent mode applies transforms without printing."""
    from goldenmatch.config.schemas import TransformConfig
    df = _sample_df()
    transformed_df = df.clone()

    mock_result = _MockTransformResult(
        df=transformed_df,
        manifest=_MockManifest(records=[
            _MockTransformRecord(
                transform="whitespace_strip",
                column="name",
                affected_rows=1,
                sample_before=[" John "],
                sample_after=["John"],
            ),
        ]),
    )

    config = TransformConfig(mode="silent")
    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True), \
         patch("goldenmatch.core.transform._do_transform", return_value=mock_result):
        result_df, fixes = run_transform(df, config)

    assert len(fixes) == 1
    captured = capsys.readouterr()
    assert captured.out == ""


def test_empty_manifest():
    """No transforms needed returns df unchanged with empty fixes."""
    df = _sample_df()
    mock_result = _MockTransformResult(
        df=df,
        manifest=_MockManifest(records=[]),
    )

    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True), \
         patch("goldenmatch.core.transform._do_transform", return_value=mock_result):
        result_df, fixes = run_transform(df)

    assert result_df.equals(df)
    assert fixes == []


def test_manifest_conversion_no_samples():
    """Manifest records without samples still produce valid fix dicts."""
    df = _sample_df()
    mock_result = _MockTransformResult(
        df=df,
        manifest=_MockManifest(records=[
            _MockTransformRecord(
                transform="unicode_nfc",
                column="name",
                affected_rows=3,
                sample_before=[],
                sample_after=[],
            ),
        ]),
    )

    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True), \
         patch("goldenmatch.core.transform._do_transform", return_value=mock_result):
        result_df, fixes = run_transform(df)

    assert len(fixes) == 1
    assert fixes[0]["fix"] == "goldenflow:unicode_nfc"
    assert "->" not in fixes[0]["detail"]


def test_do_transform_exception_graceful_degradation():
    """If goldenflow crashes, returns original df with empty fixes."""
    df = _sample_df()
    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True), \
         patch("goldenmatch.core.transform._do_transform", side_effect=RuntimeError("goldenflow bug")):
        result_df, fixes = run_transform(df)
    assert result_df.equals(df)
    assert fixes == []
