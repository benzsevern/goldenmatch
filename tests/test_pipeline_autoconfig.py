"""Tests for auto-config running after GoldenCheck/GoldenFlow in pipeline."""
from __future__ import annotations

from unittest.mock import patch

import polars as pl

from goldenmatch.config.schemas import GoldenMatchConfig


def test_zero_config_completes_with_auto_config():
    """Zero-config dedupe_df should complete successfully with auto_config=True path."""
    from goldenmatch._api import dedupe_df

    df = pl.DataFrame({
        "name": ["John Smith", "Jon Smith", "Jane Doe"],
        "email": ["john@test.com", "john@test.com", "jane@test.com"],
    })

    result = dedupe_df(df)

    assert result.total_records >= 3
    assert result.clusters is not None


def test_explicit_config_does_not_call_autoconfig():
    """When explicit config is provided, auto_configure_df is never called."""
    from goldenmatch._api import dedupe_df
    from goldenmatch.config.schemas import (
        MatchkeyConfig, MatchkeyField, BlockingConfig, BlockingKeyConfig,
    )

    df = pl.DataFrame({
        "name": ["John Smith", "Jon Smith", "Jane Doe"],
        "email": ["john@test.com", "john@test.com", "jane@test.com"],
    })

    config = GoldenMatchConfig(
        matchkeys=[
            MatchkeyConfig(
                name="exact_email",
                type="exact",
                fields=[MatchkeyField(field="email")],
            ),
        ],
        blocking=BlockingConfig(
            keys=[BlockingKeyConfig(fields=["email"])],
        ),
    )

    with patch("goldenmatch.core.autoconfig.auto_configure_df") as mock_ac:
        result = dedupe_df(df, config=config)

    mock_ac.assert_not_called()
    assert result.total_records >= 3


def test_auto_config_profiles_cleaned_data():
    """Auto-config should receive data that has been preprocessed."""
    from goldenmatch.core.autoconfig import auto_configure_df as real_ac
    from goldenmatch.core.pipeline import run_dedupe_df

    df = pl.DataFrame({
        "name": ["John Smith", "Jon Smith", "Jane Doe"],
        "phone": ["(555) 123-4567", "(555) 123-4567", "(555) 987-6543"],
    })

    config = GoldenMatchConfig()

    autoconfig_input_heights = []

    def capture_autoconfig(df_in, **kwargs):
        autoconfig_input_heights.append(df_in.height)
        return real_ac(df_in, **kwargs)

    with patch("goldenmatch.core.autoconfig.auto_configure_df", side_effect=capture_autoconfig):
        result = run_dedupe_df(df, config, auto_config=True)

    assert len(autoconfig_input_heights) == 1
    assert autoconfig_input_heights[0] == 3
