"""Tests for the GoldenMatch TUI app shell."""

from __future__ import annotations

import pytest

from goldenmatch.tui.app import GoldenMatchApp


class TestTUIApp:
    @pytest.mark.asyncio
    async def test_app_launches(self, sample_csv):
        """App should render without crashing when given a valid file."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            assert app.is_running

    @pytest.mark.asyncio
    async def test_sidebar_shows_file_info(self, sample_csv):
        """Sidebar should be present and accessible after launch."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            sidebar = app.query_one("#sidebar")
            assert sidebar is not None

    @pytest.mark.asyncio
    async def test_tabs_exist(self, sample_csv):
        """All five tab panes should be present."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            panes = app.query("TabPane")
            assert len(panes) == 5

    @pytest.mark.asyncio
    async def test_app_launches_without_files(self):
        """App should launch even with no files provided."""
        app = GoldenMatchApp(files=[])
        async with app.run_test() as pilot:
            assert app.is_running

    @pytest.mark.asyncio
    async def test_sidebar_no_files_shows_placeholder(self):
        """Sidebar should show placeholder text when no files loaded."""
        app = GoldenMatchApp(files=[])
        async with app.run_test() as pilot:
            await pilot.pause()
            sidebar = app.query_one("#sidebar")
            rendered = sidebar.render()
            assert "No files loaded" in rendered
