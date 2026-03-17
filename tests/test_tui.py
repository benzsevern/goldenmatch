"""Tests for the GoldenMatch TUI app shell."""

from __future__ import annotations

import pytest

from textual.widgets import TabbedContent

from textual.widgets import Button

from goldenmatch.tui.app import GoldenMatchApp
from goldenmatch.tui.tabs.config_tab import ConfigTab
from goldenmatch.tui.tabs.export_tab import ExportTab
from goldenmatch.tui.tabs.golden_tab import GoldenTab
from goldenmatch.tui.tabs.matches_tab import MatchesTab


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


class TestConfigTab:
    @pytest.mark.asyncio
    async def test_config_tab_renders(self, sample_csv):
        """Config tab should render when switched to."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            config_tab = app.query_one(ConfigTab)
            assert config_tab is not None

    @pytest.mark.asyncio
    async def test_config_tab_has_add_matchkey_button(self, sample_csv):
        """Config tab should have an Add Matchkey button."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            btn = app.query_one("#add-matchkey")
            assert btn is not None

    @pytest.mark.asyncio
    async def test_config_tab_renders_without_files(self):
        """Config tab should render even without files loaded."""
        app = GoldenMatchApp(files=[])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-config"
            await pilot.pause()
            config_tab = app.query_one(ConfigTab)
            assert config_tab is not None


class TestMatchesTab:
    @pytest.mark.asyncio
    async def test_matches_tab_renders(self, sample_csv):
        """Matches tab should render when switched to."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-matches"
            await pilot.pause()
            matches_tab = app.query_one(MatchesTab)
            assert matches_tab is not None

    @pytest.mark.asyncio
    async def test_matches_tab_shows_placeholder(self, sample_csv):
        """Matches tab should show placeholder when no results."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-matches"
            await pilot.pause()
            no_msg = app.query_one("#no-results-msg")
            assert no_msg is not None
            assert no_msg.display is True


class TestGoldenTab:
    @pytest.mark.asyncio
    async def test_golden_tab_renders(self, sample_csv):
        """Golden tab should render when switched to."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-golden"
            await pilot.pause()
            golden_tab = app.query_one(GoldenTab)
            assert golden_tab is not None

    @pytest.mark.asyncio
    async def test_golden_tab_shows_placeholder(self, sample_csv):
        """Golden tab should show placeholder when no results."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-golden"
            await pilot.pause()
            placeholder = app.query_one("#golden-placeholder")
            assert placeholder is not None
            assert placeholder.display is True

    @pytest.mark.asyncio
    async def test_golden_table_hidden_initially(self, sample_csv):
        """Golden table should be hidden when no results are available."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-golden"
            await pilot.pause()
            table = app.query_one("#golden-table")
            assert table.display is False


class TestExportTab:
    @pytest.mark.asyncio
    async def test_export_tab_renders(self, sample_csv):
        """Export tab should render when switched to."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-export"
            await pilot.pause()
            export_tab = app.query_one(ExportTab)
            assert export_tab is not None

    @pytest.mark.asyncio
    async def test_export_has_save_button(self, sample_csv):
        """Export tab should have a Save Config button."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-export"
            await pilot.pause()
            btn = app.query_one("#btn-save-config", Button)
            assert btn is not None

    @pytest.mark.asyncio
    async def test_export_has_preset_button(self, sample_csv):
        """Export tab should have a Save Preset button."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-export"
            await pilot.pause()
            btn = app.query_one("#btn-save-preset", Button)
            assert btn is not None

    @pytest.mark.asyncio
    async def test_export_has_run_full_button(self, sample_csv):
        """Export tab should have a Run Full Job button."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-export"
            await pilot.pause()
            btn = app.query_one("#btn-run-full", Button)
            assert btn is not None

    @pytest.mark.asyncio
    async def test_export_has_output_format_select(self, sample_csv):
        """Export tab should have an output format selector."""
        app = GoldenMatchApp(files=[str(sample_csv)])
        async with app.run_test() as pilot:
            await pilot.pause()
            tabs = app.query_one(TabbedContent)
            tabs.active = "tab-export"
            await pilot.pause()
            from textual.widgets import Select
            sel = app.query_one("#output-format", Select)
            assert sel is not None
