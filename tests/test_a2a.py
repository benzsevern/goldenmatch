"""Tests for the A2A protocol server."""

from __future__ import annotations

import json
import os

import pytest

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytestmark = pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")


# ── build_agent_card ─────────────────────────────────────────────────────────


def test_agent_card_has_required_fields():
    from goldenmatch.a2a.server import build_agent_card

    card = build_agent_card("http://localhost:8080")
    assert card["name"]
    assert card["description"]
    assert card["url"] == "http://localhost:8080"
    assert card["version"]
    assert card["provider"]["organization"] == "GoldenMatch"
    assert card["provider"]["url"] == "https://github.com/benzsevern/goldenmatch"
    assert card["capabilities"]["streaming"] is True
    assert card["capabilities"]["pushNotifications"] is False
    assert card["authentication"]["schemes"] == ["bearer"]


def test_agent_card_has_10_skills():
    from goldenmatch.a2a.server import build_agent_card

    card = build_agent_card("http://localhost:8080")
    assert len(card["skills"]) == 10


def test_agent_card_skills_have_modes():
    from goldenmatch.a2a.server import build_agent_card

    card = build_agent_card("http://localhost:8080")
    for skill in card["skills"]:
        assert "id" in skill
        assert "name" in skill
        assert "description" in skill
        assert "inputModes" in skill and len(skill["inputModes"]) > 0
        assert "outputModes" in skill and len(skill["outputModes"]) > 0


def test_agent_card_valid_json():
    from goldenmatch.a2a.server import build_agent_card

    card = build_agent_card("http://localhost:8080")
    # Round-trip through JSON to ensure serialisable
    text = json.dumps(card)
    parsed = json.loads(text)
    assert parsed["name"] == card["name"]


# ── TaskRegistry ─────────────────────────────────────────────────────────────


def test_registry_create_and_get_state():
    from goldenmatch.a2a.server import TaskRegistry

    reg = TaskRegistry()
    tid = reg.create_task("analyze_data", {"file_path": "test.csv"})
    assert reg.get_state(tid) == "submitted"


def test_registry_state_transitions():
    from goldenmatch.a2a.server import TaskRegistry

    reg = TaskRegistry()
    tid = reg.create_task("deduplicate", {})
    reg.set_state(tid, "working")
    assert reg.get_state(tid) == "working"
    reg.set_state(tid, "completed", result={"clusters": 5})
    assert reg.get_state(tid) == "completed"
    assert reg.get_result(tid) == {"clusters": 5}


def test_registry_cancel():
    from goldenmatch.a2a.server import TaskRegistry

    reg = TaskRegistry()
    tid = reg.create_task("match", {})
    reg.set_state(tid, "canceled")
    assert reg.get_state(tid) == "canceled"


def test_registry_unknown_task_raises():
    from goldenmatch.a2a.server import TaskRegistry

    reg = TaskRegistry()
    with pytest.raises(KeyError):
        reg.get_state("nonexistent-id")


def test_registry_list_tasks():
    from goldenmatch.a2a.server import TaskRegistry

    reg = TaskRegistry()
    tid1 = reg.create_task("analyze_data", {})
    tid2 = reg.create_task("deduplicate", {})
    tasks = reg.list_tasks()
    assert len(tasks) == 2
    ids = {t["id"] for t in tasks}
    assert tid1 in ids
    assert tid2 in ids


# ── dispatch_skill ───────────────────────────────────────────────────────────


def test_dispatch_analyze_data(tmp_path):
    import polars as pl
    from goldenmatch.a2a.skills import dispatch_skill

    csv_path = tmp_path / "data.csv"
    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "email": ["a@x.com", "b@x.com", "c@x.com"],
        "city": ["NYC", "LA", "NYC"],
    })
    df.write_csv(str(csv_path))

    result = dispatch_skill("analyze_data", {"file_path": str(csv_path)})
    assert "profile" in result
    assert "strategy" in result
    assert result["profile"]["row_count"] == 3


def test_dispatch_unknown_skill():
    from goldenmatch.a2a.skills import dispatch_skill

    with pytest.raises(ValueError, match="Unknown skill"):
        dispatch_skill("nonexistent_skill", {})


def test_agent_card_has_quality_and_transform_skills():
    from goldenmatch.a2a.server import build_agent_card

    card = build_agent_card("http://localhost:8080")
    skill_ids = {s["id"] for s in card["skills"]}
    assert "quality" in skill_ids
    assert "transform" in skill_ids


def test_dispatch_quality_without_goldencheck(tmp_path):
    """quality skill returns error when goldencheck is not installed."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.a2a.skills import dispatch_skill

    csv_path = tmp_path / "data.csv"
    pl.DataFrame({"name": ["Alice"]}).write_csv(str(csv_path))

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=False):
        result = dispatch_skill("quality", {"file_path": str(csv_path)})
    assert "error" in result
    assert "goldencheck" in result["error"].lower()


def test_dispatch_transform_without_goldenflow(tmp_path):
    """transform skill returns error when goldenflow is not installed."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.a2a.skills import dispatch_skill

    csv_path = tmp_path / "data.csv"
    pl.DataFrame({"name": ["Alice"]}).write_csv(str(csv_path))

    with patch("goldenmatch.core.transform._goldenflow_available", return_value=False):
        result = dispatch_skill("transform", {"file_path": str(csv_path)})
    assert "error" in result
    assert "goldenflow" in result["error"].lower()


# ── MCP agent tools: quality & transforms ───────────────────────────────────


def test_mcp_scan_quality_tool_registered():
    """scan_quality tool is in the AGENT_TOOLS list."""
    from goldenmatch.mcp.agent_tools import AGENT_TOOLS

    names = {t.name for t in AGENT_TOOLS}
    assert "scan_quality" in names
    assert "fix_quality" in names
    assert "run_transforms" in names


def test_mcp_scan_quality_without_goldencheck(tmp_path):
    """scan_quality returns error when goldencheck is not installed."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    csv_path = tmp_path / "data.csv"
    pl.DataFrame({"name": ["Alice"]}).write_csv(str(csv_path))

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=False):
        result = handle_agent_tool("scan_quality", {"file_path": str(csv_path)})

    text = result[0].text
    parsed = json.loads(text)
    assert "error" in parsed
    assert "goldencheck" in parsed["error"].lower()


def test_mcp_fix_quality_without_goldencheck(tmp_path):
    """fix_quality returns error when goldencheck is not installed."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    csv_path = tmp_path / "data.csv"
    pl.DataFrame({"name": ["Alice"]}).write_csv(str(csv_path))

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=False):
        result = handle_agent_tool("fix_quality", {"file_path": str(csv_path)})

    text = result[0].text
    parsed = json.loads(text)
    assert "error" in parsed
    assert "goldencheck" in parsed["error"].lower()


def test_mcp_run_transforms_without_goldenflow(tmp_path):
    """run_transforms returns error when goldenflow is not installed."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    csv_path = tmp_path / "data.csv"
    pl.DataFrame({"name": ["Alice"]}).write_csv(str(csv_path))

    with patch("goldenmatch.core.transform._goldenflow_available", return_value=False):
        result = handle_agent_tool("run_transforms", {"file_path": str(csv_path)})

    text = result[0].text
    parsed = json.loads(text)
    assert "error" in parsed
    assert "goldenflow" in parsed["error"].lower()


# ── File validation ────────────────────────────────────────────────────────


def test_mcp_scan_quality_file_not_found():
    """scan_quality returns actionable error for missing file."""
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True):
        result = handle_agent_tool("scan_quality", {"file_path": "/nonexistent/data.csv"})

    parsed = json.loads(result[0].text)
    assert "error" in parsed
    assert "not found" in parsed["error"].lower() or "could not read" in parsed["error"].lower()


def test_mcp_scan_quality_missing_file_path():
    """scan_quality returns error when file_path is missing."""
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True):
        result = handle_agent_tool("scan_quality", {})

    parsed = json.loads(result[0].text)
    assert "error" in parsed
    assert "file_path" in parsed["error"].lower()


def test_a2a_quality_file_not_found():
    """A2A quality skill returns error for missing file."""
    from unittest.mock import patch
    from goldenmatch.a2a.skills import dispatch_skill

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True):
        result = dispatch_skill("quality", {"file_path": "/nonexistent/data.csv"})
    assert "error" in result
    assert "not found" in result["error"].lower() or "could not read" in result["error"].lower()


def test_a2a_quality_missing_file_path():
    """A2A quality skill returns error when file_path is missing."""
    from unittest.mock import patch
    from goldenmatch.a2a.skills import dispatch_skill

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True):
        result = dispatch_skill("quality", {})
    assert "error" in result
    assert "file_path" in result["error"].lower()


# ── Happy-path tests (mocked deps) ─────────────────────────────────────────


def test_mcp_scan_quality_happy_path(tmp_path):
    """scan_quality returns correct response shape when goldencheck works."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    csv_path = tmp_path / "data.csv"
    pl.DataFrame({"name": ["Alice", "Bob"], "email": ["a@x.com", "b@x.com"]}).write_csv(str(csv_path))

    mock_issues = [
        {"rule": "ENC001", "severity": "warning", "column": "name",
         "message": "Mixed encoding", "rows_affected": 1, "confidence": 0.9},
    ]

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True), \
         patch("goldenmatch.core.quality.run_quality_check", return_value=(pl.DataFrame(), mock_issues)):
        result = handle_agent_tool("scan_quality", {"file_path": str(csv_path)})

    parsed = json.loads(result[0].text)
    assert "error" not in parsed
    assert parsed["total_records"] == 2
    assert parsed["issues_found"] == 1
    assert parsed["issues"] == mock_issues
    assert parsed["file"] == str(csv_path)


def test_mcp_fix_quality_happy_path(tmp_path):
    """fix_quality returns fixes and writes output file."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "fixed.csv"
    df = pl.DataFrame({"name": ["Alice"], "email": ["a@x.com"]})
    df.write_csv(str(csv_path))

    mock_fixes = [{"fix": "goldencheck:encoding", "column": "name",
                   "rows_affected": 1, "detail": "encoding: 1 rows"}]

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True), \
         patch("goldenmatch.core.quality.run_quality_check", return_value=(df, mock_fixes)):
        result = handle_agent_tool("fix_quality", {
            "file_path": str(csv_path), "fix_mode": "moderate",
            "output_path": str(out_path),
        })

    parsed = json.loads(result[0].text)
    assert "error" not in parsed
    assert parsed["fix_mode"] == "moderate"
    assert parsed["fixes_applied"] == 1
    assert parsed["output_path"] == str(out_path)
    assert out_path.exists()


def test_mcp_run_transforms_happy_path(tmp_path):
    """run_transforms returns transforms and writes output file."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    csv_path = tmp_path / "data.csv"
    out_path = tmp_path / "transformed.csv"
    df = pl.DataFrame({"phone": ["5551234567"]})
    df.write_csv(str(csv_path))

    mock_fixes = [{"fix": "goldenflow:phone_e164", "column": "phone",
                   "rows_affected": 1, "detail": "phone_e164: 1 rows"}]

    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True), \
         patch("goldenmatch.core.transform.run_transform", return_value=(df, mock_fixes)):
        result = handle_agent_tool("run_transforms", {
            "file_path": str(csv_path), "output_path": str(out_path),
        })

    parsed = json.loads(result[0].text)
    assert "error" not in parsed
    assert parsed["transforms_applied"] == 1
    assert parsed["output_path"] == str(out_path)
    assert out_path.exists()


def test_a2a_quality_happy_path(tmp_path):
    """A2A quality skill returns correct response with fixes."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.a2a.skills import dispatch_skill

    csv_path = tmp_path / "data.csv"
    df = pl.DataFrame({"name": ["Alice"]})
    df.write_csv(str(csv_path))

    mock_fixes = [{"fix": "goldencheck:unicode", "column": "name",
                   "rows_affected": 1, "detail": "unicode: 1 rows"}]

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True), \
         patch("goldenmatch.core.quality.run_quality_check", return_value=(df, mock_fixes)):
        result = dispatch_skill("quality", {"file_path": str(csv_path)})

    assert "error" not in result
    assert result["fixes_applied"] == 1
    assert result["total_records"] == 1


def test_a2a_transform_happy_path(tmp_path):
    """A2A transform skill returns correct response."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.a2a.skills import dispatch_skill

    csv_path = tmp_path / "data.csv"
    df = pl.DataFrame({"date": ["01/15/2024"]})
    df.write_csv(str(csv_path))

    mock_fixes = [{"fix": "goldenflow:date_iso", "column": "date",
                   "rows_affected": 1, "detail": "date_iso: 1 rows"}]

    with patch("goldenmatch.core.transform._goldenflow_available", return_value=True), \
         patch("goldenmatch.core.transform.run_transform", return_value=(df, mock_fixes)):
        result = dispatch_skill("transform", {"file_path": str(csv_path)})

    assert "error" not in result
    assert result["transforms_applied"] == 1


# ── Output write failure ───────────────────────────────────────────────────


def test_mcp_fix_quality_write_failure_preserves_results(tmp_path):
    """fix_quality preserves results when output write fails."""
    import polars as pl
    from unittest.mock import patch
    from goldenmatch.mcp.agent_tools import handle_agent_tool

    csv_path = tmp_path / "data.csv"
    df = pl.DataFrame({"name": ["Alice"]})
    df.write_csv(str(csv_path))

    mock_fixes = [{"fix": "goldencheck:encoding", "column": "name",
                   "rows_affected": 1, "detail": "test"}]

    with patch("goldenmatch.core.quality._goldencheck_available", return_value=True), \
         patch("goldenmatch.core.quality.run_quality_check", return_value=(df, mock_fixes)):
        result = handle_agent_tool("fix_quality", {
            "file_path": str(csv_path),
            "output_path": "/nonexistent/dir/out.csv",
        })

    parsed = json.loads(result[0].text)
    assert parsed["fixes_applied"] == 1
    assert "write_error" in parsed
    assert parsed["output_path"] is None
