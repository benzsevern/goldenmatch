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


def test_agent_card_has_8_skills():
    from goldenmatch.a2a.server import build_agent_card

    card = build_agent_card("http://localhost:8080")
    assert len(card["skills"]) == 8


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
