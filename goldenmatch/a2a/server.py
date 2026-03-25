"""A2A (Agent-to-Agent) protocol server for GoldenMatch.

Exposes the ER agent as an A2A-compliant service with an agent card,
task lifecycle, and skill dispatch via aiohttp.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

from aiohttp import web

from goldenmatch import __version__

# ── Agent card ───────────────────────────────────────────────────────────────

_SKILLS = [
    {
        "id": "analyze_data",
        "name": "Analyze Data",
        "description": "Profile a CSV file and recommend a matching strategy.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
    {
        "id": "configure",
        "name": "Configure",
        "description": "Generate a YAML config from data profiling.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
    {
        "id": "deduplicate",
        "name": "Deduplicate",
        "description": "Run deduplication pipeline on a single file.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
    {
        "id": "match",
        "name": "Match",
        "description": "Match records across two source files.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
    {
        "id": "explain",
        "name": "Explain",
        "description": "Explain why two records matched or did not match.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
    {
        "id": "review",
        "name": "Review",
        "description": "List borderline pairs pending human review.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
    {
        "id": "compare_strategies",
        "name": "Compare Strategies",
        "description": "Run multiple matching strategies and compare results.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
    {
        "id": "pprl",
        "name": "PPRL",
        "description": "Privacy-preserving record linkage using bloom filters.",
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    },
]


def build_agent_card(base_url: str) -> dict:
    """Return the A2A agent card JSON."""
    return {
        "name": "GoldenMatch ER Agent",
        "description": "Autonomous entity resolution agent for deduplication, matching, and data quality.",
        "url": base_url,
        "version": __version__,
        "provider": {
            "organization": "GoldenMatch",
            "url": "https://github.com/benzsevern/goldenmatch",
        },
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
        },
        "skills": _SKILLS,
        "authentication": {
            "schemes": ["bearer"],
        },
    }


# ── Task registry ───────────────────────────────────────────────────────────


class TaskRegistry:
    """In-memory task store for A2A task lifecycle."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}

    def create_task(self, skill: str, params: dict) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "skill": skill,
            "params": params,
            "state": "submitted",
            "result": None,
            "progress": None,
        }
        return task_id

    def get_state(self, task_id: str) -> str:
        """Return the current state of a task. Raises KeyError if unknown."""
        return self._tasks[task_id]["state"]

    def set_state(
        self,
        task_id: str,
        state: str,
        result: Optional[dict] = None,
        progress: Optional[float] = None,
    ) -> None:
        """Update task state, optionally setting result and progress."""
        task = self._tasks[task_id]
        task["state"] = state
        if result is not None:
            task["result"] = result
        if progress is not None:
            task["progress"] = progress

    def get_result(self, task_id: str) -> dict | None:
        """Return the task result, or None if not yet available."""
        return self._tasks[task_id]["result"]

    def list_tasks(self) -> list[dict]:
        """Return a list of all tasks with their metadata."""
        items = []
        for task_id, task in self._tasks.items():
            items.append({
                "id": task_id,
                "skill": task["skill"],
                "state": task["state"],
                "progress": task["progress"],
            })
        return items


# ── aiohttp application ─────────────────────────────────────────────────────

_registry = TaskRegistry()


async def _handle_agent_card(request: web.Request) -> web.Response:
    base_url = str(request.url.origin())
    card = build_agent_card(base_url)
    return web.json_response(card)


async def _handle_send_task(request: web.Request) -> web.Response:
    from goldenmatch.a2a.skills import dispatch_skill

    body = await request.json()
    skill_id = body.get("skill")
    params = body.get("params", {})

    if not skill_id:
        return web.json_response({"error": "Missing 'skill' field"}, status=400)

    registry: TaskRegistry = request.app["registry"]
    task_id = registry.create_task(skill_id, params)
    registry.set_state(task_id, "working")

    try:
        result = dispatch_skill(skill_id, params)
        registry.set_state(task_id, "completed", result=result)
        return web.json_response({
            "id": task_id,
            "state": "completed",
            "result": result,
        })
    except ValueError as exc:
        registry.set_state(task_id, "failed", result={"error": str(exc)})
        return web.json_response({
            "id": task_id,
            "state": "failed",
            "error": str(exc),
        }, status=400)
    except Exception as exc:
        registry.set_state(task_id, "failed", result={"error": str(exc)})
        return web.json_response({
            "id": task_id,
            "state": "failed",
            "error": str(exc),
        }, status=500)


async def _handle_get_task(request: web.Request) -> web.Response:
    task_id = request.match_info["task_id"]
    registry: TaskRegistry = request.app["registry"]
    try:
        state = registry.get_state(task_id)
        result = registry.get_result(task_id)
        return web.json_response({
            "id": task_id,
            "state": state,
            "result": result,
        })
    except KeyError:
        return web.json_response({"error": "Task not found"}, status=404)


async def _handle_cancel_task(request: web.Request) -> web.Response:
    task_id = request.match_info["task_id"]
    registry: TaskRegistry = request.app["registry"]
    try:
        registry.set_state(task_id, "canceled")
        return web.json_response({
            "id": task_id,
            "state": "canceled",
        })
    except KeyError:
        return web.json_response({"error": "Task not found"}, status=404)


@web.middleware
async def _auth_middleware(request: web.Request, handler):
    token = os.environ.get("GOLDENMATCH_AGENT_TOKEN")
    if token:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != token:
            return web.json_response({"error": "Unauthorized"}, status=401)
    return await handler(request)


def create_app(host: str = "0.0.0.0", port: int = 8080) -> web.Application:
    """Build and return the A2A aiohttp application."""
    app = web.Application(middlewares=[_auth_middleware])
    app["registry"] = TaskRegistry()
    app["host"] = host
    app["port"] = port

    app.router.add_get("/.well-known/agent.json", _handle_agent_card)
    app.router.add_post("/tasks/send", _handle_send_task)
    app.router.add_get("/tasks/{task_id}", _handle_get_task)
    app.router.add_post("/tasks/{task_id}/cancel", _handle_cancel_task)

    return app
