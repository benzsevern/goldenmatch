# GoldenMatch ER Agent -- Design Spec

## Overview

An autonomous entity resolution agent that other AI systems can discover and invoke via A2A and MCP protocols. It analyzes data, picks the right ER strategy, runs the pipeline, explains every decision, and manages a confidence-gated review queue -- all without human configuration.

**Goal:** Make GoldenMatch the go-to ER tool for AI agents.

**Protocols:**
- **MCP** -- Tool-level integration for Claude/Cursor/coding agents (expand existing server)
- **A2A** -- Agent-level discovery for any A2A-compatible framework (new)

**Trust model:** Confidence-gated
- Auto-merge: >0.95 confidence
- Review queue: 0.75-0.95 (held for human/agent approval)
- Auto-reject: <0.75

---

## A2A Agent Card

Served at `/.well-known/agent.json` by the A2A server (`goldenmatch agent-serve`):

```json
{
  "name": "goldenmatch-agent",
  "description": "Autonomous entity resolution agent. Analyzes data, selects optimal matching strategy, deduplicates records, explains decisions, manages confidence-gated review queue.",
  "url": "http://localhost:8200",
  "version": "1.0.0",
  "provider": {
    "organization": "GoldenMatch",
    "url": "https://github.com/benzsevern/goldenmatch"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "analyze_data",
      "name": "Analyze Data",
      "description": "Profile columns, detect domain, recommend matching strategy",
      "inputModes": ["application/json"],
      "outputModes": ["application/json"]
    },
    {
      "id": "configure",
      "name": "Auto-Configure",
      "description": "Generate optimal matching config from data analysis",
      "inputModes": ["application/json"],
      "outputModes": ["application/json", "text/yaml"]
    },
    {
      "id": "deduplicate",
      "name": "Deduplicate",
      "description": "Run full ER pipeline with confidence-gated output",
      "inputModes": ["application/json"],
      "outputModes": ["application/json"]
    },
    {
      "id": "match",
      "name": "Cross-Source Match",
      "description": "Match records across two datasets with intelligent strategy selection",
      "inputModes": ["application/json"],
      "outputModes": ["application/json"]
    },
    {
      "id": "explain",
      "name": "Explain Decision",
      "description": "Natural language explanation for why two records matched or didn't",
      "inputModes": ["application/json"],
      "outputModes": ["application/json", "text/plain"]
    },
    {
      "id": "review",
      "name": "Review Queue",
      "description": "Present borderline matches for approval, process decisions",
      "inputModes": ["application/json"],
      "outputModes": ["application/json"]
    },
    {
      "id": "compare_strategies",
      "name": "Compare Strategies",
      "description": "Run multiple approaches on sample data, report proxy metrics (cluster count, score distribution, review queue size). Reports F1 only if ground truth provided.",
      "inputModes": ["application/json"],
      "outputModes": ["application/json"]
    },
    {
      "id": "pprl",
      "name": "Privacy-Preserving Match",
      "description": "Bloom filter encryption for sensitive data with auto-configuration",
      "inputModes": ["application/json"],
      "outputModes": ["application/json"]
    }
  ],
  "authentication": {
    "schemes": ["bearer"]
  }
}
```

### A2A Task Endpoints

The A2A server implements the standard task lifecycle:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent card discovery |
| `/tasks/send` | POST | Submit a task (synchronous, blocks until complete) |
| `/tasks/sendSubscribe` | POST | Submit a task with SSE streaming updates |
| `/tasks/{id}` | GET | Get task status and result |
| `/tasks/{id}/cancel` | POST | Cancel a running task |

**Task states:** `submitted` -> `working` -> `completed` | `failed` | `canceled`

**SSE event format** (for `/tasks/sendSubscribe`):

```
event: task-status
data: {"id": "abc123", "state": "working", "progress": "Profiling 50,000 records..."}

event: task-status
data: {"id": "abc123", "state": "working", "progress": "Scoring fuzzy blocks (3/7)..."}

event: task-artifact
data: {"id": "abc123", "artifact": {"type": "result", "parts": [{"type": "data", "data": {...}}]}}

event: task-status
data: {"id": "abc123", "state": "completed"}
```

**Task request format:**

```json
{
  "id": "client-generated-uuid",
  "skill": "deduplicate",
  "message": {
    "role": "user",
    "parts": [
      {"type": "data", "data": {"file_path": "/data/customers.csv"}},
      {"type": "text", "text": "Deduplicate this customer list, prioritize precision"}
    ]
  }
}
```

### Authentication

Bearer token via `GOLDENMATCH_AGENT_TOKEN` env var. If not set, no auth required (local use). Token validated on every task request via a simple string comparison. No JWT/OAuth complexity in v1.

---

## A2A Server Architecture

**Separate server** from the existing REST API. The A2A server runs on its own port (default 8200) using `aiohttp` for async support:

- `goldenmatch agent-serve --port 8200` -- new CLI command
- Async request handling (SSE streaming requires it)
- Background task execution via `asyncio.create_task()`
- Task registry (in-memory dict of task_id -> state/result)

The existing REST API (`goldenmatch serve`, port 8000, `http.server.HTTPServer`) is **unchanged**. No migration needed.

This avoids the reviewer's concern about bolting SSE onto the synchronous HTTP server.

---

## MCP Server Expansion

New agent-level tools (additive, existing tools unchanged):

| Tool | Input | Output |
|------|-------|--------|
| `analyze_data` | `{"file_path": "..."}` | Domain, column profiles, strategy recommendation |
| `auto_configure` | `{"file_path": "...", "constraints": {...}}` | YAML config string |
| `deduplicate` | `{"file_path": "...", "config": {...}}` | Clusters, golden records, stats, review queue |
| `match_sources` | `{"file_a": "...", "file_b": "...", "config": {...}}` | Matched pairs, scores, explanations |
| `explain_pair` | `{"record_a": {...}, "record_b": {...}}` | Natural language explanation |
| `explain_cluster` | `{"cluster_id": 5}` | Cluster narrative |
| `review_queue` | `{"job_name": "..."}` | Borderline pairs needing approval |
| `approve_reject` | `{"pair_id": "...", "decision": "approve", "reason": "..."}` | Updated cluster |
| `compare_strategies` | `{"file_path": "...", "ground_truth": "..."}` | Strategy comparison (proxy metrics; F1 only if ground_truth provided) |
| `suggest_pprl` | `{"file_path": "..."}` | Whether PPRL is needed, recommended config |

All inputs are JSON. File paths refer to files accessible on the server. No DataFrames over JSON -- the tool loads the file internally.

**MCP state management:** New agent tools create their own `AgentSession` object (see Intelligence Layer) rather than sharing the global `_engine`/`_result` state with existing tools. This avoids stale state conflicts.

---

## Intelligence Layer

New module: `goldenmatch/core/agent.py`

### AgentSession

Encapsulates a single agent interaction. Holds its own state (data, config, results) independent of MCP/REST global state.

```python
class AgentSession:
    data: pl.DataFrame | None
    config: GoldenMatchConfig | None
    result: DedupeResult | None
    review_queue: ReviewQueue
    reasoning: dict
```

### Decision Tree (concrete logic)

```python
def select_strategy(profile: DataProfile) -> StrategyDecision:
    """Select ER strategy based on data profiling."""

    # Step 1: Check for sensitive fields -> PPRL
    sensitive_fields = [f for f in profile.fields if f.type in ("ssn", "dob", "drivers_license")]
    if sensitive_fields:
        return StrategyDecision(
            strategy="pprl",
            why=f"Detected sensitive fields: {[f.name for f in sensitive_fields]}. "
                "Recommending privacy-preserving record linkage.",
            auto_execute=False,  # Ask caller to confirm PPRL mode
        )

    # Step 2: Detect domain via domain registry
    domain = match_domain([f.name for f in profile.fields])

    # Step 3: Check for strong ID fields -> exact first
    strong_ids = [f for f in profile.fields
                  if f.uniqueness > 0.90 and f.null_rate < 0.05
                  and f.type == "string"]
    # Examples: email, phone, account_number

    # Step 4: Check for fuzzy-matchable fields
    fuzzy_candidates = [f for f in profile.fields
                        if f.type == "string"
                        and f.uniqueness < 0.90
                        and f.avg_length > 3
                        and f.null_rate < 0.50]
    # Examples: name, address, company

    # Step 5: Dataset size check
    if profile.row_count > 500_000:
        backend = "ray"
    else:
        backend = None

    # Step 6: Build strategy
    if strong_ids and not fuzzy_candidates:
        strategy = "exact_only"
        why = f"Strong ID fields ({[f.name for f in strong_ids]}) with >90% uniqueness. Exact matching sufficient."
    elif strong_ids and fuzzy_candidates:
        strategy = "exact_then_fuzzy"
        why = f"Exact match on {[f.name for f in strong_ids]}, fuzzy on {[f.name for f in fuzzy_candidates]} for residual."
    elif fuzzy_candidates:
        strategy = "fuzzy"
        why = f"No strong IDs. Fuzzy matching on {[f.name for f in fuzzy_candidates]}."
    elif domain and domain.confidence > 0.5:
        strategy = "domain_extraction"
        why = f"Detected {domain.name} domain. Using domain-specific extraction + fuzzy."
    else:
        strategy = "fuzzy"
        why = "Fallback to fuzzy matching on all string columns."

    return StrategyDecision(
        strategy=strategy,
        why=why,
        domain=domain,
        strong_ids=strong_ids,
        fuzzy_fields=fuzzy_candidates,
        backend=backend,
        auto_execute=True,
    )
```

The `auto_execute=False` for PPRL means the agent returns the recommendation and waits for the caller to confirm before proceeding with encryption. All other strategies auto-execute.

### Alternatives Reasoning

After selecting a strategy, the agent generates reasoning for alternatives not chosen:

```python
def build_alternatives(decision: StrategyDecision, profile: DataProfile) -> list[dict]:
    alternatives = []
    if decision.strategy != "pprl":
        alternatives.append({
            "strategy": "pprl",
            "why_not": "No sensitive fields detected." if not profile.has_sensitive
                       else "Already selected."
        })
    if decision.strategy != "fellegi_sunter":
        alternatives.append({
            "strategy": "fellegi_sunter",
            "why_not": "Best for high-precision use cases. Fuzzy matching gives better recall for this data."
                       if decision.strategy == "fuzzy" else "Not applicable."
        })
    return alternatives
```

### compare_strategies

When called without ground truth: runs each strategy on the full dataset, reports cluster count, score histogram, and review queue size as proxy metrics.

When called with ground truth: also computes precision, recall, F1 for each strategy.

```python
def compare_strategies(file_path: str, ground_truth: str | None = None) -> dict:
    strategies = ["exact_then_fuzzy", "fuzzy", "fellegi_sunter"]
    results = {}
    for s in strategies:
        config = build_config_for_strategy(s, profile)
        result = dedupe_df(df, config=config)
        metrics = {
            "clusters": result.total_clusters,
            "match_rate": result.match_rate,
            "review_queue_size": len([p for p in result.scored_pairs if 0.75 <= p[2] <= 0.95]),
        }
        if ground_truth:
            eval_result = evaluate_clusters(result.clusters, gt_pairs)
            metrics.update(eval_result.summary())
        results[s] = metrics
    return results
```

---

## Confidence-Gated Review Queue

New module: `goldenmatch/core/review_queue.py`

**Gating logic (applied in pipeline):**

```
score > 0.95  -> auto_merged
0.75 - 0.95   -> review_queue (held)
score < 0.75  -> auto_rejected
```

Thresholds configurable via `AgentSession` or config.

**Three storage backends (auto-detected):**

| Backend | When | Persists? |
|---------|------|-----------|
| Memory | Default, no config needed | No -- lost when process exits |
| SQLite | `.goldenmatch/` directory exists | Yes -- local file `.goldenmatch/reviews.db` |
| Postgres | `DATABASE_URL` env var set | Yes -- `goldenmatch._reviews` table |

Auto-selection: Postgres if DATABASE_URL set, else SQLite if `.goldenmatch/` exists, else memory. Storage tier communicated in every response.

**Review item schema:**

| Field | Type | Description |
|-------|------|-------------|
| job_name | str | Which job produced this pair |
| id_a | int | First record |
| id_b | int | Second record |
| score | float | Match score |
| explanation | str | NL explanation |
| status | str | pending / approved / rejected |
| decided_by | str | Who approved (human, agent name, auto) |
| decided_at | datetime | When |

**Schema migrations:** SQLite backend uses `CREATE TABLE IF NOT EXISTS` with a `schema_version` pragma. Postgres backend adds a `schema_version` column to `goldenmatch._reviews`. Future schema changes include a migration function that checks version and applies ALTER statements.

**API (same across MCP, A2A, REST):**
- `review_queue(job_name)` -> list of pending pairs with explanations
- `approve_reject(pair_id, decision, reason)` -> updates status, merges if approved
- `review_stats(job_name)` -> counts by status

**Migration from existing REST review queue:** The existing `MatchServer._review_queue` list is replaced by a `ReviewQueue(backend="memory")` instance. The existing `GET /reviews` and `POST /reviews/decide` endpoints delegate to the new `ReviewQueue` class. Same HTTP interface, new internal implementation.

---

## Repo & Branch Strategy

**Branch:** `feature/er-agent` off `main`

**New files:**

| File | Responsibility |
|------|---------------|
| `goldenmatch/core/agent.py` | Intelligence layer: AgentSession, select_strategy, build_alternatives, compare_strategies |
| `goldenmatch/core/review_queue.py` | ReviewQueue class with memory/SQLite/Postgres backends |
| `goldenmatch/a2a/__init__.py` | A2A package |
| `goldenmatch/a2a/server.py` | A2A server (aiohttp): agent card, task CRUD, SSE streaming |
| `goldenmatch/a2a/skills.py` | Skill dispatch: A2A task -> AgentSession method |
| `goldenmatch/mcp/agent_tools.py` | New agent-level MCP tools (10 tools) |
| `tests/test_agent.py` | Intelligence layer + strategy selection tests |
| `tests/test_review_queue.py` | Review queue tests (all three backends) |
| `tests/test_a2a.py` | A2A protocol tests (agent card, task lifecycle) |

**Modified files:**

| File | Change |
|------|--------|
| `goldenmatch/mcp/server.py` | Register new agent-level tools from agent_tools.py |
| `goldenmatch/api/server.py` | Replace `_review_queue` list with ReviewQueue instance |
| `goldenmatch/__init__.py` | Export AgentSession, ReviewQueue |
| `goldenmatch/cli/main.py` | Add `goldenmatch agent-serve` command (A2A server on port 8200) |
| `pyproject.toml` | Add `aiohttp` as optional dep: `pip install goldenmatch[agent]` |

**What doesn't change:** Existing pipeline, scorer, config, CLI commands, REST API server, SQL extensions.

**Merge criteria:**
- All new tests pass
- Existing test suite still passes
- A2A agent card validates against A2A spec (inputModes, outputModes, provider fields present)
- MCP tools work in Claude Desktop
- Demo: another agent discovers and invokes GoldenMatch via A2A
- Review queue works with all three storage backends
