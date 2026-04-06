---
layout: default
title: ER Agent
nav_order: 22
---

# ER Agent

GoldenMatch exposes itself as an autonomous entity resolution agent that other AI systems can discover and invoke.

An agent says "deduplicate this data" and GoldenMatch handles strategy selection, config generation, pipeline execution, and result explanation -- all without human configuration.

---

## Two Protocols

| Protocol | Port | Best For |
|----------|------|----------|
| **A2A** (Agent-to-Agent) | 8200 | AI agent frameworks (LangChain, CrewAI, AutoGen) |
| **MCP** (Model Context Protocol) | stdio | Claude Desktop, Cursor, Windsurf |

---

## Quick Start

### A2A Server

```bash
pip install goldenmatch[agent]
goldenmatch agent-serve --port 8200
```

Other agents discover GoldenMatch at:
```
GET http://localhost:8200/.well-known/agent.json
```

### MCP (Claude Desktop)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "goldenmatch": {
      "command": "goldenmatch",
      "args": ["mcp-serve", "--file", "customers.csv"]
    }
  }
}
```

---

## Agent Capabilities (10 Skills)

| Skill | What It Does |
|-------|-------------|
| `analyze_data` | Profile columns, detect domain, recommend matching strategy |
| `configure` | Generate optimal YAML config from data analysis |
| `deduplicate` | Full pipeline with confidence-gated output and reasoning |
| `match` | Cross-source matching with intelligent strategy selection |
| `explain` | Natural language explanation for any pair or cluster |
| `review` | Present borderline matches for approval |
| `compare_strategies` | Run multiple approaches, report metrics |
| `pprl` | Privacy-preserving mode for sensitive data |
| `quality` | Scan and fix data quality issues (encoding, Unicode, format violations) using GoldenCheck |
| `transform` | Normalize data formats (phone E.164, dates ISO, categorical spelling) using GoldenFlow |

---

## How It Works

When an agent calls `deduplicate`, GoldenMatch:

1. **Profiles** the data (column types, cardinality, null rates)
2. **Detects** the domain (healthcare, financial, retail, people, etc.)
3. **Selects** the best strategy:
   - Strong ID fields (email, SSN) -> exact matching
   - Fuzzy-matchable fields (name, address) -> fuzzy matching
   - Sensitive fields detected -> recommends PPRL
   - Large datasets (>500K) -> recommends Ray backend
4. **Generates** a config (matchkeys, blocking, scoring)
5. **Runs** the pipeline with confidence gating
6. **Returns** results + reasoning

### Reasoning Output

Every response includes the agent's reasoning:

```json
{
  "results": {
    "clusters": 42,
    "match_rate": "8.4%"
  },
  "reasoning": {
    "domain_detected": "people",
    "strategy_chosen": "exact_then_fuzzy",
    "why": "Email has 92% uniqueness -- strong exact key. Name has spelling variation -- jaro_winkler at 0.85.",
    "alternatives_considered": [
      {"strategy": "pprl", "why_not": "No sensitive fields detected."},
      {"strategy": "fellegi_sunter", "why_not": "Fuzzy gives better recall for this data."}
    ],
    "confidence_distribution": {
      "auto_merged": 38,
      "review_queue": 4,
      "auto_rejected": 0
    }
  },
  "storage": "memory"
}
```

---

## Confidence-Gated Review Queue

Not all matches are equal. The agent splits results by confidence:

| Confidence | Action | Count |
|-----------|--------|-------|
| > 0.95 | Auto-merged into golden records | High-confidence pairs |
| 0.75 - 0.95 | Held in review queue for approval | Borderline pairs |
| < 0.75 | Auto-rejected | Low-confidence pairs |

### Storage Tiers

| Tier | Config | Persists? |
|------|--------|-----------|
| **Memory** | Default (nothing to configure) | No |
| **SQLite** | Create a `.goldenmatch/` directory | Yes (local file) |
| **Postgres** | Set `DATABASE_URL` env var | Yes (shared DB) |

The agent auto-detects which tier is available and reports it in every response.

### Review Queue API

```python
from goldenmatch import AgentSession

session = AgentSession()
result = session.deduplicate("customers.csv")

# Check what needs review
pending = session.review_queue.list_pending("customers")
for item in pending:
    print(f"Pair ({item.id_a}, {item.id_b}): score={item.score}")
    print(f"  Explanation: {item.explanation}")

# Approve or reject
session.review_queue.approve("customers", 0, 1, decided_by="human")
session.review_queue.reject("customers", 2, 3, decided_by="human", reason="Different entities")

# Stats
print(session.review_queue.stats("customers"))
# {"pending": 2, "approved": 1, "rejected": 1}
```

---

## Python API

```python
from goldenmatch import AgentSession

session = AgentSession()

# Analyze data and get strategy recommendation
analysis = session.analyze("customers.csv")
print(analysis["strategy"])  # "exact_then_fuzzy"
print(analysis["why"])

# Deduplicate with full reasoning
result = session.deduplicate("customers.csv")
print(result["results"]["clusters"])
print(result["reasoning"]["why"])

# Compare strategies
comparison = session.compare_strategies("customers.csv")
for strategy, metrics in comparison.items():
    print(f"{strategy}: {metrics['clusters']} clusters, {metrics['match_rate']:.1%} match rate")

# Match two sources
matches = session.match_sources("new_customers.csv", "master.csv")
```

---

## MCP Tools (13 Agent-Level)

| Tool | Description |
|------|-------------|
| `analyze_data` | Profile data, detect domain, recommend strategy |
| `auto_configure` | Generate optimal config |
| `agent_deduplicate` | Full pipeline with reasoning |
| `agent_match_sources` | Cross-source matching |
| `agent_explain_pair` | Explain a pair match |
| `agent_explain_cluster` | Explain a cluster |
| `agent_review_queue` | Get pending reviews |
| `agent_approve_reject` | Process review decisions |
| `agent_compare_strategies` | Compare ER approaches |
| `suggest_pprl` | Check if PPRL is needed |
| `scan_quality` | Run GoldenCheck data quality scan, return issues without fixing |
| `fix_quality` | Run GoldenCheck scan and apply fixes (safe or moderate mode) |
| `run_transforms` | Run GoldenFlow transforms (phone E.164, dates ISO, Unicode) |

These are additive -- existing MCP tools (`suggest_config`, `list_domains`, etc.) continue to work.

---

## A2A Agent Card

```json
{
  "name": "goldenmatch-agent",
  "description": "Autonomous entity resolution agent.",
  "provider": {
    "organization": "GoldenMatch",
    "url": "https://github.com/benzsevern/goldenmatch"
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [...]
}
```

Full card at: `http://localhost:8200/.well-known/agent.json`

---

## Authentication

Set `GOLDENMATCH_AGENT_TOKEN` env var for bearer token auth. If not set, no auth required (suitable for local use).
