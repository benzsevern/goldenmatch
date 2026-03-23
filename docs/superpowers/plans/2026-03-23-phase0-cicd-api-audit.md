# Phase 0: CI/CD + API Stability Audit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automated CI/CD pipeline, audit and stabilize the public API surface, create CHANGELOG.md, and bump to v0.4.0.

**Architecture:** GitHub Actions workflow for tests + lint on every push/PR. API audit reviews all 51 core modules, 19 CLI commands, and config schemas — identifies and makes any breaking changes while still in 0.x. CHANGELOG.md tracks all changes retroactively and going forward.

**Tech Stack:** GitHub Actions, pytest, ruff, Python 3.11/3.12/3.13

---

## File Structure

### New Files
- `.github/workflows/ci.yml` — CI pipeline (test matrix + lint)
- `CHANGELOG.md` — Keep a Changelog format
- `goldenmatch/py.typed` — PEP 561 marker
- `docs/api-stability.md` — Public API surface documentation

### Modified Files
- `README.md` — Add CI badge
- `goldenmatch/__init__.py` — Version bump to 0.4.0
- `pyproject.toml` — Version bump to 0.4.0

---

## Task 1: CI/CD Pipeline

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`

- [ ] **Step 1: Create the CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
      fail-fast: false
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest --tb=short --ignore=tests/test_db.py --ignore=tests/test_reconcile.py -q

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install ruff
        run: pip install ruff

      - name: Run ruff check
        run: ruff check goldenmatch/ tests/

  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install from wheel
        run: |
          pip install build
          python -m build
          pip install dist/*.whl

      - name: Smoke test
        run: goldenmatch demo
```

Write to `.github/workflows/ci.yml`.

- [ ] **Step 2: Add CI badge to README**

Add after the existing PyPI badge line in `README.md`:

```markdown
[![CI](https://github.com/benzsevern/goldenmatch/actions/workflows/ci.yml/badge.svg)](https://github.com/benzsevern/goldenmatch/actions/workflows/ci.yml)
```

- [ ] **Step 3: Verify lint passes locally**

Run: `ruff check goldenmatch/ tests/`
Expected: No errors (or fix any that appear)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "feat: add CI pipeline -- test matrix (3.11/3.12/3.13), lint, smoke test"
```

---

## Task 2: PEP 561 Type Marker

**Files:**
- Create: `goldenmatch/py.typed`

- [ ] **Step 1: Create py.typed marker**

Create an empty file at `goldenmatch/py.typed`. This tells type checkers (mypy, pyright) that goldenmatch ships type information.

- [ ] **Step 2: Commit**

```bash
git add goldenmatch/py.typed
git commit -m "feat: add py.typed PEP 561 marker"
```

---

## Task 3: CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create CHANGELOG.md with retroactive entries**

Use [Keep a Changelog](https://keepachangelog.com/) format. Include entries for v0.3.0, v0.3.1, and the upcoming v0.4.0.

```markdown
# Changelog

All notable changes to GoldenMatch are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). Versioning follows [Semantic Versioning](https://semver.org/) (strict after v1.0.0).

## [Unreleased]

## [0.4.0] - 2026-03-23

### Added
- CI/CD pipeline: automated tests on Python 3.11/3.12/3.13, ruff lint, smoke test
- `py.typed` PEP 561 marker for type checker support
- `docs/api-stability.md` documenting the public API surface
- This CHANGELOG

### Changed
- Version policy: public API surface defined and documented ahead of 1.0 semver commitment

## [0.3.1] - 2026-03-22

### Added
- 5 new domain packs: healthcare, financial, real_estate, people, retail (7 total)
- `goldenmatch evaluate` CLI command -- precision/recall/F1 against ground truth CSV
- `goldenmatch incremental` CLI command -- match new records against existing base
- GitHub Actions "Try It" workflow for zero-install demo
- GitHub Codespaces devcontainer
- `dbt-goldenmatch` package for DuckDB-based entity resolution
- GitHub Discussions, issue templates, community standards (CoC, contributing, security)
- PyPI download badge in README

## [0.3.0] - 2026-03-21

### Added
- Fellegi-Sunter probabilistic matching with EM-trained m/u probabilities
- Learned blocking -- data-driven predicate selection
- LLM scorer with budget controls (BudgetTracker, cost caps, model tiering)
- Domain-aware feature extraction (electronics, software auto-detection)
- Custom domain registry (YAML rulebooks, MCP tools)
- Plugin architecture (scorers, transforms, connectors, golden strategies via entry points)
- Enterprise connectors: Snowflake, Databricks, BigQuery, HubSpot, Salesforce
- DuckDB backend for out-of-core processing
- Streaming/CDC mode with StreamProcessor
- Multi-table graph entity resolution
- Natural language explainability (zero LLM cost)
- Lineage tracking with streaming writer (no 10K cap)
- REST API review queue for data steward approval
- Daemon mode with health endpoint and PID file
- MCP server tools: list_domains, create_domain, test_domain, suggest_config

### Changed
- LLM scorer refactored to accept LLMScorerConfig with BudgetConfig
- Pipeline: domain extraction step between standardize and matchkeys
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG.md with retroactive entries for v0.3.0, v0.3.1"
```

---

## Task 4: API Stability Documentation

**Files:**
- Create: `docs/api-stability.md`

- [ ] **Step 1: Create API stability document**

```markdown
# API Stability

This document defines the public API surface of GoldenMatch. After v1.0.0, all items listed here are covered by [Semantic Versioning](https://semver.org/):

- **Patch releases** (1.0.x): bug fixes only, no API changes
- **Minor releases** (1.x.0): new features, no breaking changes
- **Major releases** (x.0.0): may include breaking changes

## Public API Surface

### CLI Commands

All commands registered in `goldenmatch/cli/main.py`:

| Command | Stable |
|---------|--------|
| `goldenmatch dedupe` | Yes |
| `goldenmatch match` | Yes |
| `goldenmatch evaluate` | Yes |
| `goldenmatch incremental` | Yes |
| `goldenmatch sync` | Yes |
| `goldenmatch watch` | Yes |
| `goldenmatch serve` | Yes |
| `goldenmatch mcp-serve` | Yes |
| `goldenmatch demo` | Yes |
| `goldenmatch setup` | Yes |
| `goldenmatch rollback` | Yes |
| `goldenmatch runs` | Yes |
| `goldenmatch unmerge` | Yes |
| `goldenmatch schedule` | Yes |
| `goldenmatch init` | Yes |
| `goldenmatch interactive` | Yes |
| `goldenmatch profile` | Yes |
| `goldenmatch analyze-blocking` | Yes |
| `goldenmatch config *` | Yes |

CLI flags for each command are part of the stable API. New flags may be added in minor releases. Existing flags will not be removed or have their behavior changed in minor releases.

### Config YAML Schema

All keys in `goldenmatch/config/schemas.py` Pydantic models:

- `GoldenMatchConfig` and all nested models
- `MatchkeyConfig`, `MatchkeyField`
- `BlockingConfig`, `BlockingKeyConfig`
- `GoldenRulesConfig`, `GoldenFieldRule`
- `LLMScorerConfig`, `BudgetConfig`
- `DomainConfig`
- `StandardizationConfig`, `ValidationConfig`

New config keys may be added in minor releases. Existing keys will not be removed or have their meaning changed.

### Core Module Functions

Public functions (not prefixed with `_`) in these modules:

| Module | Key Functions |
|--------|--------------|
| `core/pipeline.py` | `run_dedupe()`, `run_match()` |
| `core/scorer.py` | `find_exact_matches()`, `find_fuzzy_matches()`, `score_pair()`, `score_blocks_parallel()` |
| `core/cluster.py` | `build_clusters()`, `add_to_cluster()`, `unmerge_record()`, `unmerge_cluster()` |
| `core/blocker.py` | `build_blocks()` |
| `core/evaluate.py` | `evaluate_pairs()`, `evaluate_clusters()`, `load_ground_truth_csv()` |
| `core/match_one.py` | `match_one()` |
| `core/streaming.py` | `StreamProcessor`, `run_stream()` |
| `core/golden.py` | `build_golden_records()` |
| `core/ingest.py` | `load_file()`, `load_files()` |
| `core/standardize.py` | `apply_standardization()` |
| `core/matchkey.py` | `compute_matchkeys()` |
| `core/probabilistic.py` | `train_em()`, `score_probabilistic()` |
| `core/domain_registry.py` | `discover_rulebooks()`, `load_rulebook()`, `save_rulebook()`, `match_domain()`, `extract_with_rulebook()` |

### REST API Endpoints

All endpoints in `goldenmatch/api/`:

| Endpoint | Method | Stable |
|----------|--------|--------|
| `/match` | POST | Yes |
| `/clusters` | GET | Yes |
| `/clusters/{id}` | GET | Yes |
| `/golden/{id}` | GET | Yes |
| `/explain/{id_a}/{id_b}` | GET | Yes |
| `/stats` | GET | Yes |
| `/config` | GET | Yes |
| `/reviews` | GET | Yes |
| `/reviews/decide` | POST | Yes |
| `/health` | GET | Yes |

### MCP Tools

All tools in `goldenmatch/mcp/server.py` are part of the stable API.

### Domain Pack YAML Schema

The YAML schema for domain packs (`DomainRulebook` fields) is stable. New fields may be added. Existing fields will not be removed or have their meaning changed.

## NOT Part of the Public API

- Internal functions prefixed with `_`
- Module-level constants
- `__pycache__` and compiled bytecode
- Test fixtures and benchmark scripts
- `tui/` internals (widget implementations, screen layouts)
- `db/` internal SQL queries and table schemas (metadata table names are stable)

## Deprecation Policy

Deprecated features will:
1. Emit a `DeprecationWarning` for at least one minor version
2. Be documented in the CHANGELOG
3. Be removed no earlier than the next major version
```

- [ ] **Step 2: Commit**

```bash
git add docs/api-stability.md
git commit -m "docs: add API stability document defining public surface for semver"
```

---

## Task 5: Version Bump + Final Commit

**Files:**
- Modify: `goldenmatch/__init__.py`
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Bump version to 0.4.0**

In `goldenmatch/__init__.py`, change `__version__ = "0.3.1"` to `__version__ = "0.4.0"`.

In `pyproject.toml`, change `version = "0.3.1"` to `version = "0.4.0"`.

- [ ] **Step 2: Update CLAUDE.md**

Update the version reference from v0.3.1 to v0.4.0. Update test count if it changed.

- [ ] **Step 3: Run full test suite**

Run: `pytest --tb=short --ignore=tests/test_db.py --ignore=tests/test_reconcile.py`
Expected: 855+ tests pass, 6 skipped, 0 failures

- [ ] **Step 4: Commit**

```bash
git add goldenmatch/__init__.py pyproject.toml CLAUDE.md
git commit -m "chore: bump version to v0.4.0 -- CI/CD, API stability audit"
```

---

## Task 6: Push, Release, Publish

- [ ] **Step 1: Push to GitHub**

```bash
gh auth switch --user benzsevern
git push origin main
```

- [ ] **Step 2: Verify CI passes**

Check: `gh run list --workflow=ci.yml --limit=1`
Expected: CI workflow triggered and passing on all 3 Python versions

- [ ] **Step 3: Create GitHub release**

```bash
gh release create v0.4.0 --title "v0.4.0 -- CI/CD Pipeline + API Stability Audit" --notes "..."  --latest
```

- [ ] **Step 4: Publish to PyPI**

```bash
python -m build
source .testing/.env && python -m twine upload dist/goldenmatch-0.4.0* --skip-existing
```

- [ ] **Step 5: Switch back to work account**

```bash
gh auth switch --user benzsevern-mjh
```
