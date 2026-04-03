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
| `goldenmatch compare-clusters` | Yes |
| `goldenmatch sensitivity` | Yes |
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
| `core/compare_clusters.py` | `compare_clusters()`, `CompareResult`, `ClusterCase` |
| `core/sensitivity.py` | `run_sensitivity()`, `SensitivityResult`, `SweepParam`, `SweepPoint` |

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
