# Changelog

All notable changes to GoldenMatch are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). Versioning follows [Semantic Versioning](https://semver.org/) (strict after v1.0.0).

## [Unreleased] / TypeScript port

- **NEW**: TypeScript / Node.js port published as `goldenmatch` on npm
  - Full feature parity with Python: scorers, clustering, golden records, LLM, PPRL, probabilistic, graph ER, streaming, MCP/REST/A2A servers
  - Edge-safe core (browsers, Workers, Edge Runtime) + Node-only file/DB layer
  - 478 tests, strict TypeScript

## [1.4.1] - 2026-04-06

### Added
- **MCP tools for data quality** — `scan_quality` (scan without fixing), `fix_quality` (scan + apply fixes with safe/moderate mode), `run_transforms` (GoldenFlow phone/date/Unicode normalization). All 3 tools validate file paths, handle write failures gracefully, and include logging
- **A2A skills for data quality** — `quality` (scan + fix via GoldenCheck) and `transform` (normalize via GoldenFlow) skills added to the Agent-to-Agent protocol
- `run_transform(strict=True)` parameter — MCP/A2A handlers surface transform failures instead of silently returning unmodified data
- `_scan_only()` now returns serialized findings so MCP tools can inspect quality issues without duplicating the scan
- 10 new tests: happy-path coverage with mocked deps, file validation, write failure handling

### Fixed
- Eliminated redundant double-scan in `scan_quality` MCP handler (was scanning data twice and reaching into goldencheck internals)
- Temp file cleanup handles `PermissionError` on Windows (file locks no longer leak orphaned temp files)
- `_serialise_result` exception clause narrowed from `Exception` to `ImportError`
- `fix_quality` test assertion strengthened to check error message content

## [1.4.0] - 2026-04-06

### Added
- **Scoring & survivorship quality upgrade** — MST-based cluster auto-splitting, cluster quality labels (strong/weak/split), quality-weighted survivorship strategies, field-level provenance tracking
- **Data-driven strategy selection** — auto-config selects learned blocking (>= 5K rows), enables cross-encoder reranking (3+ fields), adjusts thresholds from data quality (null rate, string length)
- **`llm_auto` flag** — `GoldenMatchConfig.llm_auto=True` auto-enables LLM scorer ($0.05 budget) and memory store when API key detected. Applied uniformly across all config paths
- New config: `auto_split`, `quality_weighting`, `weak_cluster_threshold` in `GoldenRulesConfig`

### Fixed
- Pipeline wires `auto_split` config to `build_clusters`
- `add_to_cluster` documents oversized-flag-only behavior (callers must split)
- Threshold adjustments mutually exclusive (high-null and short-string no longer cancel out)

## [1.3.2] - 2026-04-03

### Fixed
- Auto-config: blocking keys with zero value overlap between sources are now skipped with a warning (fixes DBLP-ACM venue blocking failure where DBLP uses "VLDB" and ACM uses "Very Large Data Bases")
- Embedding scorer: falls back to token_sort when embedding model fails to load (HuggingFace auth, Vertex AI quota, missing dep, CUDA OOM) instead of crashing the pipeline

## [1.3.1] - 2026-04-03

### Added
- GoldenFlow integration: optional data transformation step in the dedupe pipeline (`pip install goldenmatch[transform]`)
- `TransformConfig` Pydantic model (enabled, mode: announced/silent/disabled)
- Pipeline step 1.4b: GoldenFlow runs after GoldenCheck, before autofix — normalizes phone numbers, dates, categoricals, unicode
- Graceful degradation: if goldenflow crashes, logs warning and continues with untransformed data
- Warning when config enables transforms but goldenflow is not installed
- 8 new tests

## [1.3.0] - 2026-04-03

### Added
- CCMS cluster comparison: `compare_clusters()` classifies each cluster from run A as unchanged, merged, partitioned, or overlapping relative to run B (based on Talburt et al., arXiv:2601.02824v1)
- `CompareResult` and `ClusterCase` dataclasses with `summary()` method
- Talburt-Wang Index (TWI) for normalized clustering similarity (1.0 = identical, approaches 0 for divergent outcomes)
- Parameter sensitivity analysis: `run_sensitivity()` sweeps config parameters and compares each run against a baseline using CCMS
- `SweepParam`, `SweepPoint`, `SensitivityResult` dataclasses with `stability_report()` for identifying optimal parameter ranges
- Supported sweep fields: `threshold` (all fuzzy matchkeys), `matchkey.<name>.threshold` (individual), `blocking.max_block_size`
- `--sample` option for sensitivity sweeps (random subsample for speed on large datasets)
- Per-point error handling: failed sweep points are logged and skipped, partial results preserved
- CLI command `goldenmatch compare-clusters` with `--details`, `--case-type` filter, `--output` JSON
- CLI command `goldenmatch sensitivity` with `--sweep field:start:stop:step` (repeatable), `--sample`, `--output`
- 16 new tests (10 comparison, 6 sensitivity)

## [1.2.7] - 2026-04-02

### Added
- Three auto-config cardinality guards to prevent failures on edge-case data:
  - Blocking: exclude near-unique columns (cardinality_ratio >= 0.95)
  - Matchkeys: skip exact matchkeys for low-cardinality columns (cardinality_ratio < 0.01)
  - Description columns: route long text to fuzzy matching (token_sort) alongside embedding
- Library comparison benchmarks: head-to-head against Splink, Dedupe, and RecordLinkage on Febrl (0.971 F1) and DBLP-ACM (0.918 F1)

### Fixed
- Auto-config no longer generates blocking keys from near-unique columns that produce single-record blocks
- Auto-config no longer creates exact matchkeys for columns with very few distinct values (e.g., gender, status)
- Description/long-text columns now get fuzzy fallback scoring instead of embedding-only

## [1.2.6] - 2026-04-01

### Added
- Iterative LLM calibration: samples ~100 pairs per round, learns optimal threshold via grid search, converges in 2-3 rounds (~200 pairs, ~$0.01) instead of scoring all candidates
- Concurrent LLM requests via ThreadPoolExecutor with configurable `max_workers` (default 5)
- Thread-safe BudgetTracker with `threading.RLock`
- ANN hybrid blocking: oversized blocks fall back to ANN sub-blocking via embeddings (embeds only unique text values)
- LLM-assisted column classification for ambiguous auto-config types
- Utility-based fuzzy field ranking (cardinality × completeness × string length)
- Price/cost/amount column name patterns to prevent zip misclassification
- `get_embedder()` GPU routing — returns VertexEmbedder when mode=vertex
- 3 new LLMScorerConfig fields: `calibration_sample_size`, `calibration_max_rounds`, `calibration_convergence_delta`
- 3 new ColumnProfile fields: `null_rate`, `cardinality_ratio`, `avg_len`
- 40 new tests (test_llm_calibration.py, test_ann_subblock.py, expanded test_autoconfig.py)

### Fixed
- ID patterns checked before phone/zip in auto-config — SalesID no longer misclassified as "phone"
- SalePrice (5-digit amounts) no longer misclassified as "zip"
- Identifier classifications authoritative over data profiling
- fiModelDesc no longer dropped from fuzzy fields on wide datasets
- Default batch_size bumped from 20 to 75
- "Never demote" behavior: LLM-rejected pairs keep original fuzzy score (was 0.0)
- Robust error handling: URLError/timeout retried, fut.result() guarded, ANN failures caught gracefully
- VertexEmbedder import failures fall back to local embedder

### Changed
- LLM scorer uses iterative calibration when candidates > calibration_sample_size (100)
- Multi-pass blocking passes ann_column/ann_top_k/ann_model to static builder
- `_classify_by_name` check order: date → email → ID → price → zip → geo → address → phone → name

## [1.2.0] - 2026-03-25

### Added
- **Autonomous ER Agent** -- GoldenMatch as a discoverable AI agent via A2A and MCP protocols
- `AgentSession` class -- profiles data, selects strategy, runs pipeline, explains reasoning
- `ReviewQueue` with confidence gating (auto-merge >0.95, review 0.75-0.95, reject <0.75)
- Three storage backends for review queue: memory (default), SQLite, Postgres
- `gate_pairs()` -- split scored pairs by confidence thresholds
- A2A server (`goldenmatch agent-serve`) with agent card, task lifecycle, SSE streaming
- 8 A2A skills: analyze_data, configure, deduplicate, match, explain, review, compare_strategies, pprl
- 10 MCP agent-level tools (additive to existing tools)
- `goldenmatch agent-serve --port 8200` CLI command
- Demo script: `python examples/agent_demo.py`
- Branch & Merge SOP added to CLAUDE.md

## [1.1.0] - 2026-03-23

### Added
- `gm.dedupe_df()` -- deduplicate a Polars DataFrame directly (no file I/O)
- `gm.match_df()` -- match two Polars DataFrames directly (no file I/O)
- `gm.score_strings()` -- score two strings with a named similarity algorithm
- `gm.score_pair_df()` -- score a pair of record dicts
- `gm.explain_pair_df()` -- explain a pair match from record dicts
- Internal: `run_dedupe_df()` and `run_match_df()` pipeline entry points
- These functions are the prerequisite for native SQL extensions (Postgres/DuckDB)
- New companion repo: [goldenmatch-extensions](https://github.com/benzsevern/goldenmatch-extensions) -- PostgreSQL extension (`goldenmatch_pg`) and DuckDB extension (`goldenmatch-duckdb`) for in-database entity resolution via SQL

## [1.0.0] - 2026-03-23

### Changed
- **Production/Stable** -- dropped Beta label. Semver strictly enforced from this release.
- Public API surface frozen: 96 exports from `import goldenmatch as gm`, 21 CLI commands, config YAML schema, REST endpoints, MCP tools. See `docs/api-stability.md`.

### Added
- Clean Python API: `gm.dedupe()`, `gm.match()`, `gm.pprl_link()`, `gm.evaluate()` with typed results
- 96 public exports covering every feature (config, pipeline, streaming, LLM, PPRL, domain, explain, etc.)
- REST API client: `gm.Client("http://localhost:8000")`
- Jupyter/notebook display: `_repr_html_()` on DedupeResult and MatchResult
- CI/CD quality gates: `goldenmatch evaluate --min-f1 0.90` exits code 1 if below threshold
- 7 runnable example scripts in `examples/`
- `goldenmatch label` CLI for interactive ground truth building

## [0.7.0] - 2026-03-23

### Added
- Ray distributed backend for large-scale entity resolution (`pip install goldenmatch[ray]`)
- `--backend ray` CLI flag for dedupe command
- `backend: ray` config option in GoldenMatchConfig
- `backends/ray_backend.py` with `score_blocks_ray()` -- drop-in replacement for ThreadPoolExecutor
- Automatic fallback to parallel scorer for small block counts (<= 4)
- Ray auto-initializes locally using all CPU cores, no user configuration needed
- Supports Ray clusters for 50M+ record workloads
- `goldenmatch label` CLI command -- interactive pair labeling to build ground truth CSV for accuracy measurement (y/n/s keyboard input)

## [0.6.0] - 2026-03-23

### Added
- Privacy-preserving record linkage (PPRL) package (`goldenmatch/pprl/`)
- Trusted third party mode: parties send encrypted bloom filters, coordinator computes similarity
- SMC mode: secret-shared dice similarity, only match bits revealed (simulated circuit)
- `goldenmatch pprl link` CLI command for cross-party linkage
- Bloom filter security levels: standard (512-bit), high (1024-bit + HMAC), paranoid (2048-bit + balanced padding)
- Per-field HMAC salting prevents cross-field correlation attacks
- Balanced bloom filter padding normalizes filter density for short strings
- Custom HMAC key support via transform parameter (`bloom_filter:2:20:512:my_key`)
- `pip install goldenmatch[pprl]` optional dependency group
- PPRL auto-configuration (`auto_configure_pprl`) -- profiles data, selects optimal fields, bloom filter parameters, and threshold automatically. 92.4% F1 on FEBRL4, 76.1% on NCVR
- MCP tools: `pprl_auto_config` (auto-configure PPRL for a dataset), `pprl_link` (run cross-party linkage)
- Vectorized PPRL similarity computation (13x speedup over row-wise scoring)
- NCVR (North Carolina Voter Registration) and FEBRL4 benchmark suites for PPRL evaluation

## [0.5.0] - 2026-03-23

### Added
- In-context LLM clustering (`mode: cluster`) -- send blocks of 50-100 borderline records to LLM for direct cluster assignment instead of pairwise yes/no scoring
- Uncertainty scores -- LLM returns confidence per cluster, surfaced in cluster metadata and review queue
- `core/llm_cluster.py` -- new module with component detection, graph splitting, structured JSON parsing, pairwise fallback
- LLMScorerConfig gains `mode`, `cluster_max_size`, `cluster_min_size` fields
- Budget-aware degradation: cluster mode -> pairwise fallback -> stop

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
