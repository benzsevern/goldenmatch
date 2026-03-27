# GoldenMatch - Agent Instructions

> This file is read by Gemini CLI and OpenAI Codex. See also: `CLAUDE.md` for Claude Code.

## Project Overview
GoldenMatch is a Python entity resolution / record linkage toolkit. It deduplicates and matches records across datasets using exact, fuzzy, probabilistic, PPRL (privacy-preserving), and LLM-assisted strategies. Published on PyPI as `goldenmatch` (v1.0.0+).

## Related Projects
- **goldenmatch-extensions** (`D:\show_case\goldenmatch-extensions`): Postgres extension + DuckDB UDFs. Separate repo (`benzsevern/goldenmatch-extensions`).
- **PyPI packages:** `goldenmatch`, `goldenmatch-duckdb`
- **GitHub:** `benzsevern/goldenmatch`, `benzsevern/goldenmatch-extensions`

## Branch & Merge SOP
- Feature work on `feature/<name>` branches, never commit directly to main.
- Merge via **squash merge PR**. PR title: `feat: <desc>` or `fix: <desc>`. PR body: summary bullets + test plan.
- Merge when tests pass and docs are updated. Delete remote branch after merge.
- Commands: `gh pr create --title "..." --body "..."`, then `gh pr merge --squash`.

## Environment
- **OS:** Windows 11, bash shell (Git Bash). Use Unix paths in scripts.
- **Python:** 3.12 at `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe`
- **Project path:** `D:\show_case\goldenmatch`
- **Polars gotcha:** `scan_csv` uses `encoding="utf8"` (not `"utf-8"`). `read_excel` needs explicit `engine="openpyxl"`.
- **Rust:** 1.94.0 at `C:\Users\bsevern\.cargo\bin`. Must set `RUSTUP_HOME="C:/Users/bsevern/.rustup"` and `CARGO_HOME="C:/Users/bsevern/.cargo"` in every bash command, plus add to PATH.
- **No admin privileges.** Cannot install system packages. Use `RUSTUP_WINDOWS_PATH_TYPE=hardlink` for Rust, user-dir installs for everything else.
- pgrx cannot build locally (needs libclang/LLVM). Use CI (Linux) for pgrx builds/tests.
- PostgreSQL 16 portable at `C:\Users\bsevern\tools\pg16portable\pgsql`.
- GCP project: `gen-lang-client-0692108803` (Vertex AI embeddings).
- Two GitHub accounts: `benzsevern` (personal, this repo) and `benzsevern-mjh` (work). Must `gh auth switch --user benzsevern` before push.

## Testing
- Run: `pytest --tb=short` from project root. All tests must pass after every change.
- **1173 tests** (+ 6 skipped), ~50s runtime. Coverage: 72%.
- Key fixtures in `tests/conftest.py`: `sample_csv`, `sample_csv_b`, `sample_parquet`.
- TUI tests use `pytest-asyncio` with `app.run_test()` pilot.
- DB tests (`test_db.py`, `test_reconcile.py`) need PostgreSQL -- skip with `--ignore` if unavailable.
- `import torch` hangs on this machine -- tests mocking GPU must patch `_has_cuda`/`_has_mps` at module level.
- `testing.postgresql` teardown errors on Windows (SIGINT) are harmless.
- CI: `.github/workflows/ci.yml` -- matrix (3.11/3.12/3.13), ruff lint (E9/F63/F7 only), smoke test. Ignores test_db, test_reconcile, test_mcp_and_watch.
- Ray tests require `ray` optional dep -- use `pytest.mark.skipif(not HAS_RAY)`.
- Windows drive letter tests: `@pytest.mark.skipif(sys.platform != "win32")`.
- Memory tests use `tmp_path` fixture for isolated SQLite.

## Architecture
- **Pipeline stages:** ingest -> column_map -> auto_fix -> validate -> standardize -> matchkeys -> block -> score -> cluster -> golden -> output
- **Key modules:**
  - `_api.py` -- DataFrame entry points: `dedupe_df()`, `match_df()`, `score_strings()`, `score_pair_df()`, `explain_pair_df()`
  - `pipeline.py` -- `_run_dedupe_pipeline()` and `_run_match_pipeline()` shared by file-based and DataFrame entry points
  - `core/agent.py` -- AgentSession, autonomous ER (profile -> detect domain -> select strategy -> run pipeline)
  - `core/review_queue.py` -- ReviewQueue with confidence gating (>0.95 auto-merge, 0.75-0.95 review, <0.75 reject)
  - `core/memory/` -- Learning Memory: persistent corrections + rule learning (store, corrections, learner)
  - `core/probabilistic.py` -- Fellegi-Sunter EM-trained m/u probabilities
  - `core/learned_blocking.py` -- data-driven predicate selection
  - `core/domain.py` -- auto-detects product subdomain, extracts brand/model/SKU
  - `a2a/` -- A2A protocol server (aiohttp), agent card, 8 skills, SSE streaming
  - `mcp/` -- MCP server for Claude Desktop
  - `tui/` -- Textual TUI (6 tabs: Data, Config, Matches, Golden, Boost, Export)
  - `cli/` -- Typer CLI (21 commands)
  - `db/` -- Postgres integration
  - `api/` -- REST API server
  - `plugins/` -- Plugin system (scorer/transform/connector/golden_strategy)
  - `connectors/` -- Snowflake, Databricks, BigQuery, HubSpot, Salesforce
  - `backends/` -- DuckDB (out-of-core), Ray (distributed)
  - `domains/` -- 7 built-in YAML domain packs
  - `config/schemas.py` -- Pydantic models; `config/loader.py` -- YAML loading
- **Directory layout:** `goldenmatch/core/` (pipeline, no Textual dep), `goldenmatch/tui/` (Textual TUI), `goldenmatch/cli/`, `goldenmatch/db/`, `goldenmatch/api/`, `goldenmatch/mcp/`, `goldenmatch/plugins/`, `goldenmatch/connectors/`, `goldenmatch/backends/`, `goldenmatch/domains/`, `dbt-goldenmatch/` (separate dbt package)

## Performance
- Exact matching: Polars self-join (not Python group_by + combinations).
- Fuzzy matching: `rapidfuzz.process.cdist` for vectorized NxN scoring.
- Fuzzy blocks scored in parallel via `ThreadPoolExecutor` (`score_blocks_parallel`). rapidfuzz.cdist releases GIL so threads give real parallelism. For <=2 blocks, runs sequentially.
- Intra-field early termination in `find_fuzzy_matches`: breaks early if no pair can reach threshold.
- Native Polars fast paths for standardizers (`_NATIVE_STANDARDIZERS`) and matchkey transforms (`_try_native_chain`).
- Clustering: iterative Union-Find (not recursive) with lazy pair_scores.
- Benchmarks: 1M exact dedupe ~7.8s; 100K fuzzy (name+zip) ~12.8s; 7,823 rec/s at 100K.
- OOM at 1M records in-memory -- use DuckDB backend or chunked processing for >500K.

## Public API
- File-based: `dedupe(config_path)`, `match(config_path)`
- DataFrame-based: `dedupe_df(df, config)`, `match_df(df_a, df_b, config)`, `score_strings(a, b, method)`, `score_pair_df(df, config)`, `explain_pair_df(df, config)`
- PPRL: `pprl_link(config_path)`
- Evaluation: `evaluate(config_path, ground_truth_path)`
- Results: `DedupeResult` / `MatchResult` have `_repr_html_()` for Jupyter
- `__init__.py` re-exports ~101 symbols

## Key Gotchas
- `import torch` hangs on machines without GPU -- use `goldenmatch.core.gpu.detect_gpu_mode()` before loading.
- Polars `scan_csv` encoding is `"utf8"` not `"utf-8"`. Polars infers zip/phone as Int64 -- must `str()` values before comparing.
- Windows drive letter paths (`C:\`) break `file:source_name` CLI parsing -- handled in `_parse_file_source`.
- Scored pairs are canonicalized as `(min(id_a, id_b), max(id_a, id_b))` throughout cluster.py, graph.py, chunked.py, ann_blocker.py -- any new code storing/looking up pairs must canonicalize too.
- Internal columns prefixed with `__` (e.g. `__row_id__`, `__source__`, `__mk_*__`).
- `run_dedupe()` return dict has NO `stats` key. `match_one()` returns empty list for exact matchkeys.
- `json.dumps(clusters)` fails with tuple keys (pair_scores) -- use str() fallback.
- Unicode em dashes / box drawing chars crash on Windows terminals -- use ASCII.
- Leipzig benchmark CSVs have invalid UTF-8 -- use `pl.read_csv(encoding="utf8-lossy", ignore_errors=True)`.
- Fellegi-Sunter EM: blocking fields must be excluded from training. u-probabilities estimated from random pairs and FIXED during EM.
- PyPI version must be bumped in both `pyproject.toml` and `goldenmatch/__init__.py`.
- `.testing/` folder is gitignored -- store credentials/API keys there.

## Remote MCP Server
- Endpoint: https://goldenmatch-mcp-production.up.railway.app/mcp/
- Smithery: https://smithery.ai/servers/benzsevern/goldenmatch
- 27 tools, Streamable HTTP transport
- Dockerfile: Dockerfile.mcp
- Local HTTP: goldenmatch mcp-serve --transport http --port 8200
