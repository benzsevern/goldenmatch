# GoldenMatch

## Environment
- Windows 11, bash shell (Git Bash) — use Unix paths in scripts
- Python 3.12 at `C:\Users\bsevern\AppData\Local\Programs\Python\Python312\python.exe`
- Project lives on D: drive: `D:\show_case\goldenmatch`
- Two GitHub accounts: `benzsevern` (personal, for this repo) and `benzsevern-mjh` (work)
- MUST `gh auth switch --user benzsevern` before push, switch back to `benzsevern-mjh` after
- Polars `scan_csv` uses `encoding="utf8"` not `"utf-8"`
- Polars `read_excel` needs explicit `engine="openpyxl"`

## Testing
- `pytest --tb=short` from project root — all tests must pass after every change
- 688 tests, run in ~35s
- Fixtures in `tests/conftest.py`: `sample_csv`, `sample_csv_b`, `sample_parquet`
- TUI tests use `pytest-asyncio` with `app.run_test()` pilot
- Benchmark scripts in `tests/bench_1m.py`, `tests/analyze_results.py` (not part of test suite)
- Synthetic test data generator: `tests/generate_synthetic.py`
- DB tests (`test_db.py`, `test_reconcile.py`) need PostgreSQL — skip with `--ignore` if not available
- `import torch` hangs on this machine — tests mocking GPU must patch `_has_cuda`/`_has_mps` at module level
- `testing.postgresql` teardown errors on Windows (SIGINT) are harmless — tests still pass

## Architecture
- Pipeline: ingest → column_map → auto_fix → validate → standardize → matchkeys → block → score → cluster → golden → output
- `goldenmatch/core/` — pipeline modules (no Textual dependency)
- `goldenmatch/tui/` — Textual TUI + MatchEngine (engine.py has no Textual dependency)
- `goldenmatch/cli/` — Typer CLI commands (17 commands, including `unmerge`)
- `goldenmatch/db/` — Postgres integration (connector, sync, reconcile, clusters, ANN index)
- `goldenmatch/api/` — REST API server (`goldenmatch serve`)
- `goldenmatch/mcp/` — MCP server for Claude Desktop (`goldenmatch mcp-serve`)
- Core modules: explainer, report, dashboard, graph, anomaly, diff, rollback, schema_match, chunked, cloud_ingest, api_connector, scheduler, gpu, vertex_embedder, llm_scorer, lineage, match_one
- Config: Pydantic models in `config/schemas.py`, YAML loading in `config/loader.py`
- `config/loader.py` normalizes golden_rules and standardization sections from flat YAML

## Performance
- Exact matching uses Polars self-join (not Python group_by + combinations)
- Fuzzy matching uses `rapidfuzz.process.cdist` for vectorized NxN scoring
- Fuzzy blocks scored in parallel via `ThreadPoolExecutor` (`score_blocks_parallel` in scorer.py)
  - rapidfuzz.cdist releases the GIL so threads give real parallelism
  - Blocks are independent — frozen `exclude_pairs` snapshot avoids races
  - For <=2 blocks, skips thread overhead and runs sequentially
  - All call sites (pipeline.py, engine.py, chunked.py) use the shared helper
- Intra-field early termination in `find_fuzzy_matches`: after each expensive field, checks if any pair can still reach threshold even with perfect remaining scores; breaks early if not
- Cross-encoder reranking (`rerank: true` on weighted matchkey): re-scores borderline pairs (threshold +/- band) with a pre-trained cross-encoder for improved precision, no training needed
- Histogram-based auto-select (`auto_select: true` on blocking): evaluates all configured keys, picks the one with smallest max_block_size and >= 50% coverage
- Dynamic block splitting: adaptive strategy auto-splits oversized blocks by highest-cardinality column when no sub_block_keys configured; picks column with most useful groups (>= 2 records), not just max cardinality
- Multi-field embedding supports `column_weights` — high-weight fields get text repeated in concatenation, biasing embeddings toward important fields
- Standardizers have native Polars fast path (`_NATIVE_STANDARDIZERS` in standardize.py)
- Matchkey transforms have native Polars fast path (`_try_native_chain` in matchkey.py)
- Clustering uses iterative Union-Find (not recursive) with lazy pair_scores
- Blocking key choice dominates fuzzy performance — coarse keys create huge blocks
- 1M exact dedupe: ~7.8s. 100K fuzzy (name+zip): ~39s via pipeline (was ~100s before parallel + early termination)
- Benchmark note: `analyze_fuzzy.py` calls `find_fuzzy_matches` directly per block (sequential), not `score_blocks_parallel` — times vary 38-55s depending on block size sampling; parallel speedup only visible through `run_dedupe`/CLI
- DBLP-ACM best: 97.2% F1 with multi-pass fuzzy (RapidFuzz), no embeddings needed
- Abt-Buy best: 81.7% F1 with Vertex AI candidates + GPT-4o-mini scorer (~$0.74)
- Abt-Buy zero-shot best: 62.8% F1 with Vertex AI name+desc embeddings (t=0.88)
- Amazon-Google best: 44.0% F1 with Vertex AI + 20-label reranking
- LLM-as-scorer approach: Vertex embeddings generate candidates, GPT-4o-mini scores borderline pairs (0.75-0.95 range), auto-accept pairs above 0.95. Dramatically outperforms fine-tuning and cross-encoder approaches on product matching.
- Cross-encoder Level 3 (300 labels, CPU): 65.5% F1 on Abt-Buy — modest improvement over Vertex baseline, massively outperformed by LLM scorer
- MiniLM fine-tuning Level 2 (300 labels): 58.7% F1 on Abt-Buy — worse than Vertex baseline because MiniLM is a weaker model
- Boost tab logistic regression reranking can hurt on product data where string-distance features are weaker than embeddings. Quality check detects this and recommends --llm-boost instead.
- Previous 84.8% Abt-Buy claim used top-1-per-record evaluation (invalid methodology) — corrected
- Multi-field embedding helps structured data (DBLP-ACM) but not product data (Abt-Buy) — descriptions differ in format
- Scale curve: 8,200 rec/s at 100K records on laptop (fuzzy + exact + golden)
- Active sampling saves ~45% labels vs random but value is in fine-tuning, not threshold learning

## Code Patterns
- Internal columns prefixed with `__` (e.g. `__row_id__`, `__source__`, `__mk_*__`)
- File specs are tuples: `(path, source_name)` or `(path, source_name, column_map)`
- `GoldenMatchConfig.get_matchkeys()` returns matchkeys from either top-level or match_settings
- Matchkey type field: use `mk.type` (not `mk.comparison`) after validation
- Scorer returns `list[tuple[int, int, float]]` — (row_id_a, row_id_b, score)
- `build_clusters` returns `dict[int, dict]` with keys: members, size, oversized, pair_scores, confidence, bottleneck_pair
- `confidence` = 0.4*min_edge + 0.3*avg_edge + 0.3*connectivity; `bottleneck_pair` = weakest link (id_a, id_b)
- `unmerge_record(record_id, clusters)` removes a record from its cluster, re-clusters remaining via stored pair_scores
- `unmerge_cluster(cluster_id, clusters)` shatters a cluster into singletons
- TUI has 6 tabs: Data, Config, Matches, Golden, Boost, Export (key 1-6)
- Boost tab: active learning with y/n/s keyboard labeling, trains LogisticRegression on labeled pairs
- `match_one(record, df, mk)` in `core/match_one.py` — single-record matching primitive for streaming
- `add_to_cluster(record_id, matches, clusters)` — incremental cluster update (join or merge)
- `ANNBlocker.add_to_index(embedding)` / `ANNBlocker.query_one(embedding)` — incremental FAISS ops
- PPRL: `bloom_filter` transform (CLK via SHA-256, configurable ngram/k/size), `dice`/`jaccard` scorers for fuzzy matching on encrypted data
- LLM scorer: `llm_score_pairs()` in `core/llm_scorer.py` — sends borderline pairs to GPT/Claude for yes/no match decisions, used as pipeline step after embedding candidate generation
- Lineage: `core/lineage.py` — `build_lineage` + `save_lineage` saves per-pair field-level explanations to `{run_name}_lineage.json` sidecar. Auto-generated when pipeline writes output.
- MCP `suggest_config` tool: analyze bad merges, identify guilty fields, suggest threshold/weight changes
- REST review queue: `GET /reviews` returns borderline pairs for steward review, `POST /reviews/decide` records approve/reject decisions
- Daemon mode: `watch_daemon()` in `db/watch.py` — adds health endpoint (HTTP /health), PID file, SIGTERM handling to watch mode

## Gotchas
- .docx files can't be read by Read tool — use `python-docx` or zipfile+XML
- Windows drive letter paths (C:\) break `file:source_name` CLI parsing — handle in `_parse_file_source`
- `ignore_errors=True` needed for `pl.read_csv` on files with junk rows
- Textual version 8.x installed (despite `>=1.0` pin) — API is stable
- gcloud CLI sometimes hangs on Windows — try with `timeout 30 gcloud ...` first, fall back to REST API if it hangs. User ADC at `~/AppData/Roaming/gcloud/application_default_credentials.json`
- Vertex AI service account needs `roles/aiplatform.user` for embeddings — grant via `gcloud projects add-iam-policy-binding`. IAM changes take 1-2 minutes to propagate.
- Vertex AI `text-embedding-004` does NOT support fine-tuning — only inference. Use Colab GPU or local CPU for model training.
- `import torch` crashes/hangs on machines without GPU — use `goldenmatch.core.gpu.detect_gpu_mode()` to check before loading
- Polars infers zip/phone as Int64 — explainer/scorer must `str()` values before comparing
- Unicode em dashes (`—`) break on Windows terminals — use ASCII (`-`) in CLI help text
- GitHub Wiki: image paths must use `https://raw.githubusercontent.com/...` URLs, page links omit `.md`
- Textual headless screenshots: `async with app.run_test(size=(W,H)) as pilot: app.save_screenshot('path.svg')`
- PyPI publishing: `source .testing/.env && python -m build && python -m twine upload dist/*`
- `.testing/` folder is gitignored — store credentials, API keys, service account JSON there
- GitHub Wiki repo uses `master` branch, main repo uses `main`
- GitHub Wiki needs `_Sidebar.md` and `_Footer.md` for custom nav/footer
- Rich terminal recording: `Console(record=True)` then `console.export_svg(title='...')`
- PyPI version must be bumped in both `pyproject.toml` and `goldenmatch/__init__.py`
- v0.2.0 is live on PyPI — `pip install goldenmatch` works
- Benchmark evaluation: always use threshold-based pair generation, NOT top-1-per-record (argmax). The latter inflates precision and produces unreproducible numbers.
- Leipzig benchmark datasets live in `tests/benchmarks/datasets/`. Run with `python tests/benchmarks/run_leipzig.py`
- For product matching: use LLM scorer (81.7% F1), not fine-tuning (58.7%) or cross-encoder (65.5%). Fine-tuning a weaker model (MiniLM) can't beat Vertex embeddings.
- For structured data (names, addresses): fuzzy matching alone reaches 97.2% — no embeddings or LLM needed.
- Adding a TUI tab: update `test_tabs_exist` in `tests/test_tui.py` — asserts exact tab count (currently 6)
- OpenAI API key: set `OPENAI_API_KEY` env var. Used by LLM scorer and LLM boost. Key stored in `.testing/.env`
