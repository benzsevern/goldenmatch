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
- 377+ tests, run in ~9s
- Fixtures in `tests/conftest.py`: `sample_csv`, `sample_csv_b`, `sample_parquet`
- TUI tests use `pytest-asyncio` with `app.run_test()` pilot
- Benchmark scripts in `tests/bench_1m.py`, `tests/analyze_results.py` (not part of test suite)
- Synthetic test data generator: `tests/generate_synthetic.py`

## Architecture
- Pipeline: ingest → column_map → auto_fix → validate → standardize → matchkeys → block → score → cluster → golden → output
- `goldenmatch/core/` — pipeline modules (no Textual dependency)
- `goldenmatch/tui/` — Textual TUI + MatchEngine (engine.py has no Textual dependency)
- `goldenmatch/cli/` — Typer CLI commands
- Config: Pydantic models in `config/schemas.py`, YAML loading in `config/loader.py`
- `config/loader.py` normalizes golden_rules and standardization sections from flat YAML

## Performance
- Exact matching uses Polars self-join (not Python group_by + combinations)
- Fuzzy matching uses `rapidfuzz.process.cdist` for vectorized NxN scoring
- Standardizers have native Polars fast path (`_NATIVE_STANDARDIZERS` in standardize.py)
- Matchkey transforms have native Polars fast path (`_try_native_chain` in matchkey.py)
- Clustering uses iterative Union-Find (not recursive) with lazy pair_scores
- Blocking key choice dominates fuzzy performance — coarse keys create huge blocks
- 1M exact dedupe: ~12s. 100K fuzzy (name+zip): ~100s

## Code Patterns
- Internal columns prefixed with `__` (e.g. `__row_id__`, `__source__`, `__mk_*__`)
- File specs are tuples: `(path, source_name)` or `(path, source_name, column_map)`
- `GoldenMatchConfig.get_matchkeys()` returns matchkeys from either top-level or match_settings
- Matchkey type field: use `mk.type` (not `mk.comparison`) after validation
- Scorer returns `list[tuple[int, int, float]]` — (row_id_a, row_id_b, score)
- `build_clusters` returns `dict[int, dict]` with keys: members, size, oversized, pair_scores

## Gotchas
- .docx files can't be read by Read tool — use `python-docx` or zipfile+XML
- Windows drive letter paths (C:\) break `file:source_name` CLI parsing — handle in `_parse_file_source`
- `ignore_errors=True` needed for `pl.read_csv` on files with junk rows
- Textual version 8.x installed (despite `>=1.0` pin) — API is stable
