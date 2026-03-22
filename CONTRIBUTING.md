# Contributing to GoldenMatch

Thanks for your interest in contributing! Here's how to get started.

## Quick Start

```bash
git clone https://github.com/benzsevern/goldenmatch.git
cd goldenmatch
pip install -e ".[dev]"
pytest --tb=short
```

Or open a [Codespace](https://github.com/benzsevern/goldenmatch/codespaces) for a pre-configured dev environment.

## Ways to Contribute

### Report Bugs

Use the [bug report template](https://github.com/benzsevern/goldenmatch/issues/new?template=bug_report.yml). Include your version, config, and the full error output.

### Suggest Features

Use the [feature request template](https://github.com/benzsevern/goldenmatch/issues/new?template=feature_request.yml) or post in [Discussions > Ideas](https://github.com/benzsevern/goldenmatch/discussions/categories/ideas).

### Add a Domain Pack

Domain packs are the easiest way to contribute. Create a YAML file in `goldenmatch/domains/`:

```yaml
name: your_domain
signals: ["keyword1", "keyword2"]
identifier_patterns:
  id_name: '\b(regex_pattern)\b'
brand_patterns:
  - Brand1
  - Brand2
attribute_patterns:
  attr_name: '(\d+)\s*unit\b'
stop_words:
  - the
  - a
normalization:
  lowercase: true
```

See existing packs in `goldenmatch/domains/` for examples. Add tests in `tests/test_domain_packs.py`.

### Fix Bugs or Add Features

1. Fork the repo
2. Create a branch (`git checkout -b fix/my-fix`)
3. Make your changes
4. Run tests (`pytest --tb=short`) -- all 855+ tests must pass
5. Submit a PR

## Development Guidelines

### Code Style

- Python 3.11+, type hints encouraged
- `ruff` for linting (configured in `pyproject.toml`)
- Line length: 100 characters
- Internal columns prefixed with `__` (e.g., `__row_id__`, `__source__`)

### Testing

- `pytest --tb=short` from project root
- New features need tests -- aim for the same patterns in existing test files
- DB tests (`test_db.py`, `test_reconcile.py`) need PostgreSQL -- skip with `--ignore` if not available

### Architecture

- `goldenmatch/core/` -- pipeline modules (no UI dependency)
- `goldenmatch/cli/` -- Typer CLI commands
- `goldenmatch/tui/` -- Textual TUI (depends on core, not the other way around)
- `goldenmatch/db/` -- database operations

Keep core modules free of UI or database dependencies.

### Commit Messages

Use conventional commits:

```
feat: add automotive domain pack
fix: handle empty blocking keys in learned blocking
docs: update benchmarks with v0.3.1 results
```

## Questions?

- [Discussions Q&A](https://github.com/benzsevern/goldenmatch/discussions/categories/q-a) for help
- [Discussions Ideas](https://github.com/benzsevern/goldenmatch/discussions/categories/ideas) for feature brainstorming
