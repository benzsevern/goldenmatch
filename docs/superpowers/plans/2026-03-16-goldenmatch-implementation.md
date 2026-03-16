# GoldenMatch Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool for high-performance record deduplication and list matching with configurable matchkeys, golden record creation, and per-field confidence scoring.

**Architecture:** Pipeline-based (ingest -> transform -> block -> compare -> threshold -> cluster -> golden -> output). Polars for vectorized data ops, rapidfuzz for fuzzy matching, Typer+Rich for CLI, Pydantic for config validation.

**Tech Stack:** Python 3.11+, Polars, Typer, Rich, PyYAML, Pydantic, rapidfuzz, jellyfish, pytest

**Spec:** `docs/superpowers/specs/2026-03-16-goldenmatch-design.md`

---

## Chunk 1: Project Scaffolding & Config Schemas

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `goldenmatch/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "goldenmatch"
version = "0.1.0"
description = "High-performance CLI for record deduplication, list matching, and golden record creation"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
authors = [{ name = "Ben Severn", email = "benzsevern@gmail.com" }]

dependencies = [
    "polars>=1.0",
    "typer>=0.12",
    "rich>=13.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "rapidfuzz>=3.0",
    "jellyfish>=1.0",
    "openpyxl>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]

[project.scripts]
goldenmatch = "goldenmatch.cli.main:app"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

```python
# goldenmatch/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create test conftest with shared fixtures**

```python
# tests/conftest.py
import tempfile
from pathlib import Path
import polars as pl
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    path = tmp_path / "sample.csv"
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "first_name": ["John", "john", "Jane", "JOHN", "Bob"],
        "last_name": ["Smith", "Smith", "Doe", "Smyth", "Jones"],
        "email": ["john@example.com", "john@example.com", "jane@test.com", "john.s@example.com", "bob@test.com"],
        "zip": ["19382", "19382", "10001", "19383", "90210"],
        "phone": ["267-555-1234", "267-555-1234", "212-555-9999", "267-555-1235", "310-555-0000"],
    })
    df.write_csv(path)
    return path


@pytest.fixture
def sample_csv_b(tmp_path) -> Path:
    path = tmp_path / "sample_b.csv"
    df = pl.DataFrame({
        "id": [101, 102, 103],
        "first_name": ["John", "Alice", "Jane"],
        "last_name": ["Smith", "Wonder", "Doe"],
        "email": ["jsmith@work.com", "alice@test.com", "jane@test.com"],
        "zip": ["19382", "30301", "10001"],
        "phone": ["267-555-1234", "404-555-1111", "212-555-9999"],
    })
    df.write_csv(path)
    return path


@pytest.fixture
def sample_parquet(tmp_path) -> Path:
    path = tmp_path / "sample.parquet"
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "first_name": ["John", "Jane", "Bob"],
        "last_name": ["Smith", "Doe", "Jones"],
        "email": ["john@example.com", "jane@test.com", "bob@test.com"],
        "zip": ["19382", "10001", "90210"],
    })
    df.write_parquet(path)
    return path
```

- [ ] **Step 4: Create tests/__init__.py**

```python
# tests/__init__.py
```

- [ ] **Step 5: Install in dev mode and verify**

Run: `cd D:/show_case/goldenmatch && pip install -e ".[dev]"`
Expected: Successful install

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml goldenmatch/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: project scaffolding with dependencies and test fixtures"
```

---

### Task 2: Pydantic Config Schemas

**Files:**
- Create: `goldenmatch/config/__init__.py`
- Create: `goldenmatch/config/schemas.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for config schemas**

```python
# tests/test_config.py
import pytest
from goldenmatch.config.schemas import (
    FieldTransform,
    MatchkeyField,
    MatchkeyConfig,
    BlockingKeyConfig,
    BlockingConfig,
    GoldenFieldRule,
    GoldenRulesConfig,
    InputFileConfig,
    InputConfig,
    OutputConfig,
    MatchSettingsConfig,
    GoldenMatchConfig,
)


class TestMatchkeyField:
    def test_exact_field_no_scorer(self):
        f = MatchkeyField(column="last_name", transforms=["lowercase", "strip"])
        assert f.column == "last_name"
        assert f.scorer is None
        assert f.weight is None

    def test_fuzzy_field_with_scorer(self):
        f = MatchkeyField(column="full_name", transforms=["lowercase"], scorer="jaro_winkler", weight=0.6)
        assert f.scorer == "jaro_winkler"
        assert f.weight == 0.6

    def test_invalid_scorer_rejected(self):
        with pytest.raises(ValueError):
            MatchkeyField(column="x", transforms=[], scorer="invalid_scorer")

    def test_invalid_transform_rejected(self):
        with pytest.raises(ValueError):
            MatchkeyField(column="x", transforms=["not_a_transform"])


class TestMatchkeyConfig:
    def test_exact_matchkey(self):
        mk = MatchkeyConfig(
            name="test",
            fields=[MatchkeyField(column="zip", transforms=["strip"])],
            comparison="exact",
        )
        assert mk.threshold is None

    def test_weighted_matchkey_requires_threshold(self):
        with pytest.raises(ValueError):
            MatchkeyConfig(
                name="test",
                fields=[MatchkeyField(column="zip", transforms=["strip"], scorer="exact", weight=1.0)],
                comparison="weighted",
                # missing threshold
            )

    def test_weighted_fields_require_scorer_and_weight(self):
        with pytest.raises(ValueError):
            MatchkeyConfig(
                name="test",
                fields=[MatchkeyField(column="zip", transforms=["strip"])],  # no scorer/weight
                comparison="weighted",
                threshold=0.85,
            )


class TestGoldenRulesConfig:
    def test_valid_strategies(self):
        rule = GoldenFieldRule(strategy="most_complete")
        assert rule.strategy == "most_complete"

    def test_most_recent_requires_date_column(self):
        with pytest.raises(ValueError):
            GoldenFieldRule(strategy="most_recent")  # no date_column

    def test_source_priority_requires_list(self):
        with pytest.raises(ValueError):
            GoldenFieldRule(strategy="source_priority")  # no source_priority list


class TestGoldenMatchConfig:
    def test_minimal_config(self):
        cfg = GoldenMatchConfig(
            matchkeys=[
                MatchkeyConfig(
                    name="simple",
                    fields=[MatchkeyField(column="email", transforms=["lowercase"])],
                    comparison="exact",
                )
            ]
        )
        assert cfg.output.format == "csv"
        assert cfg.output.directory == "."

    def test_fuzzy_without_blocking_raises(self):
        with pytest.raises(ValueError, match="[Bb]locking"):
            GoldenMatchConfig(
                matchkeys=[
                    MatchkeyConfig(
                        name="fuzzy",
                        fields=[MatchkeyField(column="name", transforms=["lowercase"], scorer="jaro_winkler", weight=1.0)],
                        comparison="weighted",
                        threshold=0.85,
                    )
                ],
                # no blocking config
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_config.py -v`
Expected: FAIL — cannot import schemas

- [ ] **Step 3: Implement config schemas**

```python
# goldenmatch/config/__init__.py
```

```python
# goldenmatch/config/schemas.py
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, field_validator, model_validator

VALID_TRANSFORMS = {
    "lowercase", "uppercase", "strip", "strip_all", "soundex", "metaphone",
    "digits_only", "alpha_only", "normalize_whitespace",
}
VALID_TRANSFORM_PREFIXES = {"substring:"}

VALID_SCORERS = {"exact", "jaro_winkler", "levenshtein", "token_sort", "soundex_match"}

VALID_STRATEGIES = {"most_recent", "source_priority", "most_complete", "majority_vote", "first_non_null"}


def _validate_transform(t: str) -> str:
    if t in VALID_TRANSFORMS:
        return t
    for prefix in VALID_TRANSFORM_PREFIXES:
        if t.startswith(prefix):
            return t
    raise ValueError(f"Invalid transform: {t!r}. Valid: {sorted(VALID_TRANSFORMS)} or substring:start:end")


class FieldTransform(BaseModel):
    column: str
    transforms: list[str]

    @field_validator("transforms")
    @classmethod
    def check_transforms(cls, v: list[str]) -> list[str]:
        return [_validate_transform(t) for t in v]


class MatchkeyField(BaseModel):
    column: str
    transforms: list[str] = []
    scorer: str | None = None
    weight: float | None = None

    @field_validator("transforms")
    @classmethod
    def check_transforms(cls, v: list[str]) -> list[str]:
        return [_validate_transform(t) for t in v]

    @field_validator("scorer")
    @classmethod
    def check_scorer(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_SCORERS:
            raise ValueError(f"Invalid scorer: {v!r}. Valid: {sorted(VALID_SCORERS)}")
        return v


class MatchkeyConfig(BaseModel):
    name: str
    description: str = ""
    fields: list[MatchkeyField]
    comparison: Literal["exact", "weighted"]
    threshold: float | None = None

    @model_validator(mode="after")
    def check_weighted_requirements(self) -> MatchkeyConfig:
        if self.comparison == "weighted":
            if self.threshold is None:
                raise ValueError("Weighted matchkeys require a threshold")
            for f in self.fields:
                if f.scorer is None or f.weight is None:
                    raise ValueError(
                        f"Field {f.column!r} in weighted matchkey {self.name!r} "
                        "requires both 'scorer' and 'weight'"
                    )
        return self


class BlockingKeyConfig(BaseModel):
    key_fields: list[FieldTransform]


class BlockingConfig(BaseModel):
    max_block_size: int = 5000
    skip_oversized: bool = False
    keys: list[BlockingKeyConfig]


class GoldenFieldRule(BaseModel):
    strategy: str
    date_column: str | None = None
    source_priority: list[str] | None = None

    @field_validator("strategy")
    @classmethod
    def check_strategy(cls, v: str) -> str:
        if v not in VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy: {v!r}. Valid: {sorted(VALID_STRATEGIES)}")
        return v

    @model_validator(mode="after")
    def check_strategy_requirements(self) -> GoldenFieldRule:
        if self.strategy == "most_recent" and self.date_column is None:
            raise ValueError("Strategy 'most_recent' requires 'date_column'")
        if self.strategy == "source_priority" and not self.source_priority:
            raise ValueError("Strategy 'source_priority' requires 'source_priority' list")
        return self


class GoldenRulesConfig(BaseModel):
    max_cluster_size: int = 100
    default: GoldenFieldRule = GoldenFieldRule(strategy="most_complete")
    field_rules: dict[str, GoldenFieldRule] = {}


class InputFileConfig(BaseModel):
    path: str
    source_name: str | None = None
    delimiter: str = ","
    encoding: str = "utf-8"
    sheet: str | None = None


class InputConfig(BaseModel):
    files: list[InputFileConfig] = []


class OutputConfig(BaseModel):
    format: Literal["csv", "xlsx", "parquet"] = "csv"
    directory: str = "."
    run_name: str | None = None


class MatchSettingsConfig(BaseModel):
    match_mode: Literal["best", "all", "none"] = "best"


class GoldenMatchConfig(BaseModel):
    input: InputConfig = InputConfig()
    mode: Literal["dedupe", "match"] | None = None
    matchkeys: list[MatchkeyConfig]
    blocking: BlockingConfig | None = None
    golden_rules: GoldenRulesConfig = GoldenRulesConfig()
    output: OutputConfig = OutputConfig()
    match_settings: MatchSettingsConfig = MatchSettingsConfig()

    @model_validator(mode="after")
    def check_blocking_for_fuzzy(self) -> GoldenMatchConfig:
        has_fuzzy = any(mk.comparison == "weighted" for mk in self.matchkeys)
        if has_fuzzy and self.blocking is None:
            raise ValueError(
                "Blocking config is required when using fuzzy (weighted) matchkeys. "
                "Without blocking, fuzzy comparison is O(n^2) and infeasible for large datasets. "
                "Add a 'blocking' section to your config."
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/config/ tests/test_config.py
git commit -m "feat: Pydantic config schemas with validation"
```

---

### Task 3: Config Loader (YAML -> Pydantic)

**Files:**
- Create: `goldenmatch/config/loader.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for YAML loading**

Append to `tests/test_config.py`:

```python
from goldenmatch.config.loader import load_config
from pathlib import Path
import yaml


class TestLoadConfig:
    def test_load_minimal_yaml(self, tmp_path):
        cfg_path = tmp_path / "goldenmatch.yaml"
        cfg_path.write_text(yaml.dump({
            "matchkeys": [{
                "name": "email_exact",
                "fields": [{"column": "email", "transforms": ["lowercase"]}],
                "comparison": "exact",
            }]
        }))
        cfg = load_config(cfg_path)
        assert len(cfg.matchkeys) == 1
        assert cfg.matchkeys[0].name == "email_exact"

    def test_load_full_yaml(self, tmp_path):
        cfg_path = tmp_path / "goldenmatch.yaml"
        cfg_path.write_text(yaml.dump({
            "matchkeys": [{
                "name": "fuzzy_name",
                "fields": [{"column": "name", "transforms": ["lowercase"], "scorer": "jaro_winkler", "weight": 1.0}],
                "comparison": "weighted",
                "threshold": 0.85,
            }],
            "blocking": {
                "max_block_size": 3000,
                "keys": [{"key_fields": [{"column": "zip", "transforms": ["strip"]}]}],
            },
            "golden_rules": {
                "max_cluster_size": 50,
                "default": {"strategy": "most_complete"},
                "field_rules": {
                    "email": {"strategy": "most_recent", "date_column": "updated_at"},
                },
            },
            "output": {"format": "parquet", "directory": "./out"},
        }))
        cfg = load_config(cfg_path)
        assert cfg.blocking.max_block_size == 3000
        assert cfg.golden_rules.field_rules["email"].strategy == "most_recent"
        assert cfg.output.format == "parquet"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("matchkeys: not_a_list")
        with pytest.raises((ValueError, Exception)):
            load_config(cfg_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_config.py::TestLoadConfig -v`
Expected: FAIL

- [ ] **Step 3: Implement loader**

```python
# goldenmatch/config/loader.py
from __future__ import annotations
from pathlib import Path
import yaml
from pydantic import ValidationError
from goldenmatch.config.schemas import (
    GoldenMatchConfig,
    GoldenRulesConfig,
    GoldenFieldRule,
)


def load_config(path: Path | str) -> GoldenMatchConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    # Normalize golden_rules: move non-special keys into field_rules
    if "golden_rules" in raw and isinstance(raw["golden_rules"], dict):
        gr = raw["golden_rules"]
        special_keys = {"max_cluster_size", "default", "field_rules"}
        field_rules = gr.pop("field_rules", {})
        for key in list(gr.keys()):
            if key not in special_keys:
                field_rules[key] = gr.pop(key)
        gr["field_rules"] = field_rules

    return GoldenMatchConfig(**raw)
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/config/loader.py tests/test_config.py
git commit -m "feat: YAML config loader with golden_rules normalization"
```

---

## Chunk 2: Transforms & Ingest

### Task 4: Field Transforms

**Files:**
- Create: `goldenmatch/utils/__init__.py`
- Create: `goldenmatch/utils/transforms.py`
- Create: `tests/test_transforms.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transforms.py
import pytest
from goldenmatch.utils.transforms import apply_transform, apply_transforms


class TestIndividualTransforms:
    def test_lowercase(self):
        assert apply_transform("SMITH", "lowercase") == "smith"

    def test_uppercase(self):
        assert apply_transform("smith", "uppercase") == "SMITH"

    def test_strip(self):
        assert apply_transform("  smith  ", "strip") == "smith"

    def test_strip_all(self):
        assert apply_transform("s m i t h", "strip_all") == "smith"

    def test_substring(self):
        assert apply_transform("smith", "substring:0:3") == "smi"

    def test_substring_beyond_length(self):
        assert apply_transform("ab", "substring:0:5") == "ab"

    def test_soundex(self):
        assert apply_transform("Smith", "soundex") == "S530"

    def test_metaphone(self):
        result = apply_transform("Smith", "metaphone")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_digits_only(self):
        assert apply_transform("555-123-4567", "digits_only") == "5551234567"

    def test_alpha_only(self):
        assert apply_transform("O'Brien-Jr.", "alpha_only") == "OBrienJr"

    def test_normalize_whitespace(self):
        assert apply_transform("John   Smith", "normalize_whitespace") == "John Smith"

    def test_null_passthrough(self):
        assert apply_transform(None, "lowercase") is None

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError):
            apply_transform("test", "invalid_transform")


class TestApplyTransforms:
    def test_chain(self):
        result = apply_transforms("  SMITH  ", ["strip", "lowercase", "substring:0:3"])
        assert result == "smi"

    def test_empty_list(self):
        assert apply_transforms("hello", []) == "hello"

    def test_null_chain(self):
        assert apply_transforms(None, ["lowercase", "strip"]) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_transforms.py -v`
Expected: FAIL

- [ ] **Step 3: Implement transforms**

```python
# goldenmatch/utils/__init__.py
```

```python
# goldenmatch/utils/transforms.py
from __future__ import annotations
import re
import jellyfish


def apply_transform(value: str | None, transform: str) -> str | None:
    if value is None:
        return None

    value = str(value)

    if transform == "lowercase":
        return value.lower()
    elif transform == "uppercase":
        return value.upper()
    elif transform == "strip":
        return value.strip()
    elif transform == "strip_all":
        return re.sub(r"\s+", "", value)
    elif transform.startswith("substring:"):
        parts = transform.split(":")
        start, end = int(parts[1]), int(parts[2])
        return value[start:end]
    elif transform == "soundex":
        return jellyfish.soundex(value)
    elif transform == "metaphone":
        return jellyfish.metaphone(value)
    elif transform == "digits_only":
        return re.sub(r"\D", "", value)
    elif transform == "alpha_only":
        return re.sub(r"[^a-zA-Z]", "", value)
    elif transform == "normalize_whitespace":
        return re.sub(r"\s+", " ", value).strip()
    else:
        raise ValueError(f"Unknown transform: {transform!r}")


def apply_transforms(value: str | None, transforms: list[str]) -> str | None:
    for t in transforms:
        value = apply_transform(value, t)
    return value
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_transforms.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/utils/ tests/test_transforms.py
git commit -m "feat: field transforms (lowercase, soundex, substring, etc.)"
```

---

### Task 5: File Ingest

**Files:**
- Create: `goldenmatch/core/__init__.py`
- Create: `goldenmatch/core/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ingest.py
import pytest
import polars as pl
from pathlib import Path
from goldenmatch.core.ingest import load_file, load_files, validate_columns


class TestLoadFile:
    def test_load_csv(self, sample_csv):
        df = load_file(sample_csv)
        assert isinstance(df, pl.LazyFrame)
        assert "first_name" in df.collect_schema().names()

    def test_load_parquet(self, sample_parquet):
        df = load_file(sample_parquet)
        assert isinstance(df, pl.LazyFrame)
        assert df.collect().height == 3

    def test_load_xlsx(self, tmp_path):
        path = tmp_path / "test.xlsx"
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_excel(path)
        df = load_file(path)
        assert isinstance(df, pl.LazyFrame)
        assert df.collect().height == 2

    def test_load_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(path)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_file(tmp_path / "missing.csv")


class TestLoadFiles:
    def test_load_multiple_with_source(self, sample_csv, sample_csv_b):
        frames = load_files([
            (sample_csv, "source_a"),
            (sample_csv_b, "source_b"),
        ])
        assert len(frames) == 2
        combined = pl.concat([f.collect() for f in frames])
        assert "__source__" in combined.columns
        assert set(combined["__source__"].unique().to_list()) == {"source_a", "source_b"}


class TestValidateColumns:
    def test_valid_columns(self, sample_csv):
        df = load_file(sample_csv)
        validate_columns(df, ["first_name", "last_name", "zip"])

    def test_missing_column_raises(self, sample_csv):
        df = load_file(sample_csv)
        with pytest.raises(ValueError, match="nonexistent"):
            validate_columns(df, ["first_name", "nonexistent"])
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_ingest.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ingest**

```python
# goldenmatch/core/__init__.py
```

```python
# goldenmatch/core/ingest.py
from __future__ import annotations
from pathlib import Path
import polars as pl


def load_file(
    path: Path | str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    sheet: str | None = None,
) -> pl.LazyFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(path, separator=delimiter, encoding=encoding)
    elif suffix == ".parquet":
        return pl.scan_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        df = pl.read_excel(path, sheet_name=sheet or 0)
        return df.lazy()
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Use .csv, .parquet, or .xlsx")


def load_files(
    file_specs: list[tuple[Path | str, str]],
) -> list[pl.LazyFrame]:
    frames = []
    for path, source_name in file_specs:
        lf = load_file(path)
        lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
        frames.append(lf)
    return frames


def validate_columns(lf: pl.LazyFrame, required: list[str]) -> None:
    schema_names = lf.collect_schema().names()
    missing = [c for c in required if c not in schema_names]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available: {schema_names}"
        )
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_ingest.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/ tests/test_ingest.py
git commit -m "feat: file ingest for CSV, Parquet, and Excel with source tagging"
```

---

## Chunk 3: Matchkey Builder & Blocker

### Task 6: Matchkey Builder

**Files:**
- Create: `goldenmatch/core/matchkey.py`
- Create: `tests/test_matchkey.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_matchkey.py
import pytest
import polars as pl
from goldenmatch.core.matchkey import build_matchkey_expr, compute_matchkeys
from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField


class TestBuildMatchkeyExpr:
    def test_single_field_exact(self):
        mk = MatchkeyConfig(
            name="email_key",
            fields=[MatchkeyField(column="email", transforms=["lowercase", "strip"])],
            comparison="exact",
        )
        expr = build_matchkey_expr(mk)
        df = pl.DataFrame({"email": ["  JOHN@TEST.COM  ", "john@test.com", "JANE@TEST.COM"]})
        result = df.with_columns(expr)
        keys = result["__mk_email_key__"].to_list()
        assert keys[0] == keys[1]  # normalized to same
        assert keys[0] != keys[2]

    def test_multi_field_concatenation(self):
        mk = MatchkeyConfig(
            name="name_zip",
            fields=[
                MatchkeyField(column="last_name", transforms=["lowercase", "substring:0:3"]),
                MatchkeyField(column="zip", transforms=["substring:0:3"]),
            ],
            comparison="exact",
        )
        expr = build_matchkey_expr(mk)
        df = pl.DataFrame({
            "last_name": ["Smith", "SMITH", "Jones"],
            "zip": ["19382", "19399", "19382"],
        })
        result = df.with_columns(expr)
        keys = result["__mk_name_zip__"].to_list()
        assert keys[0] == keys[1]  # smi||193 == smi||193
        assert keys[0] != keys[2]


class TestComputeMatchkeys:
    def test_adds_matchkey_columns(self):
        mks = [
            MatchkeyConfig(
                name="email_key",
                fields=[MatchkeyField(column="email", transforms=["lowercase"])],
                comparison="exact",
            ),
            MatchkeyConfig(
                name="zip_key",
                fields=[MatchkeyField(column="zip", transforms=["substring:0:3"])],
                comparison="exact",
            ),
        ]
        df = pl.DataFrame({"email": ["A@B.COM"], "zip": ["19382"]}).lazy()
        result = compute_matchkeys(df, mks).collect()
        assert "__mk_email_key__" in result.columns
        assert "__mk_zip_key__" in result.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_matchkey.py -v`
Expected: FAIL

- [ ] **Step 3: Implement matchkey builder**

```python
# goldenmatch/core/matchkey.py
from __future__ import annotations
import polars as pl
from goldenmatch.config.schemas import MatchkeyConfig
from goldenmatch.utils.transforms import apply_transforms


def _transform_expr(column: str, transforms: list[str]) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).map_elements(
        lambda val: apply_transforms(val, transforms),
        return_dtype=pl.Utf8,
    )


def build_matchkey_expr(mk: MatchkeyConfig) -> pl.Expr:
    if mk.comparison == "exact":
        parts = [_transform_expr(f.column, f.transforms) for f in mk.fields]
        if len(parts) == 1:
            return parts[0].alias(f"__mk_{mk.name}__")
        return pl.concat_str(parts, separator="||").alias(f"__mk_{mk.name}__")
    else:
        # Weighted matchkeys don't produce a single key column;
        # they are handled in the scorer. Return a placeholder.
        return pl.lit(None).alias(f"__mk_{mk.name}__")


def compute_matchkeys(lf: pl.LazyFrame, matchkeys: list[MatchkeyConfig]) -> pl.LazyFrame:
    exprs = []
    for mk in matchkeys:
        if mk.comparison == "exact":
            exprs.append(build_matchkey_expr(mk))
    if exprs:
        lf = lf.with_columns(exprs)
    return lf
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_matchkey.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/matchkey.py tests/test_matchkey.py
git commit -m "feat: matchkey builder with transform chaining and concatenation"
```

---

### Task 7: Blocker

**Files:**
- Create: `goldenmatch/core/blocker.py`
- Create: `tests/test_blocker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_blocker.py
import pytest
import polars as pl
from goldenmatch.core.blocker import build_blocks, BlockResult
from goldenmatch.config.schemas import BlockingConfig, BlockingKeyConfig, FieldTransform


@pytest.fixture
def blocking_config():
    return BlockingConfig(
        max_block_size=100,
        keys=[
            BlockingKeyConfig(key_fields=[
                FieldTransform(column="last_name", transforms=["lowercase", "substring:0:3"]),
            ])
        ],
    )


class TestBuildBlocks:
    def test_groups_by_block_key(self, blocking_config):
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "last_name": ["Smith", "SMITH", "Smyth", "Jones"],
            "zip": ["19382", "19382", "19382", "10001"],
        }).lazy()
        result = build_blocks(df, blocking_config)
        assert isinstance(result, list)
        # "smi" block should have 3 records, "jon" block should have 1
        sizes = sorted([r.df.collect().height for r in result])
        assert sizes == [1, 3]

    def test_oversized_block_warning(self, caplog):
        cfg = BlockingConfig(
            max_block_size=2,
            skip_oversized=False,
            keys=[BlockingKeyConfig(key_fields=[
                FieldTransform(column="key", transforms=[]),
            ])],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "key": ["A", "A", "A", "B"],
        }).lazy()
        import logging
        with caplog.at_level(logging.WARNING):
            result = build_blocks(df, cfg)
        assert any("oversized" in r.message.lower() or "exceeds" in r.message.lower() for r in caplog.records)

    def test_skip_oversized(self):
        cfg = BlockingConfig(
            max_block_size=2,
            skip_oversized=True,
            keys=[BlockingKeyConfig(key_fields=[
                FieldTransform(column="key", transforms=[]),
            ])],
        )
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "key": ["A", "A", "A", "B"],
        }).lazy()
        result = build_blocks(df, cfg)
        # Only the "B" block (size 1) should survive
        assert len(result) == 1
        assert result[0].df.collect().height == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_blocker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement blocker**

```python
# goldenmatch/core/blocker.py
from __future__ import annotations
import logging
from dataclasses import dataclass
import polars as pl
from goldenmatch.config.schemas import BlockingConfig
from goldenmatch.utils.transforms import apply_transforms

logger = logging.getLogger(__name__)


@dataclass
class BlockResult:
    block_key: str
    df: pl.LazyFrame


def _build_block_key_expr(key_config) -> pl.Expr:
    parts = []
    for ft in key_config.key_fields:
        parts.append(
            pl.col(ft.column).cast(pl.Utf8).map_elements(
                lambda val, transforms=ft.transforms: apply_transforms(val, transforms),
                return_dtype=pl.Utf8,
            )
        )
    if len(parts) == 1:
        return parts[0]
    return pl.concat_str(parts, separator="||")


def build_blocks(lf: pl.LazyFrame, config: BlockingConfig) -> list[BlockResult]:
    results = []
    seen_pairs: set[tuple] = set()

    for key_config in config.keys:
        block_expr = _build_block_key_expr(key_config).alias("__block_key__")
        blocked = lf.with_columns(block_expr).collect()

        for block_key, group in blocked.group_by("__block_key__"):
            key_str = str(block_key[0]) if isinstance(block_key, tuple) else str(block_key)
            size = group.height

            if size < 2:
                continue

            if size > config.max_block_size:
                if config.skip_oversized:
                    logger.warning(
                        f"Block {key_str!r} has {size} records (exceeds max_block_size={config.max_block_size}). Skipping."
                    )
                    continue
                else:
                    logger.warning(
                        f"Block {key_str!r} has {size} records (exceeds max_block_size={config.max_block_size}). Processing anyway."
                    )

            block_df = group.drop("__block_key__").lazy()
            results.append(BlockResult(block_key=key_str, df=block_df))

    return results
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_blocker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/blocker.py tests/test_blocker.py
git commit -m "feat: blocker with configurable max block size and skip-oversized"
```

---

## Chunk 4: Scorer & Cluster

### Task 8: Scorer

**Files:**
- Create: `goldenmatch/core/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scorer.py
import pytest
import polars as pl
from goldenmatch.core.scorer import (
    score_field,
    score_pair,
    find_exact_matches,
    find_fuzzy_matches,
)
from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField


class TestScoreField:
    def test_exact_match(self):
        assert score_field("hello", "hello", "exact") == 1.0

    def test_exact_mismatch(self):
        assert score_field("hello", "world", "exact") == 0.0

    def test_jaro_winkler_similar(self):
        score = score_field("smith", "smyth", "jaro_winkler")
        assert 0.7 < score < 1.0

    def test_jaro_winkler_identical(self):
        assert score_field("smith", "smith", "jaro_winkler") == 1.0

    def test_levenshtein_similar(self):
        score = score_field("smith", "smyth", "levenshtein")
        assert 0.5 < score < 1.0

    def test_token_sort(self):
        score = score_field("john smith", "smith john", "token_sort")
        assert score == 1.0

    def test_soundex_match(self):
        score = score_field("Smith", "Smyth", "soundex_match")
        assert score == 1.0  # both have soundex S530

    def test_null_returns_none(self):
        assert score_field(None, "hello", "exact") is None
        assert score_field("hello", None, "exact") is None


class TestScorePair:
    def test_weighted_score(self):
        fields = [
            MatchkeyField(column="name", transforms=[], scorer="exact", weight=0.6),
            MatchkeyField(column="email", transforms=[], scorer="exact", weight=0.4),
        ]
        row_a = {"name": "John", "email": "john@test.com"}
        row_b = {"name": "John", "email": "john@test.com"}
        score = score_pair(row_a, row_b, fields)
        assert score == 1.0

    def test_partial_match(self):
        fields = [
            MatchkeyField(column="name", transforms=[], scorer="exact", weight=0.6),
            MatchkeyField(column="email", transforms=[], scorer="exact", weight=0.4),
        ]
        row_a = {"name": "John", "email": "john@test.com"}
        row_b = {"name": "John", "email": "different@test.com"}
        score = score_pair(row_a, row_b, fields)
        assert score == pytest.approx(0.6)  # 0.6*1.0 / (0.6+0.4)

    def test_null_excluded_from_score(self):
        fields = [
            MatchkeyField(column="name", transforms=[], scorer="exact", weight=0.6),
            MatchkeyField(column="email", transforms=[], scorer="exact", weight=0.4),
        ]
        row_a = {"name": "John", "email": None}
        row_b = {"name": "John", "email": "john@test.com"}
        score = score_pair(row_a, row_b, fields)
        assert score == 1.0  # only name compared, perfect match


class TestFindExactMatches:
    def test_finds_duplicates(self):
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2, 3],
            "__mk_email__": ["a@b.com", "a@b.com", "c@d.com", "c@d.com"],
        }).lazy()
        mk = MatchkeyConfig(
            name="email",
            fields=[MatchkeyField(column="email", transforms=["lowercase"])],
            comparison="exact",
        )
        pairs = find_exact_matches(df, mk)
        assert len(pairs) == 2  # (0,1) and (2,3)
        assert all(score == 1.0 for _, _, score in pairs)

    def test_no_duplicates(self):
        df = pl.DataFrame({
            "__row_id__": [0, 1, 2],
            "__mk_test__": ["a", "b", "c"],
        }).lazy()
        mk = MatchkeyConfig(
            name="test",
            fields=[MatchkeyField(column="x", transforms=[])],
            comparison="exact",
        )
        pairs = find_exact_matches(df, mk)
        assert len(pairs) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scorer**

```python
# goldenmatch/core/scorer.py
from __future__ import annotations
from itertools import combinations
import polars as pl
import rapidfuzz.fuzz as fuzz
import jellyfish
from goldenmatch.config.schemas import MatchkeyConfig, MatchkeyField
from goldenmatch.utils.transforms import apply_transforms


def score_field(val_a: str | None, val_b: str | None, scorer: str) -> float | None:
    if val_a is None or val_b is None:
        return None

    val_a, val_b = str(val_a), str(val_b)

    if scorer == "exact":
        return 1.0 if val_a == val_b else 0.0
    elif scorer == "jaro_winkler":
        return rapidfuzz_jaro_winkler(val_a, val_b)
    elif scorer == "levenshtein":
        return rapidfuzz_normalized_levenshtein(val_a, val_b)
    elif scorer == "token_sort":
        return fuzz.token_sort_ratio(val_a, val_b) / 100.0
    elif scorer == "soundex_match":
        return 1.0 if jellyfish.soundex(val_a) == jellyfish.soundex(val_b) else 0.0
    else:
        raise ValueError(f"Unknown scorer: {scorer!r}")


def rapidfuzz_jaro_winkler(a: str, b: str) -> float:
    from rapidfuzz.distance import JaroWinkler
    return JaroWinkler.similarity(a, b)


def rapidfuzz_normalized_levenshtein(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.normalized_similarity(a, b)


def score_pair(
    row_a: dict,
    row_b: dict,
    fields: list[MatchkeyField],
) -> float:
    numerator = 0.0
    denominator = 0.0

    for f in fields:
        val_a = row_a.get(f.column)
        val_b = row_b.get(f.column)

        # Apply transforms
        val_a = apply_transforms(val_a, f.transforms)
        val_b = apply_transforms(val_b, f.transforms)

        field_score = score_field(val_a, val_b, f.scorer)
        if field_score is None:
            continue
        numerator += field_score * f.weight
        denominator += f.weight

    if denominator == 0:
        return 0.0
    return numerator / denominator


def find_exact_matches(
    lf: pl.LazyFrame,
    mk: MatchkeyConfig,
) -> list[tuple[int, int, float]]:
    mk_col = f"__mk_{mk.name}__"
    df = lf.collect()

    if mk_col not in df.columns:
        return []

    # Self-join on matchkey column
    pairs = []
    grouped = df.group_by(mk_col)
    for key, group in grouped:
        if group.height < 2:
            continue
        row_ids = group["__row_id__"].to_list()
        for i, j in combinations(range(len(row_ids)), 2):
            pairs.append((row_ids[i], row_ids[j], 1.0))

    return pairs


def find_fuzzy_matches(
    block_df: pl.DataFrame,
    mk: MatchkeyConfig,
) -> list[tuple[int, int, float]]:
    rows = block_df.to_dicts()
    pairs = []

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            score = score_pair(rows[i], rows[j], mk.fields)
            if score >= mk.threshold:
                pairs.append((rows[i]["__row_id__"], rows[j]["__row_id__"], score))

    return pairs
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_scorer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/scorer.py tests/test_scorer.py
git commit -m "feat: scorer with exact, fuzzy, and phonetic matching"
```

---

### Task 9: Union-Find Clustering

**Files:**
- Create: `goldenmatch/core/cluster.py`
- Create: `tests/test_cluster.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cluster.py
import pytest
from goldenmatch.core.cluster import UnionFind, build_clusters


class TestUnionFind:
    def test_basic_union(self):
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        assert uf.find(1) == uf.find(3)
        assert uf.find(1) != uf.find(4)

    def test_singleton(self):
        uf = UnionFind()
        uf.add(5)
        assert uf.find(5) == 5

    def test_clusters(self):
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(4, 5)
        uf.add(6)
        clusters = uf.get_clusters()
        assert len(clusters) == 3  # {1,2}, {3,4,5}, {6}


class TestBuildClusters:
    def test_from_pairs(self):
        pairs = [(0, 1, 1.0), (2, 3, 0.9), (3, 4, 0.85)]
        all_ids = [0, 1, 2, 3, 4, 5]
        clusters = build_clusters(pairs, all_ids, max_cluster_size=100)
        # Expect clusters: {0,1}, {2,3,4}, {5}
        multi = [c for c in clusters.values() if len(c["members"]) > 1]
        assert len(multi) == 2

    def test_oversized_cluster_flagged(self):
        pairs = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        all_ids = [0, 1, 2, 3]
        clusters = build_clusters(pairs, all_ids, max_cluster_size=2)
        oversized = [c for c in clusters.values() if c.get("oversized")]
        assert len(oversized) == 1

    def test_no_pairs_all_singletons(self):
        clusters = build_clusters([], [0, 1, 2], max_cluster_size=100)
        assert all(len(c["members"]) == 1 for c in clusters.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cluster.py -v`
Expected: FAIL

- [ ] **Step 3: Implement clustering**

```python
# goldenmatch/core/cluster.py
from __future__ import annotations
from collections import defaultdict


class UnionFind:
    def __init__(self):
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        self.add(x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def get_clusters(self) -> list[set[int]]:
        groups: dict[int, set[int]] = defaultdict(set)
        for x in self._parent:
            groups[self.find(x)].add(x)
        return list(groups.values())


def build_clusters(
    pairs: list[tuple[int, int, float]],
    all_ids: list[int],
    max_cluster_size: int = 100,
) -> dict[int, dict]:
    uf = UnionFind()
    for id_ in all_ids:
        uf.add(id_)
    for a, b, score in pairs:
        uf.union(a, b)

    raw_clusters = uf.get_clusters()
    result = {}
    for cluster_id, members in enumerate(sorted(raw_clusters, key=lambda s: min(s)), start=1):
        is_oversized = len(members) > max_cluster_size
        # Collect best scores for each pair in cluster
        pair_scores = {(a, b): score for a, b, score in pairs if a in members and b in members}
        result[cluster_id] = {
            "members": sorted(members),
            "size": len(members),
            "oversized": is_oversized,
            "pair_scores": pair_scores,
        }
    return result
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cluster.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/cluster.py tests/test_cluster.py
git commit -m "feat: Union-Find clustering with oversized cluster detection"
```

---

## Chunk 5: Golden Record Builder

### Task 10: Golden Record Builder

**Files:**
- Create: `goldenmatch/core/golden.py`
- Create: `tests/test_golden.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_golden.py
import pytest
import polars as pl
from goldenmatch.core.golden import build_golden_record, merge_field
from goldenmatch.config.schemas import GoldenFieldRule, GoldenRulesConfig


@pytest.fixture
def cluster_df():
    return pl.DataFrame({
        "__row_id__": [0, 1, 2],
        "__source__": ["crm", "legacy", "crm"],
        "email": ["john@a.com", None, "john@a.com"],
        "phone": ["555-1234", "555-1234-ext5", "555-1234"],
        "specialty": ["cardiology", "cardiology", "oncology"],
        "name": ["John Smith", "Jon Smith", "John Smith"],
        "updated_at": ["2025-01-01", "2025-06-01", "2025-03-01"],
    })


class TestMergeField:
    def test_most_complete(self):
        values = ["555-1234", "555-1234-ext5", "555-1234"]
        rule = GoldenFieldRule(strategy="most_complete")
        val, conf = merge_field(values, rule)
        assert val == "555-1234-ext5"
        assert conf == 1.0  # strictly longer

    def test_majority_vote(self):
        values = ["cardiology", "cardiology", "oncology"]
        rule = GoldenFieldRule(strategy="majority_vote")
        val, conf = merge_field(values, rule)
        assert val == "cardiology"
        assert conf == pytest.approx(2.0 / 3.0)

    def test_source_priority(self):
        values = ["john@a.com", "john@b.com"]
        sources = ["crm", "legacy"]
        rule = GoldenFieldRule(strategy="source_priority", source_priority=["legacy", "crm"])
        val, conf = merge_field(values, rule, sources=sources)
        assert val == "john@b.com"
        assert conf == 1.0  # highest priority had value

    def test_first_non_null(self):
        values = [None, "hello", "world"]
        rule = GoldenFieldRule(strategy="first_non_null")
        val, conf = merge_field(values, rule)
        assert val == "hello"
        assert conf == 0.6

    def test_all_agree(self):
        values = ["same", "same", "same"]
        rule = GoldenFieldRule(strategy="most_complete")
        val, conf = merge_field(values, rule)
        assert val == "same"
        assert conf == 1.0  # unanimous agreement

    def test_all_null(self):
        values = [None, None]
        rule = GoldenFieldRule(strategy="most_complete")
        val, conf = merge_field(values, rule)
        assert val is None
        assert conf == 0.0


class TestBuildGoldenRecord:
    def test_produces_golden(self, cluster_df):
        rules = GoldenRulesConfig(
            default=GoldenFieldRule(strategy="most_complete"),
            field_rules={
                "specialty": GoldenFieldRule(strategy="majority_vote"),
            },
        )
        golden = build_golden_record(cluster_df, rules)
        assert golden["specialty"]["value"] == "cardiology"
        assert golden["phone"]["value"] == "555-1234-ext5"
        assert "__golden_confidence__" in golden
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_golden.py -v`
Expected: FAIL

- [ ] **Step 3: Implement golden record builder**

```python
# goldenmatch/core/golden.py
from __future__ import annotations
from collections import Counter
from goldenmatch.config.schemas import GoldenFieldRule, GoldenRulesConfig
import polars as pl


def merge_field(
    values: list,
    rule: GoldenFieldRule,
    sources: list[str] | None = None,
    dates: list | None = None,
) -> tuple[any, float]:
    non_null = [v for v in values if v is not None]

    if not non_null:
        return None, 0.0

    # Check unanimous agreement
    if len(set(str(v) for v in non_null)) == 1:
        return non_null[0], 1.0

    strategy = rule.strategy

    if strategy == "most_complete":
        winner = max(non_null, key=lambda v: len(str(v)))
        lengths = [len(str(v)) for v in non_null]
        max_len = max(lengths)
        tied = lengths.count(max_len)
        confidence = 1.0 if tied == 1 else 0.7
        return winner, confidence

    elif strategy == "majority_vote":
        counts = Counter(str(v) for v in non_null)
        winner_str, count = counts.most_common(1)[0]
        # Find original value (not stringified)
        winner = next(v for v in non_null if str(v) == winner_str)
        confidence = count / len(non_null)
        return winner, confidence

    elif strategy == "source_priority":
        if not sources or not rule.source_priority:
            return non_null[0], 0.5
        for priority_idx, src in enumerate(rule.source_priority):
            for v, s in zip(values, sources):
                if s == src and v is not None:
                    confidence = max(0.1, 1.0 - priority_idx * 0.1)
                    return v, confidence
        return non_null[0], 0.3

    elif strategy == "most_recent":
        if dates is None:
            return non_null[0], 0.5
        # Pair values with dates, filter nulls
        dated = [(v, d) for v, d in zip(values, dates) if v is not None and d is not None]
        if not dated:
            return non_null[0], 0.5
        dated.sort(key=lambda x: str(x[1]), reverse=True)
        winner = dated[0][0]
        # Check if tie on date
        top_date = str(dated[0][1])
        tied = sum(1 for _, d in dated if str(d) == top_date)
        confidence = 1.0 if tied == 1 else 0.5
        return winner, confidence

    elif strategy == "first_non_null":
        return non_null[0], 0.6

    else:
        return non_null[0], 0.5


def build_golden_record(
    cluster_df: pl.DataFrame,
    rules: GoldenRulesConfig,
) -> dict:
    skip_cols = {"__row_id__", "__source__", "__block_key__"}
    skip_prefixes = ("__mk_",)

    columns = [c for c in cluster_df.columns
                if c not in skip_cols and not any(c.startswith(p) for p in skip_prefixes)]

    sources = cluster_df["__source__"].to_list() if "__source__" in cluster_df.columns else None
    golden = {}
    confidences = []

    for col in columns:
        values = cluster_df[col].to_list()
        rule = rules.field_rules.get(col, rules.default)

        dates = None
        if rule.strategy == "most_recent" and rule.date_column and rule.date_column in cluster_df.columns:
            dates = cluster_df[rule.date_column].to_list()

        value, confidence = merge_field(values, rule, sources=sources, dates=dates)
        golden[col] = {"value": value, "confidence": confidence}
        confidences.append(confidence)

    golden["__golden_confidence__"] = sum(confidences) / len(confidences) if confidences else 0.0
    return golden
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_golden.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/golden.py tests/test_golden.py
git commit -m "feat: golden record builder with per-field confidence scoring"
```

---

## Chunk 6: Output Writer & Report

### Task 11: Output Writer

**Files:**
- Create: `goldenmatch/output/__init__.py`
- Create: `goldenmatch/output/writer.py`
- Create: `goldenmatch/output/report.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_output.py
import pytest
import polars as pl
from pathlib import Path
from goldenmatch.output.writer import write_output
from goldenmatch.output.report import generate_dedupe_report, generate_match_report


class TestWriteOutput:
    def test_write_csv(self, tmp_path):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = write_output(df, tmp_path, "test", "results", "csv")
        assert path.exists()
        assert path.name == "test_results.csv"
        loaded = pl.read_csv(path)
        assert loaded.height == 2

    def test_write_parquet(self, tmp_path):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = write_output(df, tmp_path, "test", "results", "parquet")
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert loaded.height == 2

    def test_write_xlsx(self, tmp_path):
        df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = write_output(df, tmp_path, "test", "results", "xlsx")
        assert path.exists()


class TestDedupeReport:
    def test_report_contents(self):
        report = generate_dedupe_report(
            total_records=1000,
            total_clusters=150,
            cluster_sizes=[2, 2, 3, 5, 2, 2, 2, 3],
            oversized_clusters=0,
            matchkeys_used=["name_zip", "fuzzy_email"],
        )
        assert report["total_records"] == 1000
        assert report["total_clusters"] == 150
        assert report["match_rate"] == pytest.approx(0.15)


class TestMatchReport:
    def test_report_contents(self):
        report = generate_match_report(
            total_targets=500,
            matched=350,
            unmatched=150,
            scores=[0.9, 0.85, 0.92, 0.88],
        )
        assert report["hit_rate"] == pytest.approx(0.70)
        assert report["avg_score"] == pytest.approx(0.8875)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_output.py -v`
Expected: FAIL

- [ ] **Step 3: Implement output writer and report**

```python
# goldenmatch/output/__init__.py
```

```python
# goldenmatch/output/writer.py
from __future__ import annotations
from pathlib import Path
import polars as pl


def write_output(
    df: pl.DataFrame,
    directory: Path | str,
    run_name: str,
    output_type: str,
    fmt: str,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{run_name}_{output_type}.{fmt}"
    path = directory / filename

    if fmt == "csv":
        df.write_csv(path)
    elif fmt == "parquet":
        df.write_parquet(path)
    elif fmt == "xlsx":
        df.write_excel(path)
    else:
        raise ValueError(f"Unsupported output format: {fmt!r}")

    return path
```

```python
# goldenmatch/output/report.py
from __future__ import annotations
from collections import Counter


def generate_dedupe_report(
    total_records: int,
    total_clusters: int,
    cluster_sizes: list[int],
    oversized_clusters: int,
    matchkeys_used: list[str],
) -> dict:
    size_dist = Counter(cluster_sizes)
    return {
        "total_records": total_records,
        "total_clusters": total_clusters,
        "match_rate": total_clusters / total_records if total_records else 0.0,
        "cluster_size_distribution": dict(sorted(size_dist.items())),
        "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "oversized_clusters": oversized_clusters,
        "matchkeys_used": matchkeys_used,
    }


def generate_match_report(
    total_targets: int,
    matched: int,
    unmatched: int,
    scores: list[float],
) -> dict:
    return {
        "total_targets": total_targets,
        "matched": matched,
        "unmatched": unmatched,
        "hit_rate": matched / total_targets if total_targets else 0.0,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_output.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/output/ tests/test_output.py
git commit -m "feat: output writer (CSV/Parquet/Excel) and report generators"
```

---

## Chunk 7: Pipeline Orchestrator

### Task 12: Dedupe Pipeline

**Files:**
- Create: `goldenmatch/core/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_pipeline.py
import pytest
import polars as pl
from pathlib import Path
from goldenmatch.core.pipeline import run_dedupe, run_match
from goldenmatch.config.schemas import (
    GoldenMatchConfig, MatchkeyConfig, MatchkeyField,
    OutputConfig, GoldenRulesConfig, GoldenFieldRule,
)


class TestRunDedupe:
    def test_exact_dedupe_single_file(self, sample_csv, tmp_path):
        cfg = GoldenMatchConfig(
            matchkeys=[
                MatchkeyConfig(
                    name="email_key",
                    fields=[MatchkeyField(column="email", transforms=["lowercase", "strip"])],
                    comparison="exact",
                )
            ],
            output=OutputConfig(format="csv", directory=str(tmp_path), run_name="test"),
            golden_rules=GoldenRulesConfig(
                default=GoldenFieldRule(strategy="most_complete"),
            ),
        )
        results = run_dedupe(
            files=[(sample_csv, "test_source")],
            config=cfg,
            output_golden=True,
            output_clusters=True,
            output_report=True,
        )
        assert "clusters" in results
        assert "golden" in results
        assert "report" in results
        assert results["report"]["total_records"] == 5
        # john@example.com appears twice -> at least 1 cluster with size > 1
        multi = [c for c in results["clusters"].values() if c["size"] > 1]
        assert len(multi) >= 1


class TestRunMatch:
    def test_exact_match(self, sample_csv, sample_csv_b, tmp_path):
        cfg = GoldenMatchConfig(
            matchkeys=[
                MatchkeyConfig(
                    name="email_key",
                    fields=[MatchkeyField(column="email", transforms=["lowercase", "strip"])],
                    comparison="exact",
                )
            ],
            output=OutputConfig(format="csv", directory=str(tmp_path), run_name="test_match"),
        )
        results = run_match(
            target_file=(sample_csv, "targets"),
            reference_files=[(sample_csv_b, "reference")],
            config=cfg,
            output_matched=True,
            output_unmatched=True,
            output_report=True,
        )
        assert "matched" in results
        assert "unmatched" in results
        assert "report" in results
        assert results["report"]["total_targets"] == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement pipeline**

```python
# goldenmatch/core/pipeline.py
from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime
import polars as pl
from goldenmatch.config.schemas import GoldenMatchConfig
from goldenmatch.core.ingest import load_file, validate_columns
from goldenmatch.core.matchkey import compute_matchkeys, build_matchkey_expr
from goldenmatch.core.blocker import build_blocks
from goldenmatch.core.scorer import find_exact_matches, find_fuzzy_matches
from goldenmatch.core.cluster import build_clusters
from goldenmatch.core.golden import build_golden_record
from goldenmatch.output.writer import write_output
from goldenmatch.output.report import generate_dedupe_report, generate_match_report

logger = logging.getLogger(__name__)


def _add_row_ids(lf: pl.LazyFrame, offset: int = 0) -> pl.LazyFrame:
    return lf.with_row_index("__row_id__").with_columns(
        (pl.col("__row_id__") + offset).cast(pl.Int64)
    )


def _get_required_columns(config: GoldenMatchConfig) -> list[str]:
    cols = set()
    for mk in config.matchkeys:
        for f in mk.fields:
            cols.add(f.column)
    if config.blocking:
        for key in config.blocking.keys:
            for ft in key.key_fields:
                cols.add(ft.column)
    return list(cols)


def run_dedupe(
    files: list[tuple[Path | str, str]],
    config: GoldenMatchConfig,
    output_golden: bool = False,
    output_clusters: bool = False,
    output_dupes: bool = False,
    output_unique: bool = False,
    output_report: bool = False,
    across_files_only: bool = False,
) -> dict:
    logger.info("Starting dedupe pipeline")

    # 1. INGEST
    frames = []
    offset = 0
    for file_path, source_name in files:
        lf = load_file(file_path)
        required = _get_required_columns(config)
        validate_columns(lf, required)
        lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
        lf = _add_row_ids(lf, offset)
        collected = lf.collect()
        offset += collected.height
        frames.append(collected)

    combined = pl.concat(frames)
    total_records = combined.height
    logger.info(f"Loaded {total_records} records from {len(files)} file(s)")

    # 2. TRANSFORM (matchkeys)
    combined_lf = combined.lazy()
    combined_lf = compute_matchkeys(combined_lf, config.matchkeys)
    combined = combined_lf.collect()

    # 3-5. BLOCK + COMPARE + THRESHOLD
    all_pairs: list[tuple[int, int, float]] = []

    for mk in config.matchkeys:
        if mk.comparison == "exact":
            pairs = find_exact_matches(combined.lazy(), mk)
            if across_files_only:
                pairs = [
                    (a, b, s) for a, b, s in pairs
                    if combined.row(a, named=True)["__source__"] != combined.row(b, named=True)["__source__"]
                ]
            all_pairs.extend(pairs)
        else:
            # Fuzzy: requires blocking
            blocks = build_blocks(combined.lazy(), config.blocking)
            for block in blocks:
                block_df = block.df.collect()
                if across_files_only:
                    sources = block_df["__source__"].to_list()
                    if len(set(sources)) < 2:
                        continue
                pairs = find_fuzzy_matches(block_df, mk)
                if across_files_only:
                    pairs = [
                        (a, b, s) for a, b, s in pairs
                        if combined.filter(pl.col("__row_id__") == a)["__source__"][0]
                        != combined.filter(pl.col("__row_id__") == b)["__source__"][0]
                    ]
                all_pairs.extend(pairs)

    logger.info(f"Found {len(all_pairs)} matching pairs")

    # 6. CLUSTER
    all_ids = combined["__row_id__"].to_list()
    max_cluster = config.golden_rules.max_cluster_size
    clusters = build_clusters(all_pairs, all_ids, max_cluster_size=max_cluster)

    multi_clusters = {k: v for k, v in clusters.items() if v["size"] > 1}
    logger.info(f"Formed {len(multi_clusters)} clusters with duplicates")

    # 7. GOLDEN
    golden_records = []
    if output_golden or output_clusters:
        for cid, cluster_info in multi_clusters.items():
            if cluster_info["oversized"]:
                continue
            member_ids = cluster_info["members"]
            cluster_df = combined.filter(pl.col("__row_id__").is_in(member_ids))
            golden = build_golden_record(cluster_df, config.golden_rules)
            golden["__cluster_id__"] = {"value": cid, "confidence": 1.0}
            golden_records.append(golden)

    # Build result
    results = {"clusters": clusters}

    if output_golden and golden_records:
        golden_rows = []
        for g in golden_records:
            row = {k: v["value"] for k, v in g.items() if k != "__golden_confidence__"}
            row["__confidence__"] = g["__golden_confidence__"]
            golden_rows.append(row)
        results["golden"] = pl.DataFrame(golden_rows)

    if output_unique:
        singleton_ids = [c["members"][0] for c in clusters.values() if c["size"] == 1]
        results["unique"] = combined.filter(pl.col("__row_id__").is_in(singleton_ids))

    if output_dupes:
        dupe_ids = []
        for c in multi_clusters.values():
            dupe_ids.extend(c["members"])
        results["dupes"] = combined.filter(pl.col("__row_id__").is_in(dupe_ids))

    if output_report:
        cluster_sizes = [c["size"] for c in multi_clusters.values()]
        oversized_count = sum(1 for c in multi_clusters.values() if c["oversized"])
        results["report"] = generate_dedupe_report(
            total_records=total_records,
            total_clusters=len(multi_clusters),
            cluster_sizes=cluster_sizes,
            oversized_clusters=oversized_count,
            matchkeys_used=[mk.name for mk in config.matchkeys],
        )

    # 8. OUTPUT
    run_name = config.output.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output.directory)
    fmt = config.output.format

    if output_golden and "golden" in results:
        write_output(results["golden"], out_dir, run_name, "golden", fmt)
    if output_unique and "unique" in results:
        write_output(results["unique"], out_dir, run_name, "unique", fmt)
    if output_dupes and "dupes" in results:
        write_output(results["dupes"], out_dir, run_name, "dupes", fmt)

    return results


def run_match(
    target_file: tuple[Path | str, str],
    reference_files: list[tuple[Path | str, str]],
    config: GoldenMatchConfig,
    output_matched: bool = False,
    output_unmatched: bool = False,
    output_scores: bool = False,
    output_report: bool = False,
    match_mode: str = "best",
) -> dict:
    logger.info("Starting list-match pipeline")

    # Load target
    target_path, target_source = target_file
    target_lf = load_file(target_path)
    required = _get_required_columns(config)
    validate_columns(target_lf, required)
    target = _add_row_ids(target_lf).collect()
    target = target.with_columns(pl.lit(target_source).alias("__source__"))

    # Load references
    ref_frames = []
    offset = target.height
    for ref_path, ref_source in reference_files:
        ref_lf = load_file(ref_path)
        validate_columns(ref_lf, required)
        ref = _add_row_ids(ref_lf, offset).collect()
        ref = ref.with_columns(pl.lit(ref_source).alias("__source__"))
        offset += ref.height
        ref_frames.append(ref)

    all_refs = pl.concat(ref_frames)

    # Combine for matchkey computation
    combined = pl.concat([target, all_refs])
    combined_lf = compute_matchkeys(combined.lazy(), config.matchkeys)
    combined = combined_lf.collect()

    target_ids = set(target["__row_id__"].to_list())
    ref_ids = set(all_refs["__row_id__"].to_list())

    # Find matches (only cross target<->reference pairs)
    all_pairs: list[tuple[int, int, float]] = []

    for mk in config.matchkeys:
        if mk.comparison == "exact":
            pairs = find_exact_matches(combined.lazy(), mk)
            # Only keep pairs where one is target and one is reference
            pairs = [(a, b, s) for a, b, s in pairs
                     if (a in target_ids) != (b in target_ids)]
            all_pairs.extend(pairs)
        else:
            blocks = build_blocks(combined.lazy(), config.blocking)
            for block in blocks:
                block_df = block.df.collect()
                pairs = find_fuzzy_matches(block_df, mk)
                pairs = [(a, b, s) for a, b, s in pairs
                         if (a in target_ids) != (b in target_ids)]
                all_pairs.extend(pairs)

    # Normalize pairs so first ID is always the target
    normalized = []
    for a, b, s in all_pairs:
        if a in target_ids:
            normalized.append((a, b, s))
        else:
            normalized.append((b, a, s))

    # Group by target
    from collections import defaultdict
    target_matches: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for t_id, r_id, score in normalized:
        target_matches[t_id].append((r_id, score))

    # Apply match mode
    if match_mode == "best":
        for t_id in target_matches:
            target_matches[t_id] = [max(target_matches[t_id], key=lambda x: x[1])]

    matched_target_ids = set(target_matches.keys())
    unmatched_target_ids = target_ids - matched_target_ids

    results = {}
    all_scores = [s for pairs in target_matches.values() for _, s in pairs]

    if output_matched:
        matched_rows = []
        for t_id, matches in target_matches.items():
            t_row = combined.filter(pl.col("__row_id__") == t_id).to_dicts()[0]
            for r_id, score in matches:
                r_row = combined.filter(pl.col("__row_id__") == r_id).to_dicts()[0]
                row = {}
                for k, v in t_row.items():
                    if not k.startswith("__"):
                        row[f"target_{k}"] = v
                for k, v in r_row.items():
                    if not k.startswith("__"):
                        row[f"ref_{k}"] = v
                row["__match_score__"] = score
                matched_rows.append(row)
        if matched_rows:
            results["matched"] = pl.DataFrame(matched_rows)
        else:
            results["matched"] = pl.DataFrame()

    if output_unmatched:
        results["unmatched"] = combined.filter(
            pl.col("__row_id__").is_in(list(unmatched_target_ids))
        )

    if output_report:
        results["report"] = generate_match_report(
            total_targets=len(target_ids),
            matched=len(matched_target_ids),
            unmatched=len(unmatched_target_ids),
            scores=all_scores,
        )

    # Write outputs
    run_name = config.output.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output.directory)
    fmt = config.output.format

    if output_matched and "matched" in results and results["matched"].height > 0:
        write_output(results["matched"], out_dir, run_name, "matched", fmt)
    if output_unmatched and "unmatched" in results:
        write_output(results["unmatched"], out_dir, run_name, "unmatched", fmt)

    return results
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_pipeline.py -v`
Expected: All PASS

- [ ] **Step 5: Run all tests**

Run: `cd D:/show_case/goldenmatch && pytest -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: dedupe and list-match pipeline orchestrators"
```

---

## Chunk 8: CLI

### Task 13: CLI Main App + Dedupe Command

**Files:**
- Create: `goldenmatch/cli/__init__.py`
- Create: `goldenmatch/cli/main.py`
- Create: `goldenmatch/cli/dedupe.py`
- Create: `tests/test_cli_dedupe.py`

- [ ] **Step 1: Write failing CLI test**

```python
# tests/test_cli_dedupe.py
import pytest
from typer.testing import CliRunner
from goldenmatch.cli.main import app
from pathlib import Path
import yaml

runner = CliRunner()


@pytest.fixture
def simple_config(tmp_path, sample_csv):
    cfg_path = tmp_path / "goldenmatch.yaml"
    cfg_path.write_text(yaml.dump({
        "matchkeys": [{
            "name": "email_key",
            "fields": [{"column": "email", "transforms": ["lowercase"]}],
            "comparison": "exact",
        }],
        "output": {"format": "csv", "directory": str(tmp_path), "run_name": "test_run"},
    }))
    return cfg_path


class TestDedupeCommand:
    def test_basic_dedupe(self, sample_csv, simple_config, tmp_path):
        result = runner.invoke(app, [
            "dedupe", str(sample_csv),
            "--config", str(simple_config),
            "--output-all",
        ])
        assert result.exit_code == 0, result.output

    def test_missing_config(self, sample_csv):
        result = runner.invoke(app, [
            "dedupe", str(sample_csv),
            "--config", "/nonexistent/config.yaml",
        ])
        assert result.exit_code != 0

    def test_help(self):
        result = runner.invoke(app, ["dedupe", "--help"])
        assert result.exit_code == 0
        assert "dedupe" in result.output.lower() or "Dedupe" in result.output
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cli_dedupe.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI**

```python
# goldenmatch/cli/__init__.py
```

```python
# goldenmatch/cli/main.py
import typer
from goldenmatch.cli.dedupe import dedupe_cmd
from goldenmatch.cli.match import match_cmd

app = typer.Typer(
    name="goldenmatch",
    help="High-performance record deduplication and list matching CLI.",
    no_args_is_help=True,
)

app.command("dedupe")(dedupe_cmd)
app.command("match")(match_cmd)


if __name__ == "__main__":
    app()
```

```python
# goldenmatch/cli/dedupe.py
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import json

console = Console(stderr=True)


def dedupe_cmd(
    files: list[str] = typer.Argument(..., help="Input file(s) to deduplicate. Use file:source_name for source tagging."),
    config: Path = typer.Option("goldenmatch.yaml", "--config", "-c", help="Path to YAML config file"),
    output_golden: bool = typer.Option(False, "--output-golden", help="Output merged golden records"),
    output_clusters: bool = typer.Option(False, "--output-clusters", help="Output grouped clusters"),
    output_dupes: bool = typer.Option(False, "--output-dupes", help="Output duplicate records only"),
    output_unique: bool = typer.Option(False, "--output-unique", help="Output unique (non-duplicate) records"),
    output_all: bool = typer.Option(False, "--output-all", help="Output everything"),
    output_report: bool = typer.Option(False, "--output-report", help="Output summary report"),
    across_files_only: bool = typer.Option(False, "--across-files-only", help="Only match across files, not within"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Override output directory"),
    fmt: Optional[str] = typer.Option(None, "--format", help="Override output format (csv, xlsx, parquet)"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Set run name for output files"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings only"),
) -> None:
    """Find and merge duplicate records within or across files."""
    # Setup logging
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)

    from goldenmatch.config.loader import load_config
    from goldenmatch.core.pipeline import run_dedupe

    try:
        cfg = load_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise typer.Exit(1)

    # Apply CLI overrides
    if output_dir:
        cfg.output.directory = output_dir
    if fmt:
        cfg.output.format = fmt
    if run_name:
        cfg.output.run_name = run_name

    if output_all:
        output_golden = output_clusters = output_dupes = output_unique = output_report = True

    # Parse file:source pairs
    file_specs = []
    for f in files:
        if ":" in f and not (len(f) == 2 and f[1] == ":"):
            # Check it's not a Windows drive letter (e.g. C:)
            parts = f.rsplit(":", 1)
            if len(parts[0]) > 1:  # not a drive letter
                file_specs.append((Path(parts[0]), parts[1]))
            else:
                file_specs.append((Path(f), Path(f).stem))
        else:
            file_specs.append((Path(f), Path(f).stem))

    console.print(f"[bold]GoldenMatch Dedupe[/bold] - {len(file_specs)} file(s)")

    try:
        results = run_dedupe(
            files=file_specs,
            config=cfg,
            output_golden=output_golden,
            output_clusters=output_clusters,
            output_dupes=output_dupes,
            output_unique=output_unique,
            output_report=output_report,
            across_files_only=across_files_only,
        )
    except Exception as e:
        console.print(f"[red]Runtime error: {e}[/red]")
        raise typer.Exit(3)

    # Print report summary
    if output_report and "report" in results:
        r = results["report"]
        table = Table(title="Dedupe Report")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Total Records", str(r["total_records"]))
        table.add_row("Clusters Found", str(r["total_clusters"]))
        table.add_row("Match Rate", f"{r['match_rate']:.1%}")
        table.add_row("Oversized Clusters", str(r["oversized_clusters"]))
        console.print(table)

    console.print("[green]Done.[/green]")
```

- [ ] **Step 4: Create match CLI stub**

```python
# goldenmatch/cli/match.py
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

console = Console(stderr=True)


def match_cmd(
    target: str = typer.Argument(..., help="Target file to match"),
    against: list[str] = typer.Option(..., "--against", "-a", help="Reference file(s) to match against"),
    config: Path = typer.Option("goldenmatch.yaml", "--config", "-c", help="Path to YAML config file"),
    output_matched: bool = typer.Option(False, "--output-matched", help="Output matched records"),
    output_unmatched: bool = typer.Option(False, "--output-unmatched", help="Output unmatched target records"),
    output_scores: bool = typer.Option(False, "--output-scores", help="Output all scores"),
    output_all: bool = typer.Option(False, "--output-all", help="Output everything"),
    output_report: bool = typer.Option(False, "--output-report", help="Output summary report"),
    match_mode: str = typer.Option("best", "--match-mode", help="Match mode: best, all, or none"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Override output directory"),
    fmt: Optional[str] = typer.Option(None, "--format", help="Override output format"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Set run name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Compare a target list against reference file(s)."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", stream=sys.stderr)

    from goldenmatch.config.loader import load_config
    from goldenmatch.core.pipeline import run_match

    try:
        cfg = load_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Config error: {e}[/red]")
        raise typer.Exit(1)

    if output_dir:
        cfg.output.directory = output_dir
    if fmt:
        cfg.output.format = fmt
    if run_name:
        cfg.output.run_name = run_name

    if output_all:
        output_matched = output_unmatched = output_scores = output_report = True

    # Parse target
    target_path = Path(target)
    target_spec = (target_path, target_path.stem)

    # Parse references
    ref_specs = []
    for r in against:
        if ":" in r and len(r.split(":")[0]) > 1:
            parts = r.rsplit(":", 1)
            ref_specs.append((Path(parts[0]), parts[1]))
        else:
            ref_specs.append((Path(r), Path(r).stem))

    console.print(f"[bold]GoldenMatch List-Match[/bold] - {len(ref_specs)} reference(s)")

    try:
        results = run_match(
            target_file=target_spec,
            reference_files=ref_specs,
            config=cfg,
            output_matched=output_matched,
            output_unmatched=output_unmatched,
            output_scores=output_scores,
            output_report=output_report,
            match_mode=match_mode,
        )
    except Exception as e:
        console.print(f"[red]Runtime error: {e}[/red]")
        raise typer.Exit(3)

    if output_report and "report" in results:
        r = results["report"]
        table = Table(title="Match Report")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Total Targets", str(r["total_targets"]))
        table.add_row("Matched", str(r["matched"]))
        table.add_row("Unmatched", str(r["unmatched"]))
        table.add_row("Hit Rate", f"{r['hit_rate']:.1%}")
        table.add_row("Avg Score", f"{r['avg_score']:.3f}")
        console.print(table)

    console.print("[green]Done.[/green]")
```

- [ ] **Step 5: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cli_dedupe.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/cli/ tests/test_cli_dedupe.py
git commit -m "feat: CLI with dedupe and match commands via Typer"
```

---

## Chunk 9: Config Wizard & Preferences

### Task 14: Preferences Store

**Files:**
- Create: `goldenmatch/prefs/__init__.py`
- Create: `goldenmatch/prefs/store.py`
- Create: `tests/test_prefs.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prefs.py
import pytest
from pathlib import Path
from goldenmatch.prefs.store import PresetStore


@pytest.fixture
def store(tmp_path):
    return PresetStore(base_dir=tmp_path / ".goldenmatch" / "presets")


class TestPresetStore:
    def test_save_and_load(self, store, tmp_path):
        cfg_path = tmp_path / "goldenmatch.yaml"
        cfg_path.write_text("matchkeys: []")
        store.save("my_preset", cfg_path)
        loaded_path = store.load("my_preset", tmp_path / "loaded.yaml")
        assert loaded_path.exists()
        assert loaded_path.read_text() == "matchkeys: []"

    def test_list_presets(self, store, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("test: true")
        store.save("preset_a", cfg)
        store.save("preset_b", cfg)
        presets = store.list_presets()
        assert "preset_a" in presets
        assert "preset_b" in presets

    def test_delete_preset(self, store, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text("test: true")
        store.save("to_delete", cfg)
        store.delete("to_delete")
        assert "to_delete" not in store.list_presets()

    def test_load_nonexistent_raises(self, store, tmp_path):
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent", tmp_path / "out.yaml")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_prefs.py -v`
Expected: FAIL

- [ ] **Step 3: Implement store**

```python
# goldenmatch/prefs/__init__.py
```

```python
# goldenmatch/prefs/store.py
from __future__ import annotations
from pathlib import Path
import shutil


class PresetStore:
    def __init__(self, base_dir: Path | str | None = None):
        if base_dir is None:
            base_dir = Path.home() / ".goldenmatch" / "presets"
        self._dir = Path(base_dir)

    def save(self, name: str, config_path: Path | str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        dest = self._dir / f"{name}.yaml"
        shutil.copy2(str(config_path), str(dest))
        return dest

    def load(self, name: str, dest: Path | str) -> Path:
        source = self._dir / f"{name}.yaml"
        if not source.exists():
            raise FileNotFoundError(f"Preset not found: {name!r}")
        dest = Path(dest)
        shutil.copy2(str(source), str(dest))
        return dest

    def list_presets(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.yaml"))

    def delete(self, name: str) -> None:
        path = self._dir / f"{name}.yaml"
        if path.exists():
            path.unlink()

    def show(self, name: str) -> str:
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {name!r}")
        return path.read_text()
```

- [ ] **Step 4: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_prefs.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/prefs/ tests/test_prefs.py
git commit -m "feat: persistent preset store for saving/loading configs"
```

---

### Task 15: Add Config Subcommands to CLI

**Files:**
- Modify: `goldenmatch/cli/main.py`

- [ ] **Step 1: Add config commands to main.py**

Add to `goldenmatch/cli/main.py` after existing commands:

```python
# Add to goldenmatch/cli/main.py
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.syntax import Syntax
from goldenmatch.prefs.store import PresetStore

console = Console(stderr=True)
config_app = typer.Typer(name="config", help="Manage saved config presets.", no_args_is_help=True)
app.add_typer(config_app, name="config")


@config_app.command("save")
def config_save(
    name: str = typer.Argument(..., help="Preset name"),
    config: Path = typer.Option("goldenmatch.yaml", "--config", "-c"),
):
    """Save current config as a named preset."""
    store = PresetStore()
    store.save(name, config)
    console.print(f"[green]Saved preset: {name}[/green]")


@config_app.command("load")
def config_load(
    name: str = typer.Argument(..., help="Preset name"),
    dest: Path = typer.Option("goldenmatch.yaml", "--dest", "-d"),
):
    """Load a saved preset to the working directory."""
    store = PresetStore()
    try:
        store.load(name, dest)
        console.print(f"[green]Loaded preset {name!r} to {dest}[/green]")
    except FileNotFoundError:
        console.print(f"[red]Preset not found: {name!r}[/red]")
        raise typer.Exit(1)


@config_app.command("list")
def config_list():
    """List all saved presets."""
    store = PresetStore()
    presets = store.list_presets()
    if not presets:
        console.print("[dim]No saved presets.[/dim]")
        return
    for name in presets:
        console.print(f"  {name}")


@config_app.command("delete")
def config_delete(name: str = typer.Argument(...)):
    """Delete a saved preset."""
    store = PresetStore()
    store.delete(name)
    console.print(f"[green]Deleted preset: {name}[/green]")


@config_app.command("show")
def config_show(name: str = typer.Argument(...)):
    """Display contents of a saved preset."""
    store = PresetStore()
    try:
        content = store.show(name)
        syntax = Syntax(content, "yaml", theme="monokai")
        console.print(syntax)
    except FileNotFoundError:
        console.print(f"[red]Preset not found: {name!r}[/red]")
        raise typer.Exit(1)
```

- [ ] **Step 2: Run all tests**

Run: `cd D:/show_case/goldenmatch && pytest -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add goldenmatch/cli/main.py
git commit -m "feat: config subcommands for preset management"
```

---

### Task 16: Config Wizard

**Files:**
- Create: `goldenmatch/config/wizard.py`
- Create: `tests/test_wizard.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_wizard.py
import pytest
from unittest.mock import patch
from pathlib import Path
from goldenmatch.config.wizard import suggest_transforms, suggest_scorer


class TestSmartSuggestions:
    def test_name_field(self):
        transforms, scorer = suggest_transforms("last_name"), suggest_scorer("last_name")
        assert "lowercase" in transforms
        assert scorer == "jaro_winkler"

    def test_email_field(self):
        transforms, scorer = suggest_transforms("email"), suggest_scorer("email")
        assert "lowercase" in transforms
        assert scorer == "exact"

    def test_zip_field(self):
        transforms, scorer = suggest_transforms("zip_code"), suggest_scorer("zip_code")
        assert any("substring" in t for t in transforms)
        assert scorer == "exact"

    def test_phone_field(self):
        transforms, scorer = suggest_transforms("phone"), suggest_scorer("phone")
        assert "digits_only" in transforms
        assert scorer == "exact"

    def test_address_field(self):
        transforms, scorer = suggest_transforms("street_address"), suggest_scorer("street_address")
        assert "lowercase" in transforms
        assert scorer == "token_sort"

    def test_unknown_field(self):
        transforms, scorer = suggest_transforms("foobar"), suggest_scorer("foobar")
        assert transforms == ["strip"]
        assert scorer == "exact"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_wizard.py -v`
Expected: FAIL

- [ ] **Step 3: Implement wizard helpers**

```python
# goldenmatch/config/wizard.py
from __future__ import annotations
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
import polars as pl
from goldenmatch.core.ingest import load_file

console = Console()

NAME_KEYWORDS = {"name", "first", "last", "fname", "lname", "full_name"}
EMAIL_KEYWORDS = {"email", "mail", "e_mail"}
ZIP_KEYWORDS = {"zip", "postal", "postcode"}
PHONE_KEYWORDS = {"phone", "tel", "mobile", "fax"}
ADDRESS_KEYWORDS = {"address", "street", "addr", "city"}


def suggest_transforms(column_name: str) -> list[str]:
    lower = column_name.lower()
    if any(kw in lower for kw in NAME_KEYWORDS):
        return ["lowercase", "strip"]
    elif any(kw in lower for kw in EMAIL_KEYWORDS):
        return ["lowercase", "strip"]
    elif any(kw in lower for kw in ZIP_KEYWORDS):
        return ["strip", "substring:0:5"]
    elif any(kw in lower for kw in PHONE_KEYWORDS):
        return ["digits_only"]
    elif any(kw in lower for kw in ADDRESS_KEYWORDS):
        return ["lowercase", "normalize_whitespace"]
    return ["strip"]


def suggest_scorer(column_name: str) -> str:
    lower = column_name.lower()
    if any(kw in lower for kw in NAME_KEYWORDS):
        return "jaro_winkler"
    elif any(kw in lower for kw in ADDRESS_KEYWORDS):
        return "token_sort"
    return "exact"


def run_wizard(output_path: Path | None = None) -> dict:
    console.print("[bold]GoldenMatch Config Wizard[/bold]\n")

    # Step 1: Files
    file_paths = Prompt.ask("Enter file path(s), comma-separated").split(",")
    file_paths = [p.strip() for p in file_paths if p.strip()]

    # Preview first file
    if file_paths:
        first_file = load_file(file_paths[0])
        preview = first_file.head(5).collect()
        table = Table(title=f"Preview: {file_paths[0]}")
        for col in preview.columns:
            table.add_column(col)
        for row in preview.iter_rows():
            table.add_row(*[str(v) for v in row])
        console.print(table)
        columns = preview.columns
    else:
        console.print("[red]No files provided.[/red]")
        return {}

    # Step 2: Mode
    mode = Prompt.ask("Mode", choices=["dedupe", "match"], default="dedupe")

    # Step 3: Matchkey fields
    console.print(f"\n[bold]Available columns:[/bold] {', '.join(columns)}")
    selected = Prompt.ask("Select columns for matchkey (comma-separated)")
    selected_cols = [c.strip() for c in selected.split(",") if c.strip()]

    # Step 4: Comparison type
    comparison = Prompt.ask("Comparison type", choices=["exact", "weighted"], default="exact")

    fields = []
    for col in selected_cols:
        suggested_transforms = suggest_transforms(col)
        suggested_scorer = suggest_scorer(col)
        console.print(f"\n  [dim]{col}[/dim] - suggested transforms: {suggested_transforms}, scorer: {suggested_scorer}")

        use_suggested = Confirm.ask("  Use suggested settings?", default=True)
        if use_suggested:
            field = {"column": col, "transforms": suggested_transforms}
            if comparison == "weighted":
                field["scorer"] = suggested_scorer
                field["weight"] = 1.0
        else:
            transforms_str = Prompt.ask(f"  Transforms for {col} (comma-separated)", default=",".join(suggested_transforms))
            field = {"column": col, "transforms": [t.strip() for t in transforms_str.split(",")]}
            if comparison == "weighted":
                scorer = Prompt.ask(f"  Scorer for {col}", default=suggested_scorer)
                weight = float(Prompt.ask(f"  Weight for {col}", default="1.0"))
                field["scorer"] = scorer
                field["weight"] = weight
        fields.append(field)

    # Step 5: Threshold
    threshold = None
    if comparison == "weighted":
        preset = Prompt.ask("Threshold preset", choices=["strict", "moderate", "loose", "custom"], default="moderate")
        threshold = {"strict": 0.95, "moderate": 0.85, "loose": 0.70}.get(preset)
        if threshold is None:
            threshold = float(Prompt.ask("Custom threshold (0.0-1.0)"))

    matchkey = {"name": "wizard_key", "fields": fields, "comparison": comparison}
    if threshold:
        matchkey["threshold"] = threshold

    config = {"matchkeys": [matchkey]}

    # Blocking (if weighted)
    if comparison == "weighted":
        console.print("\n[bold]Blocking key[/bold] (required for fuzzy matching)")
        block_col = Prompt.ask("Column for blocking key", default=selected_cols[0])
        block_transforms = suggest_transforms(block_col)
        config["blocking"] = {
            "keys": [{"key_fields": [{"column": block_col, "transforms": block_transforms[:1] + ["substring:0:3"]}]}]
        }

    # Golden rules (dedupe only)
    if mode == "dedupe":
        if Confirm.ask("\nConfigure golden record rules?", default=True):
            default_strategy = Prompt.ask("Default merge strategy",
                                          choices=["most_complete", "majority_vote", "first_non_null"],
                                          default="most_complete")
            config["golden_rules"] = {"default": {"strategy": default_strategy}}

    # Output
    fmt = Prompt.ask("\nOutput format", choices=["csv", "xlsx", "parquet"], default="csv")
    config["output"] = {"format": fmt}

    # Save
    output_path = output_path or Path("goldenmatch.yaml")
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[green]Config saved to {output_path}[/green]")

    if Confirm.ask("Save as a named preset?", default=False):
        name = Prompt.ask("Preset name")
        from goldenmatch.prefs.store import PresetStore
        PresetStore().save(name, output_path)
        console.print(f"[green]Saved preset: {name}[/green]")

    return config
```

- [ ] **Step 4: Add init command to CLI main.py**

Add to `goldenmatch/cli/main.py`:

```python
@app.command("init")
def init_cmd(
    output: Path = typer.Option("goldenmatch.yaml", "--output", "-o", help="Output config path"),
):
    """Interactive wizard to build a GoldenMatch config file."""
    from goldenmatch.config.wizard import run_wizard
    run_wizard(output_path=output)
```

- [ ] **Step 5: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_wizard.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/config/wizard.py goldenmatch/cli/main.py tests/test_wizard.py
git commit -m "feat: interactive config wizard with smart field suggestions"
```

---

## Chunk 10: README, Final Integration Tests & Polish

### Task 17: CLI Match Tests

**Files:**
- Create: `tests/test_cli_match.py`

- [ ] **Step 1: Write match CLI test**

```python
# tests/test_cli_match.py
import pytest
from typer.testing import CliRunner
from goldenmatch.cli.main import app
import yaml

runner = CliRunner()


@pytest.fixture
def match_config(tmp_path, sample_csv, sample_csv_b):
    cfg_path = tmp_path / "goldenmatch.yaml"
    cfg_path.write_text(yaml.dump({
        "matchkeys": [{
            "name": "email_key",
            "fields": [{"column": "email", "transforms": ["lowercase"]}],
            "comparison": "exact",
        }],
        "output": {"format": "csv", "directory": str(tmp_path), "run_name": "match_test"},
    }))
    return cfg_path


class TestMatchCommand:
    def test_basic_match(self, sample_csv, sample_csv_b, match_config, tmp_path):
        result = runner.invoke(app, [
            "match", str(sample_csv),
            "--against", str(sample_csv_b),
            "--config", str(match_config),
            "--output-all",
        ])
        assert result.exit_code == 0, result.output

    def test_help(self):
        result = runner.invoke(app, ["match", "--help"])
        assert result.exit_code == 0
```

- [ ] **Step 2: Run tests**

Run: `cd D:/show_case/goldenmatch && pytest tests/test_cli_match.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_match.py
git commit -m "test: CLI match command integration tests"
```

---

### Task 18: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create a comprehensive README with:
- Project description and badges
- Installation instructions
- Quick start examples for both dedupe and match modes
- Config file reference
- Transform and scorer tables
- CLI reference
- Performance notes

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: comprehensive README with usage examples"
```

---

### Task 19: Full Test Suite & Final Push

- [ ] **Step 1: Run full test suite**

Run: `cd D:/show_case/goldenmatch && pytest -v --tb=short`
Expected: All PASS

- [ ] **Step 2: Run ruff linter**

Run: `cd D:/show_case/goldenmatch && ruff check goldenmatch/ tests/`
Expected: Clean or only minor warnings

- [ ] **Step 3: Fix any linting issues**

- [ ] **Step 4: Final commit and push**

```bash
# Switch to personal account
gh auth switch --user benzsevern
git push origin main
gh auth switch --user benzsevern-mjh
```
