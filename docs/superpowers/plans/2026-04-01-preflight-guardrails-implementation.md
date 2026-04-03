# Preflight & Guardrails Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add domain-aware auto-config, sample-based preflight with auto-downgrade, and runtime circuit breakers to prevent OOM/cost blowups on large datasets.

**Architecture:** Three layers — (1) domain detection + presets in autoconfig, (2) preflight sample runner with `RunPlan` + `SafetyPolicy` + auto-downgrade cascade, (3) `CircuitBreaker` at scoring/LLM/pipeline checkpoints. All new params default to current behavior for backwards compat.

**Tech Stack:** Python 3.12, Polars, psutil (already installed), Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-04-01-preflight-guardrails-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|---|---|
| `goldenmatch/core/domain_detector.py` | `detect_domain()` — scores column profiles against registered domain YAML packs, returns best match + confidence |
| `goldenmatch/core/preflight.py` | `RunPlan`, `ResourceProjection`, `SampleStats`, `Downgrade` dataclasses; `preflight()` sample runner; extrapolation; auto-downgrade cascade |
| `goldenmatch/core/circuit_breaker.py` | `CircuitBreaker`, `CircuitAction` classes; memory/cost/comparison checks at pipeline checkpoints |
| `tests/test_domain_detector.py` | Domain detection unit tests |
| `tests/test_preflight.py` | Preflight + downgrade unit tests |
| `tests/test_circuit_breaker.py` | Circuit breaker unit tests |

### Modified Files
| File | Change |
|---|---|
| `goldenmatch/config/schemas.py` | Add `SafetyPolicy` Pydantic model + `safety` field on `GoldenMatchConfig` |
| `goldenmatch/core/domain_registry.py` | Add `autoconfig_preset` field to `DomainRulebook`, update `load_rulebook()` |
| `goldenmatch/domains/*.yaml` (7 files) | Add `autoconfig_preset` section |
| `goldenmatch/core/autoconfig.py` | Wire domain detection + preset loading into `auto_configure_df()` |
| `goldenmatch/core/scorer.py` | Add `circuit_breaker` param to `score_blocks_parallel()` |
| `goldenmatch/core/llm_scorer.py` | Add `circuit_breaker` param to `_batch_score()` |
| `goldenmatch/core/pipeline.py` | Pass `CircuitBreaker` through `_run_dedupe_pipeline()` |
| `goldenmatch/_api.py` | Add `preflight()`, extend `dedupe_df()` with `run_preflight`/`safety`/`plan` params, add `plan` to `DedupeResult` |
| `goldenmatch/__init__.py` | Re-export `preflight`, `RunPlan`, `SafetyPolicy` |

---

## Task 1: SafetyPolicy in schemas.py

**Files:**
- Modify: `goldenmatch/config/schemas.py:407-443` (after `MemoryConfig`, before `GoldenMatchConfig`)
- Modify: `goldenmatch/config/schemas.py:410-424` (`GoldenMatchConfig` class)
- Test: `tests/test_autoconfig.py` (existing — verify no breakage)

- [ ] **Step 1: Add SafetyPolicy model to schemas.py**

Add after `MemoryConfig` (line 398), before `MatchSettingsConfig`:

```python
class SafetyPolicy(BaseModel):
    """Resource safety policy for preflight and circuit breakers."""
    max_comparisons: int = 10_000_000
    max_memory_mb: float = 4096
    max_llm_cost_usd: float = 5.00
    max_wall_time_seconds: float = 3600
    mode: Literal["conservative", "aggressive", "none"] = "conservative"
```

Add `Literal` to the `typing` import at the top of the file.

Add `safety: SafetyPolicy | None = None` field to `GoldenMatchConfig` (after `memory`).

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `pytest tests/test_autoconfig.py tests/test_autoconfig_enhanced.py -v --tb=short`
Expected: All pass (SafetyPolicy defaults to None, no impact)

- [ ] **Step 3: Commit**

```bash
git add goldenmatch/config/schemas.py
git commit -m "feat: add SafetyPolicy model to config schemas"
```

---

## Task 2: Domain Detector

**Files:**
- Create: `goldenmatch/core/domain_detector.py`
- Test: `tests/test_domain_detector.py`

- [ ] **Step 1: Write failing tests for domain detection**

Create `tests/test_domain_detector.py`:

```python
"""Tests for domain detection from column profiles."""
from __future__ import annotations

import pytest

from goldenmatch.core.autoconfig import ColumnProfile
from goldenmatch.core.domain_detector import detect_domain, DomainDetectionResult


def _make_profile(name: str, col_type: str = "string", confidence: float = 0.7) -> ColumnProfile:
    return ColumnProfile(
        name=name, dtype="String", col_type=col_type, confidence=confidence,
    )


class TestDetectDomain:
    def test_electronics_detected(self):
        profiles = [
            _make_profile("brand", "name"),
            _make_profile("model_number", "string"),
            _make_profile("sku", "identifier"),
            _make_profile("price", "numeric"),
        ]
        result = detect_domain(profiles)
        assert result.domain == "electronics"
        assert result.confidence > 0.3

    def test_people_detected(self):
        profiles = [
            _make_profile("first_name", "name"),
            _make_profile("last_name", "name"),
            _make_profile("email", "email"),
            _make_profile("phone", "phone"),
        ]
        result = detect_domain(profiles)
        assert result.domain == "people"
        assert result.confidence > 0.3

    def test_generic_fallback(self):
        profiles = [
            _make_profile("col_a", "string"),
            _make_profile("col_b", "numeric"),
            _make_profile("col_c", "string"),
        ]
        result = detect_domain(profiles)
        assert result.domain == "generic"

    def test_preset_loaded_when_available(self):
        profiles = [
            _make_profile("brand", "name"),
            _make_profile("model", "string"),
            _make_profile("sku", "identifier"),
        ]
        result = detect_domain(profiles)
        assert result.domain == "electronics"
        assert result.preset is not None
        assert "blocking" in result.preset or "scorer_overrides" in result.preset

    def test_preset_none_for_generic(self):
        profiles = [_make_profile("x", "string")]
        result = detect_domain(profiles)
        assert result.domain == "generic"
        assert result.preset is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_domain_detector.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'goldenmatch.core.domain_detector'`

- [ ] **Step 3: Implement domain_detector.py**

Create `goldenmatch/core/domain_detector.py`:

```python
"""Domain detection for auto-config — scores column profiles against registered domain packs."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from goldenmatch.core.autoconfig import ColumnProfile
from goldenmatch.core.domain_registry import discover_rulebooks, DomainRulebook

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.2  # minimum score to accept a domain match


@dataclass
class DomainDetectionResult:
    """Result of domain detection."""
    domain: str                             # domain name or "generic"
    confidence: float                       # 0.0 to 1.0
    rulebook: DomainRulebook | None = None  # matched rulebook
    preset: dict | None = None              # autoconfig_preset from YAML


def detect_domain(
    profiles: list[ColumnProfile],
    rulebooks: dict[str, DomainRulebook] | None = None,
) -> DomainDetectionResult:
    """Detect the data domain from column profiles.

    Scores each registered domain by signal overlap with column names
    and column types. Returns the best match or "generic" fallback.
    """
    if rulebooks is None:
        rulebooks = discover_rulebooks()

    if not rulebooks:
        return DomainDetectionResult(domain="generic", confidence=0.0)

    col_names_lower = [p.name.lower() for p in profiles]
    col_str = " ".join(col_names_lower)

    best_rb: DomainRulebook | None = None
    best_score = 0.0

    for rb in rulebooks.values():
        if not rb.signals:
            continue
        hits = sum(1 for s in rb.signals if s.lower() in col_str)
        score = hits / len(rb.signals)
        if score > best_score:
            best_score = score
            best_rb = rb

    if best_score < _CONFIDENCE_THRESHOLD or best_rb is None:
        logger.info("No domain detected (best score %.2f < %.2f)", best_score, _CONFIDENCE_THRESHOLD)
        return DomainDetectionResult(domain="generic", confidence=best_score)

    preset = getattr(best_rb, "autoconfig_preset", None)
    logger.info("Detected domain '%s' (confidence %.2f)", best_rb.name, best_score)
    return DomainDetectionResult(
        domain=best_rb.name,
        confidence=best_score,
        rulebook=best_rb,
        preset=preset,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_domain_detector.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/domain_detector.py tests/test_domain_detector.py
git commit -m "feat: add domain detector for auto-config"
```

---

## Task 3: DomainRulebook autoconfig_preset + YAML packs

**Files:**
- Modify: `goldenmatch/core/domain_registry.py:47-61` (`DomainRulebook` dataclass)
- Modify: `goldenmatch/core/domain_registry.py:141-160` (`load_rulebook()`)
- Modify: `goldenmatch/domains/electronics.yaml`
- Modify: `goldenmatch/domains/people.yaml`
- Modify: `goldenmatch/domains/software.yaml`
- Modify: `goldenmatch/domains/financial.yaml`
- Modify: `goldenmatch/domains/healthcare.yaml`
- Modify: `goldenmatch/domains/real_estate.yaml`
- Modify: `goldenmatch/domains/retail.yaml`
- Test: `tests/test_domain_detector.py` (preset test from Task 2)

- [ ] **Step 1: Add autoconfig_preset field to DomainRulebook**

In `goldenmatch/core/domain_registry.py`, add to the `DomainRulebook` dataclass (after `normalization` field, line 56):

```python
    autoconfig_preset: dict | None = field(default=None, repr=False)  # auto-config tuning preset from YAML
```

- [ ] **Step 2: Update load_rulebook() to read autoconfig_preset**

In `load_rulebook()` (line 150-158), add `autoconfig_preset=data.get("autoconfig_preset")` to the `DomainRulebook()` constructor:

```python
    rulebook = DomainRulebook(
        name=data.get("name", path.stem),
        signals=data.get("signals", []),
        identifier_patterns=data.get("identifier_patterns", {}),
        brand_patterns=data.get("brand_patterns", []),
        attribute_patterns=data.get("attribute_patterns", {}),
        stop_words=data.get("stop_words", []),
        normalization=data.get("normalization", {}),
        autoconfig_preset=data.get("autoconfig_preset"),
    )
```

- [ ] **Step 3: Add autoconfig_preset to electronics.yaml**

Append to `goldenmatch/domains/electronics.yaml`:

```yaml
autoconfig_preset:
  blocking:
    strategy: ann
    ann_top_k: 20
    max_block_size: 1000
    skip_oversized: true
  scorer_overrides:
    description: {scorer: ensemble, weight: 2.0}
    brand: {scorer: jaro_winkler, weight: 1.5}
    category: {scorer: token_sort, weight: 1.0}
  threshold: 0.85
  recommended_standardization:
    description: [strip, trim_whitespace]
    brand: [strip]
```

**Important:** `recommended_standardization` values must be valid standardizer names (email, phone, zip5, address, state, name_proper, name_upper, name_lower, strip, trim_whitespace), NOT matchkey transforms (lowercase, soundex, etc.). Using `lowercase` as a standardizer raises ValidationError.

- [ ] **Step 4: Add autoconfig_preset to people.yaml**

Append to `goldenmatch/domains/people.yaml`:

```yaml
autoconfig_preset:
  blocking:
    strategy: multi_pass
    max_block_size: 1000
    skip_oversized: true
  scorer_overrides:
    name: {scorer: ensemble, weight: 2.0}
    first_name: {scorer: ensemble, weight: 1.5}
    last_name: {scorer: ensemble, weight: 1.5}
    email: {scorer: exact, weight: 1.0}
    phone: {scorer: exact, weight: 0.8}
    address: {scorer: token_sort, weight: 0.8}
    zip: {scorer: exact, weight: 0.3}
  threshold: 0.80
  recommended_standardization:
    name: [strip, name_proper]
    first_name: [strip, name_proper]
    last_name: [strip, name_proper]
    email: [email]
    phone: [phone]
```

- [ ] **Step 5: Add autoconfig_preset to remaining 5 YAML packs**

For `software.yaml`:
```yaml
autoconfig_preset:
  blocking:
    strategy: ann
    ann_top_k: 20
    max_block_size: 1000
    skip_oversized: true
  scorer_overrides:
    name: {scorer: ensemble, weight: 2.0}
    version: {scorer: exact, weight: 1.0}
    platform: {scorer: exact, weight: 0.5}
  threshold: 0.80
  recommended_standardization:
    name: [strip, name_lower]
```

For `financial.yaml`:
```yaml
autoconfig_preset:
  blocking:
    strategy: static
    max_block_size: 500
    skip_oversized: true
  scorer_overrides:
    name: {scorer: ensemble, weight: 2.0}
    ticker: {scorer: exact, weight: 1.5}
    sector: {scorer: exact, weight: 0.5}
  threshold: 0.90
  recommended_standardization:
    name: [strip, name_lower]
```

For `healthcare.yaml`:
```yaml
autoconfig_preset:
  blocking:
    strategy: multi_pass
    max_block_size: 500
    skip_oversized: true
  scorer_overrides:
    name: {scorer: ensemble, weight: 2.0}
    npi: {scorer: exact, weight: 1.5}
    specialty: {scorer: token_sort, weight: 0.8}
  threshold: 0.90
  recommended_standardization:
    name: [strip, name_proper]
```

For `real_estate.yaml`:
```yaml
autoconfig_preset:
  blocking:
    strategy: multi_pass
    max_block_size: 1000
    skip_oversized: true
  scorer_overrides:
    address: {scorer: token_sort, weight: 2.0}
    city: {scorer: exact, weight: 0.5}
    zip: {scorer: exact, weight: 0.5}
  threshold: 0.85
  recommended_standardization:
    address: [strip, address]
```

For `retail.yaml`:
```yaml
autoconfig_preset:
  blocking:
    strategy: ann
    ann_top_k: 20
    max_block_size: 1000
    skip_oversized: true
  scorer_overrides:
    name: {scorer: ensemble, weight: 2.0}
    brand: {scorer: jaro_winkler, weight: 1.5}
    category: {scorer: token_sort, weight: 1.0}
  threshold: 0.85
  recommended_standardization:
    name: [strip, name_lower]
    brand: [strip]
```

- [ ] **Step 6: Run domain detector tests to verify preset loading works**

Run: `pytest tests/test_domain_detector.py -v --tb=short`
Expected: All pass (including `test_preset_loaded_when_available`)

- [ ] **Step 7: Run existing domain registry tests**

Run: `pytest tests/ -k "domain" -v --tb=short`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add goldenmatch/core/domain_registry.py goldenmatch/domains/
git commit -m "feat: add autoconfig_preset to domain YAML packs"
```

---

## Task 4: Wire Domain Detection into auto_configure_df()

**Files:**
- Modify: `goldenmatch/core/autoconfig.py:871-925` (`auto_configure_df()`)
- Modify: `goldenmatch/core/autoconfig.py:369-490` (`build_matchkeys()`)
- Test: `tests/test_autoconfig.py` (existing + new test)

- [ ] **Step 1: Write failing test for domain-aware auto-config**

Add to `tests/test_autoconfig.py` (or create new test file if preferred):

```python
def test_auto_configure_df_uses_domain_preset(tmp_path):
    """When column names match a domain, auto-config should use the preset."""
    import polars as pl
    from goldenmatch.core.autoconfig import auto_configure_df

    # Create a DataFrame with electronics-like columns
    df = pl.DataFrame({
        "brand": ["Sony", "Samsung", "LG", "Sony", "Apple"] * 100,
        "model": ["WH-1000XM5", "Galaxy S24", "OLED55", "WH-1000XM4", "iPhone 15"] * 100,
        "sku": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"] * 100,
        "price": ["299.99", "999.99", "1499.99", "249.99", "799.99"] * 100,
    })

    config = auto_configure_df(df)

    # Should have detected electronics domain and used ANN blocking
    # (the electronics preset specifies strategy: ann)
    assert config.blocking is not None
    # The domain preset should influence the config
    # At minimum, blocking should be configured
    assert config.blocking.skip_oversized is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_autoconfig.py::test_auto_configure_df_uses_domain_preset -v --tb=short`
Expected: FAIL (auto_configure_df doesn't use domain presets yet)

- [ ] **Step 3: Modify auto_configure_df() to wire in domain detection**

In `goldenmatch/core/autoconfig.py`, update `auto_configure_df()`:

```python
def auto_configure_df(df: pl.DataFrame, llm_provider: str | None = None) -> GoldenMatchConfig:
    """Auto-generate a GoldenMatchConfig from a DataFrame."""
    total_rows = df.height
    logger.info("Auto-configuring %d rows, %d columns", total_rows, len(df.columns))

    # Profile columns
    profiles = profile_columns(df, llm_provider=llm_provider)
    logger.info("Detected column types: %s", {p.name: p.col_type for p in profiles})

    # Detect domain and load preset
    from goldenmatch.core.domain_detector import detect_domain
    domain_result = detect_domain(profiles)
    preset = domain_result.preset  # dict or None

    # Build matchkeys (preset influences scorer/weight choices)
    matchkeys = build_matchkeys(profiles, df=df, preset=preset)

    # Check if embeddings are needed
    has_embeddings = any(
        f.scorer in ("embedding", "record_embedding")
        for mk in matchkeys for f in mk.fields
    )
    model = select_model(total_rows, has_embeddings)
    if model:
        for mk in matchkeys:
            for f in mk.fields:
                if f.scorer in ("embedding", "record_embedding") and not f.model:
                    f.model = model

    # Build blocking (preset may override strategy)
    has_fuzzy = any(mk.type in ("weighted", "probabilistic") for mk in matchkeys)
    if has_fuzzy:
        blocking = _build_blocking_from_preset(profiles, df, preset, llm_provider)
    else:
        blocking = None

    # Build standardization from preset
    standardization = None
    if preset and preset.get("recommended_standardization"):
        from goldenmatch.config.schemas import StandardizationConfig
        rules = {}
        for col_name, std_list in preset["recommended_standardization"].items():
            # Only apply if column exists in DataFrame
            if col_name in df.columns:
                rules[col_name] = std_list
        if rules:
            standardization = StandardizationConfig(rules=rules)

    config = GoldenMatchConfig(
        matchkeys=matchkeys,
        blocking=blocking,
        golden_rules=GoldenRulesConfig(default_strategy="most_complete"),
        standardization=standardization,
        output=OutputConfig(),
    )
    return config
```

Add helper `_build_blocking_from_preset()`:

```python
def _build_blocking_from_preset(
    profiles: list[ColumnProfile],
    df: pl.DataFrame,
    preset: dict | None,
    llm_provider: str | None,
) -> BlockingConfig:
    """Build blocking config, using domain preset as baseline if available."""
    if preset and "blocking" in preset:
        bp = preset["blocking"]
        strategy = bp.get("strategy", "multi_pass")

        # For ANN strategy, we need an ann_column — use first description/string col
        if strategy == "ann":
            text_cols = [p for p in profiles if p.col_type in ("description", "string", "name")]
            ann_col = text_cols[0].name if text_cols else None
            if ann_col:
                return BlockingConfig(
                    strategy="ann",
                    ann_column=ann_col,
                    ann_top_k=bp.get("ann_top_k", 20),
                    max_block_size=bp.get("max_block_size", 1000),
                    skip_oversized=bp.get("skip_oversized", True),
                    keys=[BlockingKeyConfig(fields=[ann_col], transforms=["lowercase"])],
                )
            # Fall through to heuristic blocking if no text column

        if strategy in ("static", "multi_pass"):
            # Use heuristic blocking but with preset's max_block_size/skip_oversized
            blocking = build_blocking(profiles, df, llm_provider=llm_provider)
            blocking.max_block_size = bp.get("max_block_size", blocking.max_block_size)
            blocking.skip_oversized = bp.get("skip_oversized", blocking.skip_oversized)
            return blocking

    # No preset — fall back to heuristic blocking
    return build_blocking(profiles, df, llm_provider=llm_provider)
```

Update `build_matchkeys()` signature to accept `preset`:

```python
def build_matchkeys(
    profiles: list[ColumnProfile],
    df: pl.DataFrame | None = None,
    preset: dict | None = None,
) -> list[MatchkeyConfig]:
```

Inside `build_matchkeys()`, when building fuzzy fields, check `preset["scorer_overrides"]` first:

```python
    scorer_overrides = preset.get("scorer_overrides", {}) if preset else {}
    preset_threshold = preset.get("threshold") if preset else None

    for p in profiles:
        if p.col_type in ("numeric", "date", "identifier"):
            continue
        if p.col_type == "description":
            description_columns.append(p)
            continue

        # Check preset override first
        override = scorer_overrides.get(p.name)
        if override:
            scorer = override.get("scorer", "token_sort")
            weight = override.get("weight", 1.0)
            transforms = ["lowercase", "strip"]
            mf = MatchkeyField(field=p.name, scorer=scorer, weight=weight, transforms=transforms)
            fuzzy_fields.append(mf)
            continue

        # Fall back to default scorer map
        scorer_info = _SCORER_MAP.get(p.col_type)
        # ... rest of existing logic
```

Use `preset_threshold` as the threshold when available:

```python
    if all_weighted:
        threshold = preset_threshold if preset_threshold else _adaptive_threshold(all_weighted)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_autoconfig.py::test_auto_configure_df_uses_domain_preset -v --tb=short`
Expected: PASS

- [ ] **Step 5: Run all autoconfig tests**

Run: `pytest tests/test_autoconfig.py tests/test_autoconfig_enhanced.py -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/core/autoconfig.py tests/test_autoconfig.py
git commit -m "feat: wire domain detection + presets into auto_configure_df"
```

---

## Task 5: Circuit Breaker

**Files:**
- Create: `goldenmatch/core/circuit_breaker.py`
- Test: `tests/test_circuit_breaker.py`

- [ ] **Step 1: Write failing circuit breaker tests**

Create `tests/test_circuit_breaker.py`:

```python
"""Tests for runtime circuit breaker."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from goldenmatch.config.schemas import SafetyPolicy
from goldenmatch.core.circuit_breaker import CircuitBreaker, CircuitAction


class TestCircuitBreakerMemory:
    def test_continue_when_memory_ok(self):
        cb = CircuitBreaker(policy=SafetyPolicy(max_memory_mb=4096))
        mock_mem = MagicMock()
        mock_mem.rss = 1_000_000_000  # ~1GB
        with patch("psutil.Process") as mock_proc:
            mock_proc.return_value.memory_info.return_value = mock_mem
            action = cb.check("scoring")
        assert action.action == "continue"

    def test_stop_when_memory_critical(self):
        cb = CircuitBreaker(policy=SafetyPolicy(max_memory_mb=1024))
        mock_mem = MagicMock()
        mock_mem.rss = 2_000_000_000  # ~2GB > 1024MB limit
        with patch("psutil.Process") as mock_proc:
            mock_proc.return_value.memory_info.return_value = mock_mem
            action = cb.check("scoring")
        assert action.action == "stop"
        assert "memory" in action.reason.lower()


class TestCircuitBreakerCost:
    def test_continue_below_budget(self):
        from goldenmatch.core.llm_budget import BudgetTracker
        from goldenmatch.config.schemas import BudgetConfig
        budget = BudgetTracker(BudgetConfig(max_cost_usd=1.00))
        cb = CircuitBreaker(
            policy=SafetyPolicy(max_llm_cost_usd=1.00),
            budget_tracker=budget,
        )
        action = cb.check("llm_scoring")
        assert action.action == "continue"

    def test_stop_when_budget_exhausted(self):
        from goldenmatch.core.llm_budget import BudgetTracker
        from goldenmatch.config.schemas import BudgetConfig
        budget = BudgetTracker(BudgetConfig(max_cost_usd=0.01))
        # Simulate spending over budget
        budget.record_usage(100000, 100000, "gpt-4o")
        cb = CircuitBreaker(
            policy=SafetyPolicy(max_llm_cost_usd=0.01),
            budget_tracker=budget,
        )
        action = cb.check("llm_scoring")
        assert action.action == "stop"


class TestCircuitBreakerComparisons:
    def test_stop_when_comparisons_exceeded(self):
        cb = CircuitBreaker(policy=SafetyPolicy(max_comparisons=1000))
        cb.comparisons_processed = 1500
        action = cb.check("scoring")
        assert action.action == "stop"
        assert "comparison" in action.reason.lower()


class TestCircuitBreakerNone:
    def test_none_circuit_breaker_is_noop(self):
        """Pipeline functions should work fine with circuit_breaker=None."""
        # This is tested implicitly — just verify the class exists
        cb = CircuitBreaker(policy=SafetyPolicy())
        assert cb is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_circuit_breaker.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement circuit_breaker.py**

Create `goldenmatch/core/circuit_breaker.py`:

```python
"""Runtime circuit breaker — monitors resource usage at pipeline checkpoints."""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

import psutil

from goldenmatch.config.schemas import SafetyPolicy

logger = logging.getLogger(__name__)


@dataclass
class CircuitAction:
    """Result of a circuit breaker check."""
    action: str  # "continue", "downgrade", "stop"
    reason: str | None = None
    component: str | None = None  # which component to downgrade


class CircuitBreakerError(Exception):
    """Raised when circuit breaker decides to stop the pipeline."""
    pass


@dataclass
class CircuitBreaker:
    """Lightweight resource monitor checked at pipeline natural boundaries."""

    policy: SafetyPolicy
    budget_tracker: object | None = None  # BudgetTracker, typed loosely to avoid circular import
    comparisons_processed: int = 0
    _warned_memory: bool = field(default=False, repr=False)
    _warned_cost: bool = field(default=False, repr=False)

    def check(self, stage: str) -> CircuitAction:
        """Check resource usage. Called at pipeline checkpoints.

        Args:
            stage: Description of current pipeline stage (for logging).

        Returns:
            CircuitAction with action="continue", "downgrade", or "stop".
        """
        # Check memory
        mem_action = self._check_memory(stage)
        if mem_action.action != "continue":
            return mem_action

        # Check comparisons
        if self.comparisons_processed > self.policy.max_comparisons:
            return CircuitAction(
                action="stop",
                reason=f"Comparison count {self.comparisons_processed:,} exceeds limit {self.policy.max_comparisons:,}",
                component="scoring",
            )

        # Check LLM cost
        if self.budget_tracker is not None:
            cost_action = self._check_cost(stage)
            if cost_action.action != "continue":
                return cost_action

        return CircuitAction(action="continue")

    def _check_memory(self, stage: str) -> CircuitAction:
        """Check process memory via psutil."""
        try:
            rss_bytes = psutil.Process().memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return CircuitAction(action="continue")

        rss_mb = rss_bytes / (1024 * 1024)

        if rss_mb > self.policy.max_memory_mb:
            # Try GC first
            gc.collect()
            rss_bytes = psutil.Process().memory_info().rss
            rss_mb = rss_bytes / (1024 * 1024)

            if rss_mb > self.policy.max_memory_mb:
                logger.error(
                    "Circuit breaker STOP at '%s': memory %.0fMB exceeds %.0fMB limit",
                    stage, rss_mb, self.policy.max_memory_mb,
                )
                return CircuitAction(
                    action="stop",
                    reason=f"Memory {rss_mb:.0f}MB exceeds {self.policy.max_memory_mb:.0f}MB limit",
                    component="memory",
                )

        # Warn at 80%
        warn_threshold = self.policy.max_memory_mb * 0.8
        if rss_mb > warn_threshold and not self._warned_memory:
            self._warned_memory = True
            logger.warning(
                "Circuit breaker WARNING at '%s': memory %.0fMB (%.0f%% of %.0fMB limit)",
                stage, rss_mb, (rss_mb / self.policy.max_memory_mb) * 100, self.policy.max_memory_mb,
            )

        return CircuitAction(action="continue")

    def _check_cost(self, stage: str) -> CircuitAction:
        """Check LLM cost against policy."""
        tracker = self.budget_tracker
        if tracker is None:
            return CircuitAction(action="continue")

        # Use budget_exhausted property from BudgetTracker
        if hasattr(tracker, "budget_exhausted") and tracker.budget_exhausted:
            logger.error(
                "Circuit breaker STOP at '%s': LLM budget exhausted", stage,
            )
            return CircuitAction(
                action="stop",
                reason="LLM budget exhausted",
                component="llm_scorer",
            )

        # Warn at 80%
        if hasattr(tracker, "budget_remaining_pct") and not self._warned_cost:
            remaining = tracker.budget_remaining_pct
            if remaining < 20:
                self._warned_cost = True
                logger.warning(
                    "Circuit breaker WARNING at '%s': LLM budget %.0f%% remaining",
                    stage, remaining,
                )

        return CircuitAction(action="continue")

    def add_comparisons(self, count: int) -> None:
        """Track comparison count."""
        self.comparisons_processed += count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_circuit_breaker.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/circuit_breaker.py tests/test_circuit_breaker.py
git commit -m "feat: add runtime circuit breaker"
```

---

## Task 6: Preflight System

**Files:**
- Create: `goldenmatch/core/preflight.py`
- Test: `tests/test_preflight.py`

- [ ] **Step 1: Write failing preflight tests**

Create `tests/test_preflight.py`:

```python
"""Tests for preflight system."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import polars as pl
import pytest

from goldenmatch.config.schemas import SafetyPolicy
from goldenmatch.core.preflight import (
    preflight,
    RunPlan,
    ResourceProjection,
    SampleStats,
    Downgrade,
    _take_sample,
    _extrapolate,
    _apply_downgrades,
    PreflightError,
)


def _make_df(n: int = 500) -> pl.DataFrame:
    """Create a simple test DataFrame."""
    return pl.DataFrame({
        "name": [f"Person {i}" for i in range(n)],
        "email": [f"p{i}@test.com" for i in range(n)],
        "zip": [f"{10000 + i % 100}" for i in range(n)],
    })


class TestSampling:
    def test_small_dataset_no_sample(self):
        df = _make_df(100)
        sample = _take_sample(df)
        assert sample.height == 100  # returns full df

    def test_default_sample_5k(self):
        df = _make_df(50000)
        sample = _take_sample(df)
        assert sample.height == 5000

    def test_one_percent_when_larger(self):
        df = _make_df(800000)
        sample = _take_sample(df)
        assert sample.height == 8000  # 1% of 800K

    def test_cap_at_10k(self):
        df = _make_df(2000000)
        sample = _take_sample(df)
        assert sample.height == 10000


class TestExtrapolation:
    def test_comparison_count_from_blocking_stats(self):
        # 3 blocks of size 10, 20, 5
        blocking_stats = {"a": 10, "b": 20, "c": 5}
        proj = _extrapolate(
            blocking_stats=blocking_stats,
            sample_comparisons=50,
            sample_peak_memory_mb=100.0,
            sample_llm_calls=5,
            sample_llm_cost=0.01,
            sample_wall_time=2.0,
            sample_rows=500,
            total_rows=5000,
        )
        # 10*9/2 + 20*19/2 + 5*4/2 = 45 + 190 + 10 = 245
        assert proj.total_comparisons == 245

    def test_risk_level_safe(self):
        proj = _extrapolate(
            blocking_stats={"a": 10},
            sample_comparisons=10,
            sample_peak_memory_mb=50.0,
            sample_llm_calls=0,
            sample_llm_cost=0.0,
            sample_wall_time=1.0,
            sample_rows=100,
            total_rows=1000,
        )
        assert proj.risk_level == "safe"


class TestDowngrades:
    def test_conservative_adds_skip_oversized(self):
        from goldenmatch.config.schemas import GoldenMatchConfig, BlockingConfig, BlockingKeyConfig, MatchkeyConfig, MatchkeyField, OutputConfig
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="test", type="weighted", threshold=0.8,
                fields=[MatchkeyField(field="name", scorer="ensemble", weight=1.0)])],
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase"])],
                skip_oversized=False,
            ),
            output=OutputConfig(),
        )
        policy = SafetyPolicy(max_comparisons=100, mode="conservative")
        proj = ResourceProjection(
            total_comparisons=1_000_000,
            estimated_memory_mb=500,
            estimated_llm_calls=0,
            estimated_llm_cost_usd=0.0,
            estimated_wall_time_seconds=60,
            risk_level="danger",
        )
        adjusted, downgrades = _apply_downgrades(config, proj, policy)
        assert adjusted.blocking.skip_oversized is True
        assert len(downgrades) > 0

    def test_safety_none_skips_downgrades(self):
        from goldenmatch.config.schemas import GoldenMatchConfig, BlockingConfig, BlockingKeyConfig, MatchkeyConfig, MatchkeyField, OutputConfig
        config = GoldenMatchConfig(
            matchkeys=[MatchkeyConfig(name="test", type="weighted", threshold=0.8,
                fields=[MatchkeyField(field="name", scorer="ensemble", weight=1.0)])],
            blocking=BlockingConfig(
                keys=[BlockingKeyConfig(fields=["name"], transforms=["lowercase"])],
            ),
            output=OutputConfig(),
        )
        policy = SafetyPolicy(mode="none")
        proj = ResourceProjection(
            total_comparisons=999_999_999,
            estimated_memory_mb=99999,
            estimated_llm_calls=0,
            estimated_llm_cost_usd=0.0,
            estimated_wall_time_seconds=99999,
            risk_level="danger",
        )
        adjusted, downgrades = _apply_downgrades(config, proj, policy)
        assert len(downgrades) == 0  # none mode skips all


class TestPreflightSmallDataset:
    def test_skips_for_small_df(self):
        df = _make_df(100)
        plan = preflight(df, safety="conservative")
        # Small datasets get a plan but with no downgrades
        assert plan is not None
        assert len(plan.downgrades) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preflight.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement preflight.py**

Create `goldenmatch/core/preflight.py`:

```python
"""Preflight system — sample-based resource estimation and auto-downgrade."""
from __future__ import annotations

import copy
import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import polars as pl
import psutil

from goldenmatch.config.schemas import (
    GoldenMatchConfig,
    SafetyPolicy,
)

logger = logging.getLogger(__name__)


class PreflightError(Exception):
    """Raised when preflight cannot find a safe configuration."""

    def __init__(self, message: str, projections: "ResourceProjection | None" = None):
        super().__init__(message)
        self.projections = projections


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class ResourceProjection:
    """Extrapolated resource estimates for the full dataset."""
    total_comparisons: int
    estimated_memory_mb: float
    estimated_llm_calls: int
    estimated_llm_cost_usd: float
    estimated_wall_time_seconds: float
    risk_level: str  # "safe", "caution", "danger"


@dataclass
class Downgrade:
    """A config modification applied to reduce resource usage."""
    component: str
    original_value: Any
    new_value: Any
    reason: str


@dataclass
class SampleStats:
    """Raw measurements from the sample run."""
    sample_size: int
    sample_comparisons: int
    sample_peak_memory_mb: float
    sample_llm_calls: int
    sample_llm_cost_usd: float
    sample_wall_time_seconds: float
    blocking_stats: dict[str, int] = field(default_factory=dict)


@dataclass
class RunPlan:
    """Complete preflight result."""
    original_config: GoldenMatchConfig
    adjusted_config: GoldenMatchConfig
    projections: ResourceProjection
    downgrades: list[Downgrade]
    sample_stats: SampleStats
    domain: str | None = None
    safety: str = "conservative"


# ── Sampling ─────────────────────────────────────────────────────────────


def _take_sample(df: pl.DataFrame, seed: int = 42) -> pl.DataFrame:
    """Take a stratified sample for preflight.

    Rules: 5K default, or 1% of dataset (whichever larger), capped at 10K.
    Returns full dataset if smaller than 5K.
    """
    n = df.height
    if n <= 5000:
        return df
    sample_size = max(5000, int(n * 0.01))
    sample_size = min(sample_size, 10000)
    return df.sample(sample_size, seed=seed)


# ── Extrapolation ────────────────────────────────────────────────────────


def _extrapolate(
    *,
    blocking_stats: dict[str, int],
    sample_comparisons: int,
    sample_peak_memory_mb: float,
    sample_llm_calls: int,
    sample_llm_cost: float,
    sample_wall_time: float,
    sample_rows: int,
    total_rows: int,
    policy: SafetyPolicy | None = None,
) -> ResourceProjection:
    """Extrapolate from sample stats + blocking stats to full dataset projections."""
    # Total comparisons from blocking stats: sum(n*(n-1)/2)
    total_comparisons = sum(
        size * (size - 1) // 2 for size in blocking_stats.values()
    )

    # Memory: base overhead + per-comparison cost
    row_ratio = total_rows / max(sample_rows, 1)
    base_overhead = sample_peak_memory_mb * 0.5  # DataFrame memory scales ~linearly
    per_comparison_mb = (
        (sample_peak_memory_mb * 0.5) / max(sample_comparisons, 1)
    )
    estimated_memory_mb = base_overhead * row_ratio + per_comparison_mb * total_comparisons

    # LLM cost scales with comparison count
    comparison_ratio = total_comparisons / max(sample_comparisons, 1)
    estimated_llm_calls = int(sample_llm_calls * comparison_ratio)
    estimated_llm_cost = sample_llm_cost * comparison_ratio

    # Wall time: rough linear extrapolation
    estimated_wall_time = sample_wall_time * comparison_ratio

    # Risk level
    p = policy or SafetyPolicy()
    if (total_comparisons > p.max_comparisons
            or estimated_memory_mb > p.max_memory_mb
            or estimated_llm_cost > p.max_llm_cost_usd):
        risk_level = "danger"
    elif (total_comparisons > p.max_comparisons * 0.5
            or estimated_memory_mb > p.max_memory_mb * 0.5
            or estimated_llm_cost > p.max_llm_cost_usd * 0.5):
        risk_level = "caution"
    else:
        risk_level = "safe"

    return ResourceProjection(
        total_comparisons=total_comparisons,
        estimated_memory_mb=estimated_memory_mb,
        estimated_llm_calls=estimated_llm_calls,
        estimated_llm_cost_usd=estimated_llm_cost,
        estimated_wall_time_seconds=estimated_wall_time,
        risk_level=risk_level,
    )


# ── Downgrade cascade ───────────────────────────────────────────────────


def _is_safe(proj: ResourceProjection, policy: SafetyPolicy) -> bool:
    """Check if projections are within policy limits."""
    return (
        proj.total_comparisons <= policy.max_comparisons
        and proj.estimated_memory_mb <= policy.max_memory_mb
        and proj.estimated_llm_cost_usd <= policy.max_llm_cost_usd
    )


def _apply_downgrades(
    config: GoldenMatchConfig,
    proj: ResourceProjection,
    policy: SafetyPolicy,
) -> tuple[GoldenMatchConfig, list[Downgrade]]:
    """Apply downgrade cascade to bring projections within safety limits.

    Returns (adjusted_config, list_of_downgrades_applied).
    """
    if policy.mode == "none":
        return config, []

    if _is_safe(proj, policy):
        return config, []

    adjusted = config.model_copy(deep=True)
    downgrades: list[Downgrade] = []

    # Conservative step 1: skip_oversized
    if adjusted.blocking and not adjusted.blocking.skip_oversized:
        adjusted.blocking.skip_oversized = True
        downgrades.append(Downgrade(
            component="blocking",
            original_value="skip_oversized=False",
            new_value="skip_oversized=True",
            reason=f"Projected {proj.total_comparisons:,} comparisons exceeds {policy.max_comparisons:,} limit",
        ))

    # Conservative step 2: halve max_block_size
    if adjusted.blocking and adjusted.blocking.max_block_size and adjusted.blocking.max_block_size > 200:
        old_val = adjusted.blocking.max_block_size
        adjusted.blocking.max_block_size = old_val // 2
        downgrades.append(Downgrade(
            component="blocking",
            original_value=f"max_block_size={old_val}",
            new_value=f"max_block_size={old_val // 2}",
            reason="Reducing block size to lower comparison count",
        ))

    # Conservative step 3: switch to ANN blocking
    if adjusted.blocking and adjusted.blocking.strategy not in ("ann", "ann_pairs"):
        old_strategy = adjusted.blocking.strategy
        adjusted.blocking.strategy = "ann"
        # Pick first text column for ann_column
        if not adjusted.blocking.ann_column:
            mk_fields = [f.field for mk in adjusted.get_matchkeys() for f in mk.fields if f.field]
            if mk_fields:
                adjusted.blocking.ann_column = mk_fields[0]
                adjusted.blocking.ann_top_k = 20
        downgrades.append(Downgrade(
            component="blocking",
            original_value=f"strategy={old_strategy}",
            new_value="strategy=ann",
            reason="ANN blocking reduces comparison count via approximate nearest neighbors",
        ))

    # Conservative step 4: switch backend to duckdb
    if proj.estimated_memory_mb > policy.max_memory_mb and adjusted.backend != "duckdb":
        try:
            import importlib
            importlib.import_module("duckdb")
            adjusted.backend = "duckdb"
            downgrades.append(Downgrade(
                component="backend",
                original_value=f"backend={config.backend}",
                new_value="backend=duckdb",
                reason=f"Projected memory {proj.estimated_memory_mb:.0f}MB exceeds {policy.max_memory_mb:.0f}MB limit",
            ))
        except ImportError:
            logger.info("DuckDB not installed, skipping backend downgrade")

    # Aggressive mode: additional downgrades
    if policy.mode == "aggressive":
        # Step 6: tighten LLM candidate band
        if adjusted.llm_scorer and adjusted.llm_scorer.enabled:
            old_lo = adjusted.llm_scorer.candidate_lo
            adjusted.llm_scorer.candidate_lo = min(old_lo + 0.05, adjusted.llm_scorer.candidate_hi - 0.05)
            downgrades.append(Downgrade(
                component="llm_scorer",
                original_value=f"candidate_lo={old_lo}",
                new_value=f"candidate_lo={adjusted.llm_scorer.candidate_lo}",
                reason="Narrowing LLM scoring band to reduce API calls",
            ))

        # Step 7: raise match threshold
        for mk in adjusted.get_matchkeys():
            if mk.type == "weighted" and mk.threshold:
                old_t = mk.threshold
                mk.threshold = min(old_t + 0.05, 0.99)
                downgrades.append(Downgrade(
                    component="threshold",
                    original_value=f"threshold={old_t}",
                    new_value=f"threshold={mk.threshold}",
                    reason="Raising threshold to reduce candidate pairs",
                ))

    return adjusted, downgrades


# ── Main preflight function ─────────────────────────────────────────────


def _get_blocking_stats(df: pl.DataFrame, config: GoldenMatchConfig) -> dict[str, int]:
    """Get block sizes from the full dataset (cheap group_by)."""
    if not config.blocking or not config.blocking.keys:
        return {}

    block_fields = config.blocking.keys[0].fields
    try:
        stats_df = df.group_by(block_fields).len()
        sizes = stats_df.get_column("len").to_list()
        keys = [str(i) for i in range(len(sizes))]
        return dict(zip(keys, sizes))
    except Exception as e:
        logger.warning("Failed to compute blocking stats: %s", e)
        return {}


def preflight(
    df: pl.DataFrame,
    *,
    config: GoldenMatchConfig | None = None,
    safety: str = "conservative",
) -> RunPlan:
    """Run preflight analysis on a DataFrame.

    Takes a sample, runs the pipeline on it, measures resources,
    extrapolates to full dataset, and auto-downgrades if needed.

    Args:
        df: Full DataFrame to analyze.
        config: Config to test. If None, auto-generates via auto_configure_df().
        safety: Safety mode — "conservative", "aggressive", or "none".

    Returns:
        RunPlan with projections, adjusted config, and any downgrades.
    """
    policy = SafetyPolicy(mode=safety)

    # Build config if not provided
    if config is None:
        from goldenmatch.core.autoconfig import auto_configure_df
        from goldenmatch._api import _detect_llm_provider
        provider = _detect_llm_provider()
        config = auto_configure_df(df, llm_provider=provider)

    original_config = config.model_copy(deep=True)

    # Detect domain (for RunPlan metadata)
    domain_name = None
    try:
        from goldenmatch.core.autoconfig import profile_columns
        from goldenmatch.core.domain_detector import detect_domain
        profiles = profile_columns(df)
        domain_result = detect_domain(profiles)
        domain_name = domain_result.domain
    except Exception:
        pass

    # For small datasets, skip the sample run
    if df.height <= 10_000:
        blocking_stats = _get_blocking_stats(df, config)
        total_comps = sum(s * (s - 1) // 2 for s in blocking_stats.values())
        proj = ResourceProjection(
            total_comparisons=total_comps,
            estimated_memory_mb=df.estimated_size() / (1024 * 1024) * 2,
            estimated_llm_calls=0,
            estimated_llm_cost_usd=0.0,
            estimated_wall_time_seconds=0.0,
            risk_level="safe",
        )
        return RunPlan(
            original_config=original_config,
            adjusted_config=config,
            projections=proj,
            downgrades=[],
            sample_stats=SampleStats(
                sample_size=df.height,
                sample_comparisons=total_comps,
                sample_peak_memory_mb=0.0,
                sample_llm_calls=0,
                sample_llm_cost_usd=0.0,
                sample_wall_time_seconds=0.0,
                blocking_stats=blocking_stats,
            ),
            domain=domain_name,
            safety=safety,
        )

    # Take sample
    sample = _take_sample(df)
    logger.info("Preflight: sampling %d/%d rows", sample.height, df.height)

    # Measure sample run
    gc.collect()
    rss_before = psutil.Process().memory_info().rss
    t0 = time.monotonic()

    try:
        from goldenmatch.core.pipeline import run_dedupe_df
        sample_result = run_dedupe_df(sample, config, source_name="__preflight__")
    except Exception as e:
        logger.warning("Preflight sample run failed: %s", e)
        # Return a conservative plan
        return RunPlan(
            original_config=original_config,
            adjusted_config=config,
            projections=ResourceProjection(
                total_comparisons=0, estimated_memory_mb=0, estimated_llm_calls=0,
                estimated_llm_cost_usd=0, estimated_wall_time_seconds=0, risk_level="caution",
            ),
            downgrades=[],
            sample_stats=SampleStats(
                sample_size=sample.height, sample_comparisons=0,
                sample_peak_memory_mb=0, sample_llm_calls=0,
                sample_llm_cost_usd=0, sample_wall_time_seconds=0,
            ),
            domain=domain_name,
            safety=safety,
        )

    t1 = time.monotonic()
    rss_after = psutil.Process().memory_info().rss
    peak_memory_mb = max(0, (rss_after - rss_before)) / (1024 * 1024)

    # Count sample comparisons from result
    sample_comparisons = len(sample_result.get("clusters", {}))
    # Better estimate: count pairs
    sample_pairs = 0
    for cluster in sample_result.get("clusters", {}).values():
        members = cluster.get("members", [])
        n = len(members)
        sample_pairs += n * (n - 1) // 2
    sample_comparisons = max(sample_pairs, 1)

    # Get blocking stats from FULL dataset (cheap)
    blocking_stats = _get_blocking_stats(df, config)

    sample_stats = SampleStats(
        sample_size=sample.height,
        sample_comparisons=sample_comparisons,
        sample_peak_memory_mb=peak_memory_mb,
        sample_llm_calls=0,  # TODO: wire BudgetTracker stats
        sample_llm_cost_usd=0.0,
        sample_wall_time_seconds=t1 - t0,
        blocking_stats=blocking_stats,
    )

    # Extrapolate
    proj = _extrapolate(
        blocking_stats=blocking_stats,
        sample_comparisons=sample_comparisons,
        sample_peak_memory_mb=peak_memory_mb,
        sample_llm_calls=0,
        sample_llm_cost=0.0,
        sample_wall_time=t1 - t0,
        sample_rows=sample.height,
        total_rows=df.height,
        policy=policy,
    )

    logger.info(
        "Preflight projections: %d comparisons, %.0fMB memory, $%.2f LLM cost, risk=%s",
        proj.total_comparisons, proj.estimated_memory_mb,
        proj.estimated_llm_cost_usd, proj.risk_level,
    )

    # Apply downgrades if needed
    adjusted_config, downgrades = _apply_downgrades(config, proj, policy)

    for d in downgrades:
        logger.warning("Preflight downgrade: %s -> %s (%s)", d.original_value, d.new_value, d.reason)

    return RunPlan(
        original_config=original_config,
        adjusted_config=adjusted_config,
        projections=proj,
        downgrades=downgrades,
        sample_stats=sample_stats,
        domain=domain_name,
        safety=safety,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preflight.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add goldenmatch/core/preflight.py tests/test_preflight.py
git commit -m "feat: add preflight system with sample runner and auto-downgrade"
```

---

## Task 7: Wire Circuit Breaker into scorer.py

**Files:**
- Modify: `goldenmatch/core/scorer.py:531-610` (`score_blocks_parallel()`)
- Test: `tests/test_circuit_breaker.py` (existing tests cover CircuitBreaker; pipeline integration tested in Task 10)

- [ ] **Step 1: Add circuit_breaker param to score_blocks_parallel()**

In `goldenmatch/core/scorer.py`, modify `score_blocks_parallel()` signature:

```python
def score_blocks_parallel(
    blocks: list,
    mk: MatchkeyConfig,
    matched_pairs: set[tuple[int, int]],
    max_workers: int = 4,
    across_files_only: bool = False,
    source_lookup: dict[int, str] | None = None,
    target_ids: set[int] | None = None,
    circuit_breaker: object | None = None,  # CircuitBreaker, typed loosely to avoid import
) -> list[tuple[int, int, float]]:
```

In the thread pool loop, after `all_pairs.extend(pairs)` and `matched_pairs.add(...)` block (around line 604-607), add circuit breaker check:

```python
            completed += 1
            # Circuit breaker check
            if circuit_breaker is not None:
                circuit_breaker.add_comparisons(len(pairs))
                action = circuit_breaker.check(f"scoring block {completed}/{total_blocks}")
                if action.action == "stop":
                    logger.warning(
                        "Circuit breaker stopped scoring at block %d/%d: %s",
                        completed, total_blocks, action.reason,
                    )
                    # Cancel remaining futures
                    for f in future_to_idx:
                        f.cancel()
                    break
```

Also add the same check in the small-block sequential path (the `if len(blocks) <= 2:` branch, around line 562-578):

```python
        for block in blocks:
            pairs = _score_one_block(...)
            # ... existing target_ids filter ...
            all_pairs.extend(pairs)
            for a, b, s in pairs:
                matched_pairs.add((min(a, b), max(a, b)))
            # Circuit breaker check
            if circuit_breaker is not None:
                circuit_breaker.add_comparisons(len(pairs))
                action = circuit_breaker.check("scoring (sequential)")
                if action.action == "stop":
                    logger.warning("Circuit breaker stopped sequential scoring: %s", action.reason)
                    break
```

- [ ] **Step 2: Run existing scorer tests to verify no breakage**

Run: `pytest tests/test_scorer.py -v --tb=short`
Expected: All pass (circuit_breaker defaults to None, no impact)

- [ ] **Step 3: Commit**

```bash
git add goldenmatch/core/scorer.py
git commit -m "feat: add circuit breaker checkpoint to score_blocks_parallel"
```

---

## Task 8: Wire Circuit Breaker into llm_scorer.py

**Files:**
- Modify: `goldenmatch/core/llm_scorer.py:249-310` (`_batch_score()`)

- [ ] **Step 1: Add circuit_breaker param to _batch_score()**

In `goldenmatch/core/llm_scorer.py`, modify `_batch_score()` signature:

```python
def _batch_score(
    candidate_indices: list[int],
    pairs: list[tuple[int, int, float]],
    row_lookup: dict[int, dict],
    cols: list[str],
    provider: str,
    api_key: str,
    model: str,
    batch_size: int,
    budget: "BudgetTracker | None" = None,
    max_workers: int = 5,
    circuit_breaker: object | None = None,
) -> dict[int, bool]:
```

After the thread pool completes each batch (where results are collected), add circuit breaker check. Find where batch results are merged back (after the `executor` block), and add:

```python
    # After processing each batch group
    if circuit_breaker is not None:
        action = circuit_breaker.check("llm_scoring")
        if action.action == "stop":
            logger.warning("Circuit breaker stopped LLM scoring: %s", action.reason)
            # Return what we have so far
            return results
```

Also find `llm_score_pairs()` and pass `circuit_breaker` through to `_batch_score()`. Add it as an optional param to `llm_score_pairs()`:

```python
def llm_score_pairs(
    pairs, df, config=None, circuit_breaker=None,
):
```

And pass it through: `_batch_score(..., circuit_breaker=circuit_breaker)`.

- [ ] **Step 2: Run existing LLM scorer tests**

Run: `pytest tests/test_llm_scorer.py -v --tb=short`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add goldenmatch/core/llm_scorer.py
git commit -m "feat: add circuit breaker checkpoint to LLM batch scoring"
```

---

## Task 9: Wire Circuit Breaker into pipeline.py

**Files:**
- Modify: `goldenmatch/core/pipeline.py:178-191` (`_run_dedupe_pipeline()` signature)
- Modify: `goldenmatch/core/pipeline.py:305-317` (fuzzy scoring call)
- Modify: `goldenmatch/core/pipeline.py:360-368` (LLM scoring call)
- Modify: `goldenmatch/core/pipeline.py:509-533` (`run_dedupe_df()` signature)

- [ ] **Step 1: Add circuit_breaker param to _run_dedupe_pipeline()**

Add `circuit_breaker: object | None = None` to the function signature.

Pass it through to `block_scorer()` call (line ~312):

```python
            pairs = block_scorer(
                blocks, mk, matched_pairs,
                across_files_only=across_files_only,
                source_lookup=source_lookup if across_files_only else None,
                circuit_breaker=circuit_breaker,
            )
```

Pass it through to `llm_score_pairs()` call (line ~367):

```python
            all_pairs = llm_score_pairs(all_pairs, collected_df, config=config.llm_scorer,
                                         circuit_breaker=circuit_breaker)
```

Add checkpoint between pipeline stages (after scoring, before clustering, around line 389):

```python
    # Circuit breaker checkpoint between stages
    if circuit_breaker is not None:
        action = circuit_breaker.check("post_scoring")
        if action.action == "stop":
            from goldenmatch.core.circuit_breaker import CircuitBreakerError
            raise CircuitBreakerError(f"Stopped after scoring: {action.reason}")
```

- [ ] **Step 2: Add circuit_breaker param to run_dedupe_df()**

Add `circuit_breaker: object | None = None` to `run_dedupe_df()` signature and pass it to `_run_dedupe_pipeline()`:

```python
def run_dedupe_df(
    df: pl.DataFrame,
    config: GoldenMatchConfig,
    source_name: str = "dataframe",
    output_golden: bool = False,
    output_clusters: bool = False,
    output_dupes: bool = False,
    output_unique: bool = False,
    output_report: bool = False,
    circuit_breaker: object | None = None,
) -> dict:
    """Run dedupe pipeline on a DataFrame directly (no file I/O)."""
    df = df.cast({col: pl.Utf8 for col in df.columns if not col.startswith("__")})
    matchkeys = config.get_matchkeys()
    lf = df.lazy()
    required = _get_required_columns(config)
    validate_columns(lf, required)
    lf = lf.with_columns(pl.lit(source_name).alias("__source__"))
    lf = _add_row_ids(lf, offset=0)
    combined_lf = lf.collect().lazy()
    return _run_dedupe_pipeline(combined_lf, config, matchkeys,
                                output_golden, output_clusters,
                                output_dupes, output_unique, output_report,
                                circuit_breaker=circuit_breaker)
```

**Note:** `_run_match_pipeline` is intentionally NOT wired with `circuit_breaker` in this version. Preflight + circuit breakers are scoped to the dedupe pipeline for v1. TODO: wire into match pipeline in a follow-up.

- [ ] **Step 3: Run existing pipeline tests**

Run: `pytest tests/test_pipeline.py -v --tb=short`
Expected: All pass (circuit_breaker defaults to None)

- [ ] **Step 4: Commit**

```bash
git add goldenmatch/core/pipeline.py
git commit -m "feat: wire circuit breaker through dedupe pipeline"
```

---

## Task 10: Public API — preflight() + extended dedupe_df()

**Files:**
- Modify: `goldenmatch/_api.py:263-324` (`dedupe_df()`)
- Modify: `goldenmatch/_api.py:32-50` (`DedupeResult`)
- Modify: `goldenmatch/__init__.py:33-46` (re-exports)

- [ ] **Step 1: Add plan field to DedupeResult**

In `goldenmatch/_api.py`, add to `DedupeResult` (after `config` field):

```python
    plan: Any = None  # RunPlan from preflight, if run
```

- [ ] **Step 2: Add preflight() to _api.py**

Add after the `dedupe()` function:

```python
def preflight(
    df: pl.DataFrame,
    *,
    config: Any | None = None,
    safety: str = "conservative",
) -> Any:
    """Run preflight analysis — sample the dataset, measure resources, project costs.

    Returns a RunPlan with projections, adjusted config, and any downgrades.
    Use the plan with dedupe_df(df, plan=plan) to skip re-running preflight.

    Args:
        df: Polars DataFrame to analyze.
        config: GoldenMatchConfig or None for auto-config.
        safety: "conservative", "aggressive", or "none".

    Returns:
        RunPlan with projections, adjusted config, and downgrades.
    """
    from goldenmatch.core.preflight import preflight as _preflight
    if isinstance(config, str):
        config = load_config(config)
    return _preflight(df, config=config, safety=safety)
```

- [ ] **Step 3: Extend dedupe_df() with run_preflight/safety/plan params**

Modify `dedupe_df()` signature:

```python
def dedupe_df(
    df: pl.DataFrame,
    *,
    config: Any | None = None,
    exact: list[str] | None = None,
    fuzzy: dict[str, float] | None = None,
    blocking: list[str] | None = None,
    threshold: float | None = None,
    llm_scorer: bool = False,
    backend: str | None = None,
    source_name: str = "dataframe",
    run_preflight: bool = True,
    safety: str = "conservative",
    plan: Any | None = None,
) -> DedupeResult:
```

Add preflight logic before the pipeline call:

```python
    from goldenmatch.core.pipeline import run_dedupe_df as _run_dedupe_df

    run_plan = plan
    circuit_breaker = None

    if plan is not None:
        # User provided a pre-computed RunPlan
        config = plan.adjusted_config
        run_plan = plan
    else:
        if isinstance(config, str):
            config = load_config(config)
        elif config is None:
            if exact or fuzzy:
                config = _build_config(exact, fuzzy, blocking, threshold, llm_scorer, backend)
            else:
                from goldenmatch.core.autoconfig import auto_configure_df
                provider = _detect_llm_provider() if llm_scorer else None
                config = auto_configure_df(df, llm_provider=provider)

        # Apply overrides
        if backend and hasattr(config, "backend"):
            config.backend = backend
        if llm_scorer and hasattr(config, "llm_scorer"):
            from goldenmatch.config.schemas import LLMScorerConfig
            config.llm_scorer = LLMScorerConfig(enabled=True)

        # Run preflight for large datasets
        if run_preflight and safety != "none" and df.height > 10_000:
            from goldenmatch.core.preflight import preflight as _preflight
            run_plan = _preflight(df, config=config, safety=safety)
            config = run_plan.adjusted_config

    # Create circuit breaker from safety policy
    if safety != "none":
        from goldenmatch.config.schemas import SafetyPolicy
        from goldenmatch.core.circuit_breaker import CircuitBreaker
        circuit_breaker = CircuitBreaker(policy=SafetyPolicy(mode=safety))

    result = _run_dedupe_df(df, config, source_name=source_name,
                             circuit_breaker=circuit_breaker)

    return DedupeResult(
        golden=result.get("golden"),
        clusters=result.get("clusters", {}),
        dupes=result.get("dupes"),
        unique=result.get("unique"),
        stats=_extract_stats(result),
        scored_pairs=_extract_pairs(result),
        config=config,
        plan=run_plan,
    )
```

- [ ] **Step 4: Add re-exports to __init__.py**

In `goldenmatch/__init__.py`, add to the `_api` imports:

```python
from goldenmatch._api import (
    ...
    preflight,
)
```

Add a new section for preflight types:

```python
# ── Preflight & Safety ──────────────────────────────────────────────────
from goldenmatch.core.preflight import RunPlan, ResourceProjection, SampleStats, Downgrade, PreflightError
from goldenmatch.config.schemas import SafetyPolicy
from goldenmatch.core.circuit_breaker import CircuitBreaker, CircuitBreakerError
```

- [ ] **Step 5: Run existing API tests**

Run: `pytest tests/ -k "api or dedupe_df" -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add goldenmatch/_api.py goldenmatch/__init__.py
git commit -m "feat: expose preflight() API, extend dedupe_df with run_preflight/safety/plan"
```

---

## Task 11: Integration Test

**Files:**
- Test: `tests/test_preflight.py` (add integration test)

- [ ] **Step 1: Add integration test**

Add to `tests/test_preflight.py`:

```python
class TestIntegration:
    def test_dedupe_df_with_preflight(self):
        """End-to-end: dedupe_df runs preflight for large datasets."""
        import goldenmatch as gm

        # Create 15K rows of people-like data
        n = 15000
        df = pl.DataFrame({
            "first_name": [f"Person{i % 500}" for i in range(n)],
            "last_name": [f"Last{i % 300}" for i in range(n)],
            "email": [f"p{i}@test.com" for i in range(n)],
            "zip": [f"{10000 + i % 100}" for i in range(n)],
        })

        result = gm.dedupe_df(df, run_preflight=True, safety="conservative")

        # Should have a plan since n > 10K
        assert result.plan is not None
        assert result.plan.projections is not None
        assert result.plan.safety == "conservative"

    def test_dedupe_df_skip_preflight(self):
        """run_preflight=False skips preflight."""
        import goldenmatch as gm

        df = pl.DataFrame({
            "name": [f"Person{i}" for i in range(15000)],
            "email": [f"p{i}@test.com" for i in range(15000)],
        })

        result = gm.dedupe_df(df, run_preflight=False)
        assert result.plan is None

    def test_dedupe_df_safety_none(self):
        """safety='none' skips preflight even for large datasets."""
        import goldenmatch as gm

        df = pl.DataFrame({
            "name": [f"Person{i}" for i in range(15000)],
            "email": [f"p{i}@test.com" for i in range(15000)],
        })

        result = gm.dedupe_df(df, safety="none")
        assert result.plan is None

    def test_standalone_preflight(self):
        """gm.preflight() returns a RunPlan."""
        import goldenmatch as gm

        df = pl.DataFrame({
            "first_name": [f"Person{i}" for i in range(500)],
            "last_name": [f"Last{i}" for i in range(500)],
            "email": [f"p{i}@test.com" for i in range(500)],
        })

        plan = gm.preflight(df)
        assert isinstance(plan, gm.RunPlan)
        assert plan.projections is not None
        assert plan.domain is not None  # should detect "people" or "generic"
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_preflight.py::TestIntegration -v --tb=short`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_preflight.py
git commit -m "test: add preflight integration tests"
```

---

## Task 12: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `pytest --tb=short -q`
Expected: All 1173+ tests pass (plus new tests), 0 failures

- [ ] **Step 2: Verify no import errors in new modules**

Run: `python -c "import goldenmatch; print(goldenmatch.preflight, goldenmatch.RunPlan, goldenmatch.SafetyPolicy, goldenmatch.CircuitBreaker)"`
Expected: Prints the function/class references without errors

- [ ] **Step 3: Final commit if any fixups needed**

Only if there were test failures to fix in steps 1-2.
